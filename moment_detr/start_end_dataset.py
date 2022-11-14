import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import copy
import h5py
import pickle
import random
import logging
from os.path import join, exists
from utils.basic_utils import load_jsonl, l2_normalize_np_array
from utils.tensor_utils import pad_sequences_1d
from moment_detr.span_utils import span_xx_to_cxw

logger = logging.getLogger(__name__)


class StartEndDataset(Dataset):
    Q_FEAT_TYPES = ["pooler_output", "last_hidden_state"]
    """One line in data loaded from data_path."
    {
      "qid": 7803,
      "query": "Man in gray top walks from outside to inside.",
      "duration": 150,
      "vid": "RoripwjYFp8_360.0_510.0",
      "relevant_clip_ids": [13, 14, 15, 16, 17],
      "relevant_windows": [[26, 36]]
    }
    """

    def __init__(self, dset_name, data_path, v_feat_dirs, q_feat_dir,
                 q_feat_type="last_hidden_state",
                 max_q_l=32, max_v_l=75, data_ratio=1.0, ctx_mode="video",
                 normalize_v=True, normalize_t=True, load_labels=True,
                 clip_len=2, max_windows=5, span_loss_type="l1", txt_drop_ratio=0,sampling_fps=0.5):
        self.dset_name = dset_name
        self.data_path = data_path
        self.data_ratio = data_ratio
        self.v_feat_dirs = v_feat_dirs \
            if isinstance(v_feat_dirs, list) else [v_feat_dirs]
        self.q_feat_dir = q_feat_dir
        self.q_feat_type = q_feat_type
        self.max_q_l = max_q_l
        self.max_v_l = max_v_l
        self.ctx_mode = ctx_mode
        self.use_tef = "tef" in ctx_mode
        self.use_video = "video" in ctx_mode
        self.normalize_t = normalize_t
        self.normalize_v = normalize_v
        self.load_labels = load_labels
        self.clip_len = clip_len
        self.max_windows = max_windows  # maximum number of windows to use as labels
        self.span_loss_type = span_loss_type
        self.txt_drop_ratio = txt_drop_ratio
        self.video_feat_cache = None
        self.vid_cache = None
        self.video_feats = self.get_video_feats()
        self.lang_feats = self.get_lang_feats()
        self.q_feat_cache = None
        self.qid_cache = None
        self.rng = np.random.default_rng(42)
        self.sampling_fps=sampling_fps
        self.meta_log = {}
        if "val" in data_path or "test" in data_path:
            assert txt_drop_ratio == 0

        # checks
        assert q_feat_type in self.Q_FEAT_TYPES

        # data
        self.data = self.load_data()

    def load_data(self):
        datalist = load_jsonl(self.data_path)
        if self.data_ratio != 1:
            n_examples = int(len(datalist) * self.data_ratio)
            datalist = datalist[:n_examples]
            logger.info("Using {}% of the data: {} examples"
                        .format(self.data_ratio * 100, n_examples))
        return datalist

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        meta = self.data[index]

        model_inputs = dict()
        if "mad_dataset" in self.data_path:
            model_inputs["query_feat"] = self._mad_get_query_feat_by_qid(meta["qid"])  # (Dq, ) or (Lq, Dq)
        else:
            model_inputs["query_feat"] = self._get_query_feat_by_qid(meta["qid"])  # (Dq, ) or (Lq, Dq)
        if self.use_video:
            #if "mad_dataset" in self.data_path:
            #    model_inputs["video_feat"], meta = self._mad_get_video_feat_by_vid(meta["vid"],
            #                                                                       meta)  # (Lv, Dv)
            model_inputs["video_feat"] = self._get_video_feat_by_vid(meta["vid"], self.sampling_fps)  # (Lv, Dv)

            ctx_l = len(model_inputs["video_feat"])
        else:
            ctx_l = self.max_v_l

        #assert self.vid_cache == self.qid_cache, "vid and qid dont match"

        if self.use_tef:
            tef_st = torch.arange(0, ctx_l, 1.0) / ctx_l
            tef_ed = tef_st + 1.0 / ctx_l
            tef = torch.stack([tef_st, tef_ed], dim=1)  # (Lv, 2)
            if self.use_video:
                model_inputs["video_feat"] = torch.cat(
                    [model_inputs["video_feat"], tef], dim=1)  # (Lv, Dv+2)
            else:
                model_inputs["video_feat"] = tef

        if self.load_labels:
            model_inputs["span_labels"] = self.get_span_labels(meta["relevant_windows"], ctx_l)  # (#windows, 2)
            if "subs_train" not in self.data_path:
                model_inputs["saliency_pos_labels"], model_inputs["saliency_neg_labels"] = \
                    self.get_saliency_labels(meta["relevant_clip_ids"], meta["saliency_scores"], ctx_l)
            else:
                model_inputs["saliency_pos_labels"], model_inputs["saliency_neg_labels"] = \
                    self.get_saliency_labels_sub_as_query(meta["relevant_windows"][0], ctx_l)  # only one gt
        return dict(meta=meta, model_inputs=model_inputs)

    def get_saliency_labels_sub_as_query(self, gt_window, ctx_l, max_n=2):
        gt_st = int(gt_window[0] / self.clip_len)
        gt_ed = max(0, min(int(gt_window[1] / self.clip_len), ctx_l) - 1)
        if gt_st > gt_ed:
            gt_st = gt_ed

        if gt_st != gt_ed:
            pos_clip_indices = random.sample(range(gt_st, gt_ed + 1), k=max_n)
        else:
            pos_clip_indices = [gt_st, gt_st]

        neg_pool = list(range(0, gt_st)) + list(range(gt_ed + 1, ctx_l))
        neg_clip_indices = random.sample(neg_pool, k=max_n)
        return pos_clip_indices, neg_clip_indices

    def get_saliency_labels(self, rel_clip_ids, scores, ctx_l, max_n=1, add_easy_negative=True):
        """Sum the scores from the three annotations, then take the two clips with the
        maximum scores as positive, and two with the minimum scores as negative.
        Args:
            rel_clip_ids: list(int), list of relevant clip ids
            scores: list([anno1_score, anno2_score, anno3_score]),
            ctx_l: int
            max_n: int, #clips to use as positive and negative, for easy and hard negative, respectively.
            add_easy_negative: bool, if True, sample eay negative outside the relevant_clip_ids.
        """
        # indices inside rel_clip_ids
        scores = np.array(scores)  # (#rel_clips, 3)
        agg_scores = np.sum(scores, 1)  # (#rel_clips, )
        sort_indices = np.argsort(agg_scores)  # increasing

        # indices in the whole video
        # the min(_, ctx_l-1) here is incorrect, but should not cause
        # much troubles since this should be rarely used.
        hard_pos_clip_indices = [min(rel_clip_ids[idx], ctx_l - 1) for idx in sort_indices[-max_n:]]
        hard_neg_clip_indices = [min(rel_clip_ids[idx], ctx_l - 1) for idx in sort_indices[:max_n]]
        easy_pos_clip_indices = []
        easy_neg_clip_indices = []
        if add_easy_negative:
            easy_neg_pool = list(set(range(ctx_l)) - set(rel_clip_ids))
            if len(easy_neg_pool) >= max_n:
                easy_pos_clip_indices = random.sample(rel_clip_ids, k=max_n)
                easy_neg_clip_indices = random.sample(easy_neg_pool, k=max_n)
            else:  # copy the hard ones
                easy_pos_clip_indices = hard_pos_clip_indices
                easy_neg_clip_indices = hard_neg_clip_indices

        pos_clip_indices = hard_pos_clip_indices + easy_pos_clip_indices
        neg_clip_indices = hard_neg_clip_indices + easy_neg_clip_indices
        return pos_clip_indices, neg_clip_indices

    def get_span_labels(self, windows, ctx_l):
        """
        windows: list([st, ed]) in seconds. E.g. [[26, 36]], corresponding st_ed clip_indices [[13, 17]] (inclusive)
            Note a maximum of `self.max_windows` windows are used.
        returns Tensor of shape (#windows, 2), each row is [center, width] normalized by video length
        """
        if len(windows) > self.max_windows:
            random.shuffle(windows)
            windows = windows[:self.max_windows]
        if self.span_loss_type == "l1":
            windows = torch.Tensor(windows) / (ctx_l * self.clip_len)  # normalized windows in xx
            windows = span_xx_to_cxw(windows)  # normalized windows in cxw
        elif self.span_loss_type == "ce":
            windows = torch.Tensor([
                [int(w[0] / self.clip_len), min(int(w[1] / self.clip_len), ctx_l) - 1]
                for w in windows]).long()  # inclusive
        else:
            raise NotImplementedError
        return windows

    def _get_query_feat_by_qid(self, qid):
        q_feat_path = join(self.q_feat_dir, f"qid{qid}.npz")
        q_feat = np.load(q_feat_path)[self.q_feat_type].astype(np.float32)
        if self.q_feat_type == "last_hidden_state":
            q_feat = q_feat[:self.max_q_l]
        if self.normalize_t:
            q_feat = l2_normalize_np_array(q_feat)
        if self.txt_drop_ratio > 0:
            q_feat = self.random_drop_rows(q_feat)
        return torch.from_numpy(q_feat)  # (D, ) or (Lq, D)

    def get_video_feats(self):
        return h5py.File(f'/nfs/data3/goldhofer/mad_dataset/CLIP_B32_frames_features_5fps.h5', 'r')

    def get_lang_feats(self):
        return h5py.File(f'/nfs/data3/goldhofer/mad_dataset/CLIP_B32_language_tokens_features.h5', 'r')

    def _mad_get_query_feat_by_qid(self, qid):

        qid = qid.split("_")[0]
        self.q_feat_cache = np.array(self.lang_feats[qid]).astype(np.float32)
        self.qid_cache = qid

        if self.q_feat_type == "last_hidden_state":
            self.q_feat_cache = self.q_feat_cache[:self.max_q_l]
            if self.q_feat_cache.shape[0] > self.max_q_l:
                logger.info(
                    f'Query feature length ({self.q_feat_cache.shape[0]}) is longer than set max query length ({self.max_q_l})"')
        if self.normalize_t:
            self.q_feat_cache = l2_normalize_np_array(self.q_feat_cache)
        if self.txt_drop_ratio > 0:
            self.q_feat_cache = self.random_drop_rows(self.q_feat_cache)
        return torch.from_numpy(self.q_feat_cache)  # (D, ) or (Lq, D)

    def random_drop_rows(self, embeddings):
        """randomly mask num_drop rows in embeddings to be zero.
        Args:
            embeddings: np.ndarray (L, D)
        """
        num_drop_rows = round(len(embeddings) * self.txt_drop_ratio)
        if num_drop_rows > 0:
            row_indices = np.random.choice(
                len(embeddings), size=num_drop_rows, replace=False)
            embeddings[row_indices] = 0
        return embeddings

    def _get_video_feat_by_vid(self, vid, sampling_fps=0.5, train=False):
        v_feat_list = []
        for _feat_dir in self.v_feat_dirs:
            _feat_path = join(_feat_dir, f"{vid}.npz")
            #_feat_path = '/nfs/data3/goldhofer/mad_dataset/clip_frame_features_transformed_dense/0.npz'
            if train:
                _feat = np.load(_feat_path)["features"][:self.max_v_l].astype(np.float32)
            else:
                _feat = np.load(_feat_path)["features"][::int(5 / sampling_fps)].astype(np.float32)
            if self.normalize_v:
                _feat = l2_normalize_np_array(_feat)
            v_feat_list.append(_feat)
        # some features are slightly longer than the others
        min_len = min([len(e) for e in v_feat_list])
        v_feat_list = [e[:min_len] for e in v_feat_list]
        v_feat = np.concatenate(v_feat_list, axis=1)
        return torch.from_numpy(v_feat)  # (Lv, D)

    def _mad_get_video_feat_by_vid(self, vid, meta):
        if vid != self.vid_cache:
            self.video_feat_cache = np.array(self.video_feats[vid]).astype(np.float32)
            self.vid_cache = vid

            if self.normalize_v:
                self.video_feat_cache = l2_normalize_np_array(self.video_feat_cache)

        video_feat_cache, meta = self._slice_window(self.video_feat_cache, meta)
        return torch.from_numpy(video_feat_cache), meta  # (Lv, D)

    def _slice_window(self, frame_features, meta):
        f_max_v_l = self.max_v_l * 5  # qv samples at 0.5FPS, MAD at 5 FPS

        f_relevant_windows = np.multiply(meta["relevant_windows"][0], 5)  # relevant windows seconds -> frames @ 5 FPS
        f_window_length = f_relevant_windows[1] - f_relevant_windows[0]

        #assert f_max_v_l > f_window_length, "moment longer then max sample length"

        random_window_offset = self.rng.random()
        f_left_offset = int(np.floor(random_window_offset * (f_max_v_l - f_window_length)))
        f_right_offset = int(f_max_v_l - f_window_length - f_left_offset)

        f_right_offset, f_left_offset = self._check_offsets(f_right_offset,
                                                            f_left_offset,
                                                            f_relevant_windows,
                                                            f_max_v_l,
                                                            frame_features)

        window = frame_features[
                 int(f_relevant_windows[0] - f_left_offset):int(f_relevant_windows[1] + f_right_offset),
                 :]

        #old_meta = copy.deepcopy(meta)
        meta = self._adjust_meta(meta,
                                 f_left_offset,
                                 f_window_length)
        #self._log_meta(old_meta, meta)
        window = self.rng.choice(window, size=self.max_v_l, replace=False, axis=0, shuffle=False)
        return window, meta

    def _check_offsets(self, f_right_offset, f_left_offset, f_relevant_windows, f_max_v_l, frame_features):
        if f_relevant_windows[0] - f_left_offset < 0:
            f_right_offset += f_left_offset
            f_left_offset = 0
        if f_relevant_windows[1] + f_right_offset > frame_features.shape[0]:
            f_left_offset += f_right_offset
            f_right_offset = 0

        #assert int(f_relevant_windows[1] + f_right_offset) - int(
        #    f_relevant_windows[0] - f_left_offset) == f_max_v_l, "Window lengths dont match"

        #assert f_relevant_windows[1] + f_right_offset != f_relevant_windows[0] - f_left_offset, "Zero window length"

        return f_right_offset, f_left_offset

    def _log_meta(self, old_meta, new_meta):
        self.meta_log[old_meta["qid"]] = {"old_meta": old_meta, "new_meta": new_meta}
        if len(self.meta_log) % 100 == 0:
            print(f'saving meta log with length: {len(self.meta_log)}')
            with open('data/meta_log.pkl', 'wb') as f:
                pickle.dump(self.meta_log, f)
        return

    def _adjust_meta(self, meta, f_left_offset, f_window_length):
        window_start = int(np.floor(f_left_offset / 5)) if int(np.floor(f_left_offset / 5)) % 2 == 0 else int(
            np.floor(f_left_offset / 5)) - 1
        new_window = [[window_start, int(window_start + f_window_length / 5)]]
        new_clip_ids = [i for i in range(int(new_window[0][0] / 2), int(new_window[0][1] / 2))]

        #assert new_window[1] - new_window[0] == meta["relevant_windows"][1] - meta["relevant_windows"][
        #    0], "adjusting windows error"
        #assert len(meta["saliency_scores"]) == len(meta["relevant_clip_ids"]), "adjusting windows saliency error"
        #assert meta["relevant_windows"][0] / 2 == meta["relevant_clip_ids"][0], "adjusting windows clip id error"

        meta["relevant_windows"] = new_window
        meta["relevant_clip_ids"] = new_clip_ids
        #meta.pop("duration")
        return meta


def start_end_collate(batch):
    batch_meta = [e["meta"] for e in batch]  # seems no need to collate ?

    model_inputs_keys = batch[0]["model_inputs"].keys()
    batched_data = dict()
    for k in model_inputs_keys:
        if k == "span_labels":
            batched_data[k] = [dict(spans=e["model_inputs"]["span_labels"]) for e in batch]
            continue
        if k in ["saliency_pos_labels", "saliency_neg_labels"]:
            batched_data[k] = torch.LongTensor([e["model_inputs"][k] for e in batch])
            continue
        batched_data[k] = pad_sequences_1d(
            [e["model_inputs"][k] for e in batch], dtype=torch.float32, fixed_length=None)
    return batch_meta, batched_data


def prepare_batch_inputs(batched_model_inputs, device, non_blocking=False):
    model_inputs = dict(
        src_txt=batched_model_inputs["query_feat"][0].to(device, non_blocking=non_blocking),
        src_txt_mask=batched_model_inputs["query_feat"][1].to(device, non_blocking=non_blocking),
        src_vid=batched_model_inputs["video_feat"][0].to(device, non_blocking=non_blocking),
        src_vid_mask=batched_model_inputs["video_feat"][1].to(device, non_blocking=non_blocking),
    )
    targets = {}
    if "span_labels" in batched_model_inputs:
        targets["span_labels"] = [
            dict(spans=e["spans"].to(device, non_blocking=non_blocking))
            for e in batched_model_inputs["span_labels"]
        ]
    if "saliency_pos_labels" in batched_model_inputs:
        for name in ["saliency_pos_labels", "saliency_neg_labels"]:
            targets[name] = batched_model_inputs[name].to(device, non_blocking=non_blocking)

    targets = None if len(targets) == 0 else targets
    return model_inputs, targets
