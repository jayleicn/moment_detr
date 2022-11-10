import json, os
import numpy as np
from tqdm import tqdm
import h5py
import pickle

root = '/nfs/data3/goldhofer/mad_dataset'
mad_transformed_train = []
mad_transformed_val = []

for mad_path in [f'{root}/annotations/MAD_val.json', f'{root}/annotations/MAD_train.json']:
    train_data = json.load(open(mad_path, 'r'))
    moment_detr_dict = {}

    cnt = 0
    for k in tqdm(list(train_data.keys())):
        lowest_clip = int(np.floor(round(train_data[k]["ext_timestamps"][0])))
        highest_clip = int(np.ceil(round(train_data[k]["ext_timestamps"][1])))
        if lowest_clip % 2 != 0:
            lowest_clip -= 1
        if highest_clip % 2 != 0:
            highest_clip += 1

        if highest_clip > train_data[k]["movie_duration"]:
            highest_clip = int(np.floor(train_data[k]["movie_duration"]))

        if highest_clip - lowest_clip > 0:

            moment_detr_dict = {"qid": k + "_" + train_data[k]["movie"],
                                "query": train_data[k]["sentence"],
                                "duration": train_data[k]["movie_duration"],
                                "vid": train_data[k]["movie"],
                                "relevant_windows": [[lowest_clip, highest_clip]],
                                "relevant_clip_ids": [i for i in
                                                      range(int(lowest_clip / 2), int(highest_clip / 2))],
                                "saliency_scores": [[0, 0, 0] for i in
                                                    range(int(lowest_clip / 2), int(highest_clip / 2))]}

            if len(moment_detr_dict["saliency_scores"]) != 0:
                assert len(moment_detr_dict["relevant_windows"][0]) != 0
                assert len(moment_detr_dict["saliency_scores"]) != 0, "saliency scores are zero"
                assert moment_detr_dict["relevant_windows"][0][0] < moment_detr_dict["relevant_windows"][0][1]
                assert 0 <= moment_detr_dict["relevant_windows"][0][0] < moment_detr_dict["relevant_windows"][0][
                    1], f'relevant window: {moment_detr_dict["relevant_windows"][0]}'

            if "train" in mad_path:
                mad_transformed_train.append(moment_detr_dict)

            else:
                mad_transformed_val.append(moment_detr_dict)

        else:
            cnt += 1
    print(f'# Clip duration for {mad_path} probably zero: {cnt}')

with open(f'{root}/annotations/MAD_train_transformed.json', "w") as f:
    f.write("\n".join([json.dumps(e) for e in mad_transformed_train]))
with open(f'{root}/annotations/MAD_val_transformed.json', "w") as f:
    f.write("\n".join([json.dumps(e) for e in mad_transformed_val]))

print(f'length train: {len(mad_transformed_train)}, length val: {len(mad_transformed_val)}')


def get_video_feats():
    return h5py.File(f'/nfs/data3/goldhofer/mad_dataset/CLIP_frames_features_5fps.h5', 'r')


def _mad_get_video_feat_by_vid(vid, meta, video_feats):
    video_feat_cache = np.array(video_feats[vid]).astype(np.float32)

    rng = np.random.default_rng(42)
    video_feat_cache, meta = _slice_window(video_feat_cache, meta, rng)
    return torch.from_numpy(video_feat_cache), meta  # (Lv, D)


def _slice_window(frame_features, meta, rng):
    max_v_l = 75
    f_max_v_l = max_v_l * 5  # qv samples at 0.5FPS, MAD at 5 FPS

    f_relevant_windows = np.multiply(meta["relevant_windows"][0], 5)  # relevant windows seconds -> frames @ 5 FPS
    f_window_length = f_relevant_windows[1] - f_relevant_windows[0]

    # assert f_max_v_l > f_window_length, "moment longer then max sample length"

    random_window_offset = rng.random()
    f_left_offset = int(np.floor(random_window_offset * (f_max_v_l - f_window_length)))
    f_right_offset = int(f_max_v_l - f_window_length - f_left_offset)

    f_right_offset, f_left_offset = _check_offsets(f_right_offset,
                                                   f_left_offset,
                                                   f_relevant_windows,
                                                   f_max_v_l,
                                                   frame_features)

    window = frame_features[
             int(f_relevant_windows[0] - f_left_offset):int(f_relevant_windows[1] + f_right_offset),
             :]

    # old_meta = copy.deepcopy(meta)
    meta = _adjust_meta(meta,
                        f_left_offset,
                        f_window_length)
    # self._log_meta(old_meta, meta)
    window = rng.choice(window, size=max_v_l, replace=False, axis=0, shuffle=False)
    return window, meta


def _check_offsets(f_right_offset, f_left_offset, f_relevant_windows, f_max_v_l, frame_features):
    if f_relevant_windows[0] - f_left_offset < 0:
        f_right_offset += f_left_offset
        f_left_offset = 0
    if f_relevant_windows[1] + f_right_offset > frame_features.shape[0]:
        f_left_offset += f_right_offset
        f_right_offset = 0

    # assert int(f_relevant_windows[1] + f_right_offset) - int(
    #    f_relevant_windows[0] - f_left_offset) == f_max_v_l, "Window lengths dont match"

    # assert f_relevant_windows[1] + f_right_offset != f_relevant_windows[0] - f_left_offset, "Zero window length"

    return f_right_offset, f_left_offset


def _log_meta(old_meta, new_meta):
    meta_log[old_meta["qid"]] = {"old_meta": old_meta, "new_meta": new_meta}
    if len(meta_log) % 100 == 0:
        print(f'saving meta log with length: {len(meta_log)}')
        with open('data/meta_log.pkl', 'wb') as f:
            pickle.dump(meta_log, f)
    return


def _adjust_meta(meta, f_left_offset, f_window_length):
    window_start = int(np.floor(f_left_offset / 5)) if int(np.floor(f_left_offset / 5)) % 2 == 0 else int(
        np.floor(f_left_offset / 5)) - 1
    new_window = [[window_start, int(window_start + f_window_length / 5)]]
    new_clip_ids = [i for i in range(int(new_window[0][0] / 2), int(new_window[0][1] / 2))]

    # assert new_window[1] - new_window[0] == meta["relevant_windows"][1] - meta["relevant_windows"][
    #    0], "adjusting windows error"
    # assert len(meta["saliency_scores"]) == len(meta["relevant_clip_ids"]), "adjusting windows saliency error"
    # assert meta["relevant_windows"][0] / 2 == meta["relevant_clip_ids"][0], "adjusting windows clip id error"

    meta["relevant_windows"] = new_window
    meta["relevant_clip_ids"] = new_clip_ids
    # meta.pop("duration")
    return meta
