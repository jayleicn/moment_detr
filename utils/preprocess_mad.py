import copy
import json, os
import numpy as np
from tqdm import tqdm
import h5py
import pickle
from pathlib import Path

def run():
    root = '/nfs/data3/goldhofer/mad_dataset'
    clip_frame_features = get_video_feats(root)
    annotation_paths = [f'{root}/annotations/MAD_val.json',
                        f'{root}/annotations/MAD_test.json', f'{root}/annotations/MAD_train.json']

    # annotation_paths = [f'{root}/annotations/MAD_val.json', f'{root}/annotations/MAD_test.json']

    rng = np.random.default_rng(42)
    save_path = f'{root}/clip_frame_features_transformed_dense/'
    Path(save_path).mkdir(parents=True, exist_ok=True)
    id_tracker = []

    fps = 5
    video_length_seconds = 150

    for annotation_path in tqdm(annotation_paths):
        annotated_data = json.load(open(annotation_path, 'r'))
        data_length = len(annotated_data)
        meta_cache = {}
        cnt = 0
        mad_transformed = []
        for k in tqdm(list(annotated_data.keys())):
            try:
                assert k not in id_tracker, f'duplicated id: {k}'
                id_tracker.append(k)

                lowest_clip = int(annotated_data[k]["ext_timestamps"][0])
                highest_clip = int(annotated_data[k]["ext_timestamps"][1])

                if lowest_clip % 2 != 0:
                   lowest_clip -= 1
                if highest_clip % 2 != 0:
                    highest_clip += 1
                # lowest_clip2, highest_clip2 = math.floor(annotated_data[k]["ext_timestamps"][0] / 2.) * 2, \
                #                             math.ceil(annotated_data[k]["ext_timestamps"][1] / 2.) * 2


                if highest_clip > annotated_data[k]["movie_duration"]:
                    print(
                        f'highest clip higher than movie duration, adjusted from {highest_clip} to {int(np.floor(annotated_data[k]["movie_duration"]))}')
                    highest_clip = int(np.floor(annotated_data[k]["movie_duration"]))

                if highest_clip > lowest_clip:

                    meta = {"qid": k,
                            "query": annotated_data[k]["sentence"],
                            "duration": annotated_data[k]["movie_duration"],
                            "vid": k,
                            "relevant_windows": [[lowest_clip, highest_clip]],
                            #TODO: may not need these 2
                            "relevant_clip_ids": [i for i in
                                                  range(int(lowest_clip / 2), int(highest_clip / 2))],
                            "saliency_scores": [[0, 0, 0] for _ in
                                                range(int(lowest_clip / 2), int(highest_clip / 2))]}

                    old_meta = copy.deepcopy(meta)
                    sliced_frame_features, meta = slice_window(clip_frame_features[annotated_data[k]["movie"]], meta,
                                                               rng, fps, video_length_seconds)
                    meta_cache = log_meta(old_meta, meta, annotated_data[k], meta_cache)

                    if check_dict(meta, annotated_data[k]):
                        mad_transformed.append(meta)
                        np.savez(f'{save_path}{k}.npz', features=sliced_frame_features)
            except Exception as e:
                print(f'Exception:\n{e}')

        save_annotations(annotation_path, root, mad_transformed)
        save_meta(meta_cache, root, annotation_path)
        print(f'# Clip duration for {annotation_path} probably zero: {cnt}')


def get_video_feats(root):
    return h5py.File(f'{root}/CLIP_L14_frames_features_5fps.h5', 'r')


def check_dict(meta, annotated_data):
    try:
        assert len(meta["relevant_windows"][0]) != 0
        assert len(meta["saliency_scores"]) != 0, "saliency scores are zero"
        assert meta["relevant_windows"][0][0] < meta["relevant_windows"][0][1]
        assert 0 <= meta["relevant_windows"][0][0] < meta["relevant_windows"][0][
            1], f'relevant window: {meta["relevant_windows"][0]}\noriginal data:\n{annotated_data}'
        return True
    except Exception as e:
        print(e)
        return False


def save_annotations(annotation_path, root, mad_transformed):
    save_path = root + "/" + annotation_path.split("/")[-1].split(".")[0] + "_transformed.json"
    with open(save_path, "w") as f:
        f.write("\n".join([json.dumps(e) for e in mad_transformed]))
    print(f'saved to: {save_path}')


def slice_window(frame_features, meta, rng, fps, max_v_l):
    f_max_v_l = max_v_l * fps  # qv samples at 0.5FPS, MAD at 5 FPS
    f_relevant_windows = np.multiply(meta["relevant_windows"][0], fps)  # relevant windows seconds -> frames @ 5 FPS
    f_window_length = f_relevant_windows[1] - f_relevant_windows[0]

    # assert f_max_v_l > f_window_length, "moment longer then max sample length"

    random_window_offset = rng.random()
    assert f_max_v_l > f_window_length, f"window length ({f_window_length}) longer than max ({f_max_v_l}), discarding datapoint"
    f_left_offset = int(np.floor(random_window_offset * (f_max_v_l - f_window_length)))
    f_right_offset = int(f_max_v_l - f_window_length - f_left_offset)

    f_right_offset, f_left_offset = check_offsets(f_right_offset,
                                                  f_left_offset,
                                                  f_relevant_windows,
                                                  f_max_v_l,
                                                  frame_features)

    window = frame_features[
             int(f_relevant_windows[0] - f_left_offset):int(f_relevant_windows[1] + f_right_offset),
             :]

    meta = adjust_meta(meta,
                       f_left_offset,
                       f_window_length,
                       fps)
    # window = rng.choice(window, size=max_v_l, replace=False, axis=0, shuffle=False) #TODO: check random
    return window, meta


def check_offsets(f_right_offset, f_left_offset, f_relevant_windows, f_max_v_l, frame_features):
    if f_relevant_windows[0] - f_left_offset < 0:
        f_right_offset += f_left_offset
        f_left_offset = 0
    if f_relevant_windows[1] + f_right_offset > frame_features.shape[0]:
        f_left_offset += f_right_offset
        f_right_offset = 0

    return f_right_offset, f_left_offset


def log_meta(old_meta, new_meta, annotated_data, meta_cache):
    meta_cache[old_meta["qid"]] = {"old_meta": old_meta, "new_meta": new_meta, "annotation": annotated_data}
    return meta_cache


def save_meta(meta_cache, root, annotation_path):
    print(f'saving meta log with length: {len(meta_cache)}')
    with open(f'{root}/{annotation_path.split("/")[-1].split(".")[0]}_meta_log.pkl', 'wb') as f:
        pickle.dump(meta_cache, f)
    print(f'saved metadata cache to: {root}/{annotation_path.split("/")[-1].split(".")[0]}_meta_log.pkl')


def adjust_meta(meta, f_left_offset, f_window_length, fps):
    window_start = int(np.floor(f_left_offset / fps)) if int(np.floor(f_left_offset / fps)) % 2 == 0 else int(
        np.floor(f_left_offset / fps)) - 1
    new_window = [[window_start, int(window_start + f_window_length / fps)]]
    new_clip_ids = [i for i in range(int(new_window[0][0] / 2), int(new_window[0][1] / 2))]

    meta["relevant_windows"] = new_window
    meta["relevant_clip_ids"] = new_clip_ids
    # meta.pop("duration")
    return meta


if __name__ == "__main__":
    run()
