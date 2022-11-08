import json, os
import numpy as np
from tqdm import tqdm

root = '/nfs/data3/goldhofer/mad_dataset'
mad_transformed_train = []
mad_transformed_val = []

for mad_path in [ f'{root}/annotations/MAD_val.json',f'{root}/annotations/MAD_train.json']:
    train_data = json.load(open(mad_path, 'r'))
    moment_detr_dict = {}

    cnt = 0
    for k in tqdm(list(train_data.keys())[0:100]):
        lowest_clip = int(np.floor(round(train_data[k]["timestamps"][0])))
        highest_clip = int(np.ceil(round(train_data[k]["timestamps"][1])))
        if lowest_clip % 2 != 0:
            lowest_clip -= 1
        if highest_clip % 2 != 0:
            highest_clip += 1

        if highest_clip > train_data[k]["movie_duration"]:
            highest_clip = int(np.floor(train_data[k]["movie_duration"]))

        if highest_clip - lowest_clip > 0:

            if "train" in mad_path:
                moment_detr_dict = {"qid": k + "_" + train_data[k]["movie"],
                                    "query": train_data[k]["sentence"],
                                    "duration": train_data[k]["movie_duration"],
                                    "vid": train_data[k]["movie"],
                                    "relevant_windows": [lowest_clip, highest_clip],
                                    "relevant_clip_ids": [i for i in
                                                          range(int(lowest_clip / 2), int(highest_clip / 2))],
                                    "saliency_scores": [[0, 0, 0] for i in
                                                        range(int(lowest_clip / 2), int(highest_clip / 2))]}
                mad_transformed_train.append(moment_detr_dict)
            else:
                moment_detr_dict = {"qid": k + "_" + train_data[k]["movie"].split("_")[0],
                                    "query": train_data[k]["sentence"],
                                    "duration": train_data[k]["movie_duration"],
                                    "vid": train_data[k]["movie"],
                                    "relevant_windows": [lowest_clip, highest_clip],
                                    "relevant_clip_ids": [i for i in
                                                          range(int(lowest_clip / 2), int(highest_clip / 2))],
                                    "saliency_scores": [[0, 0, 0] for i in
                                                        range(int(lowest_clip / 2), int(highest_clip / 2))]}
                mad_transformed_val.append(moment_detr_dict)
        else:
            cnt += 1
    print(f'# Clip duration for {mad_path} probably zero: {cnt}')

with open(f'{root}/annotations/MAD_train_transformed.json', "w") as f:
    f.write("\n".join([json.dumps(e) for e in mad_transformed_train]))
with open(f'{root}/annotations/MAD_val_transformed.json', "w") as f:
    f.write("\n".join([json.dumps(e) for e in mad_transformed_val]))

print(f'length train: {len(mad_transformed_train)}, length val: {len(mad_transformed_val)}')
