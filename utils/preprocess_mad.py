import json, os

SPLIT = 'train'
root = '/nfs/data3/goldhofer/mad_dataset'
train_data = json.load(open(f'{root}/annotations/MAD_{SPLIT}.json', 'r'))
moment_detr_dict = {}
mad_transformed = []
for k in train_data.keys():
    lowest_clip = round(train_data[k]["timestamps"][0])
    highest_clip = round(train_data[k]["timestamps"][1])
    if lowest_clip % 2 != 0:
        lowest_clip -= 1
    if highest_clip % 2 != 0:
        highest_clip += 1

    moment_detr_dict = {"qid": k + "_" + train_data[k]["movie"], "query": train_data[k]["sentence"],
                        "duration": train_data[k]["movie_duration"],
                        "vid": train_data[k]["movie"],
                        "relevant_windows": [lowest_clip, highest_clip],
                        "relevant_clip_ids": [i for i in range(int(lowest_clip/2), int(highest_clip/2))],
                        "saliency_scores": [[0,0,0] for i in range(int(lowest_clip/2), int(highest_clip/2))]}
    mad_transformed.append(moment_detr_dict)

with open(f'{root}/annotations/MAD_{SPLIT}_transformed.json', "w") as f:
    f.write("\n".join([json.dumps(e) for e in mad_transformed]))
