cd ../../
dset_name=hl
ctx_mode=video_tef
v_feat_types=clip
t_feat_type=clip 
results_root=results
exp_id=exp

######## data paths
train_path=/nfs/data3/goldhofer/mad_dataset/annotations/MAD_train_transformed.json
eval_path=/nfs/data3/goldhofer/mad_dataset/annotations/MAD_val_transformed.json
eval_split_name=val

######## setup video+text features
v_feat_dirs=(/nfs/data3/goldhofer/mad_dataset/)
v_feat_dim=512
t_feat_dir=/nfs/data3/goldhofer/mad_dataset/
t_feat_dim=512
#### training
bsz=32


PYTHONPATH=$PYTHONPATH:. python moment_detr/train.py \
--dset_name ${dset_name} \
--ctx_mode ${ctx_mode} \
--train_path ${train_path} \
--eval_path ${eval_path} \
--eval_split_name ${eval_split_name} \
--v_feat_dirs ${v_feat_dirs[@]} \
--v_feat_dim ${v_feat_dim} \
--t_feat_dir ${t_feat_dir} \
--t_feat_dim ${t_feat_dim} \
--bsz ${bsz} \
--results_root ${results_root} \
--num_workers 12 \
--n_epoch 50 \
--exp_id ${exp_id} \
${@:1}
