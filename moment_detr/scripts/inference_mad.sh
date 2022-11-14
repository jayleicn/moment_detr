ckpt_path=run_on_video/moment_detr_ckpt/model_best.ckpt

dset_name=hl
ctx_mode=video_tef
v_feat_types=clip
t_feat_type=clip
results_root=results
exp_id=exp

######## data paths
train_path=/nfs/data3/goldhofer/mad_dataset/annotations/MAD_train_transformed.json
eval_path=/nfs/data3/goldhofer/mad_dataset/annotations/MAD_test_transformed.json
eval_split_name=test

######## setup video+text features
v_feat_dirs=(/nfs/data3/goldhofer/mad_dataset/clip_frame_features_transformed_dense/)
v_feat_dim=512
t_feat_dir=/nfs/data3/goldhofer/mad_dataset/
t_feat_dim=512
bsz=256
device=1
eval_results_dir=tmp/
sampling_fps=0.5

PYTHONPATH=$PYTHONPATH:. python moment_detr/inference.py \
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
--num_workers 16 \
--n_epoch 200 \
--exp_id ${exp_id} \
--resume ${ckpt_path} \
--device ${device} \
--eval_results_dir ${eval_results_dir} \
--sampling_fps ${sampling_fps}
${@:1}