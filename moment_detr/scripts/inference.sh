ckpt_path=run_on_video/moment_detr_ckpt/model_best.ckpt
eval_split_name=val
eval_path=data/highlight_${eval_split_name}_release.jsonl
eval_results_dir=tmp/
device=1

PYTHONPATH=$PYTHONPATH:. python moment_detr/inference.py \
--resume ${ckpt_path} \
--eval_split_name ${eval_split_name} \
--eval_path ${eval_path} \
--eval_results_dir ${eval_results_dir} \
--device ${device} \
${@:3}
