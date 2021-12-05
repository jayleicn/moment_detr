To simplify feature extraction pipeline, 
this model checkpoint [model_best.ckpt](model_best.ckpt) is trained with only CLIP image and text features as input. 
It is trained from scratch, without ASR pre-training. 
It may perform worse than the model reported in the paper.

In addition to the model checkpoint, this directory also 
contains multiple files from its training process, 
including the training/evaluation log files, 
training configurations and prediction files on QVHighlights val split.
