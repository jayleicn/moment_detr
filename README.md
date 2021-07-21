# Moment-DETR

[QVHighlights: Detecting Moments and Highlights in Videos via Natural Language Queries](https://arxiv.org/abs/2107.09609)

[Jie Lei](http://www.cs.unc.edu/~jielei/), 
[Tamara L. Berg](http://tamaraberg.com/), [Mohit Bansal](http://www.cs.unc.edu/~mbansal/)


For dataset details, please check [data/README.md](data/README.md)

## Getting Started 

### Prerequisites
0. Clone this repo

```
git clone https://github.com/jayleicn/moment_detr.git
cd moment_detr
```

1. Prepare feature files

Download [moment_detr_features.tar.gz](https://drive.google.com/file/d/1Hiln02F1NEpoW8-iPZurRyi-47-W2_B9/view?usp=sharing) (8GB), 
extract it under project root directory:
```
tar -xf path/to/moment_detr_features.tar.gz
```


2. Install dependencies.

This code requires Python 3.7, PyTorch, and a few other Python libraries. 
We recommend creating conda environment and installing all the dependencies as follows:
```
# create conda env
conda create --name moment_detr python=3.7
# activate env
conda actiavte moment_detr
# install pytorch with CUDA 11.0
conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch
# install other python packages
pip install tqdm ipython easydict tensorboard tabulate scikit-learn pandas
```

### Training

Training can be launched by running the following command:
```
bash moment_detr/scripts/train.sh 
```
This will train Moment-DETR for 200 epochs on the QVHighlights train split, with SlowFast and Open AI CLIP features. The training is very fast, it can be done within 4 hours using a single RTX 2080Ti GPU. The checkpoints and other experiment log files will be written into `results`. For training under different settings, you can append additional command line flags to the command above. For example, if you want to train the model without the saliency loss (by setting the corresponding loss weight to 0):
```
bash moment_detr/scripts/train.sh --lw_saliency 0
```
For more configurable options, please checkout our config file [moment_detr/config.py](moment_detr/config.py).

### Inference
Once the model is trained, you can use the following command for inference:
```
bash moment_detr/scripts/inference.sh CHECKPOINT_PATH SPLIT_NAME  
``` 
where `CHECKPOINT_PATH` is the path to the saved checkpoint, `SPLIT_NAME` is the split name for inference, can be one of `val` and `test`.

### Pretraining and Finetuning
Moment-DETR utilizes ASR captions for weakly supervised pretraining. To launch pretraining, run:
```
bash moment_detr/scripts/pretrain.sh 
```  
This will pretrain the Moment-DETR model on the ASR captions for 100 epochs, the pretrained checkpoints and other experiment log files will be written into `results`. With the pretrained checkpoint, we can launch finetuning from a pretrained checkpoint `PRETRAIN_CHECKPOINT_PATH` as:
```
bash moment_detr/scripts/train.sh  --resume ${PRETRAIN_CHECKPOINT_PATH}
```
Note that this finetuning process is the same as standard training except that it initializes weights from a pretrained checkpoint. 


### Evaluation and Codalab Submission
Please check [standalone_eval/README.md](standalone_eval/README.md) for details.

## Acknowledgement
We thank [Linjie Li](https://scholar.google.com/citations?user=WR875gYAAAAJ&hl=en) for the helpful discussions.
This code is based on [detr](https://github.com/facebookresearch/detr) and [TVRetrieval XML](https://github.com/jayleicn/TVRetrieval). We used resources from [mdetr](https://github.com/ashkamath/mdetr), [MMAction2](https://github.com/open-mmlab/mmaction2), [CLIP](https://github.com/openai/CLIP), [SlowFast](https://github.com/facebookresearch/SlowFast) and [HERO_Video_Feature_Extractor](https://github.com/linjieli222/HERO_Video_Feature_Extractor). We thank the authors for their awesome open-source contributions. 

## LICENSE
The annotation files are under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license, see [./data/LICENSE](data/LICENSE). All the code are under [MIT](https://opensource.org/licenses/MIT) license, see [LICENSE](./LICENSE).
 
