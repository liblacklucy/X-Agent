**❗️Update:** We released the code and pre-trained weights for CVPR version of CAT-Seg! 
Some major updates are:
- We now solely utilize CLIP as the pre-trained encoders, without additional backbones (ResNet, Swin)!
- We also fine-tune the text encoder of CLIP, yielding significantly improved performance!

For further details, please check out our updated [paper](https://arxiv.org/abs/2303.11797).
Note that the demos are still running on our previous version, and will be updated soon!

## :fire:TODO
- [x] Train/Evaluation Code (Mar 21, 2023)
- [x] Pre-trained weights (Mar 30, 2023)
- [x] Code of interactive demo (Jul 13, 2023)
- [x] Release code for CVPR version (Apr 4, 2024)
- [x] Release checkpoints for CVPR version (Apr 11, 2024)
- [ ] Demo update

## Installation
Please follow [installation](INSTALL.md). 

## Data Preparation
Please follow [dataset preperation](datasets/README.md).

## Training
We provide shell scripts for training and evaluation. ```run.sh``` trains the model in default configuration and evaluates the model after training. 

To train or evaluate the model in different environments, modify the given shell script and config files accordingly.

### Training script
```bash
sh run.sh [CONFIG] [NUM_GPUS] [OUTPUT_DIR] [OPTS]

# For ViT-B variant
sh run.sh configs/vitb_384.yaml 4 output/
# For ViT-L variant
sh run.sh configs/vitl_336.yaml 4 output/
```

## Evaluation
```eval.sh``` automatically evaluates the model following our evaluation protocol, with weights in the output directory if not specified.
To individually run the model in different datasets, please refer to the commands in ```eval.sh```.

### Evaluation script
```bash
sh run.sh [CONFIG] [NUM_GPUS] [OUTPUT_DIR] [OPTS]

sh eval.sh configs/vitl_336.yaml 4 output/ MODEL.WEIGHTS path/to/weights.pth
```