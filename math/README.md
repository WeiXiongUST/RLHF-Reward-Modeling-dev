# RLHF-Reward-Modeling: Math Reward

Model
- [RLHFlow/Llama3.1-8B-PRM](https://huggingface.co/RLHFlow/Llama3.1-8B-PRM)

## Introduction

We present an impementation of process-supervision reward model, as described in the [Math Shepherd paper](https://arxiv.org/abs/2312.08935). We follow the original paper to use the hard label and SFT training pipeline.

## Installation instructions

Before starting, please make sure your linux machine has [nvidia-cuda-toolkit](https://developer.nvidia.com/cuda-toolkit) installed. The environment setup is the same as the pairwise preference model.

```shell
conda create -n prm_dev python=3.10.9
conda activate prm_dev

## Get axolotl for general model
git clone https://github.com/OpenAccess-AI-Collective/axolotl
cd axolotl
git checkout 55cc214c767741e83ee7b346e5e13e6c03b7b9fa
pip install -e .

# The test cuda version is 12.1, 12.2. You may need to update the torch version based on your cuda version...
# you may encounter underfined symbol error related to cuda and flash-attn and 2.1.2 can solve it ...
pip3 install torch==2.1.2 torchvision torchaudio
pip install flash-attn


## Get FastChat
git clone https://github.com/lm-sys/FastChat.git
cd FastChat
pip install -e .

git clone https://github.com/WeiXiongUST/RLHF-Reward-Modeling.git
pip install deepspeed
```

You also need to install wandb to record the training and log in with the huggingface accout to access Gemma.

```shell
pip install wandb
wandb login

huggingface-cli login
```

Some possible problems:

`CUDA_HOME` may not exist, unable to compile CUDA op(s)AssertionError:[end of output]

```shell
conda install nvidia/label/cuda-12.2.0::cuda-nvcc
```

## Dataset Preparation
The problem is formated as a multi-turn chat and the data should be processed into the standard format. See [RLHFlow/Mistral-PRM-Data](https://huggingface.co/datasets/RLHFlow/Mistral-PRM-Data) for an example.

## Running the Code

Running the code with Gemma-2b-it.

```shell
torchrun --nproc_per_node 8 --master_port 20001 -m axolotl.cli.train prm_hard.yaml --deepspeed ../deepspeed_configs/deepspeed_3.json
```

## Usage Example for Pairwise Comparison
