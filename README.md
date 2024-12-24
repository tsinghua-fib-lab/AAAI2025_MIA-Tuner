# ___***AAAI'25: MIA-Tuner: Adapting Large Language Models as Pre-training Text Detector***___

<div align="center">
<p>

<a href="https://arxiv.org/abs/2408.08661">
    <img src="https://img.shields.io/badge/cl.CL-2408.08661-b31b1b.svg?logo=arxiv&logoColor=b31b1b&labelColor=9a9a9a">
</a>

<a href="https://huggingface.co/collections/wjfu99/wikimia-24-benchmark-676a5bb863aa59998de65db3">
    <img src="https://img.shields.io/badge/🤗 Huggingface-WIKIMIA--24-red?style=flat_square">
</a>

</p>

_**[Wenjie Fu](https://wjfu99.github.io)<sup>1</sup>, [Huandong Wang](https://scholar.google.com/citations?user=PNbioq0AAAAJ)<sup>2</sup>, [Chen Gao](https://fi.ee.tsinghua.edu.cn/~gaochen/)<sup>2</sup>, [Guanghua Liu](https://scholar.google.com/citations?user=tTyhUQQAAAAJ)<sup>1</sup>, [Yong Li](https://fi.ee.tsinghua.edu.cn/~liyong/)<sup>2</sup>, [Tao Jiang](https://scholar.google.com/citations?user=9BEmtIwAAAAJ)<sup>1</sup>***_
<!-- <br><br> -->
<!-- (* corresponding authors, <sup>&Dagger;</sup> project leader) -->

<sup>1</sup> Huazhong University of Science and Technology &nbsp; <sup>2</sup> Tsinghua University
</div>

- [___***AAAI'25: MIA-Tuner: Adapting Large Language Models as Pre-training Text Detector***___](#aaai25-mia-tuner-adapting-large-language-models-as-pre-training-text-detector)
  - [Overview](#overview)
  - [A Quick Glance on ChatGLM\*](#a-quick-glance-on-chatglm)
  - [Requirements](#requirements)
  - [WikiMIA-24 Dataset](#wikimia-24-dataset)
  - [Running all baselines](#running-all-baselines)
  - [Running MIA-Tuner](#running-mia-tuner)
  - [Reproducing All Experiment in Our Paper](#reproducing-all-experiment-in-our-paper)
  - [Citation](#citation)

This is the official implementation of the paper "MIA-Tuner: Adapting Large Language Models as Pre-training Text Detector".
The proposed MIA-Tuner is implemented as follows.

## Overview
Instructing aligned and unaligned LLMs themselves to detect texts that have been seen during the pre-training phase. 

![The overall architecture of MIA-Tuner](./Framework.png)

## A Quick Glance on ChatGLM*

https://github.com/user-attachments/assets/b27ae9fc-3290-48ea-ab1e-b8e9abdb9c36

<small>*Please refer [./inference_on_zhipuai](/inference_on_zhipuai) for reproducing.</small>

## Requirements

- torch>=2.2.0
- accelerate==0.32.1
- transformers==4.42.4
- huggingface_hub==0.23.4
- datasets==2.20.0
- deepeval==0.21.73
- langchain==0.2.14
- Wikipedia_API==0.6.0
- numpy>=1.24.4
- scikit-learn>=1.1.3
- pyyaml>=6.0
- tqdm>=4.64.1

Dependency can be installed with the following command:

```bash
pip install -r requirements.txt
```

## WikiMIA-24 Dataset

We provide a collection of related datasets in 🤗 [Huggingface](https://huggingface.co/collections/wjfu99/wikimia-24-benchmark-676a5bb863aa59998de65db3).

## Running all baselines
In this repo, we provide an all-in-one script [run_baselines.py](run_baselines.py) for running all exiting baselines in one commond.
```bash
python run_baselines.py --model ${model} --dataset ${DATASET_NAME} --block_size ${BLOCK_SIZE}
```

## Running MIA-Tuner

* Aligned LLMs
    ```bash
    accelerate launch mia_hybrid.py -m ${model} --unaligned_model -d ${DATASET_NAME} \
    --block_size ${BLOCK_SIZE} --epochs ${EPOCHS} --batch_size ${BATCH_SIZE} --learning_rate ${LEARNING_RATE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS}
    ```

* Unligned LLMs
    ```bash
    accelerate launch mia_hybrid.py -m ${model} --unaligned_model -d ${DATASET_NAME} \
    --block_size ${BLOCK_SIZE} --epochs ${EPOCHS} --batch_size ${BATCH_SIZE} --learning_rate ${LEARNING_RATE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS}
    ```

## Reproducing All Experiment in Our Paper
All scripts for reproducing results in our paper can be found in [./exp_scripts](/exp_scripts/)

* [./exp_scripts/main_exp.sh](/exp_scripts/main_exp.sh)
* [./exp_scripts/main_exp_baselines.sh](/exp_scripts/main_exp_baselines.sh)
* [./exp_scripts/analysis_training_scale.sh](exp_scripts/analysis_training_scale.sh)
* [./exp_scripts/analysis_model_size.sh](exp_scripts/analysis_model_size.sh)
* [./exp_scripts/analysis_text_length.sh](./exp_scripts/analysis_text_length.sh)

## Citation
Please consider to cite our paper if you find MIA-Tuner helpful in your research

```bibtex
@inproceedings{fu2024membership,
    title={{MIA}-Tuner: Adapting Large Language Models as Pre-training Text Detector},
    booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
    author = {Fu, Wenjie and Wang, Huandong and Gao, Chen and Liu, Guanghua and Li, Yong and Jiang, Tao},
    year = {2025},
    address = {Philadelphia, Pennsylvania, USA}
}
```