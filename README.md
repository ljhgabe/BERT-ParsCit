

# BERT-ParsCit
<div align="center">

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)

</div>

## Description

This is the repository of ParsCit-ParsCit and is under active development at National University of Singapore (NUS), Singapore. The project was built upon a [template by ashleve](https://github.com/ashleve/lightning-hydra-template).

## How to run

Install dependencies

```bash
# clone project
git clone https://github.com/ljhgabe/BERT-ParsCit
cd BERT-ParsCit

# [OPTIONAL] create conda environment
conda create -n parscit python=3.8
conda activate parscit

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

Train model with default configuration

```bash
# train on CPU
python train.py trainer.gpus=0

# train on GPU
python train.py trainer.gpus=1
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python train.py trainer.max_epochs=20 datamodule.batch_size=64
```

## FAQ
Q. Why do I get error like `Error locating target 'src.models.bert_parscit_module.BertParsCitLitModule'` even though my file's location is correct?
   
A. Hydra may hide the full error stack trace even with `HYDRA_ERROR_FULL=1`. In this situation, the primary error source
is compilation error such as importing a library that has not been downloaded. To displace the full error stack trace, 
replace the codes using hydra to instantiate modules with your own ones.
