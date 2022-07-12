
<div align="center">
   
# BERT ParsCit

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)

</div>

## Description

This is the repository of BERT ParsCit and is under active development at National University of Singapore (NUS), Singapore. The project was built upon a [template by ashleve](https://github.com/ashleve/lightning-hydra-template).
BERT ParsCit is a BERT version of [Neural ParsCit](https://github.com/WING-NUS/Neural-ParsCit) built by researchers under [WING@NUS](https://wing.comp.nus.edu.sg/).

## How to run

Install dependencies

```bash
# clone project
git clone https://github.com/ljhgabe/BERT-ParsCit
cd BERT-ParsCit

# [OPTIONAL] create conda environment
conda create -n myenv python=3.8
conda activate myenv

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

To show the full stack trace for error occurred during training or testing

```bash
HYDRA_FULL_ERROR=1 python train.py
```


## How to Parse Reference Strings from a PDF
###  Setup Doc2Json
First prepare for the environment:
```bash
cd ./tools
python setup.py develop
```
The current `grobid2json` tool uses Grobid to first process each PDF into XML, then extracts paper components from the XML.

### Install Grobid

You will need to have Java installed on your machine. Then, you can install your own version of Grobid and get it running, or you can run the following script:

```console
bash tools/scripts/setup_grobid.sh
```

This will setup Grobid, currently hard-coded as version 0.6.1. Then run:

```console
bash tools/scripts/run_grobid.sh
```

to start the Grobid server. Don't worry if it gets stuck at 87%; this is normal and means Grobid is ready to process PDFs.

### Convert a PDF to JSON

There are a couple of test PDFs in `tools/tests/pdf/` if you'd like to try with that.

For example, you can try:

```console
python pdf2json.py input_file=tools/tests/pdf/N18-3011.pdf output_dir=output_dir/
```

This will generate a JSON file in the specified `output_dir`. If unspecified, the file will be in the `data/output/` directory from your path.

### Extract Reference Strings from a JSON File

To get reference strings, try:
```console
python extract_strings_from_json.py json_file=data/output/N18-3011.json output_dir=data/
```
This will generate a text file in the specified 'output_dir',where each line contains a reference string.

### Parse Reference Strings from a Text File
To predict the reference string tags, try:
```python
from bert_parscit import predict_for_file
res = predict_for_file("data/N18-3011_ref.txt",output_dir="data/result")
```
The prediction result is saved in `output_dir`.If unspecified, the file will be in the data/result/ directory from your path.
