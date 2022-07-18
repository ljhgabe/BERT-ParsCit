

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

## Installation

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

## Example usage

```bash
from bert_parscit import predict_for_text

result = predict_for_text("Calzolari, N. (1982) Towards the organization of lexical definitions on a database structure. In E. Hajicova (Ed.), COLING '82 Abstracts, Charles University, Prague, pp.61-64.")
```

## How to train

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

## How to Process a PDF
###  Setup Doc2Json
To setup Doc2Json, you should run:
```bash
sh bin/doc2json/scripts/run.sh
```
This will setup Doc2Json and Grobid. And after installation, it starts the Grobid server in the background by default.


### How to Parse Reference Strings

To parse reference strings from **a PDF**, try:
```python
from bert_parscit import predict_for_pdf
results, tokens, tags = predict_for_pdf(filename, output_dir, temp_dir)
```
This will generate a text file of reference strings in the specified `output_dir`.
And the JSON format of the origin PDF will be saved in the specified `temp_dir`. 
The default `output_dir` is `result/` from your path and the default `temp_dir` is `temp/` from your path.
The output `results` is a list of tagged strings, it seems like:
```
['<author>Waleed</author> <author>Ammar,</author> <author>Matthew</author> <author>E.</author> <author>Peters,</author> <author>Chandra</author> <author>Bhagavat-</author> <author>ula,</author> <author>and</author> <author>Russell</author> <author>Power.</author> <date>2017.</date> <title>The</title> <title>ai2</title> <title>system</title> <title>at</title> <title>semeval-2017</title> <title>task</title> <title>10</title> <title>(scienceie):</title> <title>semi-supervised</title> <title>end-to-end</title> <title>entity</title> <title>and</title> <title>relation</title> <title>extraction.</title> <booktitle>In</booktitle> <booktitle>ACL</booktitle> <booktitle>workshop</booktitle> <booktitle>(SemEval).</booktitle>']
```
`tokens` is a list of origin tokens of the input file and `tags` is the list of predicted tags corresponding to the input:
tokens:```[['Waleed', 'Ammar,', 'Matthew', 'E.', 'Peters,', 'Chandra', 'Bhagavat-', 'ula,', 'and', 'Russell', 'Power.', '2017.', 'The', 'ai2', 'system', 'at', 'semeval-2017', 'task', '10', '(scienceie):', 'semi-supervised', 'end-to-end', 'entity', 'and', 'relation', 'extraction.', 'In', 'ACL', 'workshop', '(SemEval).']]```
tags:```[['author', 'author', 'author', 'author', 'author', 'author', 'author', 'author', 'author', 'author', 'author', 'date', 'title', 'title', 'title', 'title', 'title', 'title', 'title', 'title', 'title', 'title', 'title', 'title', 'title', 'title', 'booktitle', 'booktitle', 'booktitle', 'booktitle']]```

You can also process **a single string** or parse strings from **a TEXT file**:
```python
from bert_parscit import predict_for_string, predict_for_text
str_results, str_tokens, str_tags = predict_for_string("Waleed Ammar, Matthew E. Peters, Chandra Bhagavat- ula, and Russell Power. 2017. The ai2 system at semeval-2017 task 10 (scienceie): semi-supervised end-to-end entity and relation extraction. In ACL workshop (SemEval).")
text_results, text_tokens, text_tags = predict_for_text(filename)
```

### To extract strings from a PDF
You can extract strings you need with the script. 
For example, to get reference strings, try:
```console
python pdf2text.py --input_file file_path --reference --output_dir output/ --temp_dir temp/
```
 
