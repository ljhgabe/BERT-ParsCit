#!/bin/bash
python bin/doc2json/setup.py develop
bash bin/doc2json/scripts/setup_grobid.sh
nohup bash bin/doc2json/scripts/run_grobid.sh

