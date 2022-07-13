import argparse
import os
from tools.doc2json.grobid2json.process_pdf import process_pdf_file


BASE_OUTPUT_DIR = "output"
BASE_TEMP_DIR = "temp"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extracting strings from JSON")
    parser.add_argument("--input_file", required=True, help="path to the input json file")
    parser.add_argument("--output_dir", default=BASE_OUTPUT_DIR, help="path to the output dir for putting json files")
    parser.add_argument("--temp_dir", default=BASE_TEMP_DIR, help="path to the temp dir for putting tei xml files")
    args = parser.parse_args()
    pdf_file = args.input_file
    output_dir = args.output_dir
    temp_dir = args.temp_dir
    res = process_pdf_file(input_file=pdf_file, temp_dir=temp_dir, output_dir=output_dir)