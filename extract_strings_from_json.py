import argparse
import json
import os



BASE_OUTPUT_DIR = 'output'
def extract_reference_strings(json_file: str, output_dir: str = BASE_OUTPUT_DIR):

    os.makedirs(output_dir, exist_ok=True)
    assert json_file[-4:] == "json"
    output_path = os.path.join(output_dir,os.path.basename(json_file)[:-5]+"_ref.txt")

    strings_file = open(output_path,"w")
    strings = []
    with open(json_file,'r') as f:
        data = json.load(f)
        for _, context in data["pdf_parse"]["bib_entries"].items():
            raw_text = context["raw_text"]
            if raw_text != "":
                strings_file.write(raw_text)
                strings_file.write("\n")
                strings.append(raw_text)
    strings_file.close()
    return strings

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extracting strings from JSON")
    parser.add_argument("--input_file", required=True, help="path to the input json file")
    parser.add_argument("--reference", action="store_true", default=False, help="to extract reference strings")
    parser.add_argument("--output_dir", default=BASE_OUTPUT_DIR, help="path to the output dir for putting json files")
    args = parser.parse_args()
    json_file = args.input_file
    output_dir = args.output_dir
    if args.reference:
        ref = extract_reference_strings(json_file, output_dir)
