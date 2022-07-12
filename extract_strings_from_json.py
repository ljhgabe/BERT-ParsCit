import json
import os

import hydra
from omegaconf import DictConfig

# file_path = r"/home/dingyx/project/s2orc-doc2json/tests/pdf/N18-3011.json"
def convert2abs(path:str):
    abs = path
    if abs and not os.path.isabs(abs):
        abs = os.path.join(
            hydra.utils.get_original_cwd(), abs
        )
    return abs

@hydra.main(config_path="configs/", config_name="extract_strings.yaml")
def extract_reference_strings(config: DictConfig):

    json_file = convert2abs(config.json_file)
    output_dir = convert2abs(config.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    assert json_file[-4:] == "json"
    output_path = os.path.join(output_dir,os.path.basename(json_file)[:-5]+"_ref.txt")
    # print(output_path)
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
    print(strings)
    return strings

if __name__ == "__main__":


    ref = extract_reference_strings()
