import os
from tools.doc2json.grobid2json.process_pdf import process_pdf_file
import hydra
from omegaconf import DictConfig

def convert2abs(path:str):
    abs = path
    if abs and not os.path.isabs(abs):
        abs = os.path.join(
            hydra.utils.get_original_cwd(), abs
        )
    return abs


@hydra.main(config_path="configs/", config_name="pdf2json.yaml")
def main(config: DictConfig):
    res = process_pdf_file(
        convert2abs(config.input_file),
        convert2abs(config.temp_dir),
        convert2abs(config.output_dir))
    return res

if __name__ == "__main__":
    res = main()