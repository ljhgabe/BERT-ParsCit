from tools.doc2json.grobid2json.process_pdf import process_pdf_file
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="configs/", config_name="pdf2json.yaml")
def main(config: DictConfig):
    res = process_pdf_file(config.input_file, config.temp_dir, config.output_dir)
    return res

if __name__ == "__main__":
    res = main()