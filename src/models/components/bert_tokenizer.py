from transformers import AutoTokenizer
from src.models.utils.bert_model_config import BERT_MODEL_CHECKPOINT
from src.models.utils.bert_model_config import MODEL_MAX_LENGTH
from src.models.utils.bert_model_path import TOKENIZER_CACHE_DIR

bert_tokenizer = AutoTokenizer.from_pretrained(
    BERT_MODEL_CHECKPOINT,
    model_max_length=MODEL_MAX_LENGTH,
    cache_dir=TOKENIZER_CACHE_DIR
)
