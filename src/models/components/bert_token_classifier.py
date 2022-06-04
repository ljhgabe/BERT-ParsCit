from pytorch_lightning import LightningModule
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification
from src.datamodules.components.class_label import label2id, id2label
from typing import Dict

from torch import nn

from src.models.utils.bert_parscit_config import BERT_MODEL_CHECKPOINT
from src.models.utils.bert_parscit_config import CACHE_DIR


class BertTokenClassifier(nn.Module):
    def __init__(
        self,
        model_checkpoint: str = BERT_MODEL_CHECKPOINT,
        cache_dir: str = CACHE_DIR
    ):
        super().__init__()

        self.model = AutoModelForTokenClassification.from_pretrained(
            model_checkpoint,
            id2label=id2label,
            label2id=label2id,
            cache_dir=cache_dir + 'models/'
        )

    def forward(self, inputs: Dict):
        outputs = self.model(**inputs)
        return outputs
