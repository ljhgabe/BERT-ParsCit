from torch import nn

from transformers import AutoModelForTokenClassification

from src.datamodules.components.class_label import LABEL_LIST, label2id, id2label
from src.models.utils.bert_parscit_config import BERT_MODEL_CHECKPOINT
from src.models.utils.bert_parscit_config import CACHE_DIR

from typing import Dict


class BertTokenClassifier(nn.Module):
    def __init__(
        self,
        model_checkpoint: str = BERT_MODEL_CHECKPOINT,
        num_labels: int = len(LABEL_LIST),
        cache_dir: str = CACHE_DIR
    ):
        super().__init__()

        self.model = AutoModelForTokenClassification.from_pretrained(
            model_checkpoint,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            cache_dir=cache_dir + 'models/',
            ignore_mismatched_sizes=True
        )

    def forward(self, inputs: Dict):
        outputs = self.model(**inputs)
        return outputs
