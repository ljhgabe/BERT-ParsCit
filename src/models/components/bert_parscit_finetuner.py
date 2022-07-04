import torch
from torch import nn
from transformers.modeling_outputs import TokenClassifierOutput
from src.datamodules.components.synthetic_label import num_labels as synthetic_labels
from src.datamodules.components.cora_label import num_labels as cora_labels
from src.models.components.bert_token_classifier import BertTokenClassifier
from src.models.utils.bert_model_config import BERT_PARSCIT_CHECKPOINT
from src.models.utils.bert_model_path import MODEL_CACHE_DIR


class BertParsCitFineTuner(nn.Module):
    def __init__(
        self,
        model_checkpoint: str = BERT_PARSCIT_CHECKPOINT,
        output_size: int = cora_labels,
        cache_dir: str = MODEL_CACHE_DIR
    ):
        super().__init__()
        self.bert_parscit: BertTokenClassifier = torch.load(model_checkpoint)
        self.bert_parscit.eval()
        self.output_size = output_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(in_features=synthetic_labels, out_features=output_size, bias=True)

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, labels=None):
        # input_ids, attention_mask, labels = inputs["input_ids"], inputs["attention_mask"], inputs["labels"]
        outputs = self.bert_parscit(input_ids, attention_mask=attention_mask)
        outputs = self.dropout(outputs[0])
        logits = self.classifier(outputs)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.output_size), labels.view(-1))
        return TokenClassifierOutput(
            loss=loss,
            logits=logits
        )

