from torch import nn
from transformers import AutoModel
from transformers.modeling_outputs import TokenClassifierOutput
from src.models.utils.bert_model_config import BERT_MODEL_CHECKPOINT
from src.models.utils.bert_model_path import MODEL_CACHE_DIR


class BertTokenClassifier(nn.Module):
    def __init__(
        self,
        model_checkpoint: str = BERT_MODEL_CHECKPOINT,
        output_size: int = 19,
        cache_dir: str = MODEL_CACHE_DIR
    ):
        super().__init__()
        self.bert_embedder = AutoModel.from_pretrained(
            model_checkpoint,
            cache_dir=cache_dir,
        )
        self.output_size = output_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(in_features=768, out_features=output_size, bias=True)

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, word_ids=None, labels=None):
        outputs = self.bert_embedder(input_ids, attention_mask=attention_mask)
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
