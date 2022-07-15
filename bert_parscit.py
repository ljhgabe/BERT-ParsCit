import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import DataCollatorForTokenClassification
from collections import Counter
from datasets import Dataset

from src.models.components.bert_token_classifier import BertTokenClassifier
from src.datamodules.components.cora_label import LABEL_NAMES
from src.models.components.bert_tokenizer import bert_tokenizer

model = BertTokenClassifier(
    model_checkpoint="allenai/scibert_scivocab_uncased",
    output_size=13,
)
model.load_state_dict(torch.load("scibert-synthetic-50k-parscit.pt"))
model.eval()

def postprocess(word_ids, predictions, label_names):
    true_word_ids = [[id for id in word_id if id != -1] for word_id in word_ids]
    true_predictions = [
        [label_names[p] for w, p in zip(word_id, prediction) if w != -1] 
        for word_id, prediction in zip(word_ids, predictions)
    ]

    grouped_true_predictions = list()

    for word_id, true_prediction in zip(true_word_ids, true_predictions):
        grouped_true_prediction = list()

        current_group_predictions = list()
        for i in range(len(word_id)):
            current_group_predictions.append(true_prediction[i])

            if i + 1 == len(word_id):
                grouped_true_prediction.append(current_group_predictions)

            elif word_id[i] != word_id[i + 1]:
                grouped_true_prediction.append(current_group_predictions)
                current_group_predictions = list()

        grouped_true_predictions.append(grouped_true_prediction)

    merged_true_predictions = list()
    for grouped_true_prediction in grouped_true_predictions:
        merged_true_prediction = list(map(lambda l: Counter(l).most_common(1)[0][0], grouped_true_prediction))
        merged_true_predictions.append(merged_true_prediction)

    return merged_true_predictions


def tokenize_and_add_word_ids(example):
    tokenized_inputs = bert_tokenizer(
        example["tokens"], truncation=True, is_split_into_words=True
    )
    all_tokens = example["tokens"]
    all_word_ids = []
    for i, tokens in enumerate(all_tokens):
        word_id = tokenized_inputs.word_ids(i)
        word_id[0], word_id[-1] = -1, -1
        all_word_ids.append(word_id)
    tokenized_inputs["word_ids"] = all_word_ids
    return tokenized_inputs


def predict_for_text(example: str):
    splitted_example = example.split()
    dict_data = {"tokens": [splitted_example]}
    dataset = Dataset.from_dict(dict_data)
    tokenized_example = dataset.map(
        lambda x: tokenize_and_add_word_ids(x),
        batched=True,
        remove_columns=dataset.column_names
    )
    dataloader = DataLoader(
        dataset=tokenized_example,
        batch_size=1,
        collate_fn=DataCollatorForTokenClassification(
            tokenizer=bert_tokenizer
        )
    )
    for batch in dataloader:
        outputs = model(**batch)
        preds = outputs.logits.argmax(dim=-1)
        word_ids = batch["word_ids"]
        true_preds = postprocess(
            word_ids=word_ids,
            predictions=preds,
            label_names=LABEL_NAMES
        )

    tokens = splitted_example
    tagged_words = []
    
    for token, label in zip(tokens, true_preds[0]):
        tagged_word = f"<{label}>{token}</{label}>"
        tagged_words.append(tagged_word)
    result = " ".join(tagged_words)
    return result
