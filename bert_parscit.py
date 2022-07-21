import os
import timeit
from typing import List, Tuple

import numpy
import torch
from torch.utils.data import DataLoader
from collections import Counter
from datasets import Dataset

from src.models.components.bert_token_classifier import BertTokenClassifier
from src.datamodules.components.cora_label import LABEL_NAMES
from src.models.components.bert_tokenizer import bert_tokenizer
from pdf2text import process_pdf_file, get_reference

BASE_OUTPUT_DIR = "result"
BASE_TEMP_DIR = "temp"

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
        word_id = [id if id is not None else -1 for id in word_id]
        all_word_ids.append(word_id)
    tokenized_inputs["word_ids"] = all_word_ids

    return tokenized_inputs

def convert_to_list(batch):
  res = []
  for i in batch:
      input_ids = i["input_ids"]
      token_type_ids = i["token_type_ids"]
      attn_mask = i["attention_mask"]
      word_ids = i["word_ids"]
      res.append([input_ids, token_type_ids, attn_mask, word_ids])
  return res


def pad(batch):
    '''Pads to the longest sample'''
    batch = convert_to_list(batch)
    get_element = lambda x: [sample[x] for sample in batch]
    seq_len = [len(tokens) for tokens in get_element(0)]
    maxlen = numpy.array(seq_len).max()

    do_pad = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch]  # 0: <pad>
    do_word_ids_pad = lambda x, seqlen: [sample[x] + [-1] * (seqlen - len(sample[x])) for sample in batch]
    input_ids = do_pad(0, maxlen)
    token_type_ids = do_pad(1, maxlen)
    attn_mask = do_pad(2, maxlen)
    word_ids = do_word_ids_pad(3, maxlen)
    LT = torch.LongTensor

    token_ids = get_element(0)
    token_ids_len = torch.LongTensor(list(map(len, token_ids)))
    _, sorted_idx = token_ids_len.sort(0, descending=True)

    input_ids = LT(input_ids)[sorted_idx]
    attn_mask = LT(attn_mask)[sorted_idx]
    token_type_ids = LT(token_type_ids)[sorted_idx]
    word_ids = LT(word_ids)[sorted_idx]

    return {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": attn_mask,
        "word_ids": word_ids
    }
    # return tok_ids, token_type_ids, attn_mask, word_ids



def predict(examples: List[List[str]]) -> Tuple[List[str], List[List[str]], List[List[str]]]:
    """

    Args:
        examples: a list of inputs for inference, where each item is a list of tokens

    Returns:
        results: a list of labels predicted by the model

    """
    dict_data = {"tokens": examples}
    dataset = Dataset.from_dict(dict_data)
    tokenized_example = dataset.map(
        lambda x: tokenize_and_add_word_ids(x),
        batched=True,
        remove_columns=dataset.column_names
    )

    dataloader = DataLoader(
        dataset=tokenized_example,
        batch_size=8,
        collate_fn=pad
    )
    results = []
    true_preds = []
    for batch in dataloader:
        outputs = model(**batch)
        preds = outputs.logits.argmax(dim=-1)
        word_ids = batch["word_ids"]
        true_pred = postprocess(
            word_ids=word_ids,
            predictions=preds,
            label_names=LABEL_NAMES
        )
        true_preds.extend(true_pred)
    tokens = examples

    for i in range(len(tokens)):
        tagged_words = []
        for token, label in zip(tokens[i], true_preds[i]):
            tagged_word = f"<{label}>{token}</{label}>"
            tagged_words.append(tagged_word)
        result = " ".join(tagged_words)
        results.append(result)
    return results, tokens, true_preds



def predict_for_string(example: str):
    splitted_example = [example.split()]
    results, tokens, preds = predict(splitted_example)

    return results, tokens, preds



def predict_for_text(filename: str, output_dir: str = BASE_OUTPUT_DIR):

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.basename(filename)

    start_time = timeit.default_timer()
    with open(filename,"r") as f:
        examples = f.readlines()
    splitted_examples = [example.split() for example in examples]
    results, tokens, preds = predict(splitted_examples)
    with open(os.path.join(output_dir, f"{output_file[:-4]}_result.txt"), "w") as output:
        for res in results:
            output.write(res+"\n")
    total_time = timeit.default_timer() - start_time
    print("total_time:",total_time)
    return results, tokens, preds


def predict_for_pdf(filename: str, output_dir: str = BASE_OUTPUT_DIR, temp_dir: str = BASE_TEMP_DIR):
    json_file = process_pdf_file(input_file=filename, temp_dir=temp_dir, output_dir=temp_dir)
    text_file = get_reference(json_file=json_file, output_dir=output_dir)
    return predict_for_text(text_file, output_dir=output_dir)