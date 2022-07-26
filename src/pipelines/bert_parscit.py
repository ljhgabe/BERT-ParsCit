import os
import timeit
from typing import List, Tuple, Optional

import numpy
import torch
from torch.utils.data import DataLoader
from collections import Counter
from datasets import Dataset

from src.models.components.bert_token_classifier import BertTokenClassifier
from src.datamodules.components.cora_label import LABEL_NAMES
from src.models.components.bert_tokenizer import bert_tokenizer
from src.utils.pdf2text import process_pdf_file, get_reference

ROOT_DIR = os.getcwd()
BASE_OUTPUT_DIR = os.path.join(ROOT_DIR, "output/result")
BASE_TEMP_DIR = os.path.join(ROOT_DIR,"output/.temp")
BASE_CACHE_DIR = os.path.join(ROOT_DIR, ".cache")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BertTokenClassifier(
    model_checkpoint="allenai/scibert_scivocab_uncased",
    output_size=13,
    cache_dir=BASE_CACHE_DIR
)

model.load_state_dict(torch.load("scibert-synthetic-50k-parscit.pt"))
model.eval()
if torch.cuda.is_available():
    model.cuda()
    print("CUDA is available.")
else:
    print("Not use CUDA.")


def dehyphen_for_str(text: str):
    text = text.replace("- ", "")
    text = text.replace("-", " ")
    return text


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
    '''Pad to the longest sample'''
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

    input_ids = LT(input_ids)
    attn_mask = LT(attn_mask)
    token_type_ids = LT(token_type_ids)
    word_ids = LT(word_ids)
    return {
        "input_ids": input_ids.to(device),
        "token_type_ids": token_type_ids.to(device),
        "attention_mask": attn_mask.to(device),
        "word_ids": word_ids.to(device)
    }
    # return tok_ids, token_type_ids, attn_mask, word_ids


def predict(examples: List[List[str]]) -> Tuple[List[str], List[List[str]], List[List[str]]]:
    """
    Parse a list of tokens obtained from reference strings.

    Args:
        examples (`List[List[str]]`):
            The inputs for inference, where each item is a list of tokens.

    Returns:
        `Tuple[List[str], List[List[str]], List[List[str]]]`:
            Tagged strings, origin tokens and labels predicted by the model.

    """

    #Prepare the dataset
    dict_data = {"tokens": examples}
    dataset = Dataset.from_dict(dict_data)

    #Tokenize for Bert
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
        #Predict the labels
        outputs = model(**batch)
        preds = outputs.logits.argmax(dim=-1)
        word_ids = batch["word_ids"]
        #Convert ids to labels and
        #merge the labels according to origin tokens.
        true_pred = postprocess(
            word_ids=word_ids,
            predictions=preds,
            label_names=LABEL_NAMES
        )
        true_preds.extend(true_pred)
    tokens = examples

    #Generate the tagged strings.
    for i in range(len(tokens)):
        tagged_words = []
        for token, label in zip(tokens[i], true_preds[i]):
            tagged_word = f"<{label}>{token}</{label}>"
            tagged_words.append(tagged_word)
        result = " ".join(tagged_words)
        results.append(result)
    return results, tokens, true_preds


def predict_for_string(example: str, dehyphen: Optional[bool] = False) -> Tuple[str, List[str], List[str]]:
    """
    Parse a reference string.

    Args:
        example (`str`): The string to parse.
        dehyphen (`Optional[bool]`): Whether to remove '-', default to `False`.
    Returns:
       `Tuple[str, List[str], List[str]]`:
            Tagged string, origin tokens and labels predicted by the model.

    """
    # remove '-' in text
    if dehyphen == True:
        example = dehyphen_for_str(example)

    splitted_example = [example.split()]
    results, tokens, preds = predict(splitted_example)

    return results[0], tokens[0], preds[0]


def predict_for_text(
        filename: str,
        output_dir: Optional[str] = BASE_OUTPUT_DIR,
        dehyphen: Optional[bool] = False
) -> Tuple[List[str], List[List[str]], List[List[str]]]:
    """

    Parse reference strings from a text and save the result as a text file.

    Args:
        filename (`str`): The path to the text file to predict.
        output_dir (`Optional[str]`): The directory to save the result file, default to `result/`.
        dehyphen (`Optional[bool]`): Whether to remove '-', default to `False`.

    Returns:
        `Tuple[List[str], List[List[str]], List[List[str]]]`:
            Tagged strings, origin tokens and labels predicted by the model.

    """
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.basename(filename)

    start_time = timeit.default_timer()

    with open(filename, "r") as f:
        examples = f.readlines()

    # remove '-' in text
    if dehyphen == True:
        examples = [dehyphen_for_str(example) for example in examples]

    splitted_examples = [example.split() for example in examples]
    results, tokens, preds = predict(splitted_examples)
    with open(os.path.join(output_dir, f"{output_file[:-4]}_result.txt"), "w") as output:
        for res in results:
            output.write(res + "\n")
    total_time = timeit.default_timer() - start_time
    print("total_time:", total_time)
    return results, tokens, preds


def predict_for_pdf(
        filename: str,
        output_dir: Optional[str] = BASE_OUTPUT_DIR,
        temp_dir: Optional[str] = BASE_TEMP_DIR,
        dehyphen: Optional[bool] = False
) -> Tuple[List[str], List[List[str]], List[List[str]]]:
    """
    Parse reference strings from a PDF and save the result as a text file.

    Args:
        filename (`str`): The path to the pdf file to parse.
        output_dir (`Optional[str]`): The directory to save the result file, default to `result/`.
        temp_dir (`Optional[str]`): The diretorcy to save intermediate file, default to `temp/`.
        dehyphen (`Optional[bool]`): Whether to remove '-', default to `False`.

    Returns:
       `Tuple[List[str], List[List[str]], List[List[str]]]`:
            Tagged strings, origin tokens and labels predicted by the model.
    """

    #Convert PDF to JSON with doc2json.
    json_file = process_pdf_file(input_file=filename, temp_dir=temp_dir, output_dir=temp_dir)
    #Extract reference strings from JSON and save them in TEXT format.
    text_file = get_reference(json_file=json_file, output_dir=output_dir)
    return predict_for_text(text_file, output_dir=output_dir, dehyphen=dehyphen)
