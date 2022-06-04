import bs4
from bs4 import BeautifulSoup
from collections import Counter


def preprocess(examples):
    all_ref_strings = examples["content"]
    processed_ref_strings = []
    new_labels = []

    for raw_string in all_ref_strings:
        # Removing white spaces in between the strings
        raw_string = raw_string.replace("> <", "><")
        soup = BeautifulSoup(raw_string, 'html.parser')

        processed_ref_string_splitted = []
        processed_ref_string_label = []
        for child in soup.children:
            # If the child is not a tag instance, skip
            if not isinstance(child, bs4.element.Tag):
                continue
            # If the child has no content, skip
            if len(str(child.contents)) == 0:
                continue

            # Get the content and label of this pair
            label = child.name
            data = str(child.contents[0])

            # If nested tag
            while isinstance(data, bs4.element.Tag):
                label = data.name
                data = str(data.contents[0])

            # Have the tokens splitted by white spaces
            tokens = data.split()

            for token in tokens:
                processed_ref_string_splitted.append(token)
                processed_ref_string_label.append(label)

        processed_ref_strings.append(processed_ref_string_splitted)
        new_labels.append(processed_ref_string_label)

    examples["tokens"] = processed_ref_strings
    examples["labels"] = new_labels
    return examples


def postprocess(input_ids, predictions, labels, label_names, tokenizer):
    true_input_ids = [[id for id in input_id if id != 0 and id != 102 and id != 103] for input_id in input_ids]
    raw_strings = [tokenizer.decode(true_input_id) for true_input_id in true_input_ids]
    tokens = [string.split() for string in raw_strings]
    word_ids = list(map(lambda t: tokenizer(t, is_split_into_words=True).word_ids(), tokens))
    true_word_ids = [[id for id in word_id if id is not None] for word_id in word_ids]
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    grouped_true_labels = list()
    grouped_true_predictions = list()

    for word_id, true_label, true_prediction in zip(true_word_ids, true_labels, true_predictions):
        grouped_true_label = list()
        grouped_true_prediction = list()

        current_group_labels = list()
        current_group_predictions = list()

        if len(word_id) != len(true_label):
            continue

        for i in range(len(word_id)):
            current_group_labels.append(true_label[i])
            current_group_predictions.append(true_prediction[i])

            if i + 1 == len(word_id):
                grouped_true_label.append(current_group_labels)
                grouped_true_prediction.append(current_group_predictions)

            elif word_id[i] != word_id[i + 1]:
                grouped_true_label.append(current_group_labels)
                grouped_true_prediction.append(current_group_predictions)

                current_group_labels = list()
                current_group_predictions = list()

        grouped_true_labels.append(grouped_true_label)
        grouped_true_predictions.append(grouped_true_prediction)

    merged_true_labels = list()
    merged_true_predictions = list()
    for grouped_true_label, grouped_true_prediction in zip(grouped_true_labels, grouped_true_predictions):
        merged_true_label = list(map(lambda l: Counter(l).most_common(1)[0][0], grouped_true_label))
        merged_true_prediction = list(map(lambda l: Counter(l).most_common(1)[0][0], grouped_true_prediction))

        merged_true_labels.append(merged_true_label)
        merged_true_predictions.append(merged_true_prediction)

    return merged_true_predictions, merged_true_labels
