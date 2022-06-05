LABEL_LIST = [
    'citation-number',
    'collection-title',
    'container-title',
    'doi',
    'issue',
    'number-of-pages',
    'volume',
    'issued',
    'year',
    'month',
    'day',
    'author',
    'editor',
    'page',
    'publisher',
    'title',
    'url',
    'year-suffix'
]

id2label = {str(i): label for i, label in enumerate(LABEL_LIST)}
label2id = {v: k for k, v in id2label.items()}
