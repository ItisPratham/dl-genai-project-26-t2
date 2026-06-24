import re, numpy as np, pandas as pd

OPTIONS = ["A", "B", "C", "D", "E"]

def split_prompt(prompt):
    text = str(prompt).strip()
    prefix, suffix = '', ''

    ci = text.find(': ')
    if ci != -1:
        prefix = text[:ci].strip()
        text = text[ci + 1:].strip()

    n_q = text.count('?')

    if n_q == 1:
        qi = text.index('?')
        suffix = text[qi + 1:].strip()
        text = text[:qi + 1].strip()

    elif n_q == 2:
        first_q = text.index('?')
        prefix = text[:first_q + 1].strip()

        text = text[first_q + 1:].strip()
        second_q = text.index('?')

        suffix = text[second_q + 1:].strip()
        text = text[:second_q + 1].strip()

    return (prefix or None), text, (suffix or None)


def clean(text, is_prompt=True):
    if(is_prompt):
        text = split_prompt(text)[1]
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def tokenize(text, is_prompt=True):
    return clean(text, is_prompt).split()


def ap_at_3(actual, predicted):
    for i, p in enumerate(predicted[:3]):
        if p == actual:
            return 1.0 / (i + 1)
    return 0.0


def map_at_3(actuals, predictions):
    return float(np.mean([ap_at_3(a, p) for a, p in zip(actuals, predictions)]))


def make_submission(ids, predictions):
    return pd.DataFrame({
        'id': ids,
        'prediction': [' '.join(p[:3]) for p in predictions]
    })