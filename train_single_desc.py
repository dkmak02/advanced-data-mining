from pathlib import Path
import json
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


DEFAULT_COLUMN = 'Description'
GENRE_COLUMN = 'Genres'

DEFAULT_GRID = {
    'tfidf__max_features': [5000, 20000],
    'clf__C': [0.1, 1.0, 10.0],
    'clf__class_weight': [None, 'balanced'],
}


def load_input(path: Path) -> pd.DataFrame:
    return pd.read_json(path, orient='records')


def extract_label(df: pd.DataFrame):
    def tokenize(genres_cell):
        if pd.isna(genres_cell):
            return []
        s = str(genres_cell)
        if not s:
            return []
        for sep in ['|', ';']:
            s = s.replace(sep, ',')
        parts = [p.strip().lower() for p in s.split(',') if p.strip()]
        parts = [p for p in parts if 'show all' not in p and p != '…' and p != '']
        return parts

    labels = df.get(GENRE_COLUMN)
    if labels is None:
        raise SystemExit(f'Genre column {GENRE_COLUMN} not found')
    return labels.apply(tokenize)


def choose_training_labels(tokens_list: pd.Series) -> pd.Series:
    all_tokens = [t for tokens in tokens_list for t in tokens]
    token_counts = Counter(all_tokens)

    def choose(tokens):
        if not tokens:
            return None
        return max(tokens, key=lambda t: token_counts.get(t, 0))

    return tokens_list.apply(choose)


def filter_min_examples(texts, labels, min_class_count=2, min_examples=30):
    mask = labels.notna() & (texts.str.strip() != '')
    texts = texts[mask]
    labels = labels[mask]
    vc = labels.value_counts()
    rare = vc[vc < min_class_count]
    if not rare.empty:
        keep = ~labels.isin(rare.index)
        texts = texts[keep]
        labels = labels[keep]
    if len(labels) < min_examples:
        return None, None
    return texts, labels


def single_label_experiment(df: pd.DataFrame, text_col: str, out_folder: Path, do_search: bool = False):
    tokens_list = extract_label(df)
    labels = choose_training_labels(tokens_list)
    texts = df[text_col].fillna('').astype(str)

    texts, labels = filter_min_examples(texts, labels)
    if texts is None:
        return {'error': 'not enough data'}

    try:
        X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42, stratify=labels)
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=20000, stop_words='english')),
        ('clf', LogisticRegression(max_iter=1000)),
    ])

    if do_search:
        gs = GridSearchCV(pipe, DEFAULT_GRID, scoring='f1_macro', cv=3, n_jobs=-1)
        gs.fit(X_train, y_train)
        pipe = gs.best_estimator_

    else:
        pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    strict_acc = accuracy_score(y_test, y_pred)

    tokens_test = extract_label(df).reindex(X_test.index)
    any_match = [str(p).lower() in [t.lower() for t in tokens_test.loc[idx]] if idx in tokens_test.index else False
                 for idx, p in zip(X_test.index, y_pred)]
    match_acc = float(sum(any_match)) / len(any_match) if any_match else 0.0

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    out_folder.mkdir(parents=True, exist_ok=True)
    try:
        import joblib
        joblib.dump(pipe, out_folder / 'single_label_pipeline.joblib')
    except Exception:
        pass

    metrics = {
        'n_train': int(len(X_train)),
        'n_test': int(len(X_test)),
        'strict_accuracy': float(strict_acc),
        'match_accuracy_any_match': float(match_acc),
    }

    preds = []
    for idx, pred in zip(X_test.index, y_pred):
        rec = {
            'Title': str(df.at[idx, 'Title']) if 'Title' in df.columns and idx in df.index else '',
            'Description': str(X_test.loc[idx]),
            'Genres_true': extract_label(df).loc[idx] if idx in df.index else [],
            'predicted': str(pred),
            'any_match': bool(str(pred).lower() in [t.lower() for t in extract_label(df).loc[idx]]) if idx in df.index else False,
        }
        preds.append(rec)

    (out_folder / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    (out_folder / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    (out_folder / 'predictions.json').write_text(json.dumps(preds, ensure_ascii=False, indent=2), encoding='utf-8')

    return metrics


def main():
    inp = Path('data/cleaned.json')
    df = load_input(inp)
    out_base = Path('artifacts') / f"experiments_{inp.stem}"
    out_folder = out_base / 'exp1_single_desc'
    print(f'Input file: {inp}\nOutput folder: {out_folder}')
    res = single_label_experiment(df, DEFAULT_COLUMN, out_folder, do_search=False)
    print('Done. Results:', res)


if __name__ == '__main__':
    main()
