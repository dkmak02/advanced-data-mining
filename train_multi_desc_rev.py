from pathlib import Path
import json
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import hamming_loss, precision_recall_fscore_support



DEFAULT_COLUMN = 'Description'
REV_COLUMN = 'AllReviews'
GENRE_COLUMN = 'Genres'

DEFAULT_GRID = {
    'tfidf__max_features': [5000, 20000],
    'clf__estimator__C': [0.1, 1.0],
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


def filter_texts_and_labels(df: pd.DataFrame, text_col: str, min_examples=30):
    tokens = extract_label(df)
    texts = df[text_col].fillna('').astype(str)
    mask = texts.str.strip() != ''
    texts = texts[mask]
    tokens = tokens[mask]
    all_tokens = [t for ts in tokens for t in ts]
    counts = Counter(all_tokens)
    allowed = {t for t, c in counts.items() if c >= 5}
    filtered = tokens.apply(lambda ts: [t for t in ts if t in allowed])
    mask2 = filtered.apply(bool)
    texts = texts[mask2]
    filtered = filtered[mask2]
    if len(texts) < min_examples:
        return None, None
    return texts, filtered


def multilabel_experiment(df: pd.DataFrame, text_col: str, out_folder: Path, do_search: bool = False):
    texts, labels = filter_texts_and_labels(df, text_col)
    if texts is None:
        return {'error': 'not enough data'}

    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

    mlb = MultiLabelBinarizer()
    Y_train = mlb.fit_transform(y_train)
    Y_test = mlb.transform(y_test)

    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=20000, stop_words='english')),
        ('clf', OneVsRestClassifier(LogisticRegression(max_iter=1000))),
    ])

    if do_search:
        gs = GridSearchCV(pipe, DEFAULT_GRID, scoring='f1_micro', cv=3, n_jobs=-1)
        gs.fit(X_train, Y_train)
        pipe = gs.best_estimator_
    else:
        pipe.fit(X_train, Y_train)

    Y_pred = pipe.predict(X_test)

    # hamming + per-class metrics
    ham = hamming_loss(Y_test, Y_pred)
    prf = precision_recall_fscore_support(Y_test, Y_pred, average='micro', zero_division=0)

    # precision@k where k = avg number of true labels
    ks = Y_test.sum(axis=1)
    avg_k = int(max(1, int(ks.mean())))

    def precision_at_k(y_true_row, y_scores_row, k):
        topk = np.argsort(y_scores_row)[-k:][::-1]
        return float(np.sum(y_true_row[topk]) / k)

    # produce probability scores if possible
    try:
        scores = pipe.decision_function(X_test)
    except Exception:
        try:
            scores = pipe.predict_proba(X_test)
        except Exception:
            scores = Y_pred

    if hasattr(scores, 'shape') and scores.shape == Y_pred.shape:
        prec_at_k = float(np.mean([precision_at_k(Y_test[i], scores[i], avg_k) for i in range(len(scores))]))
    else:
        prec_at_k = None

    metrics = {
        'n_train': int(len(X_train)),
        'n_test': int(len(X_test)),
        'hamming_loss': float(ham),
        'precision_micro': float(prf[0]),
        'recall_micro': float(prf[1]),
        'f1_micro': float(prf[2]),
        'avg_k': int(avg_k),
        'precision_at_k': prec_at_k,
    }

    out_folder.mkdir(parents=True, exist_ok=True)
    (out_folder / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')

    # build simple predictions list; ensure we never save an empty prediction
    preds = []
    # scores may be an ndarray (n_samples, n_classes) or fallback to Y_pred
    for i, idx in enumerate(X_test.index):
        true = [t for t in y_test.iloc[i]]
        pred_labels = []
        if hasattr(Y_pred, 'shape'):
            pred_idxs = np.where(Y_pred[i])[0]
            pred_labels = [mlb.classes_[pi] for pi in pred_idxs]

        # if no labels predicted, fallback to top-1 by model score (improves coverage)
        if not pred_labels:
            try:
                # scores can be ndarray or list-like
                if hasattr(scores, 'shape') and len(getattr(scores, 'shape', [])) == 2:
                    row = scores[i]
                    # if decision_function returned two-column per-class scores, argmax selects best class
                    argmax = int(np.argmax(row))
                    pred_labels = [str(mlb.classes_[argmax])]
                else:
                    # scores not available in expected shape; try to pick any positive from Y_pred
                    if hasattr(Y_pred, 'shape'):
                        nz = np.where(Y_pred[i])[0]
                        if nz.size:
                            pred_labels = [mlb.classes_[int(nz[0])]]
            except Exception:
                pred_labels = []

        preds.append({
            'Title': str(df.at[idx, 'Title']) if 'Title' in df.columns and idx in df.index else '',
            'Text': str(X_test.iloc[i]),
            'Genres_true': true,
            'predicted': pred_labels,
        })

    (out_folder / 'predictions.json').write_text(json.dumps(preds, ensure_ascii=False, indent=2), encoding='utf-8')

    try:
        import joblib
        joblib.dump({'pipeline': pipe, 'mlb': mlb}, out_folder / 'multilabel_pipeline.joblib')
    except Exception:
        pass

    return metrics


def main():
    inp = Path('data/cleaned.json')
    df = load_input(inp)
    # combine description + all reviews
    if REV_COLUMN in df.columns:
        df = df.copy()
        df['desc_all'] = df[DEFAULT_COLUMN].fillna('').astype(str) + '\n' + df[REV_COLUMN].fillna('').astype(str)
        text_col = 'desc_all'
    else:
        text_col = DEFAULT_COLUMN

    out_base = Path('artifacts') / f"experiments_{inp.stem}"
    out_folder = out_base / 'exp4_multi_desc_rev'
    print(f'Input file: {inp}\nOutput folder: {out_folder}')
    res = multilabel_experiment(df, text_col, out_folder, do_search=False)
    print('Done. Results:', res)


if __name__ == '__main__':
    main()
