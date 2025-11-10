import sys
from pathlib import Path

import pandas as pd

COLUMN = 'Genres'
ALLOWED = [
    'Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary', 'Drama',
    'Family', 'Fantasy', 'History', 'Horror', 'Music', 'Mystery', 'Romance',
    'Science Fiction', 'Thriller', 'TV Movie', 'War', 'Western'
]


def normalize(tok: str) -> str:
    return tok.strip().lower()


def clean_df_with_allowed(df: pd.DataFrame, col: str, allowed_list) -> pd.DataFrame:
    allowed_map = {normalize(g): g for g in allowed_list}

    def clean_cell(cell):
        if pd.isna(cell):
            return ''
        s = str(cell)
        for sep in ['|', ';']:
            s = s.replace(sep, ',')
        parts = [normalize(p) for p in s.split(',') if p.strip()]
        kept = [allowed_map[t] for t in parts if t in allowed_map]
        return '|'.join(kept)

    df2 = df.copy()
    df2[col] = df2[col].map(clean_cell)
    return df2


def main():
    movies_path = Path('data/movies_data.json')
    if not movies_path.exists():
        raise SystemExit(f'Missing {movies_path}. Run get-movies-data.py first.')
    df = pd.read_json(movies_path, orient='records')

    if COLUMN not in df.columns:
        print(f"Column '{COLUMN}' not found in {movies_path}.")
        sys.exit(1)

    out_dir = Path('data')
    out_path = out_dir / 'cleaned.json'
    cleaned = clean_df_with_allowed(df, COLUMN, ALLOWED)
    cleaned.to_json(out_path, orient='records', force_ascii=False, indent=2)
    print(f'Wrote {out_path} with {len(ALLOWED)} allowed genres.')


if __name__ == '__main__':
    main()
