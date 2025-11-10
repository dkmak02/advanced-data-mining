import json
import re
from pathlib import Path
from collections import Counter

INPUT = Path('data/movies_data.json')
OUTPUT = Path('data/genres.json')
SEPARATORS = r'[|,;/]'


def normalize(token: str) -> str:
    return re.sub(r'\s+', ' ', token.strip().lower())


def parse_genres_field(field) -> list:
    if field is None:
        return []
    parts = re.split(SEPARATORS, str(field))
    out = []
    for p in parts:
        n = normalize(p)
        if not n:
            continue
        if 'show all' in n:
            continue
        out.append(n)
    return out


def main():
    if not INPUT.exists():
        print(f'Input not found: {INPUT}')
        return

    with INPUT.open('r', encoding='utf-8') as f:
        data = json.load(f)

    counter = Counter()
    for obj in data:
        # prefer 'Genres' key but allow 'genres' if present
        field = obj.get('Genres') if 'Genres' in obj else obj.get('genres')
        tokens = parse_genres_field(field)
        counter.update(tokens)

    # sort counts descending
    items = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    out = {k: v for k, v in items}

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open('w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f'Wrote {len(out)} genre counts to {OUTPUT}')


if __name__ == '__main__':
    main()
