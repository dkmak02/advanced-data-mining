## Project snapshot

This repository contains three focused Python scripts that form a small scraping + cleaning pipeline:

- `get-movies-url.py` — scrapes Letterboxd popular pages and appends movie Title/URL rows to `data/movies.csv` (creates header when missing).
- `get-movies-data.py` — reads `data/movies.csv`, opens each movie page (Playwright), extracts fields and appends JSON objects into `data/movies_data.json` (writes a JSON array in streaming fashion).
- `clean-genres.py` — single-file genre normalizer using pandas. Configuration is stored as top-level constants in the script (COLUMN, ALLOWED, DROP_EMPTY).

Files to inspect for patterns: `get-movies-url.py`, `get-movies-data.py`, `clean-genres.py`.

## Big picture & data flow

1. Run `get-movies-url.py` to populate `data/movies.csv` with rows: Title,URL.
2. Run `get-movies-data.py` to fetch each URL and append movie objects into `data/movies_data.json`.
3. Run `clean-genres.py` against a file to normalize its genres column (it overwrites the input file).

Notes discovered by reading the code:
- `get-movies-data.py` produces JSON objects with keys: `Title`, `URL`, `Year`, `Genres`, `Description`, `AllReviews` (capitalized `Genres`).
- `clean-genres.py` defaults to `COLUMN = 'genres'` (lowercase). Before running `clean-genres.py` on `data/movies_data.json`, update `COLUMN = 'Genres'` (or change the JSON keys) or the script will exit with "Column X not found".
- `get-movies-data.py` creates the JSON array by writing `[
` then appending objects with trailing commas except the last one; it relies on the `is_last` flag when scheduling tasks to avoid a trailing comma. This streaming pattern is important to preserve when changing the concurrency model.

## Project-specific conventions and patterns

- Configuration-by-constant: `clean-genres.py` expects you to edit constants at the file head (COLUMN, ALLOWED, DROP_EMPTY). The script is intentionally CLI-minimal — it accepts only an input path.
- Overwrite behavior: `clean-genres.py` will overwrite the input file. Treat runs as destructive unless you keep backups.
- Streaming writes with cooperative ordering: `get-movies-data.py` appends to `data/movies_data.json` concurrently. It uses an asyncio.Semaphore and schedules workers; do not remove the `is_last` logic without ensuring the final JSON still forms a valid array.
- Minimal file-creation safety: `get-movies-url.py` and `get-movies-data.py` call `os.makedirs('data', exist_ok=True)` or otherwise ensure the `data/` folder exists.

## Dependencies & developer setup (PowerShell / Windows)

Required Python packages (inferred from imports):
- pandas
- playwright
- aiofiles
- beautifulsoup4

Recommended setup commands (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install pandas aiofiles beautifulsoup4 playwright
# Install Playwright browser binaries (required for Playwright to run)
python -m playwright install
```

If you prefer a single install line instead of individually listed packages, craft a `requirements.txt` with the above names.

## How to run (example pipeline)

1) Build the CSV of movie URLs (may take a few minutes):

```powershell
python get-movies-url.py
```

2) Scrape each movie page into JSON (Playwright requires installed browsers):

```powershell
python get-movies-data.py
```

3) Adjust `clean-genres.py` constants, then normalize genres:

```powershell
# Edit clean-genres.py: set COLUMN = 'Genres' to match get-movies-data output
python clean-genres.py data/movies_data.json
```

## Quick examples to reference in edits

- To change which column is cleaned, edit `clean-genres.py`:

```python
# at top of clean-genres.py
COLUMN = 'Genres'   # <- change to match your data file
ALLOWED = ['rock','pop','jazz', ...]
DROP_EMPTY = False
```

- `get-movies-data.py` uses a semaphore of 5; reduce to 1 if you need sequential writes instead of concurrent appends:

```python
semaphore = asyncio.Semaphore(1)
```

## Safety & caution notes for AI agents

- Don't rename keys or change the JSON structure unless you update all consumers (the column name mismatch is the most obvious fragility).
- If refactoring `get-movies-data.py` to use a single writer, preserve the streaming array format or replace it with a safe writer that writes to a temp file then atomically moves into place.
- Avoid assuming case-insensitive column names — current scripts use exact matches (e.g., `'Genres'` vs `'genres'`).

## Where to look for changes

- `get-movies-url.py` — page scraping & CSV format
- `get-movies-data.py` — Playwright usage, async flow, JSON streaming
- `clean-genres.py` — canonical single-file data-cleaning pattern and configuration-by-constants

---

If you'd like, I can: (a) add a `requirements.txt` and a short `run_pipeline.ps1` wrapper that sets up the venv and runs steps in order; or (b) update `clean-genres.py` to auto-detect `Genres`/`genres` and avoid manual edits. Which should I do next?
