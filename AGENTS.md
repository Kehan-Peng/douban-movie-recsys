# Repository Guidelines

## Project Structure & Module Organization
`app.py` is the Flask entry point and should stay focused on routing and response handling. Put data access, aggregation, and recommendation logic under `myutils/`, especially `myutils/recommender/` for content-based, collaborative, and hybrid recommendation code. HTML views live in `templates/`; static front-end assets belong in `static/css`, `static/js`, and `static/assets`. Treat `datas.csv`, `top25MovieId.csv`, `pageNum.txt`, and `doubanmovie.sql` as project data/config artifacts rather than application code. Use `spider.py`, `spider_comments.py`, and `word_cloud_picture.py` as standalone scripts.

## Build, Test, and Development Commands
This snapshot does not include a `requirements.txt`, so install dependencies from imports:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install flask pandas scikit-learn requests beautifulsoup4 wordcloud matplotlib
python app.py                 # run Flask app locally
python spider.py              # scrape Top 250 movie data into CSV
python spider_comments.py     # scrape comment pages for one movie
python word_cloud_picture.py  # generate a word cloud from CSV data
```

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indentation and snake_case for functions, variables, and modules. Use PascalCase for classes such as `HybridRecommender`. Keep route handlers thin; move reusable logic into `myutils/`. Preserve the existing template naming pattern when extending views (for example, `actor_t.html` and `comments_c.html`). Prefer small, single-purpose helper functions over large mixed-responsibility scripts.

## Testing Guidelines
No automated test suite is configured yet. For new logic, add `pytest` tests under a new `tests/` directory and name files `test_<module>.py`. At minimum, smoke-test the app with `python app.py`, verify `/`, `/movie/<id>`, and `/recommendations`, and validate scraper output files before committing.

## Commit & Pull Request Guidelines
Git history is not included in this workspace snapshot, so use clear Conventional Commit-style messages such as `feat: add movie search helper` or `fix: handle missing ratings`. PRs should include a short summary, impacted files, setup or sample data changes, and screenshots for template/static updates. Link related issues and note any database or CSV schema changes explicitly.

## Security & Configuration Tips
Do not hard-code secrets, cookies, or local absolute paths. Rate-limit scraping changes responsibly and avoid committing unnecessary generated CSVs or large temporary outputs.
