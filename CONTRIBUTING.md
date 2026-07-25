# Contributing

Thanks for taking a look. This project is a small, self-contained data analysis
in pure Python; everything below assumes a source checkout.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]        # editable install + pytest & ruff
```

## Everyday commands

| Task                    | `make`          | direct                                   |
| ----------------------- | --------------- | ---------------------------------------- |
| Build the analysis CSVs | `make data`     | `python scripts/build_dataset.py`        |
| Render the 2D figures   | `make figures`  | `python scripts/make_figures.py`         |
| Render the 3D figures   | `make viz3d`    | `python scripts/make_3d.py`              |
| Run the tests           | `make test`     | `python -m pytest`                       |
| Lint                    | `make lint`     | `ruff check .`                           |
| Format                  | `make format`   | `ruff format .`                          |
| Render the report       | `make report`   | `quarto render report/…qmd` (Quarto CLI) |

## Conventions

- **Pure Python, no JavaScript.** Figures are matplotlib and return a
  `matplotlib.figure.Figure`; 3D figures live in `figures3d.py` and share the
  `theme.py` palette and helpers. Do not reintroduce Plotly or other
  browser-rendered chart libraries.
- **Every number in the report is computed from the committed data files** by the
  code in `src/mitski_analysis/`. Add a metric in `text.py`/`stats.py`, surface
  it in `data.py`, and only then read it in a figure or the report.
- **Tests, lint, and formatting must pass** (`make test && make lint`, and
  `ruff format --check .`) before a PR; CI runs all three plus a headless
  pipeline build on Python 3.10–3.12. The whole repo, `scraper/` included, is
  linted and formatted.
- Keep the data-viz house style: fixed-order categorical colour by entity, the
  single-hue blue ramp for magnitude, recessive grey chrome, direct labels over
  legends where it reads cleanly, one axis per chart.
- Lyrics are quoted only in short excerpts, for commentary and criticism.
