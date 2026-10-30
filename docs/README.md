# Report and presentation

Built from the saved results in `reports/`; nothing here retrains a model.

- `report/main.tex` -> `report/main.pdf`: IEEE-style two-column paper. Build: `cd docs/report && pdflatex main.tex` (twice).
- `deck/portfolio_risk_ml.pptx`: 25-slide deck with speaker notes.
- `make_figures.py`, `make_tables.py`, `deck/build_deck.py`: regenerate figures, LaTeX tables and the deck.
  They need `matplotlib`, `pandas`, `pyarrow` and `python-pptx`, which are NOT in `requirements.txt`;
  use a separate virtualenv (`uv venv docenv && uv pip install --python docenv/bin/python matplotlib pandas pyarrow python-pptx`).
- Placeholders to fill before submission: author names/registration numbers and mentor (title block of
  `main.tex`, first slide of the deck).
- Table/figure numbers in the text come from the CSVs in `reports/`; if results are re-run, regenerate
  with the three scripts, then re-read the prose that quotes numbers.
