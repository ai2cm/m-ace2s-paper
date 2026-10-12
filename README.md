# M-ACE2S paper: figures, tables and configs

Code, data and configs for the paper **Variable Masking for Climate Emulation in ACE**
(Alexey Yermakov, Troy Arcomano, Jeremy McGibbon, Brian Henn, Oliver Watt-Meyer; Ai2).
<!-- TODO: add arXiv link once available -->

- **Figures:** regenerate every figure from shipped data. See [Setup](#setup) and [`FIGURES.md`](FIGURES.md).
- **Tables:** RMSE, bias and trend error of T7, T0 and TWP over the training and test periods, and the trained configurations, regenerated from shipped data and `configs/`. See [`TABLES.md`](TABLES.md).
- **Training and evaluation:** [`configs/`](configs/) has the exact config of every run behind the figures, each pinned to the [ACE](https://github.com/ai2cm/ace) commit it ran on. See [`configs/README.md`](configs/README.md). The training data is built for an Ai2-internal pipeline, so these configs will not run publicly as-is.

The model code itself lives in [ai2cm/ace](https://github.com/ai2cm/ace).

## Files

| Path | What |
|---|---|
| `paper_figures.ipynb` | Notebook: interactive plotly figures, styled as in the paper |
| `plotting.py` | Public plotting helpers used by the notebook |
| `paper_tables.ipynb` | Notebook: RMSE / bias / trend-error tables (T7, T0, TWP) and configuration table |
| `tables.py` | Table helpers used by `paper_tables.ipynb` |
| `fetch_wandb.py` | Private: rebuilds `data/paper_figures.pkl` from private wandb runs (not needed to reproduce figures) |
| `data/paper_figures.pkl` | Shipped data for all figures |
| `figures/` | PDFs as in the paper; `figures/png/` holds PNG twins |
| `FIGURES.md` | All figures, viewable on GitHub |
| `tables/` | Each table as LaTeX (`.tex`), Markdown (`.md`) and unrounded values (`.csv`) |
| `TABLES.md` | All tables, viewable on GitHub |
| `configs/` | Training and evaluation configs for every run in the figures ([README](configs/README.md)) |

## Setup

Requires Python 3.11 (pinned in `.python-version`; uv installs it if missing).
Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then from the repo root:

```bash
git clone https://github.com/ai2cm/m-ace2s-paper.git
cd m-ace2s-paper
uv sync
uv run jupyter lab paper_figures.ipynb
# or headless (regenerates figures/ and FIGURES.md):
uv run jupyter nbconvert --to notebook --execute --inplace paper_figures.ipynb
# tables (regenerates tables/ and TABLES.md):
uv run jupyter nbconvert --to notebook --execute --inplace paper_tables.ipynb
```

PDF export uses kaleido, which needs a local Chrome/Chromium. If none is installed, run
`uv run plotly_get_chrome` (or `uv run kaleido_get_chrome`).

If `data/paper_figures.pkl` is missing, the notebook calls `fetch_wandb.py`, which needs
`uv sync --extra private` and a `WANDB_API_KEY` with access to the private runs; otherwise it raises.

## Citation

<!-- TODO: replace with the arXiv BibTeX entry once available -->
```bibtex
@article{yermakov2026variable,
  title  = {Variable Masking for Climate Emulation in ACE},
  author = {Yermakov, Alexey and Arcomano, Troy and McGibbon, Jeremy and Henn, Brian and Watt-Meyer, Oliver},
  year   = {2026}
}
```

## License

Apache 2.0, see [LICENSE](LICENSE).
