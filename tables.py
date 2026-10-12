"""Tables for the masked-training (M-ACE2S) paper.

- `trend_table`: RMSE, bias and linear-trend error of an annual global-mean
  variable (T7, TWP) over the AIMIP training and test years, per
  resolution, from the annual global means in
  data/paper_figures.pkl (the series the annual-mean figures plot).
- `config_table`: the trained configurations, one row per experiment group,
  parsed from configs/*.yaml.

Each table is a DataFrame of raw values plus a LaTeX (booktabs) and a Markdown
rendering; `save` writes all three under tables/.
"""

import glob
import os
import re
from collections import namedtuple

import numpy as np
import pandas as pd
import yaml

from plotting import HERE, YEAR_SPANS, _group

# md_caption: LaTeX caption for TABLES.md when it differs from the .tex caption.
Table = namedtuple("Table", "name df tex md md_caption", defaults=(None,))

_SUPERSCRIPT = str.maketrans("-0123456789a", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹ᵃ")


def md(s):
    """The LaTeX used in these tables, as Markdown/Unicode."""
    s = re.sub(r"\\textbf\{([^}]*)\}", r"**\1**", s)
    s = re.sub(r"\\textit\{([^}]*)\}", r"*\1*", s)
    s = re.sub(r"\\(?:auto)?ref\{([^}]*)\}", r"`\1`", s)
    s = re.sub(r"\^\{([-0-9a-z]+)\}", lambda m: m.group(1).translate(_SUPERSCRIPT), s)
    for a, b in [("$-$", "−"), (r"\pm", "±"), (r"\textdegree{}", "°"), (r"\textdegree\ ", "° "),
                 (r"\Delta ", "Δ"), (r"\ldots", "…"), (r"\{", "{"), (r"\}", "}"), (r"\,", " "),
                 ("--", "–"), ("``", "\""), ("''", "\""), ("$", "")]:
        s = s.replace(a, b)
    return s


def tex(s):
    return s.replace("°", r"\textdegree{}")


def _latex(caption, label, colspec, body, setup=(r"\small",), shrink=False):
    """booktabs table; `shrink` scales a table wider than the text down to fit."""
    tabular = ([rf"\begin{{tabular}}{{{colspec}}}"] + ["  " + b for b in body]
               + [r"\end{tabular}"])
    if shrink:
        tabular = [r"\resizebox{\linewidth}{!}{%"] + tabular + ["}"]
    return "\n".join([r"\begin{table}[H]", rf"  \caption{{{caption}}}", rf"  \label{{{label}}}",
                      r"  \centering"] + ["  " + x for x in list(setup) + tabular]
                     + [r"\end{table}", ""])


# ── RMSE, bias and trend error per period ─────────────────────────────────────

# variable -> (table-name prefix, LaTeX symbol, LaTeX unit)
VARS = {
    "air_temperature_7": ("t7", "$T_7$", "K"),
    "total_water_path": ("twp", "$TWP$", r"kg\,m$^{-2}$"),
}
PERIODS = {name: (lo, hi) for name, lo, hi, _ in YEAR_SPANS if name in ("train", "test")}

Row = namedtuple("Row", "res gmr model source base")
_MAIN = {"1°": ("v5", "-v5-mstepft-bestinf"), "4°": ("v3", "-co2default-v3-bestinf")}
ROWS = [Row(res, gmr_label, model, src, f"nc-sfno-era5-{gmr}-mask{m}{suffix}")
        for res, (src, suffix) in _MAIN.items()
        for gmr, gmr_label in (("gmroff", "GMR off"), ("gmron", "GMR on"))
        for m, model in ((0, "ACE2S"), (20, "M-ACE2S"))]
COLUMNS = [(k, p) for k in ("rmse", "bias", "trend_err") for p in ("train", "test")]
DECIMALS = {"bias": 2, "rmse": 2, "trend_err": 3}


def _annual(entry, var):
    """(years, generated, target) arrays of one run's annual global mean of `var`."""
    traces = {tr["name"]: pd.Series(tr["y"], index=tr["x"], dtype=float).dropna()
              for tr in entry["annual"][f"long_46year/annual/{var}"]}
    gen, tgt = traces["gen"], traces["target"]
    years = gen.index.intersection(tgt.index)
    return years.to_numpy(), gen[years].to_numpy(), tgt[years].to_numpy()


def _trend(years, y):
    """Least-squares linear trend, per decade."""
    return 10.0 * np.polyfit(years, y, 1)[0]


def run_scores(data, var, rows=ROWS):
    """One line per (row, seed): RMSE, bias, ERA5 trend and trend error per period."""
    out = []
    for row in rows:
        for e in _group(data["sources"][row.source]["runs"])[row.base]:
            years, gen, tgt = _annual(e, var)
            rec = dict(row._asdict(), run=e["name"])
            for p, (lo, hi) in PERIODS.items():
                w = (years >= lo) & (years <= hi)
                assert w.sum() == hi - lo + 1, (e["name"], p)
                rec[f"bias_{p}"] = float(np.mean(gen[w] - tgt[w]))
                rec[f"rmse_{p}"] = float(np.sqrt(np.mean((gen[w] - tgt[w]) ** 2)))
                rec[f"era5_trend_{p}"] = _trend(years[w], tgt[w])
                rec[f"trend_err_{p}"] = _trend(years[w], gen[w]) - rec[f"era5_trend_{p}"]
            out.append(rec)
    return pd.DataFrame(out)


def _num(v, decimals):
    s = f"{v:.{decimals}f}"
    if s.lstrip("-").strip("0.") == "":  # no "-0.00"
        s = s.lstrip("-")
    return s.replace("-", "$-$")


def trend_table(data, var, res):
    """RMSE, bias and trend error of `var` for the `res` ("1°" or "4°") models."""
    prefix, symbol, unit = VARS[var]
    scores = run_scores(data, var, [r for r in ROWS if r.res == res])
    keys = ["gmr", "model"]
    g = scores.drop(columns=["res", "source", "base", "run"]).groupby(keys, sort=False)
    mean, std, n = g.mean(), g.std(), g.size()
    cols = [f"{k}_{p}" for k, p in COLUMNS]

    groups = [f"RMSE ({unit})", f"Bias ({unit})", rf"Trend error ({unit}\,decade$^{{-1}}$)"]
    body = [r"\toprule",
            "".join(rf" & \multicolumn{{2}}{{c}}{{{h}}}" for h in groups) + r" \\",
            r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}",
            r" & Train & Test & Train & Test & Train & Test \\"]
    md_lines = ["| | " + " | ".join(md(f"{h}: {p}") for h in groups for p in ("Train", "Test"))
                + " |", "|---|" + "---|" * len(cols)]
    for gmr in mean.index.unique("gmr"):
        body.append(r"\midrule")
        pair = mean.loc[gmr]
        for model in pair.index:
            key = (gmr, model)
            cells = []
            for (kind, _), c in zip(COLUMNS, cols):
                d = DECIMALS[kind]
                a = pair[c].abs().round(d)
                s = _num(mean.loc[key, c], d)
                if a.nunique() == 2 and a[model] == a.min():
                    s = rf"\textbf{{{s}}}"
                if n[key] > 1:
                    s += rf" $\pm$ {_num(std.loc[key, c], d)}"
                cells.append(s)
            label = f"{model} ({gmr})"
            body.append(" & ".join([label] + cells) + r" \\")
            md_lines.append("| " + " | ".join(md(c) for c in [label] + cells) + " |")
    body.append(r"\bottomrule")
    runs = (f"Values are mean $\\pm$ standard deviation over {n.max()} seeds"
            if n.max() > 1 else "Each model is a single training run")
    name = f"{prefix}_trend_{res.replace('°', 'deg')}"
    caption = (
        rf"Errors of the annual global-mean {symbol} of the 46-year ERA5 rollouts of the "
        rf"{tex(res)} models over the AIMIP training (1979--2008) and test (2015--2024) "
        r"periods. RMSE: root-mean-square of model minus ERA5 over the years of the "
        r"period. Bias: mean of model minus ERA5. Trend error: model minus ERA5 "
        r"least-squares linear trend. "
        rf"{runs}; bold marks the smaller magnitude of each ACE2S / M-ACE2S pair.")
    df = mean[cols].join(std[cols], rsuffix="_std").join(n.rename("n_seeds")).reset_index()
    return Table(name, df, _latex(caption, f"table:{name}", "lcccccc", body,
                                  shrink=n.max() > 1),
                 "\n".join(md_lines) + "\n")


# ── configurations ────────────────────────────────────────────────────────────

CONFIG_DIR = os.path.join(HERE, "configs")
# Substrings that would mark an anthropogenic-forcing input channel.
_FORCING_WORDS = ("co2", "ch4", "n2o", "aerosol", "ozone")
_INDICATORS = {(True, False): "channels", (False, True): "FiLM", (True, True): "both"}

# Experiment groups in table order: (name, filename-stem pattern, resolution, where shown).
GROUPS = [
    ("Main", r"gmro(n|ff)-mask(0|20)", "1°",
     r"\ref{sec:results}, \ref{appendix:additional_1deg_results}"),
    ("Main", r"gmro(n|ff)-mask(0|20)", "4°", r"\ref{appendix:additional_4deg_results}"),
    ("Mask distribution", r".*-bern\d+", "4°", r"\ref{appendix:masking_ablations}"),
    ("Mask sharing", r".*-mask(batch|sample)", "4°", r"\ref{appendix:masking_ablations}"),
    ("Mask indicator", r".*-indicator-.*", "4°", r"\ref{appendix:masking_ablations}"),
]


def _load(path):
    with open(path) as f:
        text = f.read()
    return text, yaml.safe_load(text)


def _parse(path):
    """Settings of one training config (pretrain for 1°, train for 4°)."""
    text, cfg = _load(path)
    step = cfg["stepper"]["step"]["config"]
    bad = [c for c in step["in_names"] if any(w in c.lower() for w in _FORCING_WORDS)]
    assert not bad, f"{path}: anthropogenic forcing inputs {bad}"
    drop = step.get("input_dropout") or {}
    k = (drop.get("default") or {}).get("max_masked_vars", 0)
    rates = {g["masking"]["rate"] for g in drop.get("override_groups") or []}
    if rates:
        (rate,) = rates
        masking = f"Bernoulli $p = {rate:.2f}$"
    else:
        masking = f"uniform 0--{k}" if k else "none"
    film = bool(step["builder"]["config"].get("condition_on_channel_mask"))
    (pat,) = {d["file_pattern"] for d in cfg["train_loader"]["dataset"]["concat"]}
    assert cfg["stepper_training"]["n_forward_steps"] == 1, path
    stages = f"{cfg['max_epochs']} SS"
    if path.endswith("-pretrain.yaml"):
        _, ft = _load(path.replace("-pretrain.yaml", "-finetune.yaml"))
        stages += f" + {ft['max_epochs']} MS"
    return dict(
        stem=os.path.basename(path).rsplit("-", 1)[0],
        res=re.search(r"era5-(\d)deg", pat).group(1) + "°",
        dt="1 day" if "daily" in pat else "6 h",
        gmr="on" if step.get("global_mean_removal") else "off",
        masking=masking,
        sharing=None if masking == "none" else ("sample" if drop.get("per_sample") else "batch"),
        indicator=None if masking == "none"
        else _INDICATORS[(bool(step.get("include_channel_mask_inputs")), film)],
        seeds=sorted(int(s) for s in re.findall(r"^#\s+seed (\d+):", text, re.M)),
        stages=stages)


def _join(values):
    return ", ".join(dict.fromkeys(values))


def config_rows(config_dir=CONFIG_DIR):
    """One row per experiment group, merging its configurations' settings."""
    paths = [p for p in sorted(glob.glob(os.path.join(config_dir, "*deg", "*.yaml")))
             if p.endswith(("-pretrain.yaml", "-train.yaml"))]
    configs = [_parse(p) for p in paths]
    used, rows = set(), []
    for name, pattern, res, shown in GROUPS:
        members = [c for c in configs if c["res"] == res and re.fullmatch(pattern, c["stem"])]
        assert members, name
        used.update((c["stem"], c["res"]) for c in members)
        kinds = list(dict.fromkeys(c["masking"] for c in members))
        uniform = [m.split()[1] for m in kinds if m.startswith("uniform")]
        masking = ", ".join([m for m in kinds if not m.startswith("uniform")]
                            + (["uniform " + " / ".join(uniform)] if uniform else []))
        if name == "Mask sharing":
            masking += ", per " + " / ".join(dict.fromkeys(c["sharing"] for c in members))
        if name == "Mask indicator":
            masking += ", " + " / ".join(i for i in _INDICATORS.values()
                                         if any(c["indicator"] == i for c in members))
        seeds = {tuple(c["seeds"]) for c in members}
        if len(seeds) == 1:
            (s,) = seeds
            seed_text = f"{s[0]}--{s[-1]}" if len(s) > 2 else ", ".join(map(str, s))
        else:
            assert all(len(s) == 1 for s in seeds), (name, seeds)
            seed_text = "1$^{a}$"
        rows.append(dict(experiment=name, data=f"{res}, " + _join(c["dt"] for c in members),
                         gmr=_join(sorted(c["gmr"] for c in members)),
                         masking=masking, seeds=seed_text,
                         stages=_join(c["stages"] for c in members), shown=shown,
                         n_configs=len(members)))
    unused = {(c["stem"], c["res"]) for c in configs} - used
    assert not unused, f"configs in no group: {unused}"
    return rows


CONFIG_COLUMNS = [("Experiment", "experiment"), ("Data", "data"), ("GMR", "gmr"),
                  ("Masking", "masking"), ("Seeds", "seeds"), ("Epochs", "stages"),
                  ("Shown in", "shown")]


def config_table(config_dir=CONFIG_DIR):
    rows = config_rows(config_dir)
    heads = [h for h, _ in CONFIG_COLUMNS]
    body = [r"\toprule", " & ".join(heads) + r" \\", r"\midrule"]
    md_lines = ["| " + " | ".join(md(h) for h in heads) + " |", "|" + "---|" * len(heads)]
    for r in rows:
        cells = [r[f] for _, f in CONFIG_COLUMNS]
        body.append(" & ".join(tex(c) for c in cells) + r" \\")
        md_lines.append("| " + " | ".join(md(c) for c in cells) + " |")
    body.append(r"\bottomrule")
    caption = (
        r"Trained configurations, one row per experiment; the released configuration files "
        r"give every run. Data: horizontal resolution and time step of the ERA5 training "
        r"data. Masking: uniform 0--$k$ masks a number of input channels drawn "
        r"uniformly from $\{0, \ldots, k\}$, Bernoulli masks each channel with probability "
        r"$p$; unless stated, {sharing} the model is told which "
        r"channels are masked by appended channels. Epochs: single-step (SS) and multi-step "
        r"(MS) training (\autoref{appendix:multistep_finetuning}). Shown in: section or "
        r"appendix with the results. No configuration takes "
        r"CO$_2$ or any other anthropogenic forcing as an input. $^{a}$One seed per "
        r"model: seed 1, except seed 0 for M-ACE2S with GMR.")
    # The .tex keeps the paper's wording; TABLES.md states how masks are actually shared:
    # each GPU draws its own mask (seeded by rank) and shares it across its local batch.
    tex_caption = caption.replace("{sharing}", "one mask is shared across each GPU's batch and")
    md_caption = caption.replace(
        "{sharing}", "one mask is drawn per GPU at each forward step and shared by that "
        "GPU's samples (4 at 4°, 1 in 1° pretraining, 2 in 1° fine-tuning; masks differ "
        "across GPUs), and")
    return Table("configurations", pd.DataFrame(rows),
                 _latex(tex_caption, "table:configurations", r"lllp{3.3cm}lll", body,
                        setup=[r"\small", r"\setlength{\tabcolsep}{4pt}"]),
                 "\n".join(md_lines) + "\n", md_caption)


# ── output ────────────────────────────────────────────────────────────────────

TABLE_DIR = os.path.join(HERE, "tables")


def save(table, out_dir=TABLE_DIR):
    """Write <name>.tex, <name>.md and <name>.csv under `out_dir`."""
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, table.name)
    with open(stem + ".tex", "w") as f:
        f.write(table.tex)
    with open(stem + ".md", "w") as f:
        f.write(table.md)
    table.df.to_csv(stem + ".csv", index=False, float_format="%.6g")
    return stem


def write_tables_md(tables, path=os.path.join(HERE, "TABLES.md")):
    """TABLES.md: every table in Markdown, viewable on GitHub."""
    lines = ["# Paper tables", "",
             "Generated by `paper_tables.ipynb`. LaTeX in `tables/<name>.tex`, unrounded "
             "values in `tables/<name>.csv`.", ""]
    for t in tables:
        caption = t.md_caption or re.search(r"\\caption\{(.*)\}\n", t.tex).group(1)
        lines += [f"## `{t.name}`", "", md(caption), "", t.md]
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path
