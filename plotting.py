"""Rebuild the paper's figures from data/paper_figures.pkl.

    from plotting import load_data, show
    data = load_data()
    show(data, "varmasking_v8_seed_v5/val_mean_loss-gmron.pdf")

Each figure is rebuilt the way the original analysis notebook built it: first
the full figure over every run group that notebook held (several PDF details
-- colors, axis ranges, tick sets, category order -- depend on groups that are
not drawn in the PDF), then that notebook's PDF export step, which drops the
other traces and restyles the page. The five sources are:

    v5               1° runs (VarMasking8, "-v5")
    v3               4° runs (VarMasking8, "-v3")
    bern             4° Bernoulli vs uniform masking
    batchsample      4° per-batch vs per-sample masking
    info_comparison  4° masking-information variants (VarMaskingInfoComparison)

wandb is never imported here; only `load_data` reaches for fetch_wandb.py, and
only when the pickle is missing.
"""

import math
import os
import pickle
import re
from collections import defaultdict, namedtuple

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "data", "paper_figures.pkl")


# ── data ──────────────────────────────────────────────────────────────────────

def load_data(path=DATA_PATH):
    """The pickled wandb extract; rebuilt from wandb only if it is missing."""
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    try:
        import fetch_wandb
        fetch_wandb.build(path)
    except Exception as exc:  # ImportError, missing credentials, auth, network...
        raise RuntimeError(
            f"{path} is missing and could not be rebuilt from wandb "
            f"({type(exc).__name__}: {exc}). The pickle ships with the "
            "repository -- restore it (e.g. `git checkout data/`). Rebuilding "
            "it needs private access to the ai2cm wandb projects: "
            "`uv sync --extra private`, then `uv run python fetch_wandb.py`."
        ) from exc
    with open(path, "rb") as f:
        return pickle.load(f)


# ── plotly template ───────────────────────────────────────────────────────────
# White plotting area boxed by black axis lines, applied to every figure.

FIG_BGCOLOR = "#FFFFFF"
FIG_GRIDCOLOR = "#E8E8E8"
FIG_BORDER_COLOR = "#000000"
FIG_BORDER_WIDTH = 1

_axis_style = dict(gridcolor=FIG_GRIDCOLOR, zerolinecolor=FIG_GRIDCOLOR,
                   showline=True, mirror=True,
                   linecolor=FIG_BORDER_COLOR, linewidth=FIG_BORDER_WIDTH)
pio.templates["varmasking"] = go.layout.Template(layout=dict(
    paper_bgcolor=FIG_BGCOLOR, plot_bgcolor=FIG_BGCOLOR,
    xaxis=_axis_style, yaxis=_axis_style,
))
pio.templates.default = "plotly+varmasking"


# ── shared knobs (identical in all five notebooks) ───────────────────────────

PDF_SIZE = {"width": 1100, "height": 720}
PDF_FONT_SIZES = {"title": 28, "axis_title": 22, "tick": 20, "legend": 18}
PDF_TITLE = dict(x=0.0, xref="paper", xanchor="left")
PDF_LEGEND_ITEMWIDTH = 30
PDF_LINE_WIDTH = 5.0
PDF_MARKER_SIZE = 10
PDF_LARGE_FONT_SCALE = 1.5
LINE_WIDTH = 3.5
MARKER_SIZE = 10
TARGET_COLOR = "black"
ALWAYS_VISIBLE = {"target"}
LOSS_OVERLAY = {"val/mean/loss": "train/mean/loss"}

VAR_LABELS = {
    "surface_temperature": ("<i>T</i><sub>s</sub>", "K", 0),
    "air_temperature_0": ("<i>T</i><sub>0</sub>", "K", 0),
    "air_temperature_7": ("<i>T</i><sub>7</sub>", "K", 0),
    "specific_total_water_3": ("<i>q</i><sup>T</sup><sub>3</sub>",
                               "kg kg<sup>−1</sup>", -4),
    "PRATEsfc": ("<i>P</i><sub>sfc</sub>", "kg m<sup>−2</sup> s<sup>−1</sup>", -5),
    "total_water_path": ("<i>TWP</i>", "kg m<sup>−2</sup>", 0),
}
TRAIN_Y_LABELS = {
    "aimip_checkpoint/time_mean_norm/rmse/channel_mean": ("Normalized RMSE", "dimensionless"),
    "val/mean/loss": ("Loss", "dimensionless"),
    "train/mean/loss": ("Loss", "dimensionless"),
}
EPOCH_AXIS_TITLE = "Epoch (log scale)"
EPOCH_TICKS = [1, 2, 5, 10, 20, 50, 100]

PDF_TITLES = {
    "aimip_checkpoint/time_mean_norm/rmse/channel_mean":
        "5-Year Inference Error (normalized RMSE)",
    "val/mean/loss": "Loss (solid = validation, dashed = training)",
    "long_46year/annual/air_temperature_0": "Annual-Mean Air Temperature (level 0)",
    "long_46year/annual/air_temperature_7": "Annual-Mean Air Temperature (level 7)",
    "long_46year/annual/specific_total_water_3": "Annual-Mean Specific Total Water (level 3)",
    "long_46year/annual/PRATEsfc": "Annual-Mean Surface Precipitation Rate",
    "long_46year/annual/total_water_path": "Annual-Mean Total Water Path",
    "long_46year/time_mean/bias": "46-Year Time-Mean Bias",
    "long_46year/time_mean/rmse": "46-Year Time-Mean RMSE",
}
METRIC_KEYS = list(PDF_TITLES)

# Train/val/test year bands behind the annual means: (label, first, last, fill).
YEAR_SPANS = [
    ("train", 1979, 2008, None),
    ("val", 2009, 2014, "#F0F0F0"),
    ("test", 2015, 2024, "#E0E0E0"),
]

# Bias / RMSE scatter: field-name tick labels, plotly html markup.
BIAS_FIELD_SYMBOLS = {
    "PRESsfc": ("p", "", "s"), "surface_temperature": ("T", "", "s"),
    "TMP2m": ("T", "", "2m"), "Q2m": ("q", "", "2m"),
    "UGRD10m": ("u", "", "10m"), "VGRD10m": ("v", "", "10m"),
    "USWRFtoa": ("USWRFtoa", "", ""), "ULWRFtoa": ("ULWRFtoa", "", ""),
    "USWRFsfc": ("USWRFsfc", "", ""), "ULWRFsfc": ("ULWRFsfc", "", ""),
    "DSWRFsfc": ("DSWRFsfc", "", ""), "DLWRFsfc": ("DLWRFsfc", "", ""),
    "PRATEsfc": ("P", "", ""), "LHTFLsfc": ("LHF", "", ""),
    "SHTFLsfc": ("SHF", "", ""), "TMP850": ("T", "", "850"),
    "h500": ("h", "", "500"), "total_water_path": ("TWP", "", ""),
}
BIAS_LEVEL_SYMBOLS = {
    "air_temperature": ("T", "", "{level}"),
    "specific_total_water": ("q", "T", "{level}"),
    "eastward_wind": ("u", "", "{level}"),
    "northward_wind": ("v", "", "{level}"),
}
BIAS_FIELD_LITERALS = {
    "tendency_of_total_water_path_due_to_advection":
        "∂<i>TWP</i>/∂<i>t</i>|<sub>adv</sub>",
}
BIAS_SYMLOG_MAX_TICKS = 8
BIAS_SYMLOG_MIN_TICK_GAP = 0.05
BIAS_ROWS = [("gmroff", "GMR off"), ("gmron", "GMR on")]
BIAS_ROW_OF = {v: i + 1 for i, (v, _) in enumerate(BIAS_ROWS)}
BIAS_ROW_DOMAINS = [[0.60, 0.95], [0.20, 0.55]]
PDF_BIAS_LAYOUT = {
    "height": 1000, "width": 1900,
    "margin": {"t": 90, "b": 220, "l": 90, "r": 40},
    "yaxis": {"domain": [0.52, 0.93]}, "yaxis2": {"domain": [0.0, 0.41]},
    "xaxis": {"automargin": False}, "xaxis2": {"automargin": False},
}

# The interactive figures' layouts; the PDF copies inherit whatever of these
# they do not override (notably legend entrywidth where PDF_LEGEND_ENTRYWIDTH
# is None).
_LINE_LAYOUT = dict(
    height=1450, width=1100,
    legend=dict(orientation="h", x=0, xanchor="left", y=-0.18, yanchor="top",
                font=dict(size=8), entrywidthmode="pixels", entrywidth=250),
    hoverlabel=dict(namelength=-1), margin=dict(t=190, b=720),
)
_SCATTER_LAYOUT = dict(
    height=1600, width=1100,
    legend=dict(orientation="h", x=0, xanchor="left", yref="container", y=0.16,
                yanchor="top", font=dict(size=8), entrywidthmode="pixels",
                entrywidth=250),
    hoverlabel=dict(namelength=-1), margin=dict(t=190, b=80),
)


# ── small helpers ─────────────────────────────────────────────────────────────

def _clean(vals):
    if vals is None:
        return vals
    return [None if isinstance(v, float) and not math.isfinite(v) else v for v in vals]


def _rgb(hex_c):
    r, g, b = (int(hex_c[k:k + 2], 16) for k in (1, 3, 5))
    return f"rgb({r},{g},{b})"


def _color_to_rgba(color_str, alpha=0.15):
    return color_str.replace("rgb(", "rgba(").replace(")", f",{alpha})")


def _slug(text):
    return re.sub(r"[^0-9A-Za-z._-]+", "_", text).strip("_")


def _mask_amount(name):
    m = re.search(r"-mask([0-9.]+)", name)
    return float(m.group(1)) if m else None


def _gmr_variant(name):
    return "gmron" if "gmron" in name else "gmroff"


_GMR_MARKER = {"gmron": "x", "gmroff": "circle"}
_GMR_ORDER = {"gmroff": 0, "gmron": 1}


def _base_name(run_name):
    return re.sub(r"-seed[0-9]", "", re.sub(r"^ace2-var-mask-", "", run_name))


def _group(records):
    groups = defaultdict(list)
    for rec in records:
        groups[_base_name(rec["name"])].append(rec)
    return dict(groups)


def _scalar_group_stats(entries, metric):
    """(x, mean, std) across seeds of a history metric, aligned on epoch."""
    series = []
    for e in entries:
        h = e.get("history", {}).get(metric)
        if not h or not h["value"]:
            continue
        pairs = [(x, y) for x, y in zip(h["epoch"], h["value"])
                 if x is not None and y is not None
                 and isinstance(y, (int, float)) and math.isfinite(y)]
        if pairs:
            xs, ys = zip(*pairs)
            series.append(pd.Series(list(ys), index=list(xs)))
    if not series:
        return [], [], []
    c = pd.concat(series, axis=1).sort_index()
    return c.index.tolist(), c.mean(axis=1).tolist(), c.std(axis=1).fillna(0).tolist()


def _plotly_group_stats(entries, metric):
    """(x, mean, std, first target trace) across seeds of a logged annual figure."""
    gen, target = [], None
    for e in entries:
        traces = e.get("annual", {}).get(metric)
        if not traces:
            continue
        for tr in traces:
            name = tr.get("name", "").lower()
            if "target" in name:
                if target is None:
                    target = tr
            elif "gen" in name:
                pairs = [(x, y) for x, y in zip(tr["x"], tr["y"])
                         if x is not None and y is not None]
                if pairs:
                    xs, ys = zip(*pairs)
                    gen.append(pd.Series(list(ys), index=list(xs)))
    if not gen:
        return [], [], [], target
    c = pd.concat(gen, axis=1).sort_index()
    return (c.index.tolist(), c.mean(axis=1).tolist(),
            c.std(axis=1).fillna(0).tolist(), target)


def _add_mean_std_traces(fig, trace_to_run, x, y_mean, y_std, color, name,
                         group, dash="solid", symbol="circle", band=True):
    """±1σ band (a zero-width one for single-seed groups when `band`), then the
    mean line, both tagged with `group` in trace_to_run."""
    xc, ymc = _clean(x), _clean(y_mean)
    if band:
        yu = [m + s if (m is not None and s is not None) else None
              for m, s in zip(y_mean, y_std)]
        yl = [m - s if (m is not None and s is not None) else None
              for m, s in zip(y_mean, y_std)]
        fig.add_trace(go.Scatter(
            x=xc + xc[::-1], y=_clean(yu) + _clean(yl)[::-1],
            fill="toself", fillcolor=_color_to_rgba(color, 0.15),
            line=dict(color="rgba(0,0,0,0)"), legendgroup=group,
            showlegend=False, hoverinfo="skip", name=f"{name} ±1σ"))
        trace_to_run.append(group)
    fig.add_trace(go.Scatter(
        x=xc, y=ymc, mode="lines+markers", name=name, legendgroup=group,
        connectgaps=True, line=dict(color=color, width=LINE_WIDTH, dash=dash),
        marker=dict(symbol=symbol, size=MARKER_SIZE)))
    trace_to_run.append(group)


def _add_year_spans(fig):
    """Pin x to the plotted years (+ pad) and shade the held-out years."""
    xs = [v for tr in fig.data for v in (tr.x if tr.x is not None else ())
          if isinstance(v, (int, float)) and math.isfinite(v)]
    if not xs:
        return
    lo, hi = min(xs), max(xs)
    pad = max((hi - lo) * 0.02, 0.5)
    x_lo, x_hi = lo - pad, hi + pad
    fig.update_xaxes(range=[x_lo, x_hi])
    edges = [(a - 0.5, b + 0.5, fill) for _, a, b, fill in YEAR_SPANS]
    outer_lo, outer_hi = min(e[0] for e in edges), max(e[1] for e in edges)
    for e0, e1, fill in edges:
        if fill is None:
            continue
        x0 = max(x_lo if e0 == outer_lo else e0, x_lo)
        x1 = min(x_hi if e1 == outer_hi else e1, x_hi)
        if x0 < x1:
            fig.add_vrect(x0=x0, x1=x1, layer="below", line_width=0, fillcolor=fill)


def _var_label(metric):
    var = metric.split("/")[-1]
    return VAR_LABELS.get(var, (var, "", 0))


def _diag_y_label(metric):
    symbol, unit, power = _var_label(metric)
    scale = f"10<sup>{power:d}</sup> ".replace("-", "−") if power else ""
    return f"{symbol} ({scale}{unit})" if unit else symbol


def _train_y_label(metric, y_axis_type):
    quantity, unit = TRAIN_Y_LABELS.get(metric, (metric.split("/")[-1], ""))
    notes = [n for n in (unit, "log scale" if y_axis_type == "log" else "") if n]
    return quantity + (f"<br>({', '.join(notes)})" if notes else "")


def _log_tickvals(values, min_ticks=3):
    """Coarsest mantissa set giving >= min_ticks log ticks inside the range."""
    vals = [v for v in values if isinstance(v, (int, float))
            and math.isfinite(v) and v > 0]
    if not vals:
        return None
    lo, hi = min(vals), max(vals)
    for mantissas in ([1], [1, 3], [1, 2, 5], [1, 1.5, 2, 3, 5, 7]):
        ticks = [m * 10.0 ** e
                 for e in range(math.floor(math.log10(lo)) - 1,
                                math.ceil(math.log10(hi)) + 1)
                 for m in mantissas]
        ticks = [t for t in ticks if lo <= t <= hi]
        if len(ticks) >= min_ticks:
            return ticks
    return None


def _symlog(v, t):
    return math.copysign(math.log10(1.0 + abs(v) / t), v)


def _symlog_ticks(t, vmax, max_ticks):
    """Zero plus ±10^k per decade from the threshold up to vmax, thinned."""
    lo = int(math.floor(math.log10(t)))
    hi = int(math.ceil(math.log10(vmax))) if vmax > 0 else lo
    y_max = _symlog(vmax, t) if vmax > 0 else 1.0
    decades = [k for k in range(lo, hi + 1)
               if _symlog(10.0 ** k, t) >= BIAS_SYMLOG_MIN_TICK_GAP * y_max]
    decades = decades[::max(1, math.ceil(len(decades) / max_ticks))]
    vals, texts = [0.0], ["0"]
    for k in decades:
        for sign in (1, -1):
            vals.append(_symlog(sign * 10.0 ** k, t))
            texts.append(("" if sign > 0 else "−") + f"10<sup>{k}</sup>")
    return vals, texts


def _bias_field_label(field):
    if field in BIAS_FIELD_LITERALS:
        return BIAS_FIELD_LITERALS[field]
    parts = BIAS_FIELD_SYMBOLS.get(field)
    if parts is None:
        prefix, _, level = field.rpartition("_")
        if not (level.isdigit() and prefix in BIAS_LEVEL_SYMBOLS):
            return field
        parts = tuple(p.format(level=level) for p in BIAS_LEVEL_SYMBOLS[prefix])
    base, sup, sub = parts
    return (f"<i>{base}</i>" + (f"<sup>{sup}</sup>" if sup else "")
            + (f"<sub>{sub}</sub>" if sub else ""))


def _bias_row_titles(domains, font_size):
    return [dict(text=label, x=0.0, xref="paper", xanchor="left",
                 y=dom[1], yref="paper", yanchor="bottom",
                 showarrow=False, font=dict(size=font_size))
            for (_, label), dom in zip(BIAS_ROWS, domains)]


# ── the PDF export step ───────────────────────────────────────────────────────

def _collapse_to_row(pdf, row, label):
    """Fold the two-row scatter onto `row`, keeping that row's box height."""
    keep_sfx = "" if row == 1 else str(row)
    drop_sfx = "2" if row == 1 else ""
    pdf.layout.shapes = tuple(s for s in pdf.layout.shapes
                              if str(s.yref).split(" ")[0] == f"y{keep_sfx}")
    domains = [tuple(pdf.layout[f"yaxis{s}"].domain) for s in ("", "2")]
    row_frac = domains[row - 1][1] - domains[row - 1][0]
    gap_frac = 1.0 - max(d[1] for d in domains)
    margin = pdf.layout.margin
    plot_h = pdf.layout.height - margin.t - margin.b
    row_px, gap_px = row_frac * plot_h, gap_frac * plot_h
    single_plot_h = row_px + gap_px
    pdf.layout.height = round(single_plot_h + margin.t + margin.b)
    row_top = row_px / single_plot_h
    pdf.layout[f"yaxis{keep_sfx}"].domain = (0.0, row_top)
    pdf.layout[f"xaxis{drop_sfx}"].visible = False
    pdf.layout[f"yaxis{drop_sfx}"].visible = False
    pdf.layout[f"xaxis{keep_sfx}"].showticklabels = True
    pdf.layout[f"xaxis{keep_sfx}"].matches = None
    pdf.layout.annotations = (dict(
        text=label, x=0.0, xref="paper", xanchor="left",
        y=row_top, yref="paper", yanchor="bottom",
        showarrow=False, font=dict(size=PDF_FONT_SIZES["axis_title"])),)


def _enlarge_pdf_text(pdf, scale):
    """Scale every text size, and grow top/bottom margins (and the page) to fit."""
    lay = pdf.layout

    def grow(font):
        if font.size is not None:
            font.size = font.size * scale

    grow(lay.font)
    grow(lay.title.font)
    grow(lay.legend.font)
    if lay.legend.entrywidth and lay.legend.entrywidthmode != "fraction":
        lay.legend.entrywidth = lay.legend.entrywidth * scale
    for ax in (*pdf.select_xaxes(), *pdf.select_yaxes()):
        grow(ax.title.font)
        grow(ax.tickfont)
    for ann in lay.annotations:
        grow(ann.font)
    for ax in pdf.select_xaxes():
        if ax.tickangle is not None:
            ax.tickangle = -90
    dt, db = lay.margin.t * (scale - 1), lay.margin.b * (scale - 1)
    lay.margin.t, lay.margin.b = lay.margin.t + dt, lay.margin.b + db
    lay.height = round(lay.height + dt + db)
    _share_row_y_title(pdf)


def _share_row_y_title(pdf):
    """Two-row scatters: print a y title both rows share once, as an annotation."""
    lay = pdf.layout
    if len(list(pdf.select_yaxes())) != 2:
        return
    top, bottom = lay.yaxis, lay.yaxis2
    if bottom.visible is False or not top.title.text \
            or top.title.text != bottom.title.text:
        return
    text, font = top.title.text, top.title.font
    top.title.text = bottom.title.text = ""
    tick_w = 3.6 * top.tickfont.size
    lines = text.count("<br>") + 1
    lay.margin.l = lay.margin.l + lines * font.size * 1.3 + font.size
    lay.annotations = (*lay.annotations, dict(
        text=text, textangle=-90, font=font.to_plotly_json(), showarrow=False,
        xref="paper", x=0, xanchor="right", xshift=-tick_w,
        yref="paper", yanchor="middle",
        y=(bottom.domain[0] + top.domain[1]) / 2))


def _pdf_copy(fig, trace_to_run, keep, rename, *, title, margin_b,
              entrywidth, dedupe, large, layout_updates=None, collapse=None):
    """The notebooks' save_pdf, minus the write: keep the traces `keep` accepts,
    rename them for the paper, drop buttons and the title annotation, and apply
    the PDF page, fonts and legend."""
    pdf = go.Figure(fig)
    kept = [rn for rn in trace_to_run if keep(rn)]
    pdf.data = tuple(tr for tr, rn in zip(pdf.data, trace_to_run) if keep(rn))
    names = set()
    for tr, rn in zip(pdf.data, kept):
        tr.visible = True
        tr.name = rename(tr, rn)
        if dedupe and tr.showlegend is not False:
            if tr.name in names:
                tr.showlegend = False
            names.add(tr.name)
        tr.legendrank = 1 if rn in ALWAYS_VISIBLE else 1000
        if tr.line.color != "rgba(0,0,0,0)":
            tr.line.width = PDF_LINE_WIDTH
        if tr.mode and "markers" in tr.mode:
            tr.marker.size = PDF_MARKER_SIZE
    pdf.layout.updatemenus = ()
    pdf.layout.annotations = ()
    legend = dict(orientation="h", x=0, xanchor="left", yref="container",
                  y=0.02, yanchor="bottom",
                  font=dict(size=PDF_FONT_SIZES["legend"]),
                  itemwidth=PDF_LEGEND_ITEMWIDTH)
    if entrywidth is not None:
        legend.update(entrywidthmode="pixels", entrywidth=entrywidth)
    pdf.update_layout(
        title=dict(text=title, **PDF_TITLE, font=dict(size=PDF_FONT_SIZES["title"])),
        showlegend=True, font=dict(size=PDF_FONT_SIZES["tick"]), **PDF_SIZE,
        margin={"t": 90, "b": margin_b, "l": 90, "r": 40}, legend=legend)
    for update_axes in (pdf.update_xaxes, pdf.update_yaxes):
        update_axes(title_font=dict(size=PDF_FONT_SIZES["axis_title"]),
                    tickfont=dict(size=PDF_FONT_SIZES["tick"]),
                    exponentformat="power")
    pdf.update_yaxes(automargin=True)
    if layout_updates:
        pdf.update_layout(**layout_updates)
    if collapse is not None:
        _collapse_to_row(pdf, *collapse)
    if large:
        _enlarge_pdf_text(pdf, PDF_LARGE_FONT_SCALE)
    return pdf


def _relabel(labels):
    """Rename by the trace's own name, keeping a " ±1σ" suffix."""
    def rename(tr, _rn):
        name = tr.name or ""
        base = name[:-len(" ±1σ")] if name.endswith(" ±1σ") else name
        return labels.get(base, base) + name[len(base):]
    return rename


# ── figure builders shared by the sources ─────────────────────────────────────
# A Style bundles what differs per source: colors, markers, dashes, labels,
# legend order, whether single-seed groups get a (zero-width) ±1σ band.

Style = namedtuple("Style", "color symbol dash label sort_key band line_layout train_marker")


def _line_figure(metric, groups, st):
    """Training-progress (history) or annual-mean figure over `groups`."""
    names = sorted(groups, key=st.sort_key)
    fig, trace_to_run = go.Figure(), []
    if metric.startswith("long_46year/annual/"):
        target, factor = None, 10.0 ** -_var_label(metric)[2]
        for bn in names:
            x, ym, ys, tgt = _plotly_group_stats(groups[bn], metric)
            if target is None and tgt is not None:
                target = tgt
            if not x:
                continue
            _add_mean_std_traces(
                fig, trace_to_run, x, [v * factor for v in ym],
                [v * factor for v in ys], st.color(bn), st.label(bn), bn,
                dash=st.dash(bn), symbol=st.symbol(bn), band=st.band(groups[bn]))
        if target is not None:
            fig.add_trace(go.Scatter(
                x=_clean(target["x"]),
                y=_clean([v * factor if v is not None else None for v in target["y"]]),
                mode="lines", name=st.label("target"), legendgroup="target",
                showlegend=True, line=dict(color=TARGET_COLOR, width=LINE_WIDTH)))
            trace_to_run.append("target")
        fig.update_yaxes(title_text=_diag_y_label(metric))
        fig.update_xaxes(title_text="Year")
        _add_year_spans(fig)
    else:
        overlay, all_y = LOSS_OVERLAY.get(metric), []
        for bn in names:
            entries = groups[bn]
            x, ym, ys = _scalar_group_stats(entries, metric)
            if not x:
                continue
            all_y.extend(v for v in ym if isinstance(v, (int, float)) and math.isfinite(v))
            color, symbol, dash = st.color(bn), st.symbol(bn), st.dash(bn)
            band = st.band(entries)
            _add_mean_std_traces(fig, trace_to_run, x, ym, ys, color, st.label(bn),
                                 bn, dash=dash, symbol=symbol, band=band)
            if not overlay:
                continue
            ox, om, osd = _scalar_group_stats(entries, overlay)
            if not ox:
                continue
            all_y.extend(v for v in om if isinstance(v, (int, float)) and math.isfinite(v))
            if band:
                fig.add_trace(go.Scatter(
                    x=_clean(ox) + _clean(ox)[::-1],
                    y=_clean([m + s for m, s in zip(om, osd)])
                    + _clean([m - s for m, s in zip(om, osd)])[::-1],
                    fill="toself", fillcolor=_color_to_rgba(color, 0.15),
                    line=dict(color="rgba(0,0,0,0)"), legendgroup=bn,
                    showlegend=False, hoverinfo="skip"))
                trace_to_run.append(bn)
            train_dash, train_symbol = st.train_marker(dash, symbol)
            fig.add_trace(go.Scatter(
                x=_clean(ox), y=_clean(om), mode="lines+markers",
                name=f"{st.label(bn)} (train)", legendgroup=bn, connectgaps=True,
                showlegend=False,
                line=dict(color=color, width=LINE_WIDTH, dash=train_dash),
                marker=dict(symbol=train_symbol, size=MARKER_SIZE)))
            trace_to_run.append(bn)
        y_type = "log" if all_y and all(v > 0 for v in all_y) else "linear"
        fig.update_xaxes(type="log", title_text=EPOCH_AXIS_TITLE, tickmode="array",
                         tickvals=EPOCH_TICKS, ticktext=[str(t) for t in EPOCH_TICKS])
        fig.update_yaxes(type=y_type, title_text=_train_y_label(metric, y_type))
        if y_type == "log":
            ticks = _log_tickvals(all_y)
            if ticks:
                fig.update_yaxes(tickmode="array", tickvals=ticks,
                                 ticktext=[f"{t:g}" for t in ticks])
    fig.update_layout(**st.line_layout)
    return fig, trace_to_run


def _scatter_figure(kind, groups, st):
    """Two-row (GMR off / on) time-mean bias or RMSE scatter over `groups`.
    The symlog threshold, tick range and field order come from every group."""
    prefix_key = "bias" if kind == "bias" else "rmse"
    fields = sorted({f for entries in groups.values() for e in entries
                     for f in e.get(prefix_key, {})})
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)
    trace_to_run, collected = [], []
    for bn in sorted(groups, key=st.sort_key):
        xs, ym, ys = [], [], []
        for f in fields:
            vals = [e[prefix_key][f] for e in groups[bn] if f in e.get(prefix_key, {})]
            if not vals:
                continue
            s = pd.Series(vals, dtype=float)
            xs.append(f)
            ym.append(float(s.mean()))
            ys.append(float(s.std()) if len(vals) > 1 else 0.0)
        if xs:
            collected.append((bn, xs, ym, ys))

    if kind == "bias":
        pts = [abs(v) for _, _, ym, ys in collected
               for m, sd in zip(ym, ys) for v in (m, m - sd, m + sd)]
        nz = [v for v in pts if v > 0]
        linthresh = min(nz) if nz else 1.0
        vmax = max(pts) if pts else 1.0
        ty = lambda v: _symlog(v, linthresh)  # noqa: E731
    for bn, xs, ym, ys in collected:
        color = st.color(bn)
        if kind == "bias":
            y = [ty(m) for m in ym]
            error_y = dict(type="data", array=[ty(m + sd) - ty(m) for m, sd in zip(ym, ys)],
                           arrayminus=[ty(m) - ty(m - sd) for m, sd in zip(ym, ys)],
                           visible=True, color=color, thickness=1.2, width=3)
        else:
            y = ym
            error_y = dict(type="data", array=ys, visible=True, color=color,
                           thickness=1.2, width=3)
        fig.add_trace(go.Scatter(
            x=xs, y=y, mode="markers", name=st.label(bn), legendgroup=bn,
            marker=dict(symbol=st.symbol(bn), size=MARKER_SIZE, color=color),
            error_y=error_y), row=BIAS_ROW_OF[_gmr_variant(bn)], col=1)
        trace_to_run.append(bn)
    if kind == "bias":
        for row in (1, 2):
            fig.add_hline(y=0, line=dict(color=TARGET_COLOR, width=1), row=row, col=1)
    fig.update_xaxes(title_text="", categoryorder="array", categoryarray=fields,
                     tickmode="array", tickvals=fields,
                     ticktext=[_bias_field_label(f) for f in fields], tickangle=-60)
    if kind == "bias":
        fig.update_yaxes(title_text="Bias (physical units)<br>(symlog scale)")
        tickvals, ticktext = _symlog_ticks(linthresh, vmax, BIAS_SYMLOG_MAX_TICKS)
        fig.update_yaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext)
    else:
        fig.update_yaxes(title_text="RMSE (physical units)<br>(log scale)", type="log")
    for row, dom in enumerate(BIAS_ROW_DOMAINS, start=1):
        fig.update_yaxes(domain=dom, row=row, col=1)
    fig.update_layout(**_SCATTER_LAYOUT)
    return fig, trace_to_run


def _scatter_pdf_updates():
    domains = [PDF_BIAS_LAYOUT["yaxis"]["domain"], PDF_BIAS_LAYOUT["yaxis2"]["domain"]]
    return {**PDF_BIAS_LAYOUT,
            "annotations": _bias_row_titles(domains, PDF_FONT_SIZES["axis_title"])}


def _dashed_train(dash, symbol):
    """Train-loss overlay: dashed line, same marker."""
    return "dash", symbol


# ── source: v5 (1°) ───────────────────────────────────────────────────────────

V5_RUN_NAMES = {
    "ace2-var-mask-nc-sfno-era5-gmroff-mask0-seed1-v5": "No mask",
    "ace2-var-mask-nc-sfno-era5-gmroff-mask20-seed1-v5": "Mask 20",
    "ace2-var-mask-nc-sfno-era5-gmron-mask0-seed1-v5": "No mask, GMR",
    "ace2-var-mask-nc-sfno-era5-gmron-mask20-seed0-v5": "Mask 20, GMR",
}
V5_LEGEND_LABELS = {"Target": "ERA5", "No mask": "ACE2S", "No mask, GMR": "ACE2S",
                    "Mask 20": "M-ACE2S", "Mask 20, GMR": "M-ACE2S"}
V5_LARGE_FONT_FILES = {
    "val_mean_loss-gmron", "val_mean_loss-gmroff",
    "aimip_checkpoint_time_mean_norm_rmse_channel_mean-gmron",
    "aimip_checkpoint_time_mean_norm_rmse_channel_mean-gmroff",
    *(f"long_46year_annual_{v}-{g}-mstepftbestinf"
      for v in ("air_temperature_0", "specific_total_water_3",
                "total_water_path", "PRATEsfc") for g in ("gmroff", "gmron")),
    "long_46year_time_mean_rmse-gmron-mstepftbestinf",
    "long_46year_time_mean_rmse-mstepftbestinf",
}
V5_PALETTE = ["#00bbf9", "#f15bb5", "#ff773d", "#84dd63"]
# trailing tag -> legend/filename token, in checkpoint-button order
V5_TAGS = {"-bestinf": "bestinf", "-lastepoch": "lastepoch",
           "-mstepft-lastepoch": "mstepft", "-mstepft-lastepochema": "mstepftema",
           "-mstepft-bestinf": "mstepftbestinf"}
_V5_ALL_TAGS = tuple(V5_TAGS) + ("-lastepochema", "-besttrain")


def _v5_tag(name):
    for tag in sorted(_V5_ALL_TAGS, key=len, reverse=True):
        if name.endswith(tag):
            return tag
    return None


def _v5_context(data):
    groups = _group(data["sources"]["v5"]["runs"])
    amounts = sorted({a for bn in groups if (a := _mask_amount(bn)) is not None})
    rank = {a: i for i, a in enumerate(amounts)}
    display = {_base_name(rn): label for rn, label in V5_RUN_NAMES.items()}
    display["target"] = "Target"
    tag_order = {t: i for i, t in enumerate(V5_TAGS)}

    def label(bn, with_tag=True):
        if bn in display:
            return display[bn]
        tag = _v5_tag(bn)
        if tag in V5_TAGS:
            base = display.get(bn[: -len(tag)])
            if base is None:
                return bn
            return base if (tag == "-bestinf" or not with_tag) else f"{base} ({V5_TAGS[tag]})"
        return bn

    def color(bn):
        a = _mask_amount(bn)
        return "gray" if a is None else _rgb(V5_PALETTE[rank[a] % len(V5_PALETTE)])

    def sort_key(bn):
        a = _mask_amount(bn)
        return (_GMR_ORDER[_gmr_variant(bn)], a if a is not None else 1e9,
                tag_order.get(_v5_tag(bn), len(tag_order)))

    st = Style(color=color, symbol=lambda bn: _GMR_MARKER[_gmr_variant(bn)],
               dash=lambda bn: "solid", label=label, sort_key=sort_key,
               band=lambda entries: True, line_layout=_LINE_LAYOUT,
               train_marker=_dashed_train)
    untagged = {bn: e for bn, e in groups.items() if _v5_tag(bn) is None}
    diag = {bn: e for bn, e in groups.items()
            if _v5_tag(bn) in ("-bestinf", "-mstepft-lastepoch",
                               "-mstepft-lastepochema", "-mstepft-bestinf")}
    diag.update({bn + "-lastepoch": e for bn, e in untagged.items()})
    return st, untagged, diag


def _v5_figure(data, metric, gmr, tag):
    st, untagged, diag = _v5_context(data)
    if metric.startswith("long_46year/time_mean/"):
        fig, ttr = _scatter_figure(metric.split("/")[-1], diag, st)
        updates = _scatter_pdf_updates()
        collapse = None if gmr is None else (BIAS_ROW_OF[gmr], dict(BIAS_ROWS)[gmr])
    else:
        fig, ttr = _line_figure(metric, diag if "/annual/" in metric else untagged, st)
        updates = collapse = None
    tag_full = {v: k for k, v in V5_TAGS.items()}.get(tag)

    def keep(rn):
        if rn in ALWAYS_VISIBLE:
            return True
        if tag_full is not None and _v5_tag(rn) != tag_full:
            return False
        return gmr is None or _gmr_variant(rn) == gmr

    def rename(tr, rn):
        base = st.label(rn, with_tag=False)
        base = V5_LEGEND_LABELS.get(base, base)
        return f"{base} ±1σ" if (tr.name or "").endswith("±1σ") else base

    stem = "-".join([_slug(metric)] + ([gmr] if gmr else []) + ([tag] if tag else []))
    return _pdf_copy(fig, ttr, keep, rename, title=PDF_TITLES[metric], margin_b=160,
                     entrywidth=150, dedupe=True, large=stem in V5_LARGE_FONT_FILES,
                     layout_updates=updates, collapse=collapse)


# ── sources: v3 and bern (4°) ─────────────────────────────────────────────────

V3_DISPLAY_NAMES = {
    "nc-sfno-era5-gmroff-mask0-co2default-v3": "No mask",
    "nc-sfno-era5-gmroff-mask10-co2default-v3": "Mask 10",
    "nc-sfno-era5-gmroff-mask20-co2default-v3": "Mask 20",
    "nc-sfno-era5-gmroff-mask0-co2bern75-v3": "No mask, CO2 Bern75",
    "nc-sfno-era5-gmroff-mask10-co2bern75-v3": "Mask 10, CO2 Bern75",
    "nc-sfno-era5-gmroff-mask20-co2bern75-v3": "Mask 20, CO2 Bern75",
    "nc-sfno-era5-gmron-mask0-co2default-v3": "No mask, GMR",
    "nc-sfno-era5-gmron-mask10-co2default-v3": "Mask 10, GMR",
    "nc-sfno-era5-gmron-mask20-co2default-v3": "Mask 20, GMR",
    "nc-sfno-era5-gmron-mask0-co2bern75-v3": "No mask, GMR, CO2 Bern75",
    "nc-sfno-era5-gmron-mask10-co2bern75-v3": "Mask 10, GMR, CO2 Bern75",
    "nc-sfno-era5-gmron-mask20-co2bern75-v3": "Mask 20, GMR, CO2 Bern75",
    "target": "Target",
}
BERN_DISPLAY_NAMES = {
    **V3_DISPLAY_NAMES,
    "nc-sfno-era5-gmroff-mask20-co2default-v3": "Uniform (mean 10)",
    "nc-sfno-era5-gmron-mask20-co2default-v3": "Uniform (mean 10), GMR",
    "nc-sfno-era5-gmroff-bern10-co2default-v3": "Bernoulli (mean 10)",
    "nc-sfno-era5-gmron-bern10-co2default-v3": "Bernoulli (mean 10), GMR",
}
_V3_DEFAULT = {f"nc-sfno-era5-{g}-mask{a}-{c}-v3-bestinf" for g in ("gmron", "gmroff")
               for a in (0, 10, 20) for c in ("co2default", "co2bern75")}
_BERN_DEFAULT = {f"nc-sfno-era5-{g}-{m}-co2default-v3-bestinf"
                 for g in ("gmron", "gmroff") for m in ("mask0", "mask20", "bern10")}
V3_LEGEND_LABELS = V5_LEGEND_LABELS
BERN_LEGEND_LABELS = {
    "Target": "ERA5", "No mask": "ACE2S", "No mask, GMR": "ACE2S",
    "Uniform (mean 10)": "M-ACE2S uniform masking",
    "Uniform (mean 10), GMR": "M-ACE2S uniform masking",
    "Bernoulli (mean 10)": "M-ACE2S Bernoulli masking",
    "Bernoulli (mean 10), GMR": "M-ACE2S Bernoulli masking",
}
V3_LARGE_FONT_FILES = {
    *(f"long_46year_annual_{v}-{g}"
      for v in ("air_temperature_7", "air_temperature_0", "specific_total_water_3",
                "total_water_path", "PRATEsfc") for g in ("gmroff", "gmron")),
    "val_mean_loss-gmron", "val_mean_loss-gmroff",
    "aimip_checkpoint_time_mean_norm_rmse_channel_mean-gmron",
    "aimip_checkpoint_time_mean_norm_rmse_channel_mean-gmroff",
    "long_46year_time_mean_rmse",
}
BERN_LARGE_FONT_FILES = {"long_46year_annual_air_temperature_7-gmroff-bern",
                         "long_46year_annual_air_temperature_7-gmron-bern"}
V3_PALETTE = ["#00bbf9", "#f15bb5", "#78d68c"]
V3_FIXED_MASK_COLORS = {0.0: "#00bbf9", 20.0: "#f15bb5"}
_V3_TAG_ORDER = {"-lastepoch": 0, "-besttrain": 1, "-bestinf": 2}
_V3_TAG_SUFFIX_RE = re.compile(r"-(?:lastepoch|besttrain|bestinf)(?=$|-)")
_EXCLUDE = ("-sston", "-clock50")


def _v3_tag_rank(name):
    for tag, i in _V3_TAG_ORDER.items():
        if tag in name:
            return i
    return len(_V3_TAG_ORDER)


def _co2_rank(name):
    return 1 if "co2bern75" in name else 0


def _bern_amount(name):
    m = re.search(r"-bern([0-9.]+)", name)
    return float(m.group(1)) if m else None


_FAMILY_RANK = {"none": 0, "uniform": 1, "bernoulli": 2}


def _mask_family(name):
    b = _bern_amount(name)
    if b is not None:
        return "bernoulli" if b > 0 else "none"
    a = _mask_amount(name)
    if a is None:
        return None
    return "uniform" if a > 0 else "none"


def _v3_label(display):
    def label(bn):
        if bn in display:
            return display[bn]
        m = _V3_TAG_SUFFIX_RE.search(bn)
        if not m:
            return bn
        base = display.get(bn[:m.start()] + bn[m.end():])
        if base is None:
            return bn
        return base if m.group(0) == "-bestinf" else f"{base} ({m.group(0).lstrip('-')})"
    return label


def _v3_groups(data, source, keep_tags):
    recs = [r for r in data["sources"][source]["runs"]
            if not any(p in r["name"] for p in _EXCLUDE)]
    groups = _group(recs)
    tagged = {bn: e for bn, e in groups.items() if any(t in bn for t in keep_tags)}
    untagged = {bn: e for bn, e in groups.items() if not any(t in bn for t in keep_tags)}
    return groups, tagged, untagged


def _v3_style(groups):
    amounts = sorted({a for bn in groups if (a := _mask_amount(bn)) is not None})
    rank = {a: i for i, a in enumerate(amounts)}

    def color(bn):
        a = _mask_amount(bn)
        if a is None:
            return "gray"
        if a in V3_FIXED_MASK_COLORS:
            return _rgb(V3_FIXED_MASK_COLORS[a])
        return _rgb(V3_PALETTE[rank[a] % len(V3_PALETTE)])

    def sort_key(bn):
        a = _mask_amount(bn)
        return (_GMR_ORDER[_gmr_variant(bn)], a if a is not None else 1e9,
                _v3_tag_rank(bn), _co2_rank(bn))

    return Style(color=color, symbol=lambda bn: _GMR_MARKER[_gmr_variant(bn)],
                 dash=lambda bn: "solid", label=_v3_label(V3_DISPLAY_NAMES),
                 sort_key=sort_key, band=lambda entries: True,
                 line_layout=_LINE_LAYOUT, train_marker=_dashed_train)


def _is_default(rn, default):
    if rn in ALWAYS_VISIBLE:
        return True
    if _V3_TAG_SUFFIX_RE.search(rn):
        return rn in default
    return rn in {_V3_TAG_SUFFIX_RE.sub("", g) for g in default}


def _v3_figure(data, metric, gmr):
    groups, tagged, untagged = _v3_groups(data, "v3", tuple(_V3_TAG_ORDER))
    st = _v3_style(groups)
    if metric.startswith("long_46year/time_mean/"):
        fig, ttr = _scatter_figure(metric.split("/")[-1], tagged, st)
        updates = _scatter_pdf_updates()
        collapse = None if gmr is None else (BIAS_ROW_OF[gmr], dict(BIAS_ROWS)[gmr])
    else:
        fig, ttr = _line_figure(metric, tagged if "/annual/" in metric else untagged, st)
        updates = collapse = None

    def keep(rn):
        if not _is_default(rn, _V3_DEFAULT):
            return False
        return gmr is None or rn in ALWAYS_VISIBLE or _gmr_variant(rn) == gmr

    stem = _slug(metric) + (f"-{gmr}" if gmr else "")
    return _pdf_copy(fig, ttr, keep, _relabel(V3_LEGEND_LABELS),
                     title=PDF_TITLES[metric], margin_b=160, entrywidth=150,
                     dedupe=True, large=stem in V3_LARGE_FONT_FILES,
                     layout_updates=updates, collapse=collapse)


def _bern_figure(data, metric, gmr):
    groups, tagged, untagged = _v3_groups(data, "bern", ("-bestinf",))
    # Bernoulli notebook: one color per masking family, family in the legend order.
    def color(bn):
        fam = _mask_family(bn)
        return "gray" if fam is None else _rgb(V3_PALETTE[_FAMILY_RANK[fam] % len(V3_PALETTE)])

    def sort_key(bn):
        a, b = _mask_amount(bn), _bern_amount(bn)
        return (_GMR_ORDER[_gmr_variant(bn)],
                _FAMILY_RANK.get(_mask_family(bn), len(_FAMILY_RANK)),
                a if a is not None else (b if b is not None else 1e9),
                _v3_tag_rank(bn), _co2_rank(bn))

    st = Style(color=color, symbol=lambda bn: _GMR_MARKER[_gmr_variant(bn)],
               dash=lambda bn: "solid", label=_v3_label(BERN_DISPLAY_NAMES),
               sort_key=sort_key, band=lambda entries: True,
               line_layout=_LINE_LAYOUT, train_marker=_dashed_train)
    fig, ttr = _line_figure(metric, tagged, st)

    def keep(rn):
        if not _is_default(rn, _BERN_DEFAULT):
            return False
        return gmr is None or rn in ALWAYS_VISIBLE or _gmr_variant(rn) == gmr

    stem = _slug(metric) + (f"-{gmr}" if gmr else "") + "-bern"
    return _pdf_copy(fig, ttr, keep, _relabel(BERN_LEGEND_LABELS),
                     title=PDF_TITLES[metric], margin_b=200, entrywidth=None,
                     dedupe=True, large=stem in BERN_LARGE_FONT_FILES)


# ── source: batchsample (4°) ──────────────────────────────────────────────────

BATCHSAMPLE_RUN_NAMES = {
    "ace2-var-mask-nc-sfno-era5-gmron-mask10-maskbatch-seed0-v3": "Mask 10, per-batch mask",
    "ace2-var-mask-nc-sfno-era5-gmron-mask10-masksample-seed0-v3": "Mask 10, per-sample mask",
    "ace2-var-mask-nc-sfno-era5-gmron-mask20-maskbatch-seed0-v3": "Mask 20, per-batch mask",
    "ace2-var-mask-nc-sfno-era5-gmron-mask20-masksample-seed0-v3": "Mask 20, per-sample mask",
}
BATCHSAMPLE_LEGEND_LABELS = {
    "Target": "ERA5",
    "Mask 10, per-batch mask": "Per-batch mask", "Mask 10, per-sample mask": "Per-sample mask",
    "Mask 20, per-batch mask": "Per-batch mask", "Mask 20, per-sample mask": "Per-sample mask",
}
BATCHSAMPLE_LARGE_FONT_FILES = {
    "long_46year_annual_air_temperature_7-mask10-batchsample",
    "long_46year_annual_air_temperature_7-mask20-batchsample",
}
BATCHSAMPLE_PALETTE = ["#86b6ef", "#256abf", "#104281"]
_SHARE_MODES = ["maskbatch", "masksample"]
_MODE_DASH = {"maskbatch": "solid", "masksample": "dash"}
_MODE_MARKER = {"maskbatch": "circle", "masksample": "x"}


def _share_mode(name):
    return next((m for m in _SHARE_MODES if f"-{m}" in name), None)


def _batchsample_figure(data, metric, amount):
    groups = _group(data["sources"]["batchsample"]["runs"])
    amounts = sorted({a for bn in groups if (a := _mask_amount(bn)) is not None})
    rank = {a: i for i, a in enumerate(amounts)}
    display = {_base_name(n): label for n, label in BATCHSAMPLE_RUN_NAMES.items()}
    display["target"] = "Target"
    st = Style(
        color=lambda bn: _rgb(BATCHSAMPLE_PALETTE[rank[_mask_amount(bn)] % 3]),
        symbol=lambda bn: _MODE_MARKER[_share_mode(bn)],
        dash=lambda bn: _MODE_DASH[_share_mode(bn)],
        label=lambda bn: display.get(re.sub(r"-bestinf$", "", bn), bn),
        sort_key=lambda bn: (_mask_amount(bn) if _mask_amount(bn) is not None else 1e9,
                             _SHARE_MODES.index(_share_mode(bn))
                             if _share_mode(bn) else len(_SHARE_MODES)),
        band=lambda entries: len(entries) > 1,
        line_layout=dict(_LINE_LAYOUT, height=800, margin=dict(t=120, b=180),
                         legend=dict(_LINE_LAYOUT["legend"], font=dict(size=11))),
        train_marker=lambda dash, symbol: (dash, symbol + "-open"),
    )
    tagged = {bn: e for bn, e in groups.items() if bn.endswith("-bestinf")}
    fig, ttr = _line_figure(metric, tagged, st)

    def keep(rn):  # the single-amount copy, then save_pdf's default-group filter
        return rn in ALWAYS_VISIBLE or amount is None or _mask_amount(rn) == amount

    stem = _slug(metric) + (f"-mask{amount:g}" if amount is not None else "") + "-batchsample"
    return _pdf_copy(fig, ttr, keep, _relabel(BATCHSAMPLE_LEGEND_LABELS),
                     title=PDF_TITLES[metric], margin_b=160, entrywidth=None,
                     dedupe=False, large=stem in BATCHSAMPLE_LARGE_FONT_FILES)


# ── source: info_comparison (4°) ──────────────────────────────────────────────

INFO_TYPES = {
    "concat": {"label": "Channel mask", "color": "#0072B2", "marker": "circle"},
    "film": {"label": "LayerNorm", "color": "#D55E00", "marker": "x"},
    "both": {"label": "Both", "color": "#009E73", "marker": "diamond"},
}


def _info_type(name):
    hits = [t for t in INFO_TYPES if t in name]
    return hits[0] if len(hits) == 1 else None


def _info_figure(data, metric):
    groups = _group(data["sources"]["info_comparison"]["runs"])
    order = {t: i for i, t in enumerate(INFO_TYPES)}

    def label(bn):
        if bn == "target":
            return "Target"
        t = _info_type(bn)
        return INFO_TYPES[t]["label"] if t else bn

    st = Style(
        color=lambda bn: _rgb(INFO_TYPES[_info_type(bn)]["color"]) if _info_type(bn) else "gray",
        symbol=lambda bn: INFO_TYPES[_info_type(bn)]["marker"] if _info_type(bn) else "circle",
        dash=lambda bn: "solid", label=label,
        sort_key=lambda bn: (order.get(_info_type(bn), len(order)), bn),
        band=lambda entries: len(entries) > 1, line_layout=_LINE_LAYOUT,
        train_marker=_dashed_train)
    tagged = {bn: e for bn, e in groups.items() if bn.endswith("-bestinf")}
    fig, ttr = _line_figure(metric, tagged, st)
    return _pdf_copy(fig, ttr, lambda rn: True, _relabel({}),
                     title=PDF_TITLES[metric], margin_b=160, entrywidth=150,
                     dedupe=True, large=False)


# ── registry ──────────────────────────────────────────────────────────────────

Figure = namedtuple("Figure", "section label title source pdf")

_R = "Results: 46-year Climate Rollouts"
_A = "Masked Training Ablations"
_D1 = "Additional 1° Results"
_D1T = "Additional 1° Results: Time Mean Variable Plots"
_D4 = "Additional 4° Results"
_D4T = "Additional 4° Results: Time Mean Variable Plots"
_D4M = "Additional 4° Results: Training and Checkpoint Selection Metrics"
_V5, _V3 = "varmasking_v8_seed_v5/", "varmasking_v8_seed_v3/"

FIGURES = [
    Figure(_R, "fig:46yr_era5_T7_1deg", "Annual-mean T7, GMR on, 1°", "v5",
           _V5 + "long_46year_annual_air_temperature_7-gmron-mstepftbestinf.pdf"),
    Figure(_R, "fig:46yr_era5_time_mean_rmse_gmron_1deg", "Time-mean RMSE, GMR on, 1°", "v5",
           _V5 + "long_46year_time_mean_rmse-gmron-mstepftbestinf.pdf"),
    Figure(_R, "fig:46yr_era5_training_metrics_1deg_gmron", "Training and validation loss, GMR on, 1°", "v5",
           _V5 + "val_mean_loss-gmron.pdf"),
    Figure(_R, "fig:46yr_era5_training_metrics_1deg_gmron", "5-year inference error, GMR on, 1°", "v5",
           _V5 + "aimip_checkpoint_time_mean_norm_rmse_channel_mean-gmron.pdf"),
    Figure(_A, "fig:46yr_era5_T7_bern_4deg", "Annual-mean T7, Bernoulli vs uniform masking, GMR off, 4°", "bern",
           _V3 + "long_46year_annual_air_temperature_7-gmroff-bern.pdf"),
    Figure(_A, "fig:46yr_era5_T7_bern_4deg", "Annual-mean T7, Bernoulli vs uniform masking, GMR on, 4°", "bern",
           _V3 + "long_46year_annual_air_temperature_7-gmron-bern.pdf"),
    Figure(_A, "fig:46yr_era5_T7_batch_comparison_4deg", "Annual-mean T7, per-batch vs per-sample masking, mask 10, 4°", "batchsample",
           _V3 + "long_46year_annual_air_temperature_7-mask10-batchsample.pdf"),
    Figure(_A, "fig:46yr_era5_T7_batch_comparison_4deg", "Annual-mean T7, per-batch vs per-sample masking, mask 20, 4°", "batchsample",
           _V3 + "long_46year_annual_air_temperature_7-mask20-batchsample.pdf"),
    Figure(_A, "fig:46yr_era5_T7_indicator_comparison_4deg", "Annual-mean T7, masking-information variants, 4°", "info_comparison",
           "varmasking_info_comparison/long_46year_annual_air_temperature_7.pdf"),
    Figure(_D1, "fig:46yr_era5_T7_gmroff_1deg", "Annual-mean T7, GMR off, 1°", "v5",
           _V5 + "long_46year_annual_air_temperature_7-gmroff-mstepftbestinf.pdf"),
    Figure(_D1, "fig:46yr_era5_training_metrics_1deg_gmroff", "Training and validation loss, GMR off, 1°", "v5",
           _V5 + "val_mean_loss-gmroff.pdf"),
    Figure(_D1, "fig:46yr_era5_training_metrics_1deg_gmroff", "5-year inference error, GMR off, 1°", "v5",
           _V5 + "aimip_checkpoint_time_mean_norm_rmse_channel_mean-gmroff.pdf"),
]
for _var, _short, _lab in [("air_temperature_0", "T0", "T0"),
                           ("specific_total_water_3", "q^T_3", "Q3"),
                           ("total_water_path", "TWP", "TWP"),
                           ("PRATEsfc", "P_sfc", "Psfc")]:
    for _g, _gt in (("gmroff", "GMR off"), ("gmron", "GMR on")):
        FIGURES.append(Figure(_D1, f"fig:46yr_era5_{_lab}_1deg",
                              f"Annual-mean {_short}, {_gt}, 1°", "v5",
                              _V5 + f"long_46year_annual_{_var}-{_g}-mstepftbestinf.pdf"))
FIGURES += [
    Figure(_D1T, "fig:46yr_time_mean_rmse_era5_1deg", "Time-mean RMSE, GMR off and on, 1°", "v5",
           _V5 + "long_46year_time_mean_rmse-mstepftbestinf.pdf"),
    Figure(_D1T, "fig:46yr_time_mean_bias_era5_1deg", "Time-mean bias, GMR off and on, 1°", "v5",
           _V5 + "long_46year_time_mean_bias-mstepftbestinf.pdf"),
]
for _var, _short, _lab in [("air_temperature_7", "T7", "T7"),
                           ("air_temperature_0", "T0", "T0"),
                           ("specific_total_water_3", "q^T_3", "Q3"),
                           ("total_water_path", "TWP", "TWP"),
                           ("PRATEsfc", "P_sfc", "Psfc")]:
    for _g, _gt in (("gmroff", "GMR off"), ("gmron", "GMR on")):
        FIGURES.append(Figure(_D4, f"fig:46yr_era5_{_lab}_4deg",
                              f"Annual-mean {_short}, {_gt}, 4°", "v3",
                              _V3 + f"long_46year_annual_{_var}-{_g}.pdf"))
FIGURES += [
    Figure(_D4T, "fig:46yr_time_mean_rmse_era5_4deg", "Time-mean RMSE, GMR off and on, 4°", "v3",
           _V3 + "long_46year_time_mean_rmse.pdf"),
    Figure(_D4T, "fig:46yr_time_mean_bias_era5_4deg", "Time-mean bias, GMR off and on, 4°", "v3",
           _V3 + "long_46year_time_mean_bias.pdf"),
]
for _g, _gt in (("gmron", "GMR on"), ("gmroff", "GMR off")):
    FIGURES += [
        Figure(_D4M, f"fig:46yr_era5_training_metrics_4deg_{_g}",
               f"Training and validation loss, {_gt}, 4°", "v3", _V3 + f"val_mean_loss-{_g}.pdf"),
        Figure(_D4M, f"fig:46yr_era5_training_metrics_4deg_{_g}",
               f"5-year inference error, {_gt}, 4°", "v3",
               _V3 + f"aimip_checkpoint_time_mean_norm_rmse_channel_mean-{_g}.pdf"),
    ]

_BY_PDF = {f.pdf: f for f in FIGURES}
_SLUG_TO_METRIC = {_slug(m): m for m in METRIC_KEYS}


def _strip(stem, suffix):
    return (stem[: -len(suffix)], True) if suffix and stem.endswith(suffix) else (stem, False)


def _parse(stem, suffix=""):
    """Split "<metric slug>[-gmron|-gmroff]<suffix>" into (metric, gmr)."""
    stem, _ = _strip(stem, suffix)
    gmr = None
    for g in ("gmron", "gmroff"):
        stem, hit = _strip(stem, "-" + g)
        if hit:
            gmr = g
            break
    return _SLUG_TO_METRIC[stem], gmr


def paper_figure(data, pdf_path):
    """The PDF-styled plotly figure for one of the paper's PDFs (see FIGURES)."""
    source = _BY_PDF[pdf_path].source
    stem = os.path.splitext(os.path.basename(pdf_path))[0]
    if source == "v5":
        tag = next((t for t in V5_TAGS.values() if stem.endswith("-" + t)), None)
        metric, gmr = _parse(stem, f"-{tag}" if tag else "")
        return _v5_figure(data, metric, gmr, tag)
    if source == "v3":
        return _v3_figure(data, *_parse(stem))
    if source == "bern":
        return _bern_figure(data, *_parse(stem, "-bern"))
    if source == "batchsample":
        stem, _ = _strip(stem, "-batchsample")
        m = re.search(r"-mask([0-9.]+)$", stem)
        amount = float(m.group(1)) if m else None
        return _batchsample_figure(data, _SLUG_TO_METRIC[stem[: m.start()] if m else stem], amount)
    return _info_figure(data, _SLUG_TO_METRIC[stem])


# ── output ────────────────────────────────────────────────────────────────────

_server_started = False


def _start_kaleido():
    """One shared headless Chrome for every export, as in the notebooks."""
    global _server_started
    if not _server_started:
        import kaleido
        kaleido.start_sync_server(n=4, timeout=60, mathjax=False, silence_warnings=True)
        pio.defaults.headers = {}
        _server_started = True


def save(fig, pdf_path, out_dir="figures"):
    """Write <out_dir>/<pdf_path> and a 2x PNG twin under <out_dir>/png/."""
    out_dir = os.path.join(HERE, out_dir) if not os.path.isabs(out_dir) else out_dir
    _start_kaleido()
    pdf = os.path.join(out_dir, pdf_path)
    png = os.path.join(out_dir, "png", os.path.splitext(pdf_path)[0] + ".png")
    for path in (pdf, png):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.write_image(pdf, format="pdf")
    fig.write_image(png, format="png", scale=2)
    return pdf


def show(data, pdf_path, out_dir="figures"):
    """Build, save and display one paper figure."""
    fig = paper_figure(data, pdf_path)
    save(fig, pdf_path, out_dir)
    fig.show()


def write_figures_md(path=os.path.join(HERE, "FIGURES.md")):
    """FIGURES.md: every paper figure as a PNG thumbnail linking to its PDF."""
    lines = ["# Paper figures", "",
             "The figure PDFs included by the paper, in paper order. Each "
             "thumbnail links to its PDF.", ""]
    section = None
    for f in FIGURES:
        if f.section != section:
            section = f.section
            lines += [f"## {section}", ""]
        png = "figures/png/" + os.path.splitext(f.pdf)[0] + ".png"
        lines += [f"### `{f.label}` — {f.title}", "",
                  f"[![{f.label}]({png})](figures/{f.pdf})", ""]
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
    return path
