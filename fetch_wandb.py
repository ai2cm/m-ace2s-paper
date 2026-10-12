"""Rebuild data/paper_figures.pkl from wandb (private: needs ai2cm wandb access).

    uv sync --extra private
    uv run python fetch_wandb.py

Everything the paper's figures need is pulled fresh from wandb and slimmed to
plain Python types. The run sets are frozen below as explicit name lists (the
original notebooks matched some of them by regex over the whole project), so
runs added to the projects later do not change the figures.
"""

import datetime
import json
import math
import netrc
import os
import pickle
import shutil
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PATH = os.path.join(HERE, "data", "paper_figures.pkl")
CACHE_DIR = os.path.join(HERE, ".wandb_cache")

ENTITY = "ai2cm"
PROJECT = "VarMasking8"
INFO_PROJECT = "VarMaskingInfoComparison"

TRAIN_HISTORY = [
    "aimip_checkpoint/time_mean_norm/rmse/channel_mean",
    "train/mean/loss",
    "val/mean/loss",
]
ANNUAL = [
    "long_46year/annual/air_temperature_0",
    "long_46year/annual/air_temperature_7",
    "long_46year/annual/specific_total_water_3",
    "long_46year/annual/PRATEsfc",
    "long_46year/annual/total_water_path",
]
T7 = ["long_46year/annual/air_temperature_7"]
BIAS_PREFIX = "long_46year/time_mean/bias/"
RMSE_PREFIX = "long_46year/time_mean/rmse/"

# ── frozen run sets ───────────────────────────────────────────────────────────
# 1° (v5) notebook: RUN_NAMES plus the eval siblings it plots. The finetune's own
# "-mstepft" training runs feed no paper figure and are not fetched.
V5_BASE = [
    "ace2-var-mask-nc-sfno-era5-gmroff-mask0-seed1-v5",
    "ace2-var-mask-nc-sfno-era5-gmroff-mask20-seed1-v5",
    "ace2-var-mask-nc-sfno-era5-gmron-mask0-seed1-v5",
    "ace2-var-mask-nc-sfno-era5-gmron-mask20-seed0-v5",
]
V5_TAGS = ["", "-bestinf", "-mstepft-lastepoch", "-mstepft-lastepochema",
           "-mstepft-bestinf"]
V5_RUNS = [b + t for b in V5_BASE for t in V5_TAGS]

# 4° (v3) notebooks: every seeded "-v3" run in VarMasking8 except the -sston and
# -clock50 one-offs, in the order the original notebook held them.
V3_RUNS = [
    "ace2-var-mask-nc-sfno-era5-gmroff-mask0-co2default-seed1-v3",
    "ace2-var-mask-nc-sfno-era5-gmroff-mask0-co2default-seed2-v3",
    "ace2-var-mask-nc-sfno-era5-gmroff-mask0-co2default-seed3-v3",
    "ace2-var-mask-nc-sfno-era5-gmroff-mask20-co2default-seed1-v3",
    "ace2-var-mask-nc-sfno-era5-gmroff-mask20-co2default-seed0-v3",
    "ace2-var-mask-nc-sfno-era5-gmroff-mask20-co2default-seed2-v3",
    "ace2-var-mask-nc-sfno-era5-gmroff-mask20-co2default-seed3-v3",
    "ace2-var-mask-nc-sfno-era5-gmron-mask0-co2default-seed0-v3",
    "ace2-var-mask-nc-sfno-era5-gmroff-mask20-co2default-seed4-v3",
    "ace2-var-mask-nc-sfno-era5-gmroff-mask0-co2default-seed4-v3",
    "ace2-var-mask-nc-sfno-era5-gmroff-mask0-co2default-seed0-v3",
    "ace2-var-mask-nc-sfno-era5-gmron-mask20-co2default-seed0-v3",
    "ace2-var-mask-nc-sfno-era5-gmron-mask0-co2default-seed1-v3",
    "ace2-var-mask-nc-sfno-era5-gmron-mask0-co2default-seed3-v3",
    "ace2-var-mask-nc-sfno-era5-gmron-mask20-co2default-seed4-v3",
    "ace2-var-mask-nc-sfno-era5-gmron-mask0-co2default-seed2-v3",
    "ace2-var-mask-nc-sfno-era5-gmron-mask20-co2default-seed3-v3",
    "ace2-var-mask-nc-sfno-era5-gmron-mask20-co2default-seed1-v3",
    "ace2-var-mask-nc-sfno-era5-gmron-mask20-co2default-seed2-v3",
    "ace2-var-mask-nc-sfno-era5-gmron-mask0-co2default-seed4-v3",
    "ace2-var-mask-nc-sfno-era5-gmroff-mask0-co2default-seed0-v3-besttrain",
    "ace2-var-mask-nc-sfno-era5-gmroff-mask0-co2default-seed1-v3-besttrain",
    "ace2-var-mask-nc-sfno-era5-gmroff-mask0-co2default-seed2-v3-bestinf",
    "ace2-var-mask-nc-sfno-era5-gmroff-mask0-co2default-seed3-v3-bestinf",
    "ace2-var-mask-nc-sfno-era5-gmroff-mask20-co2default-seed0-v3-bestinf",
    "ace2-var-mask-nc-sfno-era5-gmroff-mask0-co2default-seed1-v3-bestinf",
    "ace2-var-mask-nc-sfno-era5-gmroff-mask20-co2default-seed2-v3-bestinf",
    "ace2-var-mask-nc-sfno-era5-gmroff-mask0-co2default-seed2-v3-besttrain",
    "ace2-var-mask-nc-sfno-era5-gmroff-mask20-co2default-seed1-v3-besttrain",
    "ace2-var-mask-nc-sfno-era5-gmroff-mask0-co2default-seed3-v3-lastepoch",
    "ace2-var-mask-nc-sfno-era5-gmroff-mask20-co2default-seed3-v3-besttrain",
    "ace2-var-mask-nc-sfno-era5-gmroff-mask20-co2default-seed3-v3-bestinf",
    "ace2-var-mask-nc-sfno-era5-gmroff-mask20-co2default-seed2-v3-besttrain",
    "ace2-var-mask-nc-sfno-era5-gmroff-mask20-co2default-seed2-v3-lastepoch",
    "ace2-var-mask-nc-sfno-era5-gmron-mask0-co2default-seed1-v3-besttrain",
    "ace2-var-mask-nc-sfno-era5-gmroff-mask20-co2default-seed3-v3-lastepoch",
    "ace2-var-mask-nc-sfno-era5-gmron-mask0-co2default-seed0-v3-lastepoch",
    "ace2-var-mask-nc-sfno-era5-gmroff-mask20-co2default-seed1-v3-lastepoch",
    "ace2-var-mask-nc-sfno-era5-gmroff-mask20-co2default-seed0-v3-besttrain",
    "ace2-var-mask-nc-sfno-era5-gmroff-mask20-co2default-seed1-v3-bestinf",
    "ace2-var-mask-nc-sfno-era5-gmroff-mask20-co2default-seed0-v3-lastepoch",
    "ace2-var-mask-nc-sfno-era5-gmroff-mask0-co2default-seed4-v3-besttrain",
    "ace2-var-mask-nc-sfno-era5-gmroff-mask0-co2default-seed4-v3-bestinf",
    "ace2-var-mask-nc-sfno-era5-gmron-mask0-co2default-seed1-v3-bestinf",
    "ace2-var-mask-nc-sfno-era5-gmron-mask0-co2default-seed1-v3-lastepoch",
    "ace2-var-mask-nc-sfno-era5-gmroff-mask0-co2default-seed1-v3-lastepoch",
    "ace2-var-mask-nc-sfno-era5-gmron-mask0-co2default-seed0-v3-besttrain",
    "ace2-var-mask-nc-sfno-era5-gmron-mask0-co2default-seed2-v3-besttrain",
    "ace2-var-mask-nc-sfno-era5-gmron-mask0-co2default-seed0-v3-bestinf",
    "ace2-var-mask-nc-sfno-era5-gmroff-mask0-co2default-seed3-v3-besttrain",
    "ace2-var-mask-nc-sfno-era5-gmroff-mask0-co2default-seed2-v3-lastepoch",
    "ace2-var-mask-nc-sfno-era5-gmroff-mask0-co2default-seed4-v3-lastepoch",
    "ace2-var-mask-nc-sfno-era5-gmroff-mask0-co2default-seed0-v3-lastepoch",
    "ace2-var-mask-nc-sfno-era5-gmroff-mask0-co2default-seed0-v3-bestinf",
    "ace2-var-mask-nc-sfno-era5-gmron-mask0-co2default-seed2-v3-bestinf",
    "ace2-var-mask-nc-sfno-era5-gmron-mask0-co2default-seed2-v3-lastepoch",
    "ace2-var-mask-nc-sfno-era5-gmroff-mask20-co2default-seed4-v3-besttrain",
    "ace2-var-mask-nc-sfno-era5-gmron-mask0-co2default-seed4-v3-bestinf",
    "ace2-var-mask-nc-sfno-era5-gmron-mask0-co2default-seed4-v3-besttrain",
    "ace2-var-mask-nc-sfno-era5-gmron-mask20-co2default-seed0-v3-bestinf",
    "ace2-var-mask-nc-sfno-era5-gmroff-mask20-co2default-seed4-v3-bestinf",
    "ace2-var-mask-nc-sfno-era5-gmron-mask20-co2default-seed2-v3-besttrain",
    "ace2-var-mask-nc-sfno-era5-gmron-mask20-co2default-seed3-v3-besttrain",
    "ace2-var-mask-nc-sfno-era5-gmron-mask0-co2default-seed3-v3-bestinf",
    "ace2-var-mask-nc-sfno-era5-gmron-mask20-co2default-seed4-v3-besttrain",
    "ace2-var-mask-nc-sfno-era5-gmron-mask20-co2default-seed2-v3-bestinf",
    "ace2-var-mask-nc-sfno-era5-gmron-mask0-co2default-seed3-v3-lastepoch",
    "ace2-var-mask-nc-sfno-era5-gmron-mask20-co2default-seed4-v3-bestinf",
    "ace2-var-mask-nc-sfno-era5-gmron-mask0-co2default-seed4-v3-lastepoch",
    "ace2-var-mask-nc-sfno-era5-gmron-mask20-co2default-seed3-v3-lastepoch",
    "ace2-var-mask-nc-sfno-era5-gmron-mask20-co2default-seed0-v3-lastepoch",
    "ace2-var-mask-nc-sfno-era5-gmron-mask20-co2default-seed1-v3-lastepoch",
    "ace2-var-mask-nc-sfno-era5-gmron-mask20-co2default-seed3-v3-bestinf",
    "ace2-var-mask-nc-sfno-era5-gmron-mask20-co2default-seed0-v3-besttrain",
    "ace2-var-mask-nc-sfno-era5-gmron-mask20-co2default-seed2-v3-lastepoch",
    "ace2-var-mask-nc-sfno-era5-gmron-mask20-co2default-seed1-v3-bestinf",
    "ace2-var-mask-nc-sfno-era5-gmron-mask20-co2default-seed4-v3-lastepoch",
    "ace2-var-mask-nc-sfno-era5-gmron-mask20-co2default-seed1-v3-besttrain",
    "ace2-var-mask-nc-sfno-era5-gmroff-mask20-co2default-seed4-v3-lastepoch",
    "ace2-var-mask-nc-sfno-era5-gmron-mask0-co2default-seed3-v3-besttrain",
    "ace2-var-mask-nc-sfno-era5-gmroff-bern10-co2default-seed0-v3",
    "ace2-var-mask-nc-sfno-era5-gmroff-bern10-co2default-seed2-v3",
    "ace2-var-mask-nc-sfno-era5-gmroff-bern10-co2default-seed1-v3",
    "ace2-var-mask-nc-sfno-era5-gmroff-bern10-co2default-seed3-v3",
    "ace2-var-mask-nc-sfno-era5-gmroff-bern10-co2default-seed4-v3",
    "ace2-var-mask-nc-sfno-era5-gmron-bern10-co2default-seed0-v3",
    "ace2-var-mask-nc-sfno-era5-gmron-bern10-co2default-seed2-v3",
    "ace2-var-mask-nc-sfno-era5-gmron-bern10-co2default-seed1-v3",
    "ace2-var-mask-nc-sfno-era5-gmron-bern10-co2default-seed3-v3",
    "ace2-var-mask-nc-sfno-era5-gmron-bern10-co2default-seed4-v3",
    "ace2-var-mask-nc-sfno-era5-gmron-bern10-co2default-seed0-v3-bestinf",
    "ace2-var-mask-nc-sfno-era5-gmroff-bern10-co2default-seed1-v3-bestinf",
    "ace2-var-mask-nc-sfno-era5-gmroff-bern10-co2default-seed2-v3-bestinf",
    "ace2-var-mask-nc-sfno-era5-gmron-bern10-co2default-seed1-v3-bestinf",
    "ace2-var-mask-nc-sfno-era5-gmroff-bern10-co2default-seed0-v3-bestinf",
    "ace2-var-mask-nc-sfno-era5-gmron-bern10-co2default-seed2-v3-bestinf",
    "ace2-var-mask-nc-sfno-era5-gmroff-bern10-co2default-seed3-v3-bestinf",
    "ace2-var-mask-nc-sfno-era5-gmroff-bern10-co2default-seed4-v3-bestinf",
    "ace2-var-mask-nc-sfno-era5-gmron-bern10-co2default-seed3-v3-bestinf",
    "ace2-var-mask-nc-sfno-era5-gmron-bern10-co2default-seed4-v3-bestinf",
    "ace2-var-mask-nc-sfno-era5-gmron-mask10-maskbatch-seed0-v3",
    "ace2-var-mask-nc-sfno-era5-gmron-mask20-masksample-seed0-v3",
    "ace2-var-mask-nc-sfno-era5-gmron-mask20-maskbatch-seed0-v3",
    "ace2-var-mask-nc-sfno-era5-gmron-mask10-masksample-seed0-v3",
    "ace2-var-mask-nc-sfno-era5-gmron-mask20-masksample-seed0-v3-bestinf",
    "ace2-var-mask-nc-sfno-era5-gmron-mask10-masksample-seed0-v3-bestinf",
    "ace2-var-mask-nc-sfno-era5-gmron-mask20-maskbatch-seed0-v3-bestinf",
    "ace2-var-mask-nc-sfno-era5-gmron-mask10-maskbatch-seed0-v3-bestinf",
]

# Per-batch vs per-sample masking: the four configs' -bestinf eval runs.
BATCHSAMPLE_RUNS = [
    "ace2-var-mask-nc-sfno-era5-gmron-mask10-maskbatch-seed0-v3-bestinf",
    "ace2-var-mask-nc-sfno-era5-gmron-mask10-masksample-seed0-v3-bestinf",
    "ace2-var-mask-nc-sfno-era5-gmron-mask20-maskbatch-seed0-v3-bestinf",
    "ace2-var-mask-nc-sfno-era5-gmron-mask20-masksample-seed0-v3-bestinf",
]

# Masking-info comparison: the three variants' -bestinf eval runs.
INFO_RUNS = [
    "ace2-var-mask-nc-sfno-mask20-uniform-noco2-both-seed0-bestinf",
    "ace2-var-mask-nc-sfno-mask20-uniform-noco2-concat-seed0-bestinf",
    "ace2-var-mask-nc-sfno-mask20-uniform-noco2-film-seed0-bestinf",
]

_V3_TAGS = ("-lastepoch", "-besttrain", "-bestinf")


def _is_eval(name):
    return any(t in name for t in _V3_TAGS) or "-mstepft" in name


def _needs_v5(name):
    if _is_eval(name):
        return {"annual": ANNUAL, "bias": True, "rmse": True}
    # The training runs carry the training curves and, as the "lastepoch"
    # checkpoint, annual figures and time-mean scalars.
    return {"history": TRAIN_HISTORY, "annual": ANNUAL, "bias": True, "rmse": True}


def _needs_v3(name):
    if _is_eval(name):
        return {"annual": ANNUAL, "bias": True, "rmse": True}
    return {"history": TRAIN_HISTORY}


# source -> (project, [(run name, what to fetch)])
SOURCES = {
    "v5": (PROJECT, [(n, _needs_v5(n)) for n in V5_RUNS]),
    "v3": (PROJECT, [(n, _needs_v3(n)) for n in V3_RUNS]),
    "bern": (PROJECT, [(n, {"annual": T7}) for n in V3_RUNS
                       if n.endswith("-bestinf")]),
    "batchsample": (PROJECT, [(n, {"annual": T7}) for n in BATCHSAMPLE_RUNS]),
    "info_comparison": (INFO_PROJECT, [(n, {"annual": T7}) for n in INFO_RUNS]),
}


# ── credentials ───────────────────────────────────────────────────────────────

def _check_credentials():
    """Fail fast instead of letting wandb prompt for a login."""
    if os.environ.get("WANDB_API_KEY"):
        return
    host = urlparse(os.environ.get("WANDB_BASE_URL", "https://api.wandb.ai")).hostname
    path = os.environ.get("NETRC") or os.path.expanduser("~/.netrc")
    try:
        if netrc.netrc(path).authenticators(host):
            return
    except (OSError, netrc.NetrcParseError):
        pass
    raise RuntimeError(
        f"No wandb credentials: set WANDB_API_KEY or add {host} to {path}.")


# ── fetching ──────────────────────────────────────────────────────────────────

def _resolve(api, project, names):
    """Name -> the finished wandb run of that name."""
    wanted = set(names)
    found = {}
    for run in api.runs(f"{ENTITY}/{project}"):
        if run.name in wanted:
            found.setdefault(run.name, []).append(run)
    out = {}
    for name in names:
        finished = [r for r in found.get(name, []) if r.state == "finished"]
        if len(finished) != 1:
            states = [(r.id, r.state) for r in found.get(name, [])]
            raise RuntimeError(
                f"{ENTITY}/{project}: expected one finished run named {name!r}, "
                f"got {states}")
        out[name] = finished[0]
    return out


def _history(run, key):
    # One key per scan: a multi-key fetch inner-joins and drops rows whose
    # cadences differ. scan_history pages through everything (history(samples=)
    # silently caps at 10000 rows).
    epochs, values = [], []
    for row in run.scan_history(keys=["epoch", key], page_size=10000):
        val = row.get(key)
        if val is None or (isinstance(val, float) and math.isnan(val)):
            continue
        epochs.append(row.get("epoch"))
        values.append(float(val))
    return {"epoch": epochs, "value": values}


def _plotly_figure(run, ref):
    path, sha = ref["path"], ref.get("sha256", "")
    cached = os.path.join(CACHE_DIR, "plotly",
                          (sha[:20] if sha else os.path.basename(path)) + ".json")
    if not os.path.exists(cached):
        os.makedirs(os.path.dirname(cached), exist_ok=True)
        dl_dir = os.path.join(CACHE_DIR, "_dl", run.id)
        run.file(path).download(root=dl_dir, replace=True)
        shutil.copy(os.path.join(dl_dir, path), cached)
    with open(cached) as f:
        return json.load(f)


def _slim_traces(fig):
    """The target and generated curves of a logged annual-mean figure."""
    keep = []
    for tr in fig.get("data", []):
        name = tr.get("name", "")
        low = name.lower()
        if "target" in low or "gen" in low:
            keep.append({"name": name, "x": list(tr.get("x", [])),
                         "y": list(tr.get("y", []))})
    return keep


def _summary(summary, prefix, skip=()):
    return {k[len(prefix):]: float(v) for k, v in summary.items()
            if k.startswith(prefix) and isinstance(v, (int, float))
            and not isinstance(v, bool) and k[len(prefix):] not in skip}


def _fetch_run(run, needs):
    rec = {"name": run.name, "id": run.id}
    summary = dict(run.summary.items()) if (
        needs.get("annual") or needs.get("bias") or needs.get("rmse")) else {}
    if needs.get("history"):
        rec["history"] = {k: _history(run, k) for k in needs["history"]}
    if needs.get("annual"):
        rec["annual"] = {}
        for m in needs["annual"]:
            ref = summary.get(m)
            if hasattr(ref, "get") and ref.get("path"):
                rec["annual"][m] = _slim_traces(_plotly_figure(run, ref))
    if needs.get("bias"):
        rec["bias"] = _summary(summary, BIAS_PREFIX)
    if needs.get("rmse"):
        rec["rmse"] = _summary(summary, RMSE_PREFIX, skip={"channel_mean"})
    return rec


def _merge_needs(a, b):
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, list):
            out[k] = sorted(set(out.get(k, [])) | set(v))
        else:
            out[k] = out.get(k) or v
    return out


def build(path=DEFAULT_PATH):
    """Fetch every source from wandb and write the pickle to `path`."""
    _check_credentials()
    import wandb

    api = wandb.Api(timeout=60)
    needs = {}  # (project, name) -> union of what any source wants
    for project, runs in SOURCES.values():
        for name, need in runs:
            key = (project, name)
            needs[key] = _merge_needs(needs.get(key, {}), need)

    resolved = {}
    for project in sorted({p for p, _ in needs}):
        names = [n for p, n in needs if p == project]
        for name, run in _resolve(api, project, names).items():
            resolved[(project, name)] = run

    def fetch(key):
        return key, _fetch_run(resolved[key], needs[key])

    with ThreadPoolExecutor(max_workers=16) as ex:
        records = dict(ex.map(fetch, list(needs)))

    sources = {}
    for src, (project, runs) in SOURCES.items():
        out = []
        for name, need in runs:
            full = records[(project, name)]
            rec = {"name": name, "id": full["id"]}
            for k in ("history", "annual"):
                if need.get(k):
                    rec[k] = {m: full[k][m] for m in need[k] if m in full[k]}
            for k in ("bias", "rmse"):
                if need.get(k):
                    rec[k] = full[k]
            out.append(rec)
        sources[src] = {"project": project, "runs": out}

    data = {
        "meta": {
            "fetched_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "entity": ENTITY,
            "projects": sorted({p for p, _ in SOURCES.values()}),
            "runs": {src: [n for n, _ in runs] for src, (_, runs) in SOURCES.items()},
            "run_ids": {f"{p}/{n}": r.id for (p, n), r in resolved.items()},
        },
        "sources": sources,
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    return data


if __name__ == "__main__":
    d = build()
    n = sum(len(s["runs"]) for s in d["sources"].values())
    print(f"wrote {DEFAULT_PATH} ({os.path.getsize(DEFAULT_PATH) / 1e6:.2f} MB, "
          f"{n} run records)")
