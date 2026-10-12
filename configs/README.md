# Training and evaluation configs

These are the configs for every run behind the paper's figures. Each file holds the config exactly as it ran, unmodified. A header at the top of each file gives:

- the [ACE](https://github.com/ai2cm/ace) commit it ran on, with a link;
- the run command and GPU count;
- the input checkpoint;
- the paper figures it feeds;
- the run IDs.

> **The configs will not run as-is outside Ai2.** The data paths point to Ai2-internal storage. See [Data](#data).

## Layout

```
1deg/  {gmr}-{masking}-{pretrain,finetune,eval}.yaml
4deg/  {gmr}-{masking}[-{variant}]-{train,eval}.yaml
```

| Token | Meaning |
|---|---|
| `gmron` / `gmroff` | Global mean removal on / off |
| `mask0` | No masking (ACE2S baseline) |
| `mask10`, `mask20` | Uniform masking: each step masks k ~ U{0..N} input channels |
| `bern10` | Bernoulli masking: each input masked with rate 0.2326 (about 10 of 43 on average) |
| `maskbatch` / `masksample` | One mask per batch / per sample (ablation) |
| `indicator-{concat,film,both}` | How the model is told which channels are masked (ablation) |
| `pretrain` → `finetune` → `eval` | 1°: 150-epoch single-step training, then 20-epoch multi-step fine-tuning, then evaluation |
| `train` → `eval` | 4°: training, then evaluation |

The 4° main results use seeds 0–4. Those configs differ only in `seed:`, so each cell has one file whose header lists every seed's run. The 1° runs are single-seed.

## Running

1. Clone https://github.com/ai2cm/ace and check out the commit from the config header.
2. Install ACE following its README.
3. Run the command from the header, from the repo root:

```bash
# training (pretrain / finetune / train)
torchrun --nproc_per_node <N> -m fme.ace.train path/to/config.yaml
# evaluation (the script path is in the header)
python -I configs/experiments/<experiment>/run_eval_suite.py path/to/config.yaml
```

The configs use these absolute paths, which were container mounts. Replace them, or mount to them:

| Path in config | What goes there |
|---|---|
| `/results` | Output directory |
| `/climate-default/...` | ERA5 dataset and normalization stats (see [Data](#data)) |
| `/weights/training_checkpoints/best_ckpt.tar` | 1° `finetune` only: `best_ckpt.tar` from the matching `pretrain` run |
| `/ckpt.tar` | `eval` only: `training_checkpoints/best_inference_ckpt.tar` from the matching `finetune` (1°) or `train` (4°) run |

The configs log to the private wandb entity `ai2cm`. Set `logging.log_to_wandb: false`, or set your own `entity` and `project`. In eval configs these appear once per inference.

The figures use the `long_46year` inference from each eval suite.

The original filenames of the 1° configs in the ACE repo contain `4deg`, but they are 1° runs, as their data paths show.

## Data

The datasets are on Ai2-internal storage and are **not publicly available**:

- 1°: `2026-03-19-era5-1deg-8layer-1940-2025.zarr` (6-hourly) and `2026-03-19-era5-1deg-8layer-stats-1990-2019/`
- 4°: `2026-04-17-era5-4deg-8layer-daily-1940-2025.zarr` (daily) and `2026-04-17-era5-4deg-8layer-daily-stats-1990-2019/`

The ACE repo has the pipeline that produced them, at the same commits as the configs: `scripts/era5/` (ERA5 ingest) and `scripts/data_process/configs/era5-{1,4}deg-8layer-1940-2025.yaml` (stats and daily coarsening).

## Figures → configs

Each eval config loads the checkpoint from its upstream training config, as stated in its header. The 1° chain is `pretrain` → `finetune` → `eval`; the 4° chain is `train` → `eval`.

| Figure (paper label) | Shows | Configs of the plotted runs |
|---|---|---|
| `fig:46yr_era5_T7_1deg` | Annual-mean T7, GMR on, 1° | [`1deg/gmron-mask0-eval.yaml`](1deg/gmron-mask0-eval.yaml)<br>[`1deg/gmron-mask20-eval.yaml`](1deg/gmron-mask20-eval.yaml) |
| `fig:46yr_era5_time_mean_rmse_gmron_1deg` | Time-mean RMSE, GMR on, 1° | [`1deg/gmron-mask0-eval.yaml`](1deg/gmron-mask0-eval.yaml)<br>[`1deg/gmron-mask20-eval.yaml`](1deg/gmron-mask20-eval.yaml) |
| `fig:46yr_era5_training_metrics_1deg_gmron` | Training/validation loss and 5-year inference error, GMR on, 1° | [`1deg/gmron-mask0-pretrain.yaml`](1deg/gmron-mask0-pretrain.yaml)<br>[`1deg/gmron-mask20-pretrain.yaml`](1deg/gmron-mask20-pretrain.yaml) |
| `fig:46yr_era5_T7_bern_4deg` | Annual-mean T7, Bernoulli vs uniform masking, 4° | [`4deg/gmroff-mask0-eval.yaml`](4deg/gmroff-mask0-eval.yaml)<br>[`4deg/gmroff-mask20-eval.yaml`](4deg/gmroff-mask20-eval.yaml)<br>[`4deg/gmroff-bern10-eval.yaml`](4deg/gmroff-bern10-eval.yaml)<br>[`4deg/gmron-mask0-eval.yaml`](4deg/gmron-mask0-eval.yaml)<br>[`4deg/gmron-mask20-eval.yaml`](4deg/gmron-mask20-eval.yaml)<br>[`4deg/gmron-bern10-eval.yaml`](4deg/gmron-bern10-eval.yaml) |
| `fig:46yr_era5_T7_batch_comparison_4deg` | Annual-mean T7, per-batch vs per-sample masking, 4° | [`4deg/gmron-mask10-maskbatch-eval.yaml`](4deg/gmron-mask10-maskbatch-eval.yaml)<br>[`4deg/gmron-mask10-masksample-eval.yaml`](4deg/gmron-mask10-masksample-eval.yaml)<br>[`4deg/gmron-mask20-maskbatch-eval.yaml`](4deg/gmron-mask20-maskbatch-eval.yaml)<br>[`4deg/gmron-mask20-masksample-eval.yaml`](4deg/gmron-mask20-masksample-eval.yaml) |
| `fig:46yr_era5_T7_indicator_comparison_4deg` | Annual-mean T7, masking-information variants, 4° | [`4deg/gmron-mask20-indicator-concat-eval.yaml`](4deg/gmron-mask20-indicator-concat-eval.yaml)<br>[`4deg/gmron-mask20-indicator-film-eval.yaml`](4deg/gmron-mask20-indicator-film-eval.yaml)<br>[`4deg/gmron-mask20-indicator-both-eval.yaml`](4deg/gmron-mask20-indicator-both-eval.yaml) |
| `fig:46yr_era5_T7_gmroff_1deg` | Annual-mean T7, GMR off, 1° | [`1deg/gmroff-mask0-eval.yaml`](1deg/gmroff-mask0-eval.yaml)<br>[`1deg/gmroff-mask20-eval.yaml`](1deg/gmroff-mask20-eval.yaml) |
| `fig:46yr_era5_training_metrics_1deg_gmroff` | Training/validation loss and 5-year inference error, GMR off, 1° | [`1deg/gmroff-mask0-pretrain.yaml`](1deg/gmroff-mask0-pretrain.yaml)<br>[`1deg/gmroff-mask20-pretrain.yaml`](1deg/gmroff-mask20-pretrain.yaml) |
| `fig:46yr_era5_T0_1deg` | Annual-mean T0, 1° | [`1deg/gmroff-mask0-eval.yaml`](1deg/gmroff-mask0-eval.yaml)<br>[`1deg/gmroff-mask20-eval.yaml`](1deg/gmroff-mask20-eval.yaml)<br>[`1deg/gmron-mask0-eval.yaml`](1deg/gmron-mask0-eval.yaml)<br>[`1deg/gmron-mask20-eval.yaml`](1deg/gmron-mask20-eval.yaml) |
| `fig:46yr_era5_Q3_1deg` | Annual-mean q^T_3, 1° | [`1deg/gmroff-mask0-eval.yaml`](1deg/gmroff-mask0-eval.yaml)<br>[`1deg/gmroff-mask20-eval.yaml`](1deg/gmroff-mask20-eval.yaml)<br>[`1deg/gmron-mask0-eval.yaml`](1deg/gmron-mask0-eval.yaml)<br>[`1deg/gmron-mask20-eval.yaml`](1deg/gmron-mask20-eval.yaml) |
| `fig:46yr_era5_TWP_1deg` | Annual-mean TWP, 1° | [`1deg/gmroff-mask0-eval.yaml`](1deg/gmroff-mask0-eval.yaml)<br>[`1deg/gmroff-mask20-eval.yaml`](1deg/gmroff-mask20-eval.yaml)<br>[`1deg/gmron-mask0-eval.yaml`](1deg/gmron-mask0-eval.yaml)<br>[`1deg/gmron-mask20-eval.yaml`](1deg/gmron-mask20-eval.yaml) |
| `fig:46yr_era5_Psfc_1deg` | Annual-mean P_sfc, 1° | [`1deg/gmroff-mask0-eval.yaml`](1deg/gmroff-mask0-eval.yaml)<br>[`1deg/gmroff-mask20-eval.yaml`](1deg/gmroff-mask20-eval.yaml)<br>[`1deg/gmron-mask0-eval.yaml`](1deg/gmron-mask0-eval.yaml)<br>[`1deg/gmron-mask20-eval.yaml`](1deg/gmron-mask20-eval.yaml) |
| `fig:46yr_time_mean_rmse_era5_1deg` | Time-mean RMSE, 1° | [`1deg/gmroff-mask0-eval.yaml`](1deg/gmroff-mask0-eval.yaml)<br>[`1deg/gmroff-mask20-eval.yaml`](1deg/gmroff-mask20-eval.yaml)<br>[`1deg/gmron-mask0-eval.yaml`](1deg/gmron-mask0-eval.yaml)<br>[`1deg/gmron-mask20-eval.yaml`](1deg/gmron-mask20-eval.yaml) |
| `fig:46yr_time_mean_bias_era5_1deg` | Time-mean bias, 1° | [`1deg/gmroff-mask0-eval.yaml`](1deg/gmroff-mask0-eval.yaml)<br>[`1deg/gmroff-mask20-eval.yaml`](1deg/gmroff-mask20-eval.yaml)<br>[`1deg/gmron-mask0-eval.yaml`](1deg/gmron-mask0-eval.yaml)<br>[`1deg/gmron-mask20-eval.yaml`](1deg/gmron-mask20-eval.yaml) |
| `fig:46yr_era5_T7_4deg` | Annual-mean T7, 4° | [`4deg/gmroff-mask0-eval.yaml`](4deg/gmroff-mask0-eval.yaml)<br>[`4deg/gmroff-mask20-eval.yaml`](4deg/gmroff-mask20-eval.yaml)<br>[`4deg/gmron-mask0-eval.yaml`](4deg/gmron-mask0-eval.yaml)<br>[`4deg/gmron-mask20-eval.yaml`](4deg/gmron-mask20-eval.yaml) |
| `fig:46yr_era5_T0_4deg` | Annual-mean T0, 4° | [`4deg/gmroff-mask0-eval.yaml`](4deg/gmroff-mask0-eval.yaml)<br>[`4deg/gmroff-mask20-eval.yaml`](4deg/gmroff-mask20-eval.yaml)<br>[`4deg/gmron-mask0-eval.yaml`](4deg/gmron-mask0-eval.yaml)<br>[`4deg/gmron-mask20-eval.yaml`](4deg/gmron-mask20-eval.yaml) |
| `fig:46yr_era5_Q3_4deg` | Annual-mean q^T_3, 4° | [`4deg/gmroff-mask0-eval.yaml`](4deg/gmroff-mask0-eval.yaml)<br>[`4deg/gmroff-mask20-eval.yaml`](4deg/gmroff-mask20-eval.yaml)<br>[`4deg/gmron-mask0-eval.yaml`](4deg/gmron-mask0-eval.yaml)<br>[`4deg/gmron-mask20-eval.yaml`](4deg/gmron-mask20-eval.yaml) |
| `fig:46yr_era5_TWP_4deg` | Annual-mean TWP, 4° | [`4deg/gmroff-mask0-eval.yaml`](4deg/gmroff-mask0-eval.yaml)<br>[`4deg/gmroff-mask20-eval.yaml`](4deg/gmroff-mask20-eval.yaml)<br>[`4deg/gmron-mask0-eval.yaml`](4deg/gmron-mask0-eval.yaml)<br>[`4deg/gmron-mask20-eval.yaml`](4deg/gmron-mask20-eval.yaml) |
| `fig:46yr_era5_Psfc_4deg` | Annual-mean P_sfc, 4° | [`4deg/gmroff-mask0-eval.yaml`](4deg/gmroff-mask0-eval.yaml)<br>[`4deg/gmroff-mask20-eval.yaml`](4deg/gmroff-mask20-eval.yaml)<br>[`4deg/gmron-mask0-eval.yaml`](4deg/gmron-mask0-eval.yaml)<br>[`4deg/gmron-mask20-eval.yaml`](4deg/gmron-mask20-eval.yaml) |
| `fig:46yr_time_mean_rmse_era5_4deg` | Time-mean RMSE, 4° | [`4deg/gmroff-mask0-eval.yaml`](4deg/gmroff-mask0-eval.yaml)<br>[`4deg/gmroff-mask20-eval.yaml`](4deg/gmroff-mask20-eval.yaml)<br>[`4deg/gmron-mask0-eval.yaml`](4deg/gmron-mask0-eval.yaml)<br>[`4deg/gmron-mask20-eval.yaml`](4deg/gmron-mask20-eval.yaml) |
| `fig:46yr_time_mean_bias_era5_4deg` | Time-mean bias, 4° | [`4deg/gmroff-mask0-eval.yaml`](4deg/gmroff-mask0-eval.yaml)<br>[`4deg/gmroff-mask20-eval.yaml`](4deg/gmroff-mask20-eval.yaml)<br>[`4deg/gmron-mask0-eval.yaml`](4deg/gmron-mask0-eval.yaml)<br>[`4deg/gmron-mask20-eval.yaml`](4deg/gmron-mask20-eval.yaml) |
| `fig:46yr_era5_training_metrics_4deg_gmron` | Training/validation loss and 5-year inference error, GMR on, 4° | [`4deg/gmron-mask0-train.yaml`](4deg/gmron-mask0-train.yaml)<br>[`4deg/gmron-mask20-train.yaml`](4deg/gmron-mask20-train.yaml) |
| `fig:46yr_era5_training_metrics_4deg_gmroff` | Training/validation loss and 5-year inference error, GMR off, 4° | [`4deg/gmroff-mask0-train.yaml`](4deg/gmroff-mask0-train.yaml)<br>[`4deg/gmroff-mask20-train.yaml`](4deg/gmroff-mask20-train.yaml) |

## Ai2-internal

These links and IDs only work inside Ai2:

- **wandb runs and Beaker experiment IDs** are in each config header. The wandb projects (`ai2cm/VarMasking8`, `ai2cm/VarMaskingInfoComparison`) are private.
- **Launch:** gantry on Beaker via `configs/experiments/<experiment>/run-ace-train.sh` at the pinned commit. Settings were workspace `ai2/ace`, budget `ai2/atec-climate`, clusters titan/jupiter, and `--weka climate-default:/climate-default`.
- **GPUs:** 1° pretrain used 8× H100/B200; 1° finetune used 4× B200; 4° training used 2× H100/B200; evals used 1 GPU.
