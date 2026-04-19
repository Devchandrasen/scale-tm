# SCALE-TM: Leakage-Audited All-Flow Forecasting Benchmarks

This repository contains the implementation for the NeurIPS-format paper:

**SCALE-TM: Leakage-Audited All-Flow Forecasting Benchmarks for Internet Traffic Matrices**

Repository URL: https://github.com/Devchandrasen/scale-tm

The project studies reproducible Internet traffic-matrix forecasting on public OD-pair datasets. The main contribution is an evaluation and reproducibility benchmark: causal preprocessing, strong persistence/calibrated baselines, all-flow reporting, blocked chronological folds, compact modern neural controls, paired bootstrap deltas, and explicit negative stress cases.

## What Is Included

- `src/ieee_benchmark.py`: Protocol A, one-step Abilene diagnostic with per-flow damped residual forecasting.
- `src/experiment.py`: Protocol B, sampled multi-horizon Abilene WRTCN benchmark.
- `src/multidataset_benchmark.py`: Protocol C, all-flow Abilene/CERNET/GEANT linear stress test with blocked folds.
- `src/allflow_neural_benchmark.py`: Protocol D, all-flow blocked neural benchmark with PatchTST-, N-BEATS-, N-HiTS-, TimesNet-, graph-attention-, WRTCN-style controls, ensemble ablations, and paired deltas.
- `paper/neurips_main.tex`: NeurIPS-format manuscript source.
- `paper/neurips_main.pdf`: Compiled manuscript.
- `results/`: CSV outputs used by the manuscript.
- `figures/`: Figures used by the manuscript.

## Repository Layout

```text
.
├── data/
│   ├── README.md
│   └── raw/                  # user-provided DL4Internet CSVs, ignored by Git
├── figures/                  # generated paper figures
├── paper/                    # LaTeX manuscript and bibliography
├── results/                  # generated benchmark CSV outputs
├── src/                      # experiment implementations
├── requirements.txt
└── README.md
```

## Environment

The experiments were run on CPU. The local machine had a GTX 1050, but the installed PyTorch CUDA build did not support that GPU compute capability, so the reported runs use CPU paths.

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Data

Download the public DL4Internet OD-pair traffic-matrix CSVs from:

https://github.com/jwwthu/DL4Internet/tree/main/TrafficMatrixPrediction/OD_pair

Place the files in `data/raw/`:

```text
data/raw/Abilene-OD_pair.csv
data/raw/CERNET-OD_pair.csv
data/raw/GEANT-OD_pair.csv
```

The raw CSV files are not included in this repository.

## Reproduce The Paper Experiments

Run from the repository root.

Protocol A, one-step Abilene diagnostic:

```bash
python -m src.ieee_benchmark
```

Protocol B, sampled 12-step Abilene neural benchmark:

```bash
python -m src.experiment --sample-rows 8064 --top-flows 12 \
  --epochs 18 --stride 3
```

Protocol C, all-flow multi-dataset linear stress test:

```bash
python -m src.multidataset_benchmark --sample-rows 2880 \
  --bootstrap-samples 300 --ensemble-step 0.1
```

Protocol D, blocked all-flow neural benchmark:

```bash
python -m src.allflow_neural_benchmark --sample-rows 2880 \
  --datasets Abilene,CERNET,GEANT --epochs 2 --batch-size 512 \
  --bootstrap-samples 250 --ensemble-step 0.1 --blocked-folds \
  --output-prefix allflow_neural_blocked
```

The blocked Protocol D run used all OD flows from Abilene, CERNET, and GEANT, three blocked chronological folds, compact neural controls, ensemble ablations, and paired bootstrap deltas. On the local CPU setup it took `2,192.35` seconds.

## Key Output Files

- `results/allflow_neural_blocked_summary.csv`: blocked Protocol D WAPE summary.
- `results/allflow_neural_blocked_paired_deltas_summary.csv`: paired bootstrap deltas for Protocol D.
- `results/multidataset_benchmark_summary.csv`: Protocol C all-flow linear stress-test summary.
- `results/summary_metrics.csv`: Protocol B summary.
- `figures/allflow_neural_blocked_wape.png`: blocked Protocol D figure.

## Build The Paper

The paper uses the NeurIPS LaTeX style included in `paper/`.

```bash
cd paper
latexmk -pdf -interaction=nonstopmode neurips_main.tex
```

The compiled PDF is written to `paper/neurips_main.pdf`.

## Main Result Summary

Under blocked all-flow Protocol D:

- Abilene: SCALE-TM neural ensemble is best, `15.19` WAPE.
- CERNET: SCALE-TM neural ensemble is best, `14.62` WAPE.
- GEANT: N-BEATS-Small is best, `38.68` WAPE; SCALE-TM is explicitly reported as a negative case.

The paper does not claim unconditional state of the art. It argues that public traffic-matrix forecasting needs strong calibrated baselines, causal preprocessing, all-flow reporting, modern neural controls, paired tests, and negative stress cases before larger topology-specific models can be credited with genuine gains.
