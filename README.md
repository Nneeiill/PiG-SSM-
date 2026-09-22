# PiG-SSM code

This repository contains the PiG-SSM model architecture and selected code for data preparation, training, evaluation, transfer learning, and gate ablation. It does not contain material data, pretrained weights, fitted scalers, checkpoints, or experimental outputs. The numerical results in the paper require the corresponding data and checkpoints.

## Layout and input data

Clone the repository as `organized_pig_ssm` inside a working directory. Put the material data outside the repository, as sibling directories:

```text
working-directory/
├── organized_pig_ssm/       # this repository
├── alloy-1/                # source-domain CSV files (provided separately)
├── 1-0/                    # first target-domain CSV directory (optional)
└── 14-0/                   # second target-domain CSV directory (optional)
```

The controlled source-domain protocol expects complete, headerless 101-row loading histories, with six strain columns followed by six stress columns (an optional 13th column may contain PEEQ). Source filenames follow the pattern `1-0-Job_TCT1.csv`. The scripts use physical stress units of MPa. The grouped split is made at the complete-file level using SHA-256 content hashes; scalers are fitted on the training split.

Run the commands below from `organized_pig_ssm`. Python dependencies are PyTorch, NumPy, scikit-learn, and joblib. Use a PyTorch build appropriate to your CPU or CUDA environment.

```bash
python -m pip install torch numpy scikit-learn joblib
```

## Source-domain preprocessing and training

Generate a deterministic 70/10/20 manifest (replace `42` with `123` or `2026` for the other paper seeds):

```bash
python -m revision_pipeline.split_manifest \
  --dataset alloy-1 --seed 42 \
  --output outputs/manifests/alloy-1_seed42.json
```

Train the full model and evaluate its held-out test split once:

```bash
python train_pig_ssm.py \
  --split_manifest outputs/manifests/alloy-1_seed42.json \
  --seed 42 --output_root outputs/full/seed_42 \
  --evaluate_test_once
```

The output directory contains the best checkpoint, train-fitted scalers, material mapping, training history, validation and test metrics, and run provenance. These generated files should remain local.

## Gate ablation and evaluation

Run one ablation at a time, using the same manifest and seed:

```bash
python E1/code/e1_ablation_train.py \
  --ablation full \
  --split_manifest outputs/manifests/alloy-1_seed42.json \
  --seed 42 --output_root outputs/ablation \
  --evaluate_test_once
```

Repeat with `--ablation dynamic_no_gate_reg`, `static_gate`, and `no_gate`. The four run directories are created under `outputs/ablation/<variant>/seed_42/`. After all four runs, collect their held-out metrics:

```bash
python E1/code/evaluate_e1_ablation.py \
  --split_manifest outputs/manifests/alloy-1_seed42.json \
  --results_root outputs/ablation \
  --variants full dynamic_no_gate_reg static_gate no_gate \
  --seeds 42 --device cpu \
  --output_csv outputs/ablation_metrics.csv \
  --output_json outputs/ablation_metrics.json
```

## Cross-material transfer

The target-data efficiency experiment compares scratch and transfer models on the same target split. It requires a locally trained source checkpoint, source material mapping, and source scalers. For example, with a target directory `../1-0` and all available target training data:

```bash
python E4/code/e4_transfer_data_efficiency.py \
  --data_dir ../1-0 --mode scratch --ratio 1.0 --seed 42 \
  --base_mapping_path outputs/full/seed_42/material_mapping.json \
  --scaler_x_path outputs/full/seed_42/scaler_x.joblib \
  --scaler_y_path outputs/full/seed_42/scaler_y.joblib \
  --lr_scratch 1e-4 --lr_transfer 1e-4 \
  --protocol_version revision-v2-2026-08-24 \
  --output_dir outputs/transfer_comparison

python E4/code/e4_transfer_data_efficiency.py \
  --data_dir ../1-0 --mode transfer --ratio 1.0 --seed 42 \
  --base_mapping_path outputs/full/seed_42/material_mapping.json \
  --pretrained_path outputs/full/seed_42/best_model.pth \
  --scaler_x_path outputs/full/seed_42/scaler_x.joblib \
  --scaler_y_path outputs/full/seed_42/scaler_y.joblib \
  --lr_scratch 1e-4 --lr_transfer 1e-4 \
  --protocol_version revision-v2-2026-08-24 \
  --output_dir outputs/transfer_comparison
```

Set `--ratio` to `0.1`, `0.5`, or `1.0`; repeat for each target and seed. Each run writes its own split lists, validation history, checkpoint, and test metrics. `train_pig_ssm_transfer.py` is also included as a direct fine-tuning entry point; run `python train_pig_ssm_transfer.py --help` for its arguments.

The release provides code and commands for inspection and for running these workflows with suitable local data. It does not bundle the source or target datasets needed to reproduce the published numbers.
