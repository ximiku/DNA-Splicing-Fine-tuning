# DNABERT2 Splice-Site Prediction

This project builds a transcript-level splice-site dataset from GRCh38 and GENCODE, then fine-tunes DNABERT2 for three-way classification:

- `0 = non_splice`
- `1 = donor`
- `2 = acceptor`

Each example is a 401 bp DNA window centered on a candidate genomic position. Negative-strand windows are reverse-complemented so the sequence is always oriented in transcript direction. `P(donor) + P(acceptor)` is the splice-site probability.

See [RUN_COMMANDS.md](RUN_COMMANDS.md) for the exact environment, smoke-test, training, evaluation, and prediction commands.

## Main Files

- `src/build_splice_dataset.py`: GTF parsing, transcript-level intron inference, positives, hard negatives, splits, dataset export.
- `src/train_dnabert2.py`: DNABERT2 full fine-tuning with HuggingFace Trainer.
- `src/baselines.py`: char k-mer TF-IDF + logistic regression baseline.
- `src/evaluate.py`: metrics from saved prediction tables.
- `src/predict.py`: single-position FASTA inference.

## Data

Expected raw inputs:

- `data/raw/GRCh38.p14.genome.fa`
- `data/raw/GRCh38.p14.genome.fa.fai`
- `data/raw/gencode.v49.basic.annotation.gtf`

Processed datasets are written under `data/processed/`; model outputs and metrics are written under `outputs/`.

## Repository Contents

This GitHub repository is intentionally ordinary Git only. It does not use Git
LFS. The uploaded files are intended to let collaborators inspect the code,
configuration, report, metrics, evaluation history, and lightweight result
tables without downloading multi-GB model or reference files.

Uploaded content includes:

- Source code under `src/`
- Experiment configs under `configs/`
- Environment files: `environment.yml` and `requirements.txt`
- Reproduction notes: `RUN_COMMANDS.md`
- Project reports under `docs/`
- Lightweight processed-data metadata and small parquet files
- Smoke-test processed dataset files
- Output metrics, trainer state, tokenizer/config files, and prediction parquet
- FASTA index file: `data/raw/GRCh38.p14.genome.fa.fai`

The following files are not uploaded:

| Path or pattern | Reason |
|---|---|
| `data/raw/GRCh38.p14.genome.fa` | Raw GRCh38 reference FASTA, about 3.35 GB |
| `data/raw/gencode.v49.basic.annotation.gtf` | Raw GENCODE annotation GTF, about 2.57 GB |
| `outputs/**/*.safetensors` | Fine-tuned model weights; previously considered for Git LFS, omitted to avoid LFS storage/bandwidth requirements |
| `outputs/**/optimizer.pt` | Optimizer states; previously considered for Git LFS, omitted because they are large and not needed for report review |
| `data/processed/dnabert2_splice_401/train.parquet` | Full training split parquet is larger than GitHub's ordinary-Git comfort range |
| `data/processed/dnabert2_splice_401/**/*.arrow` | Full processed Arrow shards/caches are regenerated artifacts |
| `data/processed/dnabert2_splice_401/**/cache-*.arrow` | HuggingFace Dataset cache files |
| `.cache/` | Local HuggingFace/runtime cache |
| `__pycache__/`, `.pytest_cache/`, `.ipynb_checkpoints/` | Local Python and notebook caches |

To fully reproduce the full training run, place the omitted raw files back at
the expected paths, rebuild the processed dataset if needed, and rerun the
commands in `RUN_COMMANDS.md`. The report and saved metrics in this repository
document the completed run.

## Extension Points

Useful next experiments include chromosome-held-out split, LoRA, frozen DNABERT2 embeddings with a linear probe, negative-type ablations, sequence logos, ROC/PR curves, confusion matrices, embedding visualizations, and attribution heatmaps.
