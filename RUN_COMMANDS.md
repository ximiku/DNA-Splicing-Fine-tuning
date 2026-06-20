# DNABERT2 Splice-Site Project Commands

## Environment

```bash
conda env create -f environment.yml
conda activate dna-fine-tune
python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())
PY
```

If the CUDA 12.8 wheel is not suitable for the server, install the PyTorch wheel recommended by <https://pytorch.org/get-started/locally/> first, then run:

```bash
pip install -r requirements.txt
```

## Smoke Test

```bash
python -m src.build_splice_dataset --config configs/dataset_smoke.yaml
python -m src.baselines --config configs/baseline.yaml
python -m src.train_dnabert2 --config configs/dnabert2_smoke.yaml
```

## Full Dataset And Training

```bash
python -m src.build_splice_dataset --config configs/dataset.yaml
torchrun --nproc_per_node=2 -m src.train_dnabert2 --config configs/dnabert2_full.yaml
```

## Evaluate And Predict

```bash
python -m src.evaluate --predictions outputs/dnabert2_full/test_predictions.parquet --output outputs/dnabert2_full/eval_metrics.json
python -m src.predict --model_dir outputs/dnabert2_full/final_model --fasta data/raw/GRCh38.p14.genome.fa --chrom chr1 --pos 123456 --strand +
```

The `pos` argument is 1-based. For training labels, donor/acceptor anchors use the intron-side motif coordinate.

By default the configs disable DNABERT2's bundled Triton flash-attention path because the upstream remote code targets an older Triton API. Set `model.disable_triton_flash_attention: false` only in an environment where that remote kernel is known to compile.
