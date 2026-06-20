# DNABERT2 基因剪接位点预测

本项目用于完成“基于 DNA foundation model 的基因剪接位点预测算法”。项目从 GRCh38 人类参考基因组和 GENCODE GTF 注释中构建 transcript-level 剪接位点数据集，并对 DNABERT2 进行三分类 full fine-tuning。

模型输入是一段以候选基因组位置为中心的 401 bp DNA 序列窗口。对于负链转录本，序列会先进行 reverse complement，使模型看到的输入始终按照转录方向排列。模型输出三个类别的概率：

| 标签 | 类别 | 含义 |
|---:|---|---|
| 0 | `non_splice` | 非剪接位点 |
| 1 | `donor` | 供体剪接位点 |
| 2 | `acceptor` | 受体剪接位点 |

其中 `P(donor) + P(acceptor)` 可作为二分类意义下的 splice-site probability。

## 当前结果

全量数据集共包含 1,414,738 条样本：

| 类别 | 数量 |
|---|---:|
| `non_splice` | 707,369 |
| `donor` | 351,969 |
| `acceptor` | 355,400 |

主要测试集结果：

| 指标 | 数值 |
|---|---:|
| 三分类 accuracy | 0.966906 |
| 三分类 macro-F1 | 0.966328 |
| AUROC non-splice | 0.994666 |
| AUROC donor | 0.997210 |
| AUROC acceptor | 0.997111 |

更完整的项目说明、数据构建细节、训练过程和结果分析见 [docs/dnabert2_splice_site_project_report.md](docs/dnabert2_splice_site_project_report.md)。

## 代码结构

| 路径 | 说明 |
|---|---|
| `src/build_splice_dataset.py` | 解析 GTF，推导 transcript-level intron，构建正负样本并保存数据集 |
| `src/sequence_utils.py` | FASTA 读取、坐标转换、窗口截取、负链反向互补 |
| `src/train_dnabert2.py` | DNABERT2 三分类 full fine-tuning |
| `src/evaluate.py` | 根据保存的预测表重新计算评估指标 |
| `src/predict.py` | 单点 FASTA 推理 |
| `src/baselines.py` | 字符 k-mer TF-IDF + logistic regression baseline |
| `configs/` | 数据构建、训练、smoke test 和 baseline 配置 |
| `outputs/` | 已上传的轻量级指标、预测表、trainer state 和配置文件 |
| `docs/` | 项目报告和说明文档 |

完整命令参考 [RUN_COMMANDS.md](RUN_COMMANDS.md)。

## 数据说明

原始数据默认放在：

```text
data/raw/GRCh38.p14.genome.fa
data/raw/GRCh38.p14.genome.fa.fai
data/raw/gencode.v49.basic.annotation.gtf
```

本仓库只上传了 `data/raw/GRCh38.p14.genome.fa.fai`，没有上传完整 FASTA 和 GTF，因为它们是多 GB 原始文件，不适合普通 Git 仓库。

构建后的数据默认保存在：

```text
data/processed/dnabert2_splice_401
```

本仓库上传了数据集摘要、split metadata、测试/验证 parquet、smoke 数据集等轻量文件；没有上传 full training split 的大文件和 HuggingFace Arrow/cache 中间文件。

## 仓库上传内容

本仓库使用普通 Git，不使用 Git LFS。上传内容主要用于让协作者查看代码、配置、报告、指标、验证历史和轻量结果表。

已上传内容包括：

- `src/` 源代码
- `configs/` 实验配置
- `environment.yml` 和 `requirements.txt`
- `RUN_COMMANDS.md` 复现命令
- `docs/` 项目报告
- `data/raw/GRCh38.p14.genome.fa.fai`
- `data/processed/` 中的轻量 metadata、summary、smoke 数据和部分 parquet
- `outputs/` 中的 metrics、trainer state、tokenizer/config 文件和 prediction parquet

未上传内容如下：

| 路径或模式 | 原因 |
|---|---|
| `data/raw/GRCh38.p14.genome.fa` | GRCh38 原始 FASTA，约 3.35 GB |
| `data/raw/gencode.v49.basic.annotation.gtf` | GENCODE 原始 GTF，约 2.57 GB |
| `outputs/**/*.safetensors` | fine-tuned 模型权重；原本考虑用 Git LFS，后改为不上传以避免 LFS 存储和带宽限制 |
| `outputs/**/optimizer.pt` | optimizer state，体积较大，协作者阅读报告通常不需要 |
| `data/processed/dnabert2_splice_401/train.parquet` | full training split parquet 较大，不适合普通 Git |
| `data/processed/dnabert2_splice_401/**/*.arrow` | HuggingFace Dataset Arrow shard，可由处理流程重建 |
| `data/processed/dnabert2_splice_401/**/cache-*.arrow` | HuggingFace Dataset cache 文件 |
| `.cache/` | 本地 HuggingFace/runtime cache |
| `__pycache__/`、`.pytest_cache/`、`.ipynb_checkpoints/` | 本地运行缓存 |

如果需要完整复现训练，请将缺失的 FASTA/GTF 放回上述 `data/raw/` 路径，必要时重新构建 processed dataset，然后按 [RUN_COMMANDS.md](RUN_COMMANDS.md) 执行。

## 快速开始

创建环境：

```bash
conda env create -f environment.yml
conda activate clin-jepa
```

运行 smoke 数据构建：

```bash
python -m src.build_splice_dataset --config configs/dataset_smoke.yaml
```

运行 DNABERT2 smoke training：

```bash
python -m src.train_dnabert2 --config configs/dnabert2_smoke.yaml
```

重新评估已保存预测表：

```bash
python -m src.evaluate \
  --predictions outputs/dnabert2_full/test_predictions.parquet \
  --output outputs/dnabert2_full/eval_metrics.json
```

## 后续可扩展方向

- chromosome-held-out split，用于更严格检验跨染色体泛化能力
- LoRA 或 frozen DNABERT2 embedding + linear probe
- 按 `negative_type` 做更细的困难负样本分析
- donor/acceptor sequence logo
- ROC/PR 曲线、confusion matrix 和 embedding 可视化
- attribution / saliency heatmap，用于解释模型关注的序列区域
