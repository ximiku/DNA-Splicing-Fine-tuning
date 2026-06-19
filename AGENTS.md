# AGENTS.md

## 项目目标

本项目用于完成“基于 DNA foundation model 的基因剪接位点预测算法”。目标是利用人类参考基因组 FASTA 与 GENCODE GTF 注释，构建剪接位点预测数据集，并基于 DNABERT2 进行监督微调，预测输入 DNA 序列中心位置属于：

- non-splice
- donor site
- acceptor site

模型最终应能对任意给定基因组位置附近的 DNA 序列输出三分类概率，并可进一步合并 donor/acceptor 概率得到 splice-site probability。

## 数据与任务框架

默认数据位于：

- `data/raw/GRCh38.p14.genome.fa`
- `data/raw/GRCh38.p14.genome.fa.fai`
- `data/raw/gencode.v49.basic.annotation.gtf`

数据构建采用全基因组范围。核心样本为 401 bp DNA window，即以候选位置为中心，上下游各 200 bp。负链样本需要反向互补，使输入序列统一到转录方向。

正样本应基于 transcript-level exon/intron 结构推导 donor 与 acceptor，而不是仅依赖孤立 exon 端点。负样本包括随机基因组背景、含 GT/AG motif 的困难负样本、内含子内部困难负样本。数据集中应保留必要元数据，例如 chrom、pos、strand、label、site_type、negative_type、gene_id、transcript_id 等，便于追溯和调试。

当前默认数据划分可采用随机分层划分。后续可扩展为 chromosome-held-out split，用于更严格检验跨染色体泛化能力。

## 模型方向

主模型使用 DNABERT2，例如 HuggingFace 上的 DNABERT2 权重。DNABERT2 直接接收原始 DNA 字符串，经其 tokenizer 编码，不需要手动 k-mer tokenization。

首要训练方式为 full fine-tuning。后续可考虑 frozen embedding/linear probe、LoRA 等参数高效微调方式作为扩展实验。


## 推荐代码结构

建议保持模块化结构：

- `src/build_splice_dataset.py`：解析 GTF、构造正负样本、保存 processed dataset
- `src/sequence_utils.py`：FASTA 读取、坐标转换、窗口截取、反向互补
- `src/split_dataset.py`：训练集、验证集、测试集划分
- `src/train_dnabert2.py`：DNABERT2 三分类微调
- `src/evaluate.py`：统一评估
- `src/predict.py`：单点或批量推理
- `src/visualize.ipynb`：可视化接口
- `configs/`：保存数据构造、训练、评估相关配置
- `outputs/`：保存模型、指标、日志和中间结果
- `data/processed/`：保存构建后的数据集

## 评估指标

三分类任务至少关注：

- accuracy
- macro-F1
- per-class precision/recall/F1
- confusion matrix 数据
- one-vs-rest AUROC / AUPRC，可在实现条件允许时加入

对于负样本，应尽量保留 negative_type，并在评估时支持分类型查看 random、motif hard negative、intronic hard negative 上的表现。

## 工程习惯

代码应优先使用相对路径和配置文件，便于在 GPU 服务器上复现。长流程应有命令行参数、进度条、日志输出和随机种子。耗时步骤可将中间结果缓存到 `data/processed/`，训练输出统一保存到 `outputs/`。关键数据处理函数应包含简短 docstring，特别是 GTF 坐标、pyfaidx 切片坐标、中心锚点定义和负链 reverse-complement 的逻辑。