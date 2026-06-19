# ====================================================================
# DNABERT FULL PIPELINE 
# ====================================================================

import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from pyfaidx import Fasta
from sklearn.model_selection import train_test_split

from datasets import Dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


# ====================================================================
# STEP 1: LOAD GENOME
# ====================================================================

BASE = "/Users/zheheng/Desktop/生信大作业"

GTF_FILE = os.path.join(BASE, "gencode.v49.basic.annotation.gtf")
FASTA_FILE = os.path.join(BASE, "GRCh38.p14.genome.fa")

genome = Fasta(FASTA_FILE)


def normalize_chrom(chrom, genome_keys):
    if chrom in genome_keys:
        return chrom
    if "chr" + chrom in genome_keys:
        return "chr" + chrom
    if chrom.startswith("chr") and chrom[3:] in genome_keys:
        return chrom[3:]
    return None


# ====================================================================
# STEP 2: PARSE GTF
# ====================================================================

def parse_gtf(gtf_file, genome):

    donors, acceptors = [], []
    genome_keys = list(genome.keys())

    with open(gtf_file) as f:
        for line in tqdm(f):

            if line.startswith("#"):
                continue

            parts = line.strip().split("\t")
            if len(parts) < 8:
                continue

            chrom, feature = parts[0], parts[2]
            start, end, strand = int(parts[3]), int(parts[4]), parts[6]

            if feature != "exon":
                continue

            chrom = normalize_chrom(chrom, genome_keys)
            if chrom is None:
                continue

            donors.append([chrom, end, strand])
            acceptors.append([chrom, start, strand])

    df = pd.DataFrame(donors + acceptors, columns=["chrom","pos","strand"])
    df["label"] = 1
    df["type"] = "splice"

    print("✔ Splice sites:", len(df))
    return df


# ====================================================================
# STEP 3: BUILD DATASET (POS + NEG)
# ====================================================================

WINDOW = 50


def get_seq(genome, chrom, pos, strand):
    try:
        s = max(pos - WINDOW, 1)
        e = pos + WINDOW

        seq = genome[chrom][s:e].seq.upper()

        if len(seq) != 2 * WINDOW:
            return None

        if strand == "-":
            comp = str.maketrans("ACGT","TGCA")
            seq = seq.translate(comp)[::-1]

        return seq
    except:
        return None


def build_positive(df, genome):
    out = []
    for _, r in tqdm(df.iterrows(), total=len(df)):
        seq = get_seq(genome, r.chrom, r.pos, r.strand)
        if seq:
            out.append({"sequence": seq, "label": 1})
    return out


def build_negative(genome, df, n):
    out = []
    chroms = list(genome.keys())

    for _ in tqdm(range(n)):
        chrom = random.choice(chroms)
        pos = random.randint(WINDOW+1, len(genome[chrom])-WINDOW-1)
        strand = random.choice(["+","-"])

        seq = get_seq(genome, chrom, pos, strand)
        if seq:
            out.append({"sequence": seq, "label": 0})

    return out


def build_dataset(df, genome):

    pos = build_positive(df, genome)
    neg = build_negative(genome, df, len(pos))

    data = pd.DataFrame(pos + neg)

    # =========================
    # FIX: labels统一（关键）
    # =========================
    data["labels"] = data["label"].astype(int)
    data = data.drop(columns=["label"])

    print("✔ dataset ready")
    print(data["labels"].value_counts())

    return data


# ====================================================================
# STEP 4: HF DATASET
# ====================================================================

def make_hf(df):

    train, test = train_test_split(
        df,
        test_size=0.2,
        stratify=df["labels"],
        random_state=42
    )

    return (
        Dataset.from_pandas(train.reset_index(drop=True)),
        Dataset.from_pandas(test.reset_index(drop=True))
    )


# ====================================================================
# STEP 5: DNABERT TRAINING
# ====================================================================

# =========================================================
# DNABERT TRAINING STABLE VERSION
# =========================================================

import numpy as np
import torch
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


# =========================================================
# softmax（稳定数值版本）
# =========================================================
def softmax(x):
    x = np.array(x)
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


# =========================================================
# metrics（防 NaN + 稳定 AUC）
# =========================================================
def compute_metrics(eval_pred):

    logits, labels = eval_pred

    logits = np.array(logits)
    labels = np.array(labels)

    probs = softmax(logits)[:, 1]
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)

    # 防止 AUC 报错（极端 batch）
    try:
        auc = roc_auc_score(labels, probs)
    except:
        auc = 0.0

    return {
        "accuracy": acc,
        "f1": f1,
        "auc": auc
    }


# =========================================================
# tokenization（DNABERT关键修复点）
# =========================================================
def tokenize(ds, tokenizer):

    def fn(batch):
        return tokenizer(
            batch["sequence"],   # ⚠️ 确保你的列叫 sequence
            padding="max_length",
            truncation=True,
            max_length=128
        )

    ds = ds.map(fn, batched=True)

    # ❗删除原始sequence（避免Trainer报错）
    ds = ds.remove_columns(["sequence"])

    return ds


# =========================================================
# TRAIN FUNCTION（最终稳定版）
# =========================================================
def train(train_ds, test_ds):

    print("Loading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    print("Tokenizing...")
    train_ds = tokenize(train_ds, tokenizer)
    test_ds = tokenize(test_ds, tokenizer)

    train_ds.set_format("torch")
    test_ds.set_format("torch")

    print("Loading model...")
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2
    )

    # =====================================================
    # TrainingArguments（兼容旧版本 + 新版本）
    # =====================================================
    args = TrainingArguments(
        output_dir="./dnabert",

        # ===== evaluation（兼容修复）=====
        evaluation_strategy="steps",
        eval_steps=200,   # ❗避免太频繁导致卡死

        logging_steps=50,

        save_steps=200,

        learning_rate=2e-5,

        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,

        num_train_epochs=2,
        weight_decay=0.01,

        report_to="none",

        load_best_model_at_end=True,
        metric_for_best_model="auc",
        greater_is_better=True
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    print("\n🚀 START TRAINING")
    trainer.train()

    print("\n====================")
    print("FINAL EVAL")
    print("====================")

    result = trainer.evaluate()
    print(result)

    return trainer, model

# ====================================================================
# RUN ALL
# ====================================================================

df = parse_gtf(GTF_FILE, genome)

dataset = build_dataset(df, genome)

train_ds, test_ds = make_hf(dataset)

trainer, model = train(train_ds, test_ds)