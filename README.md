# ViDRILL - Vietnamese Deep Retrieval in the Expansive Legal Landscape

🇻🇳 **Vietnamese Document Retrieval System for VLSP 2025**

## 📍 Overview

**ViDRILL** is an advanced Vietnamese document retrieval system developed for the VLSP 2025 competition. It combines state-of-the-art retrieval methods, including embedding-based models, BM25, and large language models (LLMs), to achieve high performance in information retrieval from Vietnamese corpora.

In addition to standard retrieval pipelines, ViDRILL supports **LLM fine-tuning with reinforcement learning (PPO)** for self-guided query expansion and rewriting, enabling more precise document search and ranking.

## 🎯 Features

* **Multi-stage Retrieval Pipeline**: Optimized multi-step search for maximum accuracy
* **Hybrid Search Methods**:

  * Dense retrieval using E5 and GTE embeddings
  * Sparse retrieval using BM25
  * Neural reranking with BGE rerankers
* **LLM Self-Searching & Query Rewriting**:

  * Fine-tune LLMs to rewrite queries for better retrieval
  * PPO-based reinforcement learning for self-guided search
* **Vietnamese Language Support**: Specially optimized for Vietnamese text
* **Scalable Architecture**: Handles large corpora using Qdrant vector database

## 🏗️ Architecture

```
ViDRILL/
├── pipeline/           # Main retrieval pipeline
├── training/           # Model training modules
│   ├── retrieval/      # Dense retrieval training
│   ├── rerank/         # Reranking model training  
│   ├── llm/            # LLM fine-tuning (SFT or GRPO)
│   └── DeepRetrieval/  # LLM query rewriting training
│   └── ViSearch-R1/    # LLM query and auto searching
├── prepare_data/       # Data preprocessing utilities
├── eval/               # Evaluation scripts
└── config/             # Training and system configurations
```

## 🚀 Setup and Usage

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/xndien2004/ViDrill.git
cd ViDRILL

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

```bash
# Preprocess corpus and build database
cd prepare_data
bash scripts/main.sh

# Build vector database
python build_db_corpus.py
```

### 3. Training Models

#### Dense Retrieval Models

```bash
cd training/retrieval

# Run one of the following scripts depending on which model you want to train:
# E5 Multilingual Dense Retrieval
bash e5-instruct.sh

# GTE Dense Retrieval
# bash gte.sh

# BGE-M3 Dense Retrieval
# bash bge-m3.sh
```

#### Reranking Models

```bash
cd training/rerank

# Run one of the following reranking scripts:
# bash bge-reranker.sh
# bash bge-reranker-mini.sh
```

Bạn có thể chỉnh phần này trong README để làm rõ là **có thể chọn 1 trong 2 phương pháp LLM fine-tuning (SFT hoặc GRPO)** để từ các top-k documents liên quan nhất, LLM học cách chọn tài liệu phù hợp. Ví dụ như sau:

---

#### LLM Components (Choose One: SFT or GRPO)

You can fine-tune the LLM to select the most relevant documents from top-k retrieval results using **either Supervised Fine-Tuning (SFT) or GRPO (PPO-style reinforcement learning)**.

**Option 1: Supervised Fine-Tuning (SFT)**

```bash
cd training/llm
# Train LLM with supervised examples to select the most relevant documents
bash scripts/train_sft.sh
```

**Option 2: GRPO Fine-Tuning (Self-Searching / RLHF)**

```bash
cd training/llm
# Train LLM using GRPO to optimize self-guided document selection from top-k
bash scripts/train_grpo.sh
```

*Notes:*

* Both methods use the top-k retrieved documents as input.
* The LLM learns to select to pick the most relevant document(s).
* Choose **one method** depending on your workflow or experiment needs.


#### LLM Query Rewriting with PPO ()

To fine-tune the LLM to rewrite queries and select the most relevant documents, follow these steps:

1. **Prepare Data**
   Format the training and retrieval data for LLM fine-tuning:

```bash
cd training/DeepRetrieval
# Run data processing script
bash scripts/run_data_process.sh
```

2. **Start Retrieval Server**
   Provide top-k retrieved documents as input to the LLM during training:

```bash
# Start retrieval server
bash scripts/run_retrieval_server.sh
```

3. **Train LLM with PPO (GRPO)**
   Fine-tune the LLM to improve query rewriting and document selection using rewards from correct retrieval and formatting, following the methodology in [arXiv:2503.00223](https://arxiv.org/pdf/2503.00223):

```bash
# Train LLM for query rewriting and reward-based optimization
bash scripts/train_ppo.sh
```

**Notes:**

* The LLM uses **top-k documents from the retrieval server** to learn which documents are most relevant.
* Rewards are assigned based on **retrieval correctness** and **document formatting**, enabling the LLM to improve both query rewriting and selection accuracy.
* Only run **train\_ppo.sh** after data processing and the retrieval server are ready.

---

### 4. Running the Retrieval Pipeline

#### Encode Corpus

```bash
cd pipeline/scripts
bash encode_corpus.sh
```

#### Run Retrieval

```bash
# Hybrid: E5 + Sentence-BERT + BM25
bash main_e5_sentence_bm25.sh

# Dense only: E5 + Sentence-BERT
bash main_e5_sentence.sh
```

### 5. Evaluation

```bash
cd eval
bash scripts/eval.sh
```

## 📊 Pipeline Components

### Dense Retrieval

* **E5 Multilingual**: [`intfloat/multilingual-e5-large-instruct`](training/retrieval/e5-instruct.sh)
* **GTE Models**: Fine-tuned GTE embeddings for Vietnamese
* **Sentence-BERT**: Vietnamese sentence transformers

### Sparse Retrieval

* **BM25**: Traditional keyword-based retrieval
* **Hybrid Scoring**: Combines dense and sparse retrieval results

### Reranking

* **BGE Reranker**: Cross-encoder neural reranking models
* **LLM Reranking**: LLM-based reranking
* **PPO Self-Searching**: LLM fine-tuned with reinforcement learning to autonomously improve retrieval

### Query Processing

* **HyDE**: Hypothetical Document Embeddings
* **Query Expansion & Rewriting**: LLM-based query enhancement and rewriting

## 📁 Key Files

* [`encoder.py`](encoder.py) - Utilities for document encoding
* [`search.py`](search.py) - Main search engine functionality
* [`combine_ranking_score.py`](conbine_ranking_score.py) - Methods to combine retrieval scores
* [`pipeline/main_e5_sentence_bm25.py`](pipeline/main_e5_sentence_bm25.py) - Hybrid retrieval pipeline script
* [`eval/eval.py`](eval/eval.py) - Evaluation metrics and scripts
* [`training/llm/scripts/train_ppo.sh`](training/llm/scripts/train_ppo.sh) - LLM PPO training script
* [`training/query_rewrite/scripts/train_rewriter.sh`](training/query_rewrite/scripts/train_rewriter.sh) - Query rewriting training script

## 🛠️ Configuration

All training configurations are stored in the [`config/`](config/) directory:

* [`ds_zero1.json`](config/ds_zero1.json) - DeepSpeed ZeRO stage 1
* [`ds_zero2.json`](config/ds_zero2.json) - DeepSpeed ZeRO stage 2
* [`ppo_config.json`](config/ppo_config.json) - PPO RL training configuration for LLM

## 🤝 Contributing

This project is developed for the VLSP 2025 competition. For questions, issues, or contributions, please follow the competition guidelines and submit issues or pull requests accordingly.
