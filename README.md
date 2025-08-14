# ViDRILL - Vietnamese Deep Retrieval in the Expansive Legal Landscape

🇻🇳 **Vietnamese Document Retrieval System for VLSP 2025**

## 📍 Overview

**ViDRILL** is an advanced Vietnamese document retrieval system developed for the VLSP 2025 competition. It combines state-of-the-art retrieval methods, including embedding-based models, BM25, and large language models (LLMs), to achieve high performance in information retrieval from Vietnamese corpora.

In addition to standard retrieval pipelines, ViDRILL supports **LLM fine-tuning with reinforcement learning (PPO / GRPO)** for self-guided query expansion, rewriting, and reasoning, enabling precise document selection and ranking.

---

## 🎯 Features

* **Multi-stage Retrieval Pipeline**: Optimized multi-step search for maximum accuracy
* **Hybrid Search Methods**:

  * Dense retrieval using E5 and GTE embeddings
  * Sparse retrieval using BM25
  * Neural reranking with BGE rerankers
* **LLM Self-Searching & Query Rewriting**:

  * Fine-tune LLMs to rewrite queries for better retrieval
  * PPO/GRPO-based reinforcement learning for self-guided search
  * Interleaved reasoning + search with ViSearch-R1
* **Vietnamese Language Support**: Specially optimized for Vietnamese text
* **Scalable Architecture**: Handles large corpora using Qdrant vector database

---

## 🏗️ Architecture

```
ViDRILL/
├── pipeline/           # Main retrieval pipeline
├── training/           # Model training modules
│   ├── retrieval/      # Dense retrieval training
│   ├── rerank/         # Reranking model training  
│   ├── llm/            # LLM fine-tuning (SFT or GRPO)
│   ├── DeepRetrieval/  # LLM query rewriting training
│   └── ViSearch-R1/    # LLM reasoning + self-searching training
├── prepare_data/       # Data preprocessing utilities
├── eval/               # Evaluation scripts
└── config/             # Training and system configurations
```

---

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

---

## 3. Training Models

### Dense Retrieval Models

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

### Reranking Models

```bash
cd training/rerank

# Run one of the following reranking scripts:
# bash bge-reranker.sh
# bash bge-reranker-mini.sh
```

### LLM Components (Choose One: SFT or GRPO)

Fine-tune the LLM to select the most relevant documents from top-k retrieval results.

**Option 1: Supervised Fine-Tuning (SFT)**

```bash
cd training/llm
bash scripts/train_sft.sh
```

**Option 2: GRPO Fine-Tuning (Self-Searching / RLHF)**

```bash
cd training/llm
bash scripts/train_grpo.sh
```

*Notes:*

* Both methods use top-k retrieved documents as input.
* The LLM learns to pick the most relevant document(s).
* Choose **one method** depending on workflow or experiments.

---

### LLM Query Rewriting with PPO (DeepRetrieval)

```bash
cd training/DeepRetrieval

# 1. Prepare Data
bash scripts/run_data_process.sh

# 2. Start Retrieval Server
bash scripts/run_retrieval_server.sh

# 3. Train LLM with PPO (GRPO)
bash scripts/train_ppo.sh
```

*Notes:*

* Uses top-k retrieved documents to learn relevance.
* Rewards are based on retrieval correctness and proper formatting.
* Run **train\_ppo.sh** only after data processing and retrieval server are ready.

---

### ViSearch-R1: Reasoning & Self-Searching LLMs

ViSearch-R1 is an RL framework to train **interleaved reasoning-and-searching LLMs**, inspired by [arXiv:2503.09516](https://arxiv.org/abs/2503.09516).

```bash
cd training/ViSearch-R1

# 1. Prepare Data
bash scripts/run_data_process.sh

# 2. Start Retrieval Server
bash scripts/run_retrieval_server.sh

# 3. Train LLM with RL (PPO / GRPO / Reinforce)
bash scripts/train_ppo.sh
```

*Notes:*

* LLM reasons over top-k documents and decides when to call search engines.
* Rewards: correctness of retrieved documents, output formatting, reasoning efficiency.
* Supports various RL algorithms, LLMs (e.g., LLaMA3, Qwen2.5), and search engines (local or online).

---

## 4. Running the Retrieval Pipeline

### Encode Corpus

```bash
cd pipeline/scripts
bash encode_corpus.sh
```

### Run Retrieval

```bash
# Hybrid: E5 + Sentence-BERT + BM25
bash main_e5_sentence_bm25.sh

# Dense only: E5 + Sentence-BERT
bash main_e5_sentence.sh
```

---

## 5. Evaluation

```bash
cd eval
bash scripts/eval.sh
```

---

## 📊 Pipeline Components

### Dense Retrieval

* **E5 Multilingual**: [`intfloat/multilingual-e5-large-instruct`](training/retrieval/e5-instruct.sh)
* **GTE Models**: Fine-tuned GTE embeddings for Vietnamese
* **Sentence-BERT**: Vietnamese sentence transformers

### Sparse Retrieval

* **BM25**: Traditional keyword-based retrieval
* **Hybrid Scoring**: Combine dense and sparse results

### Reranking

* **BGE Reranker**: Cross-encoder neural reranking
* **LLM Reranking**: LLM-based reranking
* **PPO Self-Searching**: LLM fine-tuned with RL to autonomously improve retrieval

### Query Processing

* **HyDE**: Hypothetical Document Embeddings
* **Query Expansion & Rewriting**: LLM-based

---

## 🛠️ Configuration

* [`ds_zero1.json`](config/ds_zero1.json) - DeepSpeed ZeRO stage 1
* [`ds_zero2.json`](config/ds_zero2.json) - DeepSpeed ZeRO stage 2

## 🔗 References

* ViSearch-R1 Paper: [arXiv:2503.09516](https://arxiv.org/abs/2503.09516)
* DeepRetrieval Paper: [arXiv:2503.00223](https://arxiv.org/pdf/2503.00223)
---

## 🤝 Contributing

This project is developed for the VLSP 2025 competition. For questions, issues, or contributions, follow competition guidelines and submit issues or pull requests accordingly.
