# ViDRILL - Vietnamese Deep Retrieval in the Expansive Legal Landscape

🇻🇳 **Vietnamese Document Retrieval System for VLSP 2025**

## 📍 Overview

**ViDRILL** is an advanced Vietnamese document retrieval system developed for the VLSP 2025 competition. It combines traditional retrieval methods, including embedding-based models and BM25, to achieve high performance in information retrieval from Vietnamese corpora.

While **our official VLSP 2025 pipeline does not yet incorporate LLM-based retrieval**, ViDRILL **supports fine-tuning LLMs** for:

* Query rewriting
* Document selection (retrieval)
* Reranking

This design allows future integration of LLM-enhanced retrieval to improve accuracy and reasoning.

---

## 🎯 Features

* **Multi-stage Retrieval Pipeline**: Optimized multi-step search for maximum accuracy

* **Hybrid Search Methods**:

  * Dense retrieval using E5 and GTE embeddings
  * Sparse retrieval using BM25
  * Neural reranking with BGE rerankers

* **LLM Fine-Tuning Support** *(for future pipeline integration)*:

  * Query rewriting for more effective search
  * Document selection from top-k retrieval results
  * PPO/GRPO-based reinforcement learning for self-guided search
  * Interleaved reasoning + search with ViSearch-R1

* **Vietnamese Language Support**: Optimized for Vietnamese text

* **Scalable Architecture**: Handles large corpora using Qdrant vector database

---

## 🏗️ Architecture

```
ViDRILL/
├── pipeline/           # Main retrieval pipeline (current VLSP 2025 pipeline does not use LLM)
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

## 3. Training Models (Optional / Future LLM Pipeline)

### Dense Retrieval Models

```bash
cd training/retrieval

# Choose one model to train:
bash e5-instruct.sh    # E5 Dense Retrieval
# bash gte.sh           # GTE Dense Retrieval
# bash bge-m3.sh        # BGE-M3 Dense Retrieval
```

### Reranking Models

```bash
cd training/rerank

# Choose one reranking script:
# bash bge-reranker.sh
# bash bge-reranker-mini.sh
```

### LLM Fine-Tuning (Optional)

**Option 1: Supervised Fine-Tuning (SFT)**

```bash
cd training/llm
bash scripts/train_sft.sh
```

**Option 2: GRPO / PPO Reinforcement Learning**

```bash
cd training/llm
bash scripts/train_grpo.sh
```

*Notes:*

* Supports using top-k retrieved documents to train LLMs for document selection.
* Current VLSP 2025 pipeline does not use these LLM models, but they are ready for future integration.

---

### DeepRetrieval: Query Rewriting with PPO (Optional)

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

* Designed for improving query rewriting and selection accuracy.
* Can be integrated into the pipeline in the future.

---

### ViSearch-R1: Reasoning & Self-Searching LLMs (Optional)

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

* LLM learns to reason over retrieved documents and call search engines.
* Supports various RL algorithms and LLMs.
* Currently for experimental/future usage; not in the VLSP 2025 pipeline.

---

## 4. Running the Retrieval Pipeline (Current VLSP 2025)

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

* **E5 Multilingual**
* **GTE Models**
* **Sentence-BERT**

### Sparse Retrieval

* **BM25**
* **Hybrid Scoring**

### Reranking

* **BGE Reranker**
* **LLM Reranking / PPO Self-Searching** *(future support)*

### Query Processing

* **HyDE**
* **Query Expansion & Rewriting** *(future LLM support)*

---

## 🛠️ Configuration

* [`ds_zero1.json`](config/ds_zero1.json) - DeepSpeed ZeRO stage 1
* [`ds_zero2.json`](config/ds_zero2.json) - DeepSpeed ZeRO stage 2
* [`ppo_config.json`](config/ppo_config.json) - RL training config for LLMs (future use)

---

## 🔗 References

* ViSearch-R1 Paper: [arXiv:2503.09516](https://arxiv.org/abs/2503.09516)
* DeepRetrieval Paper: [arXiv:2503.00223](https://arxiv.org/pdf/2503.00223)

---

## 🤝 Contributing

This project is developed for the VLSP 2025 competition.

* The **current competition pipeline does not yet integrate LLM-based retrieval**, but the system supports training LLMs for query rewriting, retrieval, and reranking.
* Future pipeline versions can leverage these LLM components to improve retrieval accuracy and reasoning.
* For questions, issues, or contributions, follow competition guidelines and submit issues or pull requests.