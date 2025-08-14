# ViDRILL - Vietnamese Document Retrieval and Information Lookup Language

🇻🇳 **Vietnamese Document Retrieval System for VLSP 2025**

## 📍 Overview

**ViDRILL** is an advanced Vietnamese document retrieval system developed for the VLSP 2025 competition. It combines state-of-the-art retrieval methods, including embedding-based models, BM25, and large language models (LLMs), to achieve high performance in information retrieval from Vietnamese corpora.

## 🎯 Features

* **Multi-stage Retrieval Pipeline**: Optimized multi-step search for maximum accuracy
* **Hybrid Search Methods**:

  * Dense retrieval using E5 and GTE embeddings
  * Sparse retrieval using BM25
  * Neural reranking with BGE rerankers
* **Vietnamese Language Support**: Specially optimized for Vietnamese text
* **Large Language Model Integration**: Leverages LLMs for query expansion and reranking
* **Scalable Architecture**: Supports large corpora using Qdrant vector database

## 🏗️ Architecture

```
ViDRILL/
├── pipeline/           # Main retrieval pipeline
├── training/           # Model training modules
│   ├── retrieval/      # Dense retrieval training
│   ├── rerank/         # Reranking model training  
│   └── llm/            # LLM fine-tuning
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
# Preprocess corpus and build the database
cd prepare_data
bash scripts/main.sh

# Build vector database
python build_db_corpus.py
```

### 3. Training Models

#### Dense Retrieval Models

```bash
cd training/retrieval
bash e5-instruct.sh
```

#### Reranking Models

```bash
cd training/rerank
bash bge-reranker.sh
```

#### LLM Components

```bash
cd training/llm
bash scripts/train_sft.sh
```

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

### Query Processing

* **HyDE**: Hypothetical Document Embeddings
* **Query Expansion**: LLM-based query enhancement

## 📁 Key Files

* [`encoder.py`](encoder.py) - Utilities for document encoding
* [`search.py`](search.py) - Main search engine functionality
* [`combine_ranking_score.py`](conbine_ranking_score.py) - Methods to combine retrieval scores
* [`pipeline/main_e5_sentence_bm25.py`](pipeline/main_e5_sentence_bm25.py) - Hybrid retrieval pipeline script
* [`eval/eval.py`](eval/eval.py) - Evaluation metrics and scripts

## 🛠️ Configuration

All training configurations are stored in the [`config/`](config/) directory:

* [`ds_zero1.json`](config/ds_zero1.json) - DeepSpeed ZeRO stage 1
* [`ds_zero2.json`](config/ds_zero2.json) - DeepSpeed ZeRO stage 2

### Custom Model Training

Refer to the [`training/`](training/) directory for detailed scripts and configuration examples.

## 🤝 Contributing

This project is developed for the VLSP 2025 competition. For questions, issues, or contributions, please follow the competition guidelines and submit issues or pull requests accordingly.