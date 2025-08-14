# ViDRILL - Vietnamese Document Retrieval and Information Lookup Language

🇻🇳 **Vietnamese Document Retrieval System for VLSP 2025**

## 📍 Overview

ViDRILL là một hệ thống tìm kiếm và truy xuất thông tin tiếng Việt tiên tiến, được phát triển cho cuộc thi VLSP 2025. Hệ thống kết hợp nhiều phương pháp tìm kiếm hiện đại như embedding models, BM25, và large language models để đạt hiệu suất cao trong việc truy xuất thông tin từ corpus tiếng Việt.

## 🎯 Features

- **Multi-stage Retrieval Pipeline**: Kết hợp nhiều giai đoạn tìm kiếm để tối ưu độ chính xác
- **Hybrid Search Methods**: 
  - Dense retrieval với E5, GTE embeddings
  - Sparse retrieval với BM25
  - Neural reranking với BGE-reranker
- **Vietnamese Language Support**: Tối ưu hóa đặc biệt cho tiếng Việt
- **Large Language Model Integration**: Sử dụng LLM cho query expansion và reranking
- **Scalable Architecture**: Hỗ trợ xử lý corpus lớn với Qdrant vector database

## 🏗️ Architecture

```
ViDRILL/
├── pipeline/           # Main retrieval pipeline
├── training/          # Model training components
│   ├── retrieval/     # Dense retrieval training
│   ├── rerank/        # Reranking model training  
│   └── llm/          # LLM fine-tuning
├── prepare_data/      # Data preprocessing utilities
├── eval/             # Evaluation scripts
└── config/           # Training configurations
```

## 🚀 Setup and Usage

### 1. Installation

```bash
# Clone repository
git clone <repository-url>
cd ViDRILL

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

```bash
# Prepare corpus and build database
cd prepare_data
bash scripts/main.sh

# Build vector database
python build_db_corpus.py
```

### 3. Training Models

#### Train Dense Retrieval Models
```bash
cd training/retrieval
bash e5-instruct.sh
```

#### Train Reranking Models  
```bash
cd training/rerank
bash bge-reranker.sh
```

#### Train LLM Components
```bash
cd training/llm
bash scripts/train_sft.sh
```

### 4. Running Pipeline

#### Encode Corpus
```bash
cd pipeline/scripts
bash encode_corpus.sh
```

#### Run Retrieval Pipeline
```bash
# E5 + Sentence-BERT + BM25 hybrid
bash main_e5_sentence_bm25.sh

# E5 + Sentence-BERT only
bash main_e5_sentence.sh
```

### 5. Evaluation

```bash
cd eval
bash scripts/eval.sh
```

## 📊 Pipeline Components

### Dense Retrieval
- **E5 Multilingual**: [`intfloat/multilingual-e5-large-instruct`](training/retrieval/e5-instruct.sh)
- **GTE Models**: Fine-tuned GTE embeddings for Vietnamese
- **Sentence-BERT**: Vietnamese sentence transformers

### Sparse Retrieval  
- **BM25**: Traditional keyword matching
- **Hybrid Scoring**: Combination with dense retrieval

### Reranking
- **BGE Reranker**: Cross-encoder reranking models
- **LLM Reranking**: Large language model based reranking

### Query Processing
- **HyDE**: Hypothetical Document Embeddings
- **Query Expansion**: LLM-based query enhancement

## 📁 Key Files

- [`encoder.py`](encoder.py) - Document encoding utilities
- [`search.py`](search.py) - Main search functionality  
- [`conbine_ranking_score.py`](conbine_ranking_score.py) - Score combination methods
- [`pipeline/main_e5_sentence_bm25.py`](pipeline/main_e5_sentence_bm25.py) - Hybrid retrieval pipeline
- [`eval/eval.py`](eval/eval.py) - Evaluation metrics and scripts

## 🛠️ Configuration

Training configurations are stored in [`config/`](config/):
- [`ds_zero1.json`](config/ds_zero1.json) - DeepSpeed ZeRO stage 1
- [`ds_zero2.json`](config/ds_zero2.json) - DeepSpeed ZeRO stage 2

### Training Custom Models
See [`training/`](training/) directory for detailed training scripts and configurations.

## 🤝 Contributing

This project is developed for VLSP 2025 competition. For questions or issues, please refer to the competition guidelines.
