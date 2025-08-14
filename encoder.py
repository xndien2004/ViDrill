from FlagEmbedding import BGEM3FlagModel, FlagReranker, LayerWiseFlagLLMReranker
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel

from typing import List, Tuple, Union
import torch


class BGEM3Encoder:
    def __init__(self, model_name: str, use_fp16: bool = True):
        self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16)
        self.use_fp16 = use_fp16

    def encode(self, query_text: Union[str, List[str]], type: str = "query"):
        is_batch = isinstance(query_text, list)
        emb = self.model.encode(query_text, return_dense=True, return_sparse=True, return_colbert_vecs=False)
        emb_sparse = emb['lexical_weights']
        dense_vec = emb['dense_vecs']
        
        if not is_batch:
            indices = list(emb_sparse.keys())
            values = list(emb_sparse.values())

            return dense_vec, indices, values
        else:
            dense_out, indices_out, values_out = [], [], []
            for dense_vec, sparse_vec in zip(dense_vec, emb_sparse):
                dense_out.append(dense_vec)
                indices_out.append(list(sparse_vec.keys()))
                values_out.append(list(sparse_vec.values()))
            return dense_out, indices_out, values_out
    
    def get_dimension(self) -> int:
        dense_vec, _, _ = self.encode("dummy input")
        return dense_vec.shape[0]

class E5Encoder:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name, device="cuda:0")

    def encode(self, query_text: Union[str, List[str]], type: str = "query"):
        """Encode the query text into dense vectors"""
        if not isinstance(query_text, str):
            query_text = [f"{type}: " + text for text in query_text]
        else:
            query_text = f"{type}: " + query_text
        dense_vec = self.model.encode(query_text, normalize_embeddings=True)
        return dense_vec, [], []
    
    def get_dimension(self) -> int:
        dense_vec, _, _ = self.encode("dummy input")
        return dense_vec.shape[0]
    
class E5InstructEncoder:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name, device="cuda:0")
        self.task = "Với một truy vấn về luật Việt Nam, truy xuất các đoạn văn liên quan có chứa câu trả lời cho truy vấn đó."

    def get_detailed_instruct(self, task_description: str, query: str) -> str:
        return f'Instruct: {task_description}\nQuery: {query}'

    def encode(self, query_text: Union[str, List[str]], type: str = "query"):
        """Encode the query text into dense vectors"""
        if not isinstance(query_text, str):
            if type == "query":
                query_text = [self.get_detailed_instruct(self.task, text) for text in query_text]
        else:
            if type == "query": 
                query_text = self.get_detailed_instruct(self.task, query_text)
        dense_vec = self.model.encode(query_text, normalize_embeddings=True)
        return dense_vec, [], []
    
    def get_dimension(self) -> int:
        dense_vec, _, _ = self.encode("dummy input")
        return dense_vec.shape[0]
    
class SentenceEncoder:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name, trust_remote_code=True)

    def encode(self, query_text: Union[str, List[str]], type: str = "query"):
        """Encode the query text into dense vectors"""
        dense_vec = self.model.encode(query_text, normalize_embeddings=True)
        return dense_vec, [], []
    
    def get_dimension(self) -> int:
        dense_vec, _, _ = self.encode("dummy input")
        return dense_vec.shape[0]

    
# reranking models
class BGEM3Reranker:
    def __init__(self, model_name: str, use_fp16: bool = True):
        self.model = FlagReranker(model_name, use_fp16=use_fp16)

    def calculate_scores(self, pairs:List[List[str]]) -> List[float]:
        """Calculate scores for the given query and passages"""
        if not isinstance(pairs, list):
            pairs = [pairs]
        scores = self.model.compute_score(pairs, normalize=True)
        return scores

    def rerank(self, query: str, passages: List[Tuple[str, str]], top_k: int = -1) -> List[Tuple[float, str, str]]:
        """Rerank the passages based on the query

        Args:
            query: The input query string
            passages: A list of tuples (index, context)
            top_k: How many top passages to return. If -1, return all.

        Returns:
            A list of tuples: (score, index, context)
        """
        pairs = [[query, passage[1]] for passage in passages]
        scores = self.calculate_scores(pairs)
        
        # Combine with scores: [(score, index, context)]
        scored_passages = [(score, passage[0], passage[1]) for passage, score in zip(passages, scores)]
        
        # Sort by score descending
        ranked_passages = sorted(scored_passages, key=lambda x: x[0], reverse=True)
        
        if top_k > 0:
            ranked_passages = ranked_passages[:top_k]
            
        return ranked_passages

class BGELLMReranker:
    def __init__(self, model_name: str='BAAI/bge-reranker-v2-minicpm-layerwise', use_fp16: bool = True):
        self.model = LayerWiseFlagLLMReranker(model_name, use_fp16=use_fp16)

    def calculate_scores(self, pairs:List[List[str]]) -> List[float]:
        """Calculate scores for the given query and passages"""
        if not isinstance(pairs, list):
            pairs = [pairs]
        scores = self.model.compute_score(pairs, cutoff_layers=[28])
        return scores

    def rerank(self, query: str, passages: List[Tuple[str, str]], top_k: int = -1) -> List[Tuple[float, str, str]]:
        """Rerank the passages based on the query

        Args:
            query: The input query string
            passages: A list of tuples (index, context)
            top_k: How many top passages to return. If -1, return all.

        Returns:
            A list of tuples: (score, index, context)
        """
        pairs = [[query, passage[1]] for passage in passages]
        scores = self.calculate_scores(pairs)
        
        # Combine with scores: [(score, index, context)]
        scored_passages = [(score, passage[0], passage[1]) for passage, score in zip(passages, scores)]
        
        # Sort by score descending
        ranked_passages = sorted(scored_passages, key=lambda x: x[0], reverse=True)
        
        if top_k > 0:
            ranked_passages = ranked_passages[:top_k]
            
        return ranked_passages