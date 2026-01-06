import json
import sqlite3
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import faiss
from sentence_transformers import SentenceTransformer
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.optim as optim
from tqdm import tqdm

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare train/val dataset for a reranker task [rag]")
    parser.add_argument(
        "--reproduce_train_dataset",
        type=str,
        default="data/reranker_train.jsonl",
        help="The path to the reranker training dataset",
    )
    parser.add_argument(
        "--reproduce_eval_dataset",
        type=str,
        default="data/reranker_val.jsonl",
        help="The path to the reranker validation dataset",
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default="data/corpus.txt",
        help="The path to corpus",
    )
    parser.add_argument(
        "--path_to_retriever",
        type=str,
        default="./models/retriever",
        help="The path to retriever",
    )
    parser.add_argument(
        "--output_train_dataset",
        type=str,
        default="data/reranker_train.jsonl",
        help="Where to store the prepared train dataset",
    )
    parser.add_argument(
        "--output_eval_dataset",
        type=str,
        default="data/reranker_val.jsonl",
        help="Where to store the prepared val dataset",
    )
    parser.add_argument(
        "--top_k",
        type=str,
        default=10,
        help="Num of retrieved candiadates",
    )

    args = parser.parse_args()

    return args



class PassageRetriever:
    """use FAISS to retrieve passages"""
    def __init__(self, corpus_path: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.passages = []
        self.embeddings = None
        self.index = None
        self._load_corpus(corpus_path)
        self._build_faiss_index()
    
    def _load_corpus(self, corpus_path: str):
        """load corpus.txt"""
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.passages.append(line)
        print(f"Loaded {len(self.passages)} passages")
    
    def _build_faiss_index(self):
        print("Building FAISS index...")
        self.embeddings = self.model.encode(self.passages, batch_size=128, show_progress_bar=True)
        self.embeddings = np.asarray(self.embeddings).astype('float32')
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings)
        print(f"FAISS index built with {self.index.ntotal} vectors")
    
    def retrieve(self, queries: List[str], top_k: int = 10) -> List[List[str]]:
        """retrieve topk passages by queries"""
        query_embeddings = self.model.encode(queries, batch_size=128, show_progress_bar=False)
        query_embeddings = np.asarray(query_embeddings).astype('float32')
        
        distances, indices = self.index.search(query_embeddings, top_k)
        
        results = []
        for idx_list in indices:
            candidates = [self.passages[i] for i in idx_list if i < len(self.passages)]
            results.append(candidates)
        return results


def prepare_reranker_data(
    input_path: str,
    output_path: str,
    retriever: PassageRetriever,
    top_k: int = 10
):
    """
    training data for reranker
    input: {"qid": "...", "question": "...", "answer": {"text": "..."}, ...}
    output: {"query": "...", "positive": "...", "negatives": [...]}
    """
    data_list = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line.strip())
            question = sample.get("question") or sample.get("rewrite")
            answer_text = sample.get("answer", {}).get("text")
            
            if not question or not answer_text:
                continue
            
            # retrieve top_k
            candidates = retriever.retrieve([question], top_k=top_k)[0]
            
            # pos of gold_answer  in the candition
            positive_idx = -1
            for idx, cand in enumerate(candidates):
                if answer_text in cand:
                    positive_idx = idx
                    break
            
            # 如果找不到全部匹配，找部分匹配 (至少50個字)
            if positive_idx == -1 and len(answer_text) > 10:
                for idx, cand in enumerate(candidates):
                    if answer_text[:50] in cand or cand.find(answer_text[:50]) != -1:
                        positive_idx = idx
                        break
            
            # 如果還是沒找到就跳過
            if positive_idx == -1:
                continue
            
            # construct data
            positive = candidates[positive_idx]
            negatives = candidates[:positive_idx] + candidates[positive_idx+1:]
            
            reranker_sample = {
                "query": question,
                "positive": positive,
                "negatives": negatives,
                "qid": sample.get("qid")
            }
            data_list.append(reranker_sample)
    
    # save data
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data_list:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Prepared {len(data_list)} reranker training samples -> {output_path}")
    return len(data_list)





if __name__ == "__main__":

    # configuration
    CORPUS_PATH = args.corpus 
    RETRIEVER_MODEL= args.path_to_retriever
    TRAIN_DATA_PATH = args.reproduca_train_dataset
    VAL_DATA_PATH = args.reproduce_eval_dataset 
    RERANKER_TRAIN_PATH = args.output_train_dataset
    RERANKER_VAL_PATH = args.output_eval_dataset
    TOP_K=10
    
    
    #Initializing Retriever 
    retriever = PassageRetriever(corpus_path=CORPUS_PATH, model_name=RETRIEVER_MODEL)
    
    #Preparing Reranker Training Data
    train_count = prepare_reranker_data(TRAIN_DATA_PATH, RERANKER_TRAIN_PATH, retriever, top_k=TOP_K)
    val_count = prepare_reranker_data(VAL_DATA_PATH, RERANKER_VAL_PATH, retriever, top_k=TOP_K)
    
    print(len(train_count))
    print(len(val_count))
