import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.optim as optim
from tqdm import tqdm
import argparse
import wandb




def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a encoder model on a Reranker task [rag]")
    parser.add_argument(
        "--reranker_train_dataset",
        type=str,
        default="data/reranker_train.jsonl",
        help="The path to the reranker training dataset",
    )
    parser.add_argument(
        "--reranker_eval_dataset",
        type=str,
        default="data/reranker_val.jsonl",
        help="The path to the reranker validation dataset",
    )
    parser.add_argument(
        "--output_model_dir",
        type=str,
        default="./models/reranker/",
        help="Where to store the final model",
    )

    args = parser.parse_args()

    return args



# ============================================================
# Reranker Training
# ============================================================

class RerankerDataset(Dataset):
    """Reranker training dataset"""
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line.strip())
                self.samples.append(sample)
        
        print(f"Loaded {len(self.samples)} samples from {data_path}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        query = sample["query"]
        positive = sample["positive"]
        negatives = sample["negatives"]
        
        # Positive sample encoding
        pos_encoding = self.tokenizer(
            query,
            positive,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Negative sample encoding
        if negatives:
            neg = negatives[np.random.randint(0, len(negatives))]
        else:
            neg = "no candidates"
        
        neg_encoding = self.tokenizer(
            query,
            neg,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'pos_input_ids': pos_encoding['input_ids'].squeeze(),
            'pos_attention_mask': pos_encoding['attention_mask'].squeeze(),
            'neg_input_ids': neg_encoding['input_ids'].squeeze(),
            'neg_attention_mask': neg_encoding['attention_mask'].squeeze(),
        }


class RerankerEvaluator:
    """Evaluator for MRR@K"""
    def __init__(self, model, tokenizer, device, max_length: int = 512):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
    
    def compute_score(self, query: str, passage: str) -> float:
        """single query-passage pair score"""
        encoding = self.tokenizer(
            query,
            passage,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=encoding['input_ids'].to(self.device),
                attention_mask=encoding['attention_mask'].to(self.device)
            )
            score = outputs.logits.squeeze().item()
        
        return score
    
    def compute_mrr_at_k(self, data_path: str, k: int = 10) -> float:
        """
        compute MRR@K
        MRR = (1/|Q|) * sum(1/rank) --> rank: the ranking of the correct passage
        """
        mrr_sum = 0.0
        count = 0
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line.strip())
                query = sample["query"]
                positive = sample["positive"]
                negatives = sample["negatives"][:k-1]  # top-k中除了positive的其他candiate
                
                # 計算所有 candidate 的得分
                candidates_with_scores = []
                
                # positive的得分
                pos_score = self.compute_score(query, positive)
                candidates_with_scores.append((pos_score, True))  # (score, is_positive)
                
                # negative的得分
                for neg in negatives:
                    neg_score = self.compute_score(query, neg)
                    candidates_with_scores.append((neg_score, False))
                
                # 按得分排序（從高到低）
                candidates_with_scores.sort(key=lambda x: x[0], reverse=True)
                
                # 找正確答案的排名
                rank = -1
                for idx, (score, is_pos) in enumerate(candidates_with_scores):
                    if is_pos:
                        rank = idx + 1
                        break
                
                # 計算MRR
                if rank > 0 and rank <= k:
                    mrr_sum += 1.0 / rank
                
                count += 1
        
        mrr_at_k = mrr_sum / count if count > 0 else 0.0
        return mrr_at_k


def train_reranker(
    train_data_path: str,
    val_data_path: str,
    model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    output_dir: str = "reranker_model",
    num_epochs: int = 3,
    batch_size: int = 24,
    learning_rate: float = 2.5e-5,
    eval_steps: int = 100,
    log_steps: int = 4,  # 每N步記錄一次train_loss
    warmup_steps: int = 0,
    max_length: int = 512,
    project_name: str = "reranker-rag",
    run_name: str = "reranker-training"
):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    wandb.init(
        project=project_name,
        name=run_name,
        config={
            "model": model_name,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "eval_steps": eval_steps,
            "max_length": max_length,
        }
    )
    
    # model & tokenizer
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    model = model.to(device)
    
    # Load data
    print("Loading datasets...")
    train_dataset = RerankerDataset(train_data_path, tokenizer, max_length)
    val_dataset = RerankerDataset(val_data_path, tokenizer, max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # optimizer & learning_rate
    total_steps = num_epochs * len(train_loader)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    
    # Evaluator
    evaluator = RerankerEvaluator(model, tokenizer, device, max_length)
    
    # train-loop
    global_step = 0
    best_val_mrr = 0.0
    best_model_dir = None
    
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    
    for epoch in range(num_epochs):
        # training
        model.train()
        epoch_loss = 0.0
        step_count = 0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training")
        
        for batch_idx, batch in enumerate(train_bar):
            optimizer.zero_grad()
            
            # forward propagation
            pos_outputs = model(
                input_ids=batch['pos_input_ids'].to(device),
                attention_mask=batch['pos_attention_mask'].to(device)
            )
            neg_outputs = model(
                input_ids=batch['neg_input_ids'].to(device),
                attention_mask=batch['neg_attention_mask'].to(device)
            )
            
            # Contrastive loss
            pos_scores = pos_outputs.logits.squeeze()
            neg_scores = neg_outputs.logits.squeeze()
            
            margin = 1.0
            loss = torch.clamp(margin - (pos_scores - neg_scores), min=0).mean()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            step_count += 1
            global_step += 1
            
            # progress bar
            train_bar.set_postfix({
                "loss": loss.item(),
                "lr": optimizer.param_groups[0]['lr']
            })
            
            # Evaluation
            if global_step % eval_steps == 0:
                print(f"\n[Step {global_step}] Running evaluation...")
                model.eval()
                
                with torch.no_grad():
                    val_mrr = evaluator.compute_mrr_at_k(val_data_path, k=10)
                
                # wandb log
                wandb.log({
                    "train/loss": loss.item(),
                    "eval/mrr@10": val_mrr,
                    "global_step": global_step,
                    "learning_rate": optimizer.param_groups[0]['lr']
                }, step=global_step)
                
                print(f"[Step {global_step}] Eval MRR@10: {val_mrr:.4f}")
                
                # best model on MRR
                if val_mrr > best_val_mrr:
                    best_val_mrr = val_mrr
                    best_model_dir = f"{output_dir}/best_model"
                    Path(best_model_dir).mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(best_model_dir)
                    tokenizer.save_pretrained(best_model_dir)
                    print(f"✓ Best model saved (MRR@10: {val_mrr:.4f})")
                
                model.train()
        
        avg_epoch_loss = epoch_loss / step_count
        
        # eval after epoch
        print(f"\n[Epoch {epoch+1}] Running final evaluation...")
        model.eval()
        
        with torch.no_grad():
            epoch_val_mrr = evaluator.compute_mrr_at_k(val_data_path, k=10)
        
        # epoch_eval
        wandb.log({
            "epoch": epoch + 1,
            "train/loss": avg_epoch_loss,
            "eval/mrr@10": epoch_val_mrr,
        }, step=global_step)
        
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Avg Train Loss: {avg_epoch_loss:.4f}")
        print(f"  Val MRR@10: {epoch_val_mrr:.4f}")
        print(f"  Best MRR@10 so far: {best_val_mrr:.4f}\n")
        
    
    print("="*60)
    print("Training Completed!")
    print(f"Best MRR@10: {best_val_mrr:.4f}")
    if best_model_dir:
        print(f"Best model saved at: {best_model_dir}")
    print("="*60)
    
    # 记录最终结果到WandB
    wandb.log({
        "best_mrr@10": best_val_mrr,
        "best_model_path": best_model_dir
    })
    
    wandb.finish()
    
    return model, best_model_dir


# ============================================================
# 主函数
# ============================================================

if __name__ == "__main__":


    args = parse_args()
    RERANKER_TRAIN_PATH = args.reranker_train_dataset
    RERANKER_VAL_PATH = args.reranker_eval_dataset
    OUTPUT_DIR = args.output_model_dir
    
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    # 训练参数
    model, best_dir = train_reranker(
        train_data_path=RERANKER_TRAIN_PATH,
        val_data_path=RERANKER_VAL_PATH,
        model_name="cross-encoder/ms-marco-MiniLM-L-12-v2",
        output_dir=OUTPUT_DIR,
        num_epochs=3,
        batch_size=16,
        learning_rate=2e-5,
        eval_steps=40,  
        warmup_steps=50,
        max_length=512,
        project_name="reranker-rag",
        run_name="reranker-training-v1"
    )
    
    print(f"\nBest model saved at: {best_dir}")
