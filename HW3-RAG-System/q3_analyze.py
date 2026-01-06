import json
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import List, Dict

def load_result(filepath: str) -> Dict:
    """Load result JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_retrieval_scores(result_files: List[str]):
    """
    深度分析檢索分數特性
    """
    
    results = {}
    for file in result_files:
        prompt_name = file.split('/')[-2]  # e.g., results_new_rer_p1 -> p1
        results[prompt_name] = load_result(file)
    
    # ========== 1. Reranker vs FAISS 排名變化 ==========
    
    for prompt_name, data in results.items():
        print(f"\n {prompt_name}:")
        
        rank_improvements = []
        rank_degradations = []
        no_changes = []
        gold_not_in_faiss = 0
        
        for record in data['records']:
            gold_pids = set(record.get('gold_pids', []))
            if not gold_pids:
                continue
            
            # position of `gold doc` in FAISS 
            faiss_gold_positions = []
            for gold_pid in gold_pids:
                # check whether if contained retrieved 
                for idx, doc in enumerate(record['retrieved']):
                    if doc['pid'] == gold_pid:
                        # corresponding FAISS 
                        # 通過 faiss_distances 來推斷
                        faiss_rank = None
                        reranker_rank = idx + 1
                        faiss_gold_positions.append((gold_pid, reranker_rank))
                        break
            
            if not faiss_gold_positions:
                gold_not_in_faiss += 1
        
        print(f"  **  Gold docs not retrieved by FAISS: {gold_not_in_faiss}  **")
    
    
    
    # ==========  Reranker Top-1 Scores vs. Generation Behavior ==========
    print("--"*80)
    print("Reranker Top-1 Scores vs. Generation Behavior")
    print("--"*80)
    
    for prompt_name, data in results.items():
        print(f"\n ----------- {prompt_name} -----------")
        
        cannot_answer_top1_scores = []
        valid_answer_top1_scores = []
        
        cannot_answer_gold_ranks = []
        valid_answer_gold_ranks = []
        
        for record in data['records']:
            generated = record.get('generated', '').strip()
            is_cannot_answer = 'CANNOTANSWER' in generated.upper()
            
            # Top-1 分數
            if record['retrieved']:
                top1_score = record['retrieved'][0]['score']
                if is_cannot_answer:
                    cannot_answer_top1_scores.append(top1_score)
                else:
                    valid_answer_top1_scores.append(top1_score)
            
            # Gold doc 排名
            gold_pids = set(record.get('gold_pids', []))
            for idx, doc in enumerate(record['retrieved']):
                if doc['pid'] in gold_pids:
                    rank = idx + 1
                    if is_cannot_answer:
                        cannot_answer_gold_ranks.append(rank)
                    else:
                        valid_answer_gold_ranks.append(rank)
                    break
        
        if cannot_answer_top1_scores:
            print(f"  CANNOTANSWER cases:")
            print(f"    Count: {len(cannot_answer_top1_scores)}")
            print(f"    Avg Top-1 Score: {np.mean(cannot_answer_top1_scores):.4f}")
            if cannot_answer_gold_ranks:
                print(f"    Avg Gold Rank: {np.mean(cannot_answer_gold_ranks):.2f}")
        
        if valid_answer_top1_scores:
            print(f"\n  Valid Answer cases:")
            print(f"    Count: {len(valid_answer_top1_scores)}")
            print(f"    Avg Top-1 Score: {np.mean(valid_answer_top1_scores):.4f}")
            if valid_answer_gold_ranks:
                print(f"    Avg Gold Rank: {np.mean(valid_answer_gold_ranks):.2f}")
        
        # 統計差異
        if cannot_answer_top1_scores and valid_answer_top1_scores:
            score_diff = np.mean(valid_answer_top1_scores) - np.mean(cannot_answer_top1_scores)
            print(f"\n  ** Score Difference (Valid - Cannot): {score_diff:.4f}")
    


    # ========== Gold Document Rank Distribution ==========
    print("--"*80)
    print("Gold Document Rank Distribution")
    print("--"*80)

    for prompt_name, data in results.items():
        print(f"\n {prompt_name}:")

        rank_distribution = defaultdict(int)
        rank_to_scores = defaultdict(list)

        for record in data['records']:
            gold_pids = set(record.get('gold_pids', []))

            for idx, doc in enumerate(record['retrieved']):
                if doc['pid'] in gold_pids:
                    rank = idx + 1
                    rank_distribution[rank] += 1
                    rank_to_scores[rank].append(doc['score'])

        print(f"  Gold Doc Rank Distribution:")
        for rank in sorted(rank_distribution.keys())[:5]:
            count = rank_distribution[rank]
            avg_score = np.mean(rank_to_scores[rank])
            print(f"    Rank {rank}: {count} docs (avg score: {avg_score:.4f})")

        if len(rank_distribution) > 5:
            print(f"    Rank 6-10: {sum(rank_distribution[r] for r in range(6, 11))} docs")


if __name__ == "__main__":
    result_files = [
        'results_new_rer_p1/result.json',
        'results_new_rer_p2/result.json',
        'results_new_rer_p3/result.json',
        'results_new_rer_p4/result.json',
        'results_new_rer_p5/result.json',
        'results_new_rer_p6/result.json',
        'results_new_rer_p1_simple3/result.json',
    ]
    
    analyze_retrieval_scores(result_files)
