import os
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import json
from tqdm import tqdm
import re
import numpy as np
import csv
import random
import argparse

# environment variable
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

# bracket for Chinese
PAIRS = {
    '「':'」', '『':'』', '《':'》', '（':'）', '【':'】', '〈':'〉'
}

# set seed
def set_all_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Step 2 (QA) inferenece")

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="finetuned model"
    )
    parser.add_argument(
        "--test_file",
        type=str,
        required=True,
        help="test file from step 1 (Multiple Choice)"
    )

    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="the final submission file"
    )
    return parser.parse_args()

# Normalization & clean the blank
def normalize_spaces(s: str) -> str:
    s = re.sub(r'(?<=([\u4e00-\u9fff]))\s+(?=([\u4e00-\u9fff]))', '', s)
    s = re.sub(r'([「『《（【〈])\s+', r'\1', s)
    s = s.replace(" ", "")
    return s

def balance_pairs(s: str) -> str:
    for open_p, close_p in PAIRS.items():
        o_count = s.count(open_p)
        c_count = s.count(close_p)
        if o_count < c_count:
            s = open_p + s
        elif o_count > c_count:
            s = s + close_p
    return s

def advanced_answer_extraction(start_logits, end_logits, input_ids, tokenizer, 
                             token_type_ids, max_answer_length=30):
    """ select candiates """
    
    batch_size = start_logits.size(0)
    predictions = []
    
    for i in range(batch_size):
        start_scores = start_logits[i].cpu().numpy()
        end_scores = end_logits[i].cpu().numpy()
        token_types = token_type_ids[i].cpu().numpy()
        
        # the beginning pos of `context` (where token_type_id = 1 )
        context_mask = (token_types == 1)
        if not context_mask.any():
            predictions.append("")
            continue
            
        context_start = np.where(context_mask)[0][0]
        context_end = np.where(context_mask)[0][-1]
        
        # search only in context 
        start_scores[:context_start] = -10000
        end_scores[:context_start] = -10000
        start_scores[context_end+1:] = -10000
        end_scores[context_end+1:] = -10000
        
        # set `k`
        top_k = 10
        start_indices = np.argsort(start_scores)[-top_k:][::-1]
        end_indices = np.argsort(end_scores)[-top_k:][::-1]
        
        best_answer = ""
        best_score = -float('inf')
        candidates = []
        
        # collecting all the possible answer
        for start_idx in start_indices:
            for end_idx in end_indices:
                if (start_idx >= context_start and end_idx >= start_idx and 
                    start_idx <= context_end and end_idx <= context_end):
                    length = end_idx - start_idx + 1
                    if 1 <= length <= max_answer_length:  # at least 1 token
                        score = start_scores[start_idx] + end_scores[end_idx]
                        pred_tokens = input_ids[i][start_idx:end_idx + 1]
                        answer = tokenizer.decode(pred_tokens, skip_special_tokens=True)
                        answer = answer.strip()
                        
                        if answer and not is_invalid_answer(answer):
                            candidates.append((answer, score, length))
        
        # Choose the best answer: shorter and high-score first
        if candidates:
            # sort by score & give higher reward for shorter
            candidates.sort(key=lambda x: x[1] + (max_answer_length - x[2]) * 0.1, reverse=True)
            best_answer = candidates[0][0]
        
        predictions.append(best_answer)
    
    return predictions

def is_invalid_answer(answer):
    """ check the invalid answer"""
    answer_cand1 = answer.replace("。","")
    answer_cand2 = answer.replace("、","")
    answer_cand3 = answer.replace("，","")
    if not answer or len(answer.strip())==0 or len(answer_cand1)==0 or len(answer_cand2)==0 or len(answer_cand3) == 0:
        return True
    
    # cannot include the question
    question_indicators = ['什麼', '哪個', '誰', '何時', '為什麼', '怎麼', '多少', '？']
    if any(indicator in answer for indicator in question_indicators):
        return True
    
    # exclude the longer
    if len(answer) > 100:
        return True
    
    # exclude multiple-sentence
    if answer.count('。') > 2 or answer.count('，') > 3:
        return True
    
    return False

# load the test data
def load_test_data(file_path):
    """load the test data"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # check the test data & adjustment
    test_data = []
    for item in data:
        # adjustment by data
        if isinstance(item, dict):
            if "question" in item and "context" in item:
                # QA-format
                test_data.append({
                    "id": item.get("id", ""),
                    "question": item["question"], 
                    "context": item["context"]
                })
            elif "paragraphs" in item:
                # SQuAD-format
                for paragraph in item["paragraphs"]:
                    for qa in paragraph["qas"]:
                        test_data.append({
                            "id": qa["id"],
                            "question": qa["question"],
                            "context": paragraph["context"]
                        })
    
    return test_data

def process_batch_inference(batch_entries, model, tokenizer, device):
    """ inference in batch """
    batch_questions = [entry["question"] for entry in batch_entries]
    batch_contexts = [entry["context"] for entry in batch_entries]
    batch_ids = [entry["id"] for entry in batch_entries]
    
    inputs = tokenizer(
        batch_questions,
        batch_contexts,
        return_tensors="pt",
        truncation="only_second",
        max_length=512,
        padding=True,
        return_token_type_ids=True
    )
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits
    
    # with `advanced answer extraction`
    predictions = advanced_answer_extraction(
        start_logits, end_logits, inputs["input_ids"], 
        tokenizer, inputs["token_type_ids"]
    )
    
    # cleaned the answer
    cleaned_predictions = []
    for pred in predictions:
        cleaned = normalize_spaces(pred)
        cleaned = balance_pairs(cleaned)
        # If the empty after cleaning, try the original
        if not cleaned and pred:
            cleaned = pred.strip()
        cleaned_predictions.append(cleaned)
    
    return batch_ids, cleaned_predictions

def main():
    args = parse_args()
    set_all_seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"With: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_path)
    model.to(device)
    model.eval()
    
    if device.type == "cuda":
        model.half()

    try:
        test_data = load_test_data(args.test_file)
        print(f"# of Test data: {len(test_data)}")
    except FileNotFoundError:
        print(f"Cannot find the file: {args.test_file}")
        print("please check `--test_file`")
        return
    
    # batch process & inference
    batch_size = 16
    all_predictions = []
    
    print("Inference...")
    for i in tqdm(range(0, len(test_data), batch_size), desc="Inference Process"):
        batch = test_data[i:i+batch_size]
        batch_ids, predictions = process_batch_inference(batch, model, tokenizer, device)
        
        for id_, pred in zip(batch_ids, predictions):
            all_predictions.append({"id": id_, "answer": pred})
        
        if i % (batch_size ) == 0 and predictions:
            print(f"example prediction: {predictions[0]}")
    
    
    # write submission csv
    with open(args.output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['id', 'answer']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for pred in all_predictions:
            writer.writerow(pred)
    
    
    print(f"Inference finish!")
    print(f"Save in: {args.output_file}")
    print(f"# of total data: {len(all_predictions)}")
    
    # show some weird (empty)
    empty_count = sum(1 for pred in all_predictions if not pred["answer"])
    print(f"# of empty: {empty_count} ({empty_count/len(all_predictions)*100:.1f}%)")

if __name__ == "__main__":
    main()
