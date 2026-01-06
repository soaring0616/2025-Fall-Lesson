#!/usr/bin/env python3

import argparse
import json
import logging
import os
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np

from transformers import (
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    DataCollatorForMultipleChoice,
    default_data_collator,
)
from datasets import Dataset
from accelerate import Accelerator
import random

def set_all_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def setup_logging():
    """set log"""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    return logging.getLogger(__name__)


def load_test_data(file_path):
    """load test and turn to Dataset"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
        for item in raw_data:
            data.append({
                "id": item["id"],
                "question": item["question"],
                "ending0": item["ending0"],
                "ending1": item["ending1"], 
                "ending2": item["ending2"],
                "ending3": item["ending3"],
            })
    return Dataset.from_list(data)


def preprocess_function(examples, tokenizer, max_length=512):
    """preprocess function"""
    # 4 options for each question (q+opt1 / q+opt2 / ...)
    first_sentences = [[q] * 4 for q in examples["question"]]
    
    # create the list of options
    second_sentences = [
        [examples['ending0'][i], examples['ending1'][i], 
         examples['ending2'][i], examples['ending3'][i]]
        for i in range(len(examples["question"]))
    ]

    # flatten
    first_sentences = [item for sublist in first_sentences for item in sublist]
    second_sentences = [item for sublist in second_sentences for item in sublist]
    
    # tokenization
    tokenized_examples = tokenizer(
        first_sentences,
        second_sentences,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt"
    )

    # re-form as `[batch_size, 4, seq_len]`
    batch_size = len(examples["question"])
    tokenized_inputs = {}
    
    for key, values in tokenized_examples.items():
        tokenized_inputs[key] = values.view(batch_size, 4, -1)
    
    return tokenized_inputs


class InferenceRunner:
    """class for Inferece"""
    
    def __init__(self, model_path, test_file, batch_size=8, max_length=512):
        self.model_path = model_path
        self.test_file = test_file
        self.batch_size = batch_size
        self.max_length = max_length
        self.logger = setup_logging()
        
        # Initiail Accelerator
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        
        # load model & tokenizer
        self._load_model_and_tokenizer()
        
        # load & preprocess data
        self._prepare_data()
    
    def _load_model_and_tokenizer(self):
        """load model & tokenizer"""
        self.logger.info(f"Load model: {self.model_path}")
        
        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # load config & model
        config = AutoConfig.from_pretrained(self.model_path)
        self.model = AutoModelForMultipleChoice.from_pretrained(
            self.model_path, config=config
        )
        
        # model -> eval
        self.model.eval()
        
        self.logger.info("Finish model load!")
    
    def _prepare_data(self):
        """prepare test data"""
        self.logger.info(f"Load test data: {self.test_file}")
        
        # Load data
        raw_dataset = load_test_data(self.test_file)
        
        # preprocess data
        def tokenize_function(examples):
            return preprocess_function(examples, self.tokenizer, self.max_length)
        
        # BATCH-preprocess data
        processed_dataset = raw_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=raw_dataset.column_names,
            desc="data-preprocess"
        )
        
        #`Data_collator for multiple-choice` need `golden-truth`
        #
        #data_collator = DataCollatorForMultipleChoice(
        #    tokenizer=self.tokenizer,
        #    pad_to_multiple_of=8 if self.accelerator.mixed_precision != "no" else None,
        #    return_tensors="pt"
        #)
        #
        # Use default
        data_collator = default_data_collator
        
        # DataLoader
        self.dataloader = DataLoader(
            processed_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=data_collator,
            num_workers=4,
            pin_memory=True
        )
        
        # Accelerator w/ model & data-loader
        self.model, self.dataloader = self.accelerator.prepare(
            self.model, self.dataloader
        )
        
        self.logger.info(f"Finish the data-preprocess, with # of samples:{len(processed_dataset)}")
    
    def predict(self, output_file="predictions.json", save_detailed=False):
        """ Prediction """
        self.logger.info("Inference...")
        
        all_predictions = []
        all_logits = []
        all_probabilities = []
        
        # Get ID with the raw-data
        with open(self.test_file, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        
        progress_bar = tqdm(
            self.dataloader, 
            desc="Inference process ", 
            disable=not self.accelerator.is_local_main_process
        )
        
        with torch.no_grad():
            for batch in progress_bar:
                # forward propagation
                outputs = self.model(**batch)
                logits = outputs.logits
                
                # probabilities
                probabilities = torch.softmax(logits, dim=-1)
                
                # Obtain the prediction results
                predictions = torch.argmax(logits, dim=-1)
                
                # Collection the results
                predictions_gathered = self.accelerator.gather_for_metrics(predictions)
                logits_gathered = self.accelerator.gather_for_metrics(logits)
                probabilities_gathered = self.accelerator.gather_for_metrics(probabilities)
                
                # turn to CPU and adding to the list
                all_predictions.extend(predictions_gathered.cpu().numpy())
                all_logits.extend(logits_gathered.cpu().numpy())
                all_probabilities.extend(probabilities_gathered.cpu().numpy())
        
        # Save the prediction
        if save_detailed:
            # Detail version
            prediction_data = []
            for i, pred in enumerate(all_predictions):
                if i < len(raw_data):
                    pred_item = {
                        "id": raw_data[i]["id"],
                        "question": raw_data[i]["question"],
                        "endings": [
                            raw_data[i]["ending0"],
                            raw_data[i]["ending1"],
                            raw_data[i]["ending2"],
                            raw_data[i]["ending3"]
                        ],
                        "prediction": int(pred),
                        "predicted_answer": raw_data[i][f"ending{pred}"],
                        "confidence_scores": all_logits[i].tolist(),
                        "probabilities": all_probabilities[i].tolist(),
                        "max_probability": float(np.max(all_probabilities[i]))
                    }
                    prediction_data.append(pred_item)
        else:
            # Pipeline version
            prediction_data = []
            for i, pred in enumerate(all_predictions):
                if i < len(raw_data):
                    if int(pred) == 0:
                        context = raw_data[i]["ending0"]
                    elif int(pred) == 1:
                        context = raw_data[i]["ending1"]
                    elif int(pred) == 2:
                        context = raw_data[i]["ending2"]
                    else:
                        context = raw_data[i]["ending3"]
                    pred_item = {
                        "id": raw_data[i]["id"],
                        "question": raw_data[i]["question"],
                        "context": context,
                    }
                    prediction_data.append(pred_item)
        
        # Save the results
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(prediction_data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Save in the inference file: {output_file}")
        self.logger.info(f"# of total samples: {len(all_predictions)}")
        
        # Statistics
        statistics = {
            "total_samples": len(all_predictions),
            "prediction_distribution": {
                f"option_{i}": int(np.sum(np.array(all_predictions) == i))
                for i in range(4)
            },
            "average_confidence": float(np.mean([np.max(probs) for probs in all_probabilities])),
            "confidence_std": float(np.std([np.max(probs) for probs in all_probabilities]))
        }
        
        return prediction_data, statistics
    
    def predict_single(self, question, endings):
        """predict single sample for testing"""

        sample = {
            "question": [question],
            "ending0": [endings[0]],
            "ending1": [endings[1]],
            "ending2": [endings[2]],
            "ending3": [endings[3]]
        }
        
        processed = preprocess_function(sample, self.tokenizer, self.max_length)
        
        for key in processed:
            processed[key] = processed[key].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**processed)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            prediction = torch.argmax(logits, dim=-1)
        
        return {
            "prediction": int(prediction.cpu().item()),
            "probabilities": probabilities.cpu().numpy().tolist()[0],
            "confidence_scores": logits.cpu().numpy().tolist()[0]
        }


def parse_args():
    """ parameter """
    parser = argparse.ArgumentParser(description="Inference for multiple choixe")
    
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True,
        help="Inferencing model"
    )
    parser.add_argument(
        "--test_file",
        type=str,
        required=True,
        help="testing file"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="batch size"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="max length"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="predictions.json",
        help="prediction output"
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Detail for analysis"
    )
    parser.add_argument(
        "--stats_file",
        type=str,
        #default="prediction_stats.json",
        default=None,
        help="Statsical files"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    set_all_seed(42)
    
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"model not found: {args.model_path}")
    
    if not os.path.exists(args.test_file):
        raise FileNotFoundError(f"test-file not found: {args.test_file}")
    
    inference_runner = InferenceRunner(
        model_path=args.model_path,
        test_file=args.test_file,
        batch_size=args.batch_size,
        max_length=args.max_length
    )
    
    # inferece process
    predictions, statistics = inference_runner.predict(
        output_file=args.output_file,
        save_detailed=args.detailed
    )
    
    if args.stats_file is not None:
        with open(args.stats_file, "w", encoding="utf-8") as f:
            json.dump(statistics, f, ensure_ascii=False, indent=2)
    
        print("\n=== 推理統計 ===")
        print(f"總樣本數: {statistics['total_samples']}")
        print(f"平均信心度: {statistics['average_confidence']:.4f}")
        print(f"信心度標準差: {statistics['confidence_std']:.4f}")
        print("\n預測分布:")
        for option, count in statistics['prediction_distribution'].items():
            percentage = (count / statistics['total_samples']) * 100
            print(f"  {option}: {count} ({percentage:.1f}%)")
    
    if inference_runner.accelerator.is_main_process:
        print(f"\n推理完成！")
        print(f"預測結果保存至: {args.output_file}")
        print(f"統計信息保存至: {args.stats_file}")


if __name__ == "__main__":
    main()
