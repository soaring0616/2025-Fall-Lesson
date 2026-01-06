import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from peft import PeftModel
from utils import get_prompt, get_zero_shot_prompt
import argparse
import os
import random

def set_all_seed(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_all_seed(42)

def perplexity(
    model,
    tokenizer,
    data,
    max_length=2048,
):
    data_size = len(data)
    instructions = [get_zero_shot_prompt(x["instruction"]) for x in data]
    outputs = [x["output"] for x in data]

    # Tokenize data individually to handle errors better
    tokenized_instructions = {"input_ids": [], "attention_mask": []}
    tokenized_outputs = {"input_ids": [], "attention_mask": []}
    
    valid_indices = []
    
    for i in range(data_size):
        try:
            # Tokenize each sample individually
            tok_inst = tokenizer(instructions[i], add_special_tokens=False)
            tok_out = tokenizer(outputs[i], add_special_tokens=False)
            
            # Check if tokenization returned valid results
            if (not tok_inst["input_ids"] or not tok_out["input_ids"]):
                print(f"Warning: Skipping sample {i} due to empty tokenization")
                print(f"  Instruction length: {len(instructions[i])}")
                print(f"  Output length: {len(outputs[i])}")
                continue
            
            tokenized_instructions["input_ids"].append(tok_inst["input_ids"])
            tokenized_instructions["attention_mask"].append(tok_inst["attention_mask"])
            tokenized_outputs["input_ids"].append(tok_out["input_ids"])
            tokenized_outputs["attention_mask"].append(tok_out["attention_mask"])
            valid_indices.append(i)
            
        except Exception as e:
            print(f"Warning: Error tokenizing sample {i}: {e}")
            print(f"  Instruction: {instructions[i][:100] if len(instructions[i]) > 0 else 'EMPTY'}...")
            print(f"  Output: {outputs[i][:100] if len(outputs[i]) > 0 else 'EMPTY'}...")
            continue

    if len(valid_indices) == 0:
        raise ValueError("No valid samples found in the dataset!")
    
    print(f"Processing {len(valid_indices)} valid samples out of {data_size} total")

    # 

    ## Format data
    output_masks = []
    formatted_input_ids = []
    formatted_attention_masks = []

    for i in range(len(valid_indices)):
        instruction_input_ids = []
    
        # 只在 bos_token_id 存在時才添加
        if tokenizer.bos_token_id is not None:
            instruction_input_ids.append(tokenizer.bos_token_id)
    
        instruction_input_ids.extend(tokenized_instructions["input_ids"][i])
        
        output_input_ids = tokenized_outputs["input_ids"][i].copy()
    
        # 只在 eos_token_id 存在時才添加
        if tokenizer.eos_token_id is not None:
          output_input_ids.append(tokenizer.eos_token_id)
    
        combined_input_ids = instruction_input_ids + output_input_ids
        combined_attention_mask = [1] * len(combined_input_ids)
        output_mask = [0] * len(instruction_input_ids) + [1] * len(output_input_ids)
    
        # Truncate to max_length
        combined_input_ids = combined_input_ids[:max_length]
        combined_attention_mask = combined_attention_mask[:max_length]
        output_mask = output_mask[:max_length]
    
        formatted_input_ids.append(torch.tensor(combined_input_ids))
        formatted_attention_masks.append(torch.tensor(combined_attention_mask))
        output_masks.append(torch.tensor(output_mask))

    # Calculate ppl
    ppls = []
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    for idx in tqdm(range(len(valid_indices))):
        input_ids = formatted_input_ids[idx].unsqueeze(0).cuda()
        attn_mask = formatted_attention_masks[idx].unsqueeze(0).cuda()
        output_mask = output_masks[idx].unsqueeze(0).cuda()
        label = input_ids

        with torch.no_grad():
            out_logits = model(input_ids, attention_mask=attn_mask).logits

        shift_logits = out_logits[..., :-1, :].contiguous()
        shift_label = label[..., 1:].contiguous()
        shift_output_mask = output_mask[..., 1:].contiguous()
        perplexity_batch = torch.exp(
            (
                loss_fct(shift_logits.transpose(1, 2), shift_label) * shift_output_mask
            ).sum(1)
            / shift_output_mask.sum(1)
        )
        ppls += perplexity_batch.tolist()
    return {"perplexities": ppls, "mean_perplexity": np.mean(ppls)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="Qwen/Qwen3-4B",
        help="The path to the base model.",
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        default="",
        required=True,
        help="Path to test data.",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    with open(args.test_data_path, "r") as f:
        data = json.load(f)

    # Validate data
    print(f"Loaded {len(data)} samples from {args.test_data_path}")
    for i, sample in enumerate(data[:3]):  # Check first 3 samples
        print(f"\nSample {i}:")
        print(f"  Instruction: {sample.get('instruction', 'MISSING')[:100]}...")
        print(f"  Output: {sample.get('output', 'MISSING')[:100]}...")

    model.eval()
    ppl = perplexity(model, tokenizer, data)
    print("\n" + "="*50)
    print("Mean perplexity:", ppl["mean_perplexity"])
    print("="*50)