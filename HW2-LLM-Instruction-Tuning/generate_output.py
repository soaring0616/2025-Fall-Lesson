import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from peft import PeftModel
from utils import get_prompt, get_bnb_config
import argparse
from tqdm import tqdm

import os
import random
import numpy as np
import torch

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

def generate_to_json(
    model,
    tokenizer,
    input_path="input.json",
    output_path="result.json",
    max_length=2048,
    gen_max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
):

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    model.eval()
    results = []

    for x in tqdm(data):
        instruction = x["instruction"]
        uid = x.get("id", None)

        # 準備 prompt
        prompt = get_prompt(instruction)

        # tokenize
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(model.device)

        # generate
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=gen_max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        # decode
        decoded_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        # 取出真正的回覆部分（避免包含 prompt）
        if instruction in decoded_text:
            decoded_text = decoded_text.split(instruction, 1)[-1].strip()

        results.append({
            "id": uid,
            "output": decoded_text,
        })

    # 輸出到 JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"✅ 已輸出結果至 {output_path}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="Qwen/Qwen3-4B",
        help="The path to the base model." "",
    )
    parser.add_argument(
        "--peft_path",
        type=str,
        required=True,
        help="Path to the saved PEFT checkpoint.",
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        default="",
        required=True,
        help="Path to test data.",
    )
    parser.add_argument(
        "--output_data_path",
        type=str,
        default="",
        required=True,
        help="Path to output data.",
    )
    args = parser.parse_args()

    # Load model
    bnb_config = get_bnb_config()

    if args.base_model_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model_path,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        if tokenizer.bos_token_id is None:
            tokenizer.bos_token_id = tokenizer.eos_token_id
    else:
        model_name = "Qwen/Qwen3-4B"
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, quantization_config=bnb_config
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        if tokenizer.bos_token_id is None:
            tokenizer.bos_token_id = tokenizer.eos_token_id

    # Load LoRA
    model = PeftModel.from_pretrained(model, args.peft_path)

    generate_to_json(model, tokenizer, input_path=args.test_data_path, output_path=args.output_data_path)