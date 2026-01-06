from typing import List
import re

def get_inference_system_prompt() -> str:
    """get system prompt for generation"""
    prompt = f"""You are a helpful AI assistant. Your task is to answer questions based on the provided context passages."""
    return prompt

def get_inference_user_prompt(query : str, context_list : List[str]) -> str:
    """Create the user prompt for generation given a query and a list of context passages."""
    context_text = "\n***\n".join([
        f"Passage {i+1}:\n{passage}"
        for i, passage in enumerate(context_list)
        ])
    prompt = f"""Context Information:\n{context_text}\nQuestion: {query}\n\nBased on the context passages above, please provide a clear and accurate answer to the question. If the answer cannot be found in the provided context, please state that explicitly."""
    return prompt

def parse_generated_answer(pred_ans: str) -> str:
    """Extract the actual answer from the model's generated text."""
    if "<|im_start|>assistant" in pred_ans:
        pred_ans = pred_ans.split("<|im_start|>assistant")[-1]
    elif "assistant\n" in pred_ans:
        pred_ans = pred_ans.split("assistant\n")[-1]

    pred_ans = pred_ans.replace("<|im_end|>", "").replace("<|endoftext|>", "")

    pred_ans = pred_ans.strip()

    return pred_ans

