from typing import List
import re
    
def get_inference_system_prompt() -> str:
    """get system prompt for generation"""
    prompt = f"""You are an extraction model.\n Your only job is to copy the exact answer text from the provided passages. \n Do not explain, summarize, rephrase, guess, or include external knowledge.\n Assess confidence: HIGH (clear answer), MEDIUM (partial match), LOW (uncertain). \n Output `CANNOTANSWER` only for LOW confidence cases.\n\n\n Confidence rules:\n- HIGH: Answer explicitly stated, no ambiguity. \n- MEDIUM: Answer implied or requires minor inference. \n- LOW: Answer vague or contradictory → output CANNOTANSWER"""
    return prompt

def get_inference_user_prompt(query : str, context_list : List[str]) -> str:
    """Create the user prompt for generation given a query and a list of context passages."""
    context_text = "\n***\n".join([
        f"Passage {i+1}:\n{passage}" 
        for i, passage in enumerate(context_list)
        ])

    prompt = f"""Context Information:\n{context_text}\nQuestion: {query}\n\nInstructions: 1. Copy the exact answer phrase directly from the passages above. 2. Do NOT paraphrase, expand, or add explanations. 3. Assess your confidence:\n - HIGH confidence: Extract and output the phrase \n - MEDIUM confidence: Extract but note it's partial \n - LOW confidence: Output CANNOTANSWER 4. If the answer is not in the context, output exactly: \"CANNOTANSWER\". 5. Your answer must be one continuous text span or CANNOTANSWER. \n\nAnswer:"""
    return prompt

def parse_generated_answer(pred_ans: str) -> str:
    """Extract the actual answer from the model's generated text."""
    if "<|im_start|>assistant" in pred_ans:
        pred_ans = pred_ans.split("<|im_start|>assistant")[-1]
    elif "assistant\n" in pred_ans:
        pred_ans = pred_ans.split("assistant\n")[-1]
    
    pred_ans = re.sub(r'<think>.*?</think>', '', pred_ans, flags=re.DOTALL)
    pred_ans = pred_ans.replace("<|im_end|>", "").replace("<|endoftext|>", "")
    
    pred_ans = pred_ans.strip()
    
    return pred_ans
