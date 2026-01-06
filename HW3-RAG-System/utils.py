from typing import List
import re

def get_inference_system_prompt() -> str:
    """get system prompt for generation"""
    prompt = f"""You are a precise and context-grounded QA model. 
Your task is to answer questions **strictly based on the provided context passages**.

Follow these rules carefully:
1. Use information only from the context passages.
2. If multiple passages are relevant, synthesize them **concisely** (≤25 words).
3. When quoting or extracting, prefer the **exact phrasing** from the passage.
4. If the answer cannot be found or inferred with HIGH confidence, output exactly: CANNOTANSWER.
5. Avoid adding explanations, paraphrasing, or external knowledge.
6. Respond in one clear sentence only.

Example 1:
Context: "The album gained critical acclaim and achieved gold status."
Question: "Was the album successful?"
Answer: "The album gained critical acclaim and achieved gold status."

Example 2:
Context: "Hackett starred as the title character on NBC-TV's Stanley, a 1956-57 situation comedy which ran for 19 weeks on Monday evenings at 8:30 pm EST."
Question: "Was Stanley a character Buddy Hackett played?"
Answer: "Hackett starred as the title character on NBC-TV's Stanley, a 1956-57"

Example 3:
Context: "The game's art is made up of stick figure drawings."
Question: "Did he write it in Vegas?" 
Answer: CANNOTANSWER
"""
    return prompt


def get_inference_user_prompt(query: str, context_list: List[str]) -> str:
    """Create the user prompt for generation given a query and a list of context passages."""
    context_text = "\n***\n".join([
        f"Passage {i+1}:\n{passage}"
        for i, passage in enumerate(context_list)
    ])
    prompt = f"""Context Information:\n{context_text}\n\nQuestion: {query}\n\nAnswer:"""
    return prompt


def parse_generated_answer(pred_ans: str) -> str:
    """Extract the actual answer from the model's generated text."""
    # 去除系統標籤
    if "<|im_start|>assistant" in pred_ans:
        pred_ans = pred_ans.split("<|im_start|>assistant")[-1]
    elif "assistant\n" in pred_ans:
        pred_ans = pred_ans.split("assistant\n")[-1]

    # 移除多餘特殊符號與思考標記
    pred_ans = re.sub(r'<think>.*?</think>', '', pred_ans, flags=re.DOTALL)
    pred_ans = pred_ans.replace("<|im_end|>", "").replace("<|endoftext|>", "")

    # 過濾多行或多餘字詞（僅保留第一行）
    pred_ans = pred_ans.strip().split("\n")[0]

    # 清除回答開頭的冗餘標記（例如 "Answer:"）
    pred_ans = re.sub(r'^[Aa]nswer:\s*', '', pred_ans)

    return pred_ans

