## utlis.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, TaskType

def get_prompt(instruction: str) -> str:
    sys_prompt = "你是一個專精中國古文（文言文）與現代中文的語言專家，\
            擅長文言文與白話文之間的準確翻譯與風格轉換。\
            你的任務將會是`將中文古文（文言文）精準翻譯為中文白話文`，\
            或是`將中文白話文精準翻譯為中文古文（文言文）`。\
            要求：\
            - 翻譯時，請產生最符合歷史語言習慣的中文輸出\
            - 請注意，只能使用 ** 繁體中文 ** 。\
            - 請檢查所有資訊是否準確，並在回答時保持簡潔，不需要任何其他回饋。\
            - 若任務開頭有「文言文翻譯：」，請以「答案：」開頭，並只給相對應的正確回答\
            以下是你本次的任務：\n\n "
    return sys_prompt+instruction

def get_few_shot_prompt(instruction: str) -> str:
    sys_prompt = "你是一個專精中國古文（文言文）與現代中文的語言專家，\
            擅長文言文與白話文之間的準確翻譯與風格轉換。\
            你的任務將會是`將中文古文（文言文）精準翻譯為中文白話文`，\
            或是`將中文白話文精準翻譯為中文古文（文言文）`。\
            要求：\
            - 翻譯時，請產生最符合歷史語言習慣的中文輸出\
            - 請注意，只能使用 ** 繁體中文 ** 。\
            - 請檢查所有資訊是否準確，並在回答時保持簡潔，不需要任何其他回饋。\
            - 若任務開頭有「文言文翻譯：」，請以「答案：」開頭，並只給相對應的正確回答\
            \n\n以下是第一個範例：\
            - 沒過十天，鮑泉果然被拘捕。\n幫我把這句話翻譯成文言文\
            - 後未旬，果見囚執。\
            \n\n以下是第二個範例：\
            - 文言文翻譯：\n明日，趙用賢疏入。\
            - 答案：第二天，趙用賢的疏奏上。\
            \n\n以下是第三個範例：\
            - 祀明堂，遷給事中兼龍圖閣學士。\n翻譯成現代文：\
            - 祭祀明堂，升任給事中，兼龍圖閣學士。\
            以下是你本次的任務：\n\n "
    return sys_prompt+instruction

def get_zero_shot_prompt(instruction: str) -> str:
    sys_prompt = "你是一個專精中國古文（文言文）與現代中文的語言專家，\
            擅長文言文與白話文之間的準確翻譯與風格轉換。\
            你的任務將會是`將中文古文（文言文）精準翻譯為中文白話文`，\
            或是`將中文白話文精準翻譯為中文古文（文言文）`。\
            要求：\
            - 翻譯時，請產生最符合歷史語言習慣的中文輸出\
            - 請注意，只能使用 ** 繁體中文 ** 。\
            - 請檢查所有資訊是否準確，並在回答時保持簡潔，不需要任何其他回饋。\
            - 若任務開頭有「文言文翻譯：」，請以「答案：」開頭，並只給相對應的正確回答\
            以下是你本次的任務：\n\n "
    return sys_prompt+instruction

def get_bnb_config() -> BitsAndBytesConfig:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
      )
    return bnb_config

def get_lora_config() -> LoraConfig:
    lora_config = LoraConfig(
            r=8,                   #Rank
            lora_alpha=16,         #Scaling
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
    )
    return lora_config