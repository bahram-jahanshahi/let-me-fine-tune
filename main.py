from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW
from transformers import pipeline
import torch
import os

# Set the environment variable to disable parallelism
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

model_id = "meta-llama/Llama-3.2-1B-Instruct"
# model_id = "meta-llama/Llama-2-7b"
device = "mps"

tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map=device)


generation_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

#output = generation_pipeline("Why Sky is blue?", max_new_tokens=100)

#print(output)

input_prompts = [   "Why the Sky is blue tell me?", 
                    "Why is sky blue?", ]

tokenized_input_prompts = tokenizer(input_prompts, return_tensors="pt", padding=True, truncation=True).to(device)  

print(tokenized_input_prompts['input_ids'].shape)
print(tokenized_input_prompts['input_ids'])
print(tokenized_input_prompts['attention_mask'].shape)
print(tokenized_input_prompts['attention_mask'])

decoded = tokenizer.batch_decode(tokenized_input_prompts['input_ids'])
print(decoded)

prompt = [
    {
        "role": "system",
        "content": "you are ai assistant speaks like a pirate"
    },
    {
        "role": "user",
        "content": "Where does the sun rise?"
    }
]
print("-"*50)
chat = tokenizer.apply_chat_template(
    prompt,
    add_generation_prompt=True,
    tokenize=False,
    padding=True,
    return_tensors="pt",
)

print(chat)

print("-"*50)
tokenized_chat = tokenizer.apply_chat_template(
    prompt,
    add_generation_prompt=True,
    tokenize=True,
    padding=True,
    return_tensors="pt",
)

print(tokenized_chat)

text = "How are you?"

input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
out = model(input_ids = input_ids)
print(out)