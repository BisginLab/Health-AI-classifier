import pandas as pd
import json
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from trl import SFTTrainer
from peft import LoraConfig, TaskType
# === Step 1: Load and Prepare Dataset ===
df = pd.read_csv("Github repo/Train_Suicide_Detection.csv")  # Must have 'text' and 'class' (int 1–10)
df = df.dropna(subset=["text", "class"])
df = df.iloc[:50000]
df['class'] = [0 if x == "non-suicide" else 1 for x in df['class']]

examples = []
for _, row in df.iterrows():
    text = row["text"]
    score = str(int(row["class"]))  # convert to string
    prompt = f"Give a prediction as to whether the following message is suicidal(1) or not(0):\n'{text}'"
    examples.append({
        "conversations": [
            {"from": "user", "value": prompt},
            {"from": "assistant", "value": score}
        ]
    })
dataset = Dataset.from_list(examples)
# === Step 2: Load Model and Tokenizer ===
model_name = "meta-llama/Llama-3.2-3B-Instruct"  # or your favorite decoder model
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_skip_modules=None
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)
# === Step 3: Define LoRA Config ===
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "v_proj"]  # Use correct names for your model
)
# === Step 4: Define Training Arguments ===
training_args = TrainingArguments(
    output_dir="./lora_llama_sft",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=2e-5,
    bf16=True,
    save_total_limit=2,
    report_to="none"
)
# === Step 5: SFTTrainer with LoRA ===
trainer = SFTTrainer(
    model=model,
    #tokenizer=tokenizer,
    train_dataset=dataset,
    args=training_args,
    dataset_text_field="conversations",
    max_seq_length=512,
    peft_config=peft_config
)
trainer.train()