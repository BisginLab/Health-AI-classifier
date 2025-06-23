from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
import peft
import accelerate
from sklearn.model_selection import train_test_split

df = "/home/umflint.edu/brayclou/Github repo/datasets/Train_Suicide_Detection.csv"
model_name = "meta-llama/Llama-3.2-3B-Instruct"

train_df, dev_df = train_test_split(df, test_size=0.2, random_state=35)

#Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

#Load Dataset
dataset = load_dataset("csv", data_files=train_df)

def tokenize_dataset(datapoint):
    return tokenizer(datapoint["text"], padding="max_length", truncation=True)
def to_binary(datapoint):
    return 1 if datapoint["class"] == "suicide" else 0