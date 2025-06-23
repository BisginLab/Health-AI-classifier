import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
from huggingface_hub import login
import os
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from transformers import pipeline
from huggingface_hub import HfApi
import gc

#This is the current most up-to-date version of the classifier, built toward swappable huggingface pretrained models.

load_dotenv()
# user = os.getenv('username')
token = os.getenv('token')

os.environ["TORCH_USE_CUDA_DSA"] = "1"
torch.cuda.empty_cache()
torch.cuda.synchronize()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

#with open('creds.txt', "r", encoding="utf-8") as file:
    #token = file.read()
# token="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"
login(token)
# api = HfApi()
# api.set_access_token(token)

print("Huggingface credentials passed...")

# load dataset
df = pd.read_csv('./datasets/Test_Suicide_Detection.csv', header=0)
# change labels to binary
df['class'] = [0 if x == "non-suicide" else 1 for x in df['class']]

# Initialize a dataframe made of only the feature column
df = df[2:]
print(df.columns)

print("Dataset Loaded...")
try:
    model_path = "meta-llama/Llama-3.2-3B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.config.pad_token_id = tokenizer.pad_token_id

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    # Use a pipeline as a high-level helper
    # pipe = pipeline("text-generation", model="/home/umflint.edu/brayclou/llama-csv-finetuned")
except MemoryError as e:
    print("Memory error encountered: ", e)

# initialize variables outside for loop scope
# list of actual binary label
y = []
# list of predicted binary label, parallel array
y_pred = []
# non-binary label list for 1-10 predictions
scores = []
# combined output
out = []


def get_preds(input_text):
    # prompt_message = [
    #     {"role": "system",
    #      "content": 'You are a health specialist.  Answer the following text by stating whether or not the patient is suicidal or not by giving rating of 1 to 10 formatted as number/10'},
    #     {"role": "user", "content": input_text},
    # ]
    #alternate
    prompt_message = (
        'You are a health specialist. Answer the following text by stating whether or not the patient is suicidal '
        'or not by giving rating of 1 to 10 formatted as number/10.\n\nPatient text: "'
        + input_text + '"\n'
    )

    # Get the model's prediction
    with torch.no_grad():
        response = pipe(prompt_message, max_new_tokens=50)
    returned_response = response[0]['generated_text'].replace(prompt_message, '').strip()#[2]["content"]
    return returned_response


try:
    # the next 6 lines are used to save the output every 1000 batches, then release the memory to avoid CUDA memory crash
    temp_results = []
    flush_interval = 1000  # Flush every 1000 batches
    output_file = "./llama_pred_labels.csv"

    if os.path.exists(output_file):
        os.remove(output_file)  # Ensure clean start

    batch_size = 2
    sub_size = len(df)
    threshold = 6
    sub_df = df#.iloc[50000 : sub_size] #NOTE: iloc gets commented out to process the entire dataset
    for i in tqdm(range(0, len(sub_df), batch_size), desc="Generating Responses"):
        batch = sub_df.iloc[i: i + batch_size]
        txt = [t for t in batch["text"].tolist() if isinstance(t, str) and len(t.strip()) > 0]
        if len(txt) == 0:
            print("Warning: Empty batch detected. Skipping iteration:", i)
            continue
        true = batch["class"].tolist()
        sequences = batch["text"].apply(get_preds)
        df_ids = batch["Unnamed: 0"].tolist()  # Assuming this is the ID column

        for text, true1, response, df_id in zip(txt, true, sequences, df_ids):
            # if the response triggers gpt's apology message, return a max score of 10 USE FOR GPT ONLY
            # if re.search(r".*sorry.*feeling this way.*", response, re.IGNORECASE):
            #     output_text = "10/10"
            # else:

            output_text = response
            # Extract scores from the output text
            # for match in re.finditer(r'\b(\d{1,2})/10\b', output_text):
            match = re.search(r'\b(\d{1,2})/10\b', output_text)
            if match:
                score = int(match.group(1))
                y_pred.append(score / 10)
                y.append(int(true1))
                temp_results.append([int(df_id), int(true1), score / 10])

            # store output
            # out.append([(text), output_text]) NOTE: this is commented out to save memory, please uncomment for debugging
        
        # Free GPU memory
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Flush to CSV every 1000 batches

    
    # After loop: save remaining results
    if temp_results:
        df_chunk = pd.DataFrame(temp_results, columns=["ID", "True", "Predicted"])
        df_chunk.to_csv(output_file, mode='a', index=False, header=not os.path.exists(output_file))

    print(f"All predictions saved to {output_file}")

except Exception as e:
    # output the error without damaging the process
    print("Error encountered:", e)

# #Handles gracefully exiting upon failed prediction step
# if len(y) == 0 or len(y_pred) == 0:
#     print("Error: No valid predictions. Using placeholder labels.")
#     y = [0]
#     y_pred = [0]

# # convert binary to numpy array
# y = np.array(y)
# y_pred = np.array(y_pred)

# print(f"True answers:      {y}")
# print(f"Predicted answers: {y_pred}")
# pred_labels = pd.DataFrame(columns=['True', 'Predicted'])
# pred_labels['True'] = y
# pred_labels['Predicted'] = y_pred
# pred_labels.to_csv('Github repo/llama_pred_scale_labels.csv', index=False)

# # Save the results(or the predictions?) to a csv named preds.csv
# # my_df = pd.DataFrame(out)
# # my_df.to_csv('Github repo/llama_pred_texts.csv', index=False,
# #              header=False)

#This is the current most up-to-date version of the classifier, built toward swappable huggingface pretrained models.