from datasets import load_dataset
from datasets import Dataset
from torch.utils.data import DataLoader
import json
import pandas as pd
import os
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from huggingface_hub import login, snapshot_download, hf_hub_download
from accelerate import infer_auto_device_map, init_empty_weights, load_checkpoint_and_dispatch

### ============ Parameters ==================
device = "cuda:0" if torch.cuda.is_available() else "cpu"

model_name = "pankajmathur/orca_mini_v3_7b"
### ============ Parameters ==================


### Prompt generation template >>> begin

# GENERAL_INSTRUCTION = """
#     The task is to provide the overall score for the given summary with reference to the given article on a continuous scale from 0 to 10
#     along with explanation in JSON format with "score" and "explanation" keys as follows: {"score": <float-value>, "explanation": <explanation-text>}.
#     Where a score of 0 means the summary is "irrelevant, factually incorrect and not readable" and score of 10 means "relevant, factually correct, good readability".
#     You must justify the score that you provided with clear and concise reason within 2 sentences interms of justifying the relevance, readability, factuality metrics.
#     The article text and summary text is given in triple backticks ``` with ### Article: and ### Summary: as prefix respectively.
#     Note: The generated response must be in json format without any missed braces or incomplete text. Also, it should not provide any additional information other than JSON output.
#     """

GENERAL_INSTRUCTION = """
You will be given one summary written for a news article.

Your task is to assign the single score for the summary on continuous scale from 0 to 10 along with explanation. 

Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed. You must justify the score that you provided with clear and concise reason within 2 sentences interms of justifying the relevance, fluency, coherence and consistency metrics.

The article text and summary text is given in triple backticks ``` with "Source Text:" and "Summary:" as prefix respectively.

Evaluation Criteria:
1) Relevance (1-5) - selection of important content from the source. The summary should include only important information from the source document. Annotators were instructed to penalize summaries which contained redundancies and excess information. Here, 1 is the lowest and 5 is the highest.
2) Consistency (1-5) - the factual alignment between the summary and the summarized source. A factually consistent summary contains only statements that are entailed by the source document. Annotators were also asked to penalize summaries that contained hallucinated facts. Here, 1 is the lowest and 5 is the highest
3) Coherence (1-5) - the collective quality of all sentences. We align this dimension with the DUC quality question of structure and coherence whereby "the summary should be well-structured and well-organized. The summary should not just be a heap of related information, but should build from sentence to a coherent body of information about a topic.". Here, 1 is the lowest and 5 is the highest.
4) Fluency (1-3): the quality of the summary in terms of grammar, spelling, punctuation, word choice, and sentence structure.
    - 1: Poor. The summary has many errors that make it hard to understand or sound unnatural.
    - 2: Fair. The summary has some errors that affect the clarity or smoothness of the text, but the main points are still comprehensible.
    - 3: Good. The summary has few or no errors and is easy to read and follow.

Evaluation Steps:
1. Read the summary and the source document carefully.
2. Compare the summary to the source document and identify the main points of the article.
3. Assign scores for Relevance, Consistency, Coherence and Fluency based on the Evaluation Criteria.
4. By utilizing the generated scores of Relevance, Readability, Coherence and Fluency, aggregate these scores to assign the single score for the summary on continuous scale from 0 to 10 along with explanation in JSON format with "score" and "explanation" keys as follows: {"score": <float-value>, "explanation": <explanation-text>}.
"""

MODEL_INPUT_TEMPLATE = {
    'prompts_input_with_output': "### Instruction: {}\n\nSource Text: ```{}```\nSummary: ```{}```\nResponse: {}",
    'prompts_input_without_output': "### Instruction: {}\n\nSource Text: ```{}```\nSummary: ```{}```\nResponse: ",
    'output_separator':  "Response: "
}

prompt_input_with_output = MODEL_INPUT_TEMPLATE['prompts_input_with_output']
prompts_input_without_output = MODEL_INPUT_TEMPLATE['prompts_input_without_output']

### Prompt generation template <<< end


def prepare_dataset(df, split_type = 'train'):
    final_data = []
    
    for i in range(df.shape[0]):
        instruction = ""

        text = df['SRC'].iloc[i]
        summary = ""
        if split_type in ['train', 'dev']:
            summary = df['HYP'].iloc[i]
        else:
            summary = df['TGT'].iloc[i]

        score = -1
        if split_type == "train":
            score = df['Score'].iloc[i]
            instruction = prompt_input_with_output.format(GENERAL_INSTRUCTION, text, summary, score)
        else:
            instruction = prompts_input_without_output.format(GENERAL_INSTRUCTION, text, summary)

        final_data.append({'text': instruction})

    # print(final_data[-1])
    return final_data



# df = pd.read_csv("data/summ/summ_train.tsv", delimiter = '\t')
# modified_df = pd.DataFrame(data = prepare_dataset(df, 'train'), columns = ['text'])
# dataset = Dataset.from_pandas(modified_df)
# modified_df = pd.DataFrame(data = prepare_dataset(df, 'dev'), columns = ['text'])
# train_dataset = Dataset.from_pandas(modified_df)
# del modified_df

# df = pd.read_csv("data/summ/summ_dev.tsv", delimiter = '\t')
# modified_df = pd.DataFrame(data = prepare_dataset(df, 'dev'), columns = ['text'])
# dev_dataset = Dataset.from_pandas(modified_df)

df = pd.read_csv("data/summ/summ_test.tsv", delimiter = '\t')
modified_df = pd.DataFrame(data = prepare_dataset(df, 'test'), columns = ['text'])
test_dataset = Dataset.from_pandas(modified_df)

# print("Train Sample: ", dataset['text'][0])
# print("Train Data Sample: ", train_dataset['text'][0])
# print("Dev Sample: ", dev_dataset['text'][0])
print("Test Sample: ", test_dataset['text'][0])

### =============== Model Loading ===========================
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant = True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    trust_remote_code=True,
    device_map = "auto"
)
model.config.use_cache = False
model.eval()

### ==================== Inference ========================================
final_outputs = []

# for i in tqdm(range(10)):
start = 0
end = len(test_dataset)
for i in tqdm(range(start, end)):
    instruction = test_dataset['text'][i]

    try:
        inputs = tokenizer(instruction, return_tensors="pt", padding=True).to(device)

        predicted_output = "### Response:"
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=512, temperature = 0.4, top_p = 0.6, top_k = 10, do_sample=True)
            predicted_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

        final_output = predicted_output.split('Response:', 1)[1].strip().replace('```', '')

        # print("Predicted Output: ", predicted_output)
        # print("Final Output: ", final_output)
        final_outputs.append({'Index': str(i+1), 'Output': final_output})

    except Exception as e:
        print(e)
        final_outputs.append({'Index': str(i+1), 'Output': predicted_output})

df = pd.DataFrame(data = final_outputs, columns = ['Index', 'Output'])
print(df.shape)
print(df.head())
df.to_csv('output.csv', sep='\t')