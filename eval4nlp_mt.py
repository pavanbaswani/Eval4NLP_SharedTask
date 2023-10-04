from datasets import load_dataset
from datasets import Dataset
import json
import pandas as pd
import os
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from huggingface_hub import login
from accelerate import infer_auto_device_map, init_empty_weights

### ============ Parameters ==================
device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "pankajmathur/orca_mini_v3_7b"

dataset_name = "en_de" ### ["en_de", "en_es", "en_zh"]
### ============ Parameters ==================


### Prompt generation template >>> begin

# GENERAL_INSTRUCTION = """
#     The task is to score a translated text from {English} to {German} with respect to the source sentence on a continous scale from 0 to 100,
#     along with explaination in JSON format with "score" and "explanation" keys as follows: {"score": <float-value>, "explanation": <explanation-text>}.
#     Where a score of zero means "no meaning preserved and poor translation quality" and score of one hundred means "excellant translation quality with perfect meaning and grammar".
#     You must justify the score that you provided with clear and concise reason within 2 sentences interms of justifying the adequacy, fluency, faithfulness metrics.
#     The source sentence and target sentence is given in triple backticks  with ### source sentence: and ### target sentence: as prefix respectively.
# Note: The generated response must be in json format without any missed braces or incomplete text. Also, it should not provide any additional information other than JSON output.
#     """

GENERAL_INSTRUCTION = """
You will be given one translated sentence in {Spanish} for a source sentence in {English}.

Your task is to assign the single score for the translation on continuous scale from 0 to 100 along with explanation.

Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed. For explanation, you must justify the score that you provided with clear and concise reason within 2 sentences interms of justifying the adequacy, fluency and faithfulness metrics.

The source text and translation text is given in triple backticks ``` with "Source Text:" and "Translation:" as prefix respectively.

Evaluation Criteria:
1) Adequacy (1-5) - the correspondence of the target text to the source text, including the expressive means in translation. Annotators were instructed to penalize translation which contained misinformation, redundancies and excess information. Here, 1 is the lowest and 5 is the highest.
2) Faithfulness (1-5) - translation faithfulness to the meaning depends on how the translator interprets the speaker's intention and does not imply that one should never or always translate literally.  Here, 1 is the lowest and 5 is the highest.
3) Fluency (1-3): the quality of the translation in terms of grammar, spelling, punctuation, word choice, and sentence structure.
    - 1: Poor. The translation has many errors that make it hard to understand or sound unnatural.
    - 2: Fair. The translation has some errors that affect the clarity or smoothness of the text, but the main points are still comprehensible.
    - 3: Good. The translation has few or no errors and is easy to read and follow.

Evaluation Steps:
1. Read the translation and the source document carefully.
2. Compare the translation to the source text.
3. Assign scores for Adequacy, Faithfulness and Fluency based on the Evaluation Criteria.
4. By utilizing the generated scores of Adequacy, Faithfulness and Fluency, aggregate these scores to assign the single score for the translation on continuous scale from 0 to 100 along with explanation in JSON format with "score" and "explanation" keys as follows: {"score": <float-value>, "explanation": <explanation-text>}.
"""

MODEL_INPUT_TEMPLATE = {
    'prompts_input_with_output': "### Instruction: {}\n### Source Text: ```{}```\n### Translation: ```{}```\n### Response: {}",
    'prompts_input_without_output': "### Instruction: {}\n### Source Text: ```{}```\n### Translation: ```{}```\n### Response: ",
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
        translation = ""
        if split_type in ['train', 'dev']:
            translation = df['HYP'].iloc[i]
        else:
            translation = df['TGT'].iloc[i]

        score = -1
        if split_type == "train":
            score = df['Score'].iloc[i]
            instruction = prompt_input_with_output.format(GENERAL_INSTRUCTION, text, translation, score)
        else:
            instruction = prompts_input_without_output.format(GENERAL_INSTRUCTION, text, translation)

        final_data.append({'text': instruction})

    # print(final_data[-1])
    return final_data



# df = pd.read_csv("data/" + dataset_name + "/" + dataset_name + "_train.tsv", delimiter = '\t')
# modified_df = pd.DataFrame(data = prepare_dataset(df, 'train'), columns = ['text'])
# dataset = Dataset.from_pandas(modified_df)
# modified_df = pd.DataFrame(data = prepare_dataset(df, 'dev'), columns = ['text'])
# train_dataset = Dataset.from_pandas(modified_df)
# del modified_df

# df = pd.read_csv("data/" + dataset_name + "/" + dataset_name + "_dev.tsv", delimiter = '\t')
# modified_df = pd.DataFrame(data = prepare_dataset(df, 'dev'), columns = ['text'])
# dev_dataset = Dataset.from_pandas(modified_df)

df = pd.read_csv("data/" + dataset_name + "/" + dataset_name + "_test.tsv", delimiter = '\t')
modified_df = pd.DataFrame(data = prepare_dataset(df, 'test'), columns = ['text'])
test_dataset = Dataset.from_pandas(modified_df)

# print("Train Sample: ", dataset['text'][0])
# print("Train Data Sample: ", train_dataset['text'][0])
# print("Dev Sample: ", dev_dataset['text'][0])
print("Test Sample: ", test_dataset['text'][0])


### =============== Model Loading ===========================
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    device_map = "auto"
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    trust_remote_code=True,
)
model.config.use_cache = False
model.eval()

### ==================== Inference ========================================
final_outputs = []

start = 0
end = len(test_dataset)

for i in tqdm(range(start, end)):
    instruction = test_dataset['text'][i]

    try:
        inputs = tokenizer(instruction, return_tensors="pt").to(device)

        outputs = model.generate(**inputs, max_new_tokens=512)
        predicted_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

        final_output = predicted_output.split('### Response:', 1)[1].strip().replace('```', '')

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