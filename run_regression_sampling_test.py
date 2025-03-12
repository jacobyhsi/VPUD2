import json
import requests
import re
import argparse
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional
from datasets import load_from_disk
from sklearn.model_selection import train_test_split

from vpud_utils import calculate_entropy
from regression_data_processing import parse_features_to_note, create_x_row_from_x_features, create_x_row_from_x_range
from bayesian_optimisation import new_candidate

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument("--seed", default=123)
parser.add_argument("--num_x_values", default="1")
parser.add_argument("--x_values", default=None)
parser.add_argument("--x_range", default=None)
parser.add_argument("--seed_num", default="5")
parser.add_argument("--data", default="logistic_regression_3")
parser.add_argument("--feature", default="x1")
parser.add_argument("--shots", default=3)
parser.add_argument("--perm_num", default=10) # number of permutations of D
parser.add_argument("--llm", default="llama70b-nemo")
parser.add_argument("--run_name", default="fewshot")
parser.add_argument("--save_directory", default="other")
parser.add_argument("--port", default=5000)
args = parser.parse_args()
seed = int(args.seed)
np.random.seed(seed)
num_x_values = int(args.num_x_values)
x_features = args.x_values
x_range = args.x_range
shots = int(args.shots)
num_permutations = int(args.perm_num)
run_name = args.run_name
save_directory = args.save_directory
port = args.port
pd.set_option('display.max_columns', None)

################################################################################################
############################################ LLM ###############################################
################################################################################################

def get_response(prompt, label_keys, seed):
    # Add label_keys to the payload
    payload = {
        'prompt': prompt,
        'label_keys': label_keys,
        'seed': seed
    }
    
    # Send POST request to the server
    response = requests.post(f'http://localhost:{port}/predict', json=payload).json()
    
    # Extract response text and probabilities from the server's response
    response_text = response.get('response_text', "")
    probabilities = response.get('probabilities', [])
    
    return response_text, probabilities

################################################################################################
############################################ LLM ###############################################
################################################################################################

################################################################################################
########################################## Prompts #############################################
################################################################################################

def short_prompt(incontext_examples: list[str], example: str, *args, **kwargs):
    incontext_examples_str = "\n".join(incontext_examples)
    
    prompt = f"""{incontext_examples_str}\n {example} <output>"""
    
    return prompt

def note_label_prompt(note: str, label: str):
    prompt = f""" {note} <output>{label}</output>"""
    
    return prompt

def note_label_df_to_icl_examples(
        note_label_df: pd.DataFrame,
        seed: int,
        z_note: Optional[str] = None,
        u_label: Optional[str|int] = None,
    ):
    """
    Converts a DataFrame of notes and labels to incontext examples for LLM.
    Shuffles the DataFrame before converting.
    
    If z_note and u_label are provided, the z_note and u_label will be added to data as well.
    """
    
    if z_note is not None and u_label is not None:
        z_note_label_df = pd.DataFrame([{"note": z_note, "label": u_label}])
        note_label_df = pd.concat([note_label_df, z_note_label_df], ignore_index=True)
        
    note_label_df = note_label_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    incontext_examples = []
    
    for _, row in note_label_df.iterrows():
        incontext_examples.append(note_label_prompt(row['note'], row['label']))
    
    return incontext_examples


################################################################################################
########################################## Prompts #############################################
################################################################################################

################################################################################################
##################################### Data Preprocessing #######################################
################################################################################################
data_path = f'ToyRegression/logistic_regression_data/{args.data}.csv'

data = pd.read_csv(data_path, index_col=0)

label_name = "y"
label_keys = ["0", "1"]
data = data.rename(columns={label_name: 'label'})
data['label'] = data['label'].astype(int)

feature_columns = [col for col in data.columns if col != 'label']

data, test_data = train_test_split(data, test_size=0.2, random_state=seed)

data["note"] = data.apply(lambda row: parse_features_to_note(row, feature_columns), axis=1)

print("Features:", feature_columns)

selected_feature = args.feature
print("Feature to vary:", selected_feature)

# exit() # here first to check whats the feature column names

if x_features is not None:
    x_row = create_x_row_from_x_features(x_features, feature_columns)
    num_x_values = len(x_row)
elif x_range is not None:
    x_row = create_x_row_from_x_range(x_range, feature_columns)
    num_x_values = len(x_row)
else:
    x_row = data.sample(n=num_x_values, random_state=seed)
    data = data.drop(x_row.index)
D_rows = data.sample(n=shots, random_state=seed)

D_note_label_df = D_rows[['note', 'label']]

D = "\n".join(f" {row['note']} <output>{row['label']}</output>" for _, row in D_rows.iterrows())

D_rows.to_csv(f"results/{save_directory}/D_{run_name}_{args.data}.csv", index=False)

################################################################################################
##################################### Data Preprocessing #######################################
################################################################################################

################################################################################################
########################################## Sampling ############################################
################################################################################################

num_llm_seeds = int(args.seed_num)

for j in range(num_x_values):
    x = x_row['note'].iloc[j]
    x_y = x_row['label'].iloc[j]
    print("x:", x)
    
    # Initialize p(y|x)
    pyx_probs_dictionary = {label: np.zeros((num_permutations, num_llm_seeds)) for label in label_keys}
    
    # ----- Processing p(y|x) -----
    for perm_seed in range(num_permutations):
        for llm_seed in range(num_llm_seeds):
            ## p(y|x)
            print(f"\np(y|x) Seed {llm_seed + 1}/{num_llm_seeds} Permutation {perm_seed + 1}/{num_permutations}")

            prompt_pyx = short_prompt(note_label_df_to_icl_examples(D_note_label_df, perm_seed), x)
            
            # print("Prompt for p(y|x,D):")
            # print(prompt_pyx)

            # Get the prediction and probabilities from the model
            pred_pyx, probs_pyx = get_response(prompt_pyx, label_keys, seed=llm_seed)
            # print("pred_p(y|x):", pred_pyx)
            # print("probs_p(y|x):", probs_pyx)
            
            # Accumulate probabilities for puz
            for label, prob in probs_pyx.items():
                pyx_probs_dictionary[label][perm_seed, llm_seed] += prob
                
        # pyx_probs_dictionary = {label: prob / num_llm_seeds for label, prob in pyx_probs_dictionary.items()}
    # print("\nAveraged puzx probabilities:", avg_pyx_probs)
    
    for label in label_keys:
        pyx_probs_dictionary[label] = pyx_probs_dictionary[label].squeeze()
        print(pyx_probs_dictionary[label])
        mean = np.mean(pyx_probs_dictionary[label])
        print(f"Mean for {label}: {mean}")
        standard_deviation = np.std(pyx_probs_dictionary[label])
        print(f"Standard deviation for {label}: {standard_deviation}")
        cumulative_sum = np.cumsum(pyx_probs_dictionary[label])
        cumulative_mean = cumulative_sum / np.arange(1, num_permutations + 1)
        print(f"Cumulative mean for {label}: {cumulative_mean}")
        