import math
import re
import numpy as np

def calculate_entropy(probs):
    # Calculate entropy using all probabilities in the dictionary
    entropy = 0.0
    for p in probs.values():
        if p > 0:
            entropy -= p * math.log2(p)
    return round(entropy, 5)

def calculate_kl_divergence(probs1, probs2):
    # Calculate KL divergence using all probabilities in the dictionaries
    # Positive infinity is returned if the KL divergence is infinite
    kl_divergence = 0.0
    for key in probs1.keys():
        if probs1[key] > 0:
            if probs2[key] == 0:
                return float('inf')
            kl_divergence += probs1[key] * math.log2(probs1[key] / probs2[key])
     
    return round(kl_divergence, 5)

def extract_label(predicted_output: str):
    match = re.search(r'<output>\s*(.*?)\s*</output>', predicted_output, re.DOTALL | re.IGNORECASE)
    if match:
        extracted_label = match.group(1).strip()
    else:
        print("Could not find output tags in the response.")
        raise ValueError("Invalid response format.")
    return extracted_label