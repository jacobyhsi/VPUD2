import pandas as pd
import numpy as np
import ast
from itertools import product

def parse_features_to_note(row, feature_columns: list[str]):
    note = []
    for feature in feature_columns:
        note.append(f"{feature} = {row[feature]}")
    # join note with ;
    return "; ".join(note)

def create_x_row_from_x_features(x_features: str, feature_columns: list[str]):
    x_features = ast.literal_eval(x_features)
    x_row = pd.DataFrame(x_features)
    x_row["label"] = 0
    x_row["note"] = x_row.apply(lambda row: parse_features_to_note(row, feature_columns), axis=1)
    
    print("X Row:\n", x_row)
    
    return x_row

def create_x_row_from_x_range(x_range: str, feature_columns: list[str]):
    """
    Create x_row grid for a given x_range.
    
    x_range is a string with the format "start, end, step" for each feature.
    
    Example:
    x_range = "{'x1': [0, 10, 0.2],'x2': [1, 5, 1]}"
    """
    
    x_range = ast.literal_eval(x_range)
    x_row = pd.DataFrame()
    
    for feature, (start, end, step) in x_range.items():
        x_range[feature] = np.round(np.arange(float(start), float(end), float(step)),1)

    values = product(*x_range.values())
    x_row = pd.DataFrame(values, columns=x_range.keys())
    
    x_row["label"] = 0
    x_row["note"] = x_row.apply(lambda row: parse_features_to_note(row, feature_columns), axis=1)
    
    print("X Row:\n", x_row)
    
    return x_row