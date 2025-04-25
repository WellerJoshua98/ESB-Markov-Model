import numpy as np
import pandas as pd
from pathlib import Path

def get_file_in_same_directory_pathlib(filename):
    # Get the directory of the current script
    dir_path = Path(__file__).parent
    # Construct the full path to the file
    file_path = dir_path / filename
    return str(file_path)

# Load your datasets
filename = 'ESB_adoption_dataset_v6_update_august_2023.xlsx'
file_path = get_file_in_same_directory_pathlib(filename)
xls = pd.ExcelFile(file_path)

# Load the "Sheet 1. District-level data"
district_level_data = pd.read_excel(xls, sheet_name='Sheet 1. District-level data')

# Load the transition data for ESB adoption phases
transition_data_clean = district_level_data[['1b. Local Education Agency (LEA) or entity name', 
                                             '3a. Number of ESBs committed',
                                             '3c. Number of ESBs awarded', 
                                             '3d. Number of ESBs ordered', 
                                             '3e. Number of ESBs delivered', 
                                             '3f. Number of ESBs operating']]

# Merge with district-level characteristics (students and free/reduced lunch percentage)
merged_data = transition_data_clean.merge(district_level_data[['1b. Local Education Agency (LEA) or entity name', 
                                                             '4b. Number of students in district', 
                                                             '4e. Percentage of students in district eligible for free or reduced lunch']], 
                                          on='1b. Local Education Agency (LEA) or entity name', 
                                          how='inner')

# Add the "Committed" state (3a. Number of ESBs committed)
state_columns = {
    'Committed': '3a. Number of ESBs committed',
    'Awarded': '3c. Number of ESBs awarded',
    'Ordered': '3d. Number of ESBs ordered',
    'Delivered': '3e. Number of ESBs delivered',
    'Operating': '3f. Number of ESBs operating'
}

# Initialize the transition matrix with 5 states (Committed, Awarded, Ordered, Delivered, Operating)
weighted_transition_matrix = np.zeros((5, 5))

# Create a state-to-index mapping for easier access
state_to_index = {
    'Committed': 0,
    'Awarded': 1,
    'Ordered': 2,
    'Delivered': 3,
    'Operating': 4
}

# Iterate through the cleaned data and calculate weighted transitions
print(merged_data.columns)
for i in range(1, len(merged_data)):
    prev_row = merged_data.iloc[i-1]
    curr_row = merged_data.iloc[i]
    
    # Get the number of students and the percentage of students eligible for free/reduced lunch (to weight the transition)
    num_students = prev_row['4b. Number of students in district']
    perc_free_reduced = prev_row['4e. Percentage of students in district eligible for free or reduced lunch']
    
    # Calculate the weight for the transition based on both characteristics
    weight = num_students * (perc_free_reduced / 100)  # Weighting by the number of students and the percentage eligible for free/reduced lunch
    
    # Check transitions for each state, weighted by the combined characteristic
    for phase, column in state_columns.items():
        prev_count = prev_row[column]
        curr_count = curr_row[column]
        
        # If there is a transition (i.e., a change from one phase to another), update the transition matrix
        if prev_count != curr_count:
            prev_state_idx = state_to_index[phase]
            curr_state_idx = state_to_index[phase]
            weighted_transition_matrix[prev_state_idx, curr_state_idx] += weight

# Normalize the transition matrix to get probabilities, taking into account both characteristics
weighted_transition_matrix_prob = weighted_transition_matrix / weighted_transition_matrix.sum(axis=1, keepdims=True)

# Create a pandas DataFrame for better display
transition_matrix_df = pd.DataFrame(weighted_transition_matrix_prob, 
                                    columns=list(state_columns.keys()), 
                                    index=list(state_columns.keys()))

# Display the weighted transition matrix using pandas
print(transition_matrix_df)