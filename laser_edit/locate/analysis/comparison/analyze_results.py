import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import re

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Read the two CSV files
df1 = pd.read_csv('/home/hyeryung/data/mucoco/laser_edit/locate/evaluation_results/evaluation_summary.csv')
df2 = pd.read_csv('/home/hyeryung/data/mucoco/laser_edit/llm_experiments/locate_with_llm/evaluation_results/evaluation_summary.csv')

# Merge method and prompt_type columns into prompt_type/method
# First, ensure both columns exist (fill missing with empty string)
if 'method' not in df1.columns:
    df1['method'] = ''
if 'prompt_type' not in df1.columns:
    df1['prompt_type'] = ''
if 'method' not in df2.columns:
    df2['method'] = ''
if 'prompt_type' not in df2.columns:
    df2['prompt_type'] = ''

# Create the merged column
df1['prompt_type/method'] = df1['method'].fillna('') + df1['prompt_type'].fillna('')
df2['prompt_type/method'] = df2['method'].fillna('') + df2['prompt_type'].fillna('')

# Concatenate the two dataframes
df = pd.concat([df1, df2], ignore_index=True)

# Filter rows based on num_examples
# For toxic spans: num_examples >= 115
# For inconsistent spans: num_examples >= 300
df = df[
    ((df['task'] == 'toxic') & (df['num_examples'] >= 115)) |
    ((df['task'] == 'inconsistent') & (df['num_examples'] >= 300)) | 
    ((df['task'] == 'toxic_extended') & (df['num_examples'] >= 90))
]

# Convert execution_seconds to numeric (in case it's stored as string)
# This will convert empty strings and invalid values to NaN
df['execution_seconds'] = pd.to_numeric(df['execution_seconds'], errors='coerce')
# Remove rows with missing execution_seconds (NaN after conversion)
df = df.dropna(subset=['execution_seconds'])

# Identify result types
# Locate/edit locate results have methods like "gradient_norm_max_num_tokens_X" or "attention_max_num_tokens_X"
# LLM results have prompt_type values
df['result_type'] = df.apply(
    lambda row: 'locate' if ('gradient_norm' in str(row['method']) or 'attention' in str(row['method'])) 
                else 'llm',
    axis=1
)

# Filter LLM results to only include those with "type" in the filename
llm_mask = df['result_type'] == 'llm'
df_llm = df[llm_mask]
df_llm_filtered = df_llm[df_llm['prediction_file'].str.contains('type', case=False, na=False)]
df = pd.concat([df[~llm_mask], df_llm_filtered], ignore_index=True)

# Drop rows for o4-mini and gpt4.1-mini models
df = df[~df['model'].str.contains('o4-mini|gpt-4.1-mini', case=False, na=False, regex=True)]

# Drop CoT prompts for gpt-5 series models (reasoning models)
df = df[~((df['model'].str.contains('gpt-5', case=False, na=False)) & 
          (df['prompt_type'].str.contains('cot', case=False, na=False)))]

# Extract compute function and max_num_tokens for locate results
def extract_compute_function(method_str):
    if pd.isna(method_str) or method_str == '':
        return None
    method_str = str(method_str)
    if 'gradient_norm' in method_str:
        return 'gradient_norm'
    elif 'attention' in method_str:
        return 'attention'
    return None

def extract_max_num_tokens(method_str):
    if pd.isna(method_str) or method_str == '':
        return None
    method_str = str(method_str)
    # Extract number after max_num_tokens_
    match = re.search(r'max_num_tokens_(\d+)', method_str)
    if match:
        return int(match.group(1))
    return None

df['compute_function'] = df['method'].apply(extract_compute_function)
df['max_num_tokens'] = df['method'].apply(extract_max_num_tokens)

# Clean prompt_type to remove type1_v{number} patterns and trailing underscores
def clean_prompt_type(prompt_type_str):
    if pd.isna(prompt_type_str) or prompt_type_str == '':
        return prompt_type_str
    prompt_type_str = str(prompt_type_str)
    # Remove type1_v{number} pattern (e.g., "type1_v3", "type1_v1")
    cleaned = re.sub(r'type1_v\d+', '', prompt_type_str)
    # Remove trailing underscores
    cleaned = cleaned.rstrip('_')
    return cleaned.strip()

df['prompt_type_cleaned'] = df['prompt_type'].apply(clean_prompt_type)

# Create a unique identifier for coloring
# For locate results: model + compute_function
# For llm results: model
df['color_key'] = df.apply(
    lambda row: f"{row['model']}_{row['compute_function']}" if row['result_type'] == 'locate' and row['compute_function'] is not None
    else f"{row['model']}_llm" if row['result_type'] == 'llm'
    else f"{row['model']}_other",
    axis=1
)

# Get unique color keys and assign colors
unique_keys = df['color_key'].unique()
colors = plt.cm.tab20(np.linspace(0, 1, len(unique_keys)))
color_map = dict(zip(unique_keys, colors))

# Plot 1: Execution time vs Mean F1 (separate plots for each task)
tasks = df['task'].unique()
for task in tasks:
    task_df = df[df['task'] == task]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot scatter points
    for key in unique_keys:
        key_df = task_df[task_df['color_key'] == key]
        if len(key_df) > 0:
            ax.scatter(key_df['execution_seconds'], key_df['mean_f1'], 
                      c=color_map[key], label=key, s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
            
            # Add labels on circles
            for idx, row in key_df.iterrows():
                if row['result_type'] == 'locate' and pd.notna(row['max_num_tokens']):
                    # Display max_num_tokens for locate results - positioned to the left
                    label_text = str(int(row['max_num_tokens']))
                    ax.annotate(label_text, 
                              (row['execution_seconds'], row['mean_f1']),
                              xytext=(-10, 0), textcoords='offset points',
                              fontsize=8, ha='right', va='center')
                elif row['result_type'] == 'llm' and pd.notna(row['prompt_type_cleaned']) and row['prompt_type_cleaned'] != '':
                    # Display cleaned prompt_type for llm results - positioned slightly above
                    label_text = str(row['prompt_type_cleaned'])
                    ax.annotate(label_text, 
                              (row['execution_seconds'], row['mean_f1']),
                              xytext=(0, 8), textcoords='offset points',
                              fontsize=8, ha='center', va='bottom')
    
    # Connect dots for locate results (same model + compute function, sorted by max_num_tokens)
    locate_df = task_df[task_df['result_type'] == 'locate']
    for model in locate_df['model'].unique():
        for compute_func in ['gradient_norm', 'attention']:
            model_compute_df = locate_df[
                (locate_df['model'] == model) & 
                (locate_df['compute_function'] == compute_func) &
                (locate_df['max_num_tokens'].notna())
            ]
            if len(model_compute_df) > 1:
                # Sort by max_num_tokens
                model_compute_df = model_compute_df.sort_values('max_num_tokens')
                key = f"{model}_{compute_func}"
                ax.plot(model_compute_df['execution_seconds'], model_compute_df['mean_f1'],
                       color=color_map.get(key, 'gray'), linestyle='--', alpha=0.5, linewidth=1.5)
    
    # Connect dots for llm results (same model)
    llm_df = task_df[task_df['result_type'] == 'llm']
    for model in llm_df['model'].unique():
        model_df = llm_df[llm_df['model'] == model]
        if len(model_df) > 1:
            # Sort by execution_seconds for better visualization
            model_df = model_df.sort_values('execution_seconds')
            key = f"{model}_llm"
            ax.plot(model_df['execution_seconds'], model_df['mean_f1'],
                   color=color_map.get(key, 'gray'), linestyle='--', alpha=0.5, linewidth=1.5)
    
    # # Add horizontal line for gpt-5 0shot performance (inconsistent task only)
    # # Note: gpt-5 (not gpt-5-nano) for inconsistent task
    # if task == 'inconsistent':
    #     gpt5_0shot = task_df[
    #         (task_df['model'].str.contains('gpt-5', case=False, na=False)) &
    #         (~task_df['model'].str.contains('gpt-5-nano', case=False, na=False)) &
    #         (task_df['result_type'] == 'llm') &
    #         (task_df['prompt_type_cleaned'].str.contains('0shot', case=False, na=False))
    #     ]
    #     if len(gpt5_0shot) > 0:
    #         # Get the mean_f1 value (take the first one if multiple exist)
    #         mean_f1_value = gpt5_0shot.iloc[0]['mean_f1']
    #         print(f"DEBUG: Found gpt-5 (not nano) 0shot for inconsistent task. Mean F1: {mean_f1_value}")
    #         print(f"DEBUG: Model: {gpt5_0shot.iloc[0]['model']}, Prompt type cleaned: {gpt5_0shot.iloc[0]['prompt_type_cleaned']}")
    #         # Draw horizontal line with a distinct color (red) and high zorder to appear on top
    #         ax.axhline(y=mean_f1_value, color='red', linestyle='-', linewidth=2.5, alpha=0.8, 
    #                   label='GPT-5 0shot baseline', zorder=10)
    #     else:
    #         print(f"DEBUG: No gpt-5 (not nano) 0shot found for inconsistent task")
    #         print(f"DEBUG: Available models in inconsistent task: {task_df[task_df['result_type'] == 'llm']['model'].unique()}")
    #         print(f"DEBUG: Available prompt_type_cleaned in inconsistent task: {task_df[task_df['result_type'] == 'llm']['prompt_type_cleaned'].unique()}")
    
    # # Add horizontal line for gpt-5-nano 5shot performance (toxic task only)
    # if task == 'toxic':
    #     gpt5_nano_5shot = task_df[
    #         (task_df['model'].str.contains('gpt-5-nano', case=False, na=False)) &
    #         (task_df['result_type'] == 'llm') &
    #         (task_df['prompt_type_cleaned'].str.contains('5shot', case=False, na=False))
    #     ]
    #     if len(gpt5_nano_5shot) > 0:
    #         # Get the mean_f1 value (take the first one if multiple exist)
    #         mean_f1_value = gpt5_nano_5shot.iloc[0]['mean_f1']
    #         print(f"DEBUG: Found gpt-5-nano 5shot for toxic task. Mean F1: {mean_f1_value}")
    #         print(f"DEBUG: Model: {gpt5_nano_5shot.iloc[0]['model']}, Prompt type cleaned: {gpt5_nano_5shot.iloc[0]['prompt_type_cleaned']}")
    #         # Draw horizontal line with a distinct color (red) and high zorder to appear on top
    #         ax.axhline(y=mean_f1_value, color='red', linestyle='-', linewidth=2.5, alpha=0.8, 
    #                   label='GPT-5-nano 5shot baseline', zorder=10)
    #     else:
    #         print(f"DEBUG: No gpt-5-nano 5shot found for toxic task")
    #         print(f"DEBUG: Available models in toxic task: {task_df[task_df['result_type'] == 'llm']['model'].unique()}")
    #         print(f"DEBUG: Available prompt_type_cleaned in toxic task: {task_df[task_df['result_type'] == 'llm']['prompt_type_cleaned'].unique()}")
    
    ax.set_xscale('log')
    ax.set_xlabel('Cost (Latency in log seconds)', fontsize=12)
    ax.set_ylabel('Mean F1', fontsize=12)
    ax.set_title(f'Execution Time vs Mean F1 - {task.capitalize()} Spans', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'/home/hyeryung/data/mucoco/laser_edit/locate/analyze_results/execution_time_vs_f1_{task}.png', dpi=300, bbox_inches='tight')
    plt.close()

# Plot 2: Precision vs Recall (separate plots for each task)
for task in tasks:
    task_df = df[df['task'] == task]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot scatter points
    for key in unique_keys:
        key_df = task_df[task_df['color_key'] == key]
        if len(key_df) > 0:
            ax.scatter(key_df['mean_precision'], key_df['mean_recall'], 
                      c=color_map[key], label=key, s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
            
            # Add labels on circles
            for idx, row in key_df.iterrows():
                if row['result_type'] == 'locate' and pd.notna(row['max_num_tokens']):
                    # Display max_num_tokens for locate results - positioned to the left
                    label_text = str(int(row['max_num_tokens']))
                    ax.annotate(label_text, 
                              (row['mean_precision'], row['mean_recall']),
                              xytext=(-10, 0), textcoords='offset points',
                              fontsize=8, ha='right', va='center')
                elif row['result_type'] == 'llm' and pd.notna(row['prompt_type_cleaned']) and row['prompt_type_cleaned'] != '':
                    # Display cleaned prompt_type for llm results - positioned slightly above
                    label_text = str(row['prompt_type_cleaned'])
                    ax.annotate(label_text, 
                              (row['mean_precision'], row['mean_recall']),
                              xytext=(0, 8), textcoords='offset points',
                              fontsize=8, ha='center', va='bottom')
    
    # Connect dots for locate results (same model + compute function, sorted by max_num_tokens)
    locate_df = task_df[task_df['result_type'] == 'locate']
    for model in locate_df['model'].unique():
        for compute_func in ['gradient_norm', 'attention']:
            model_compute_df = locate_df[
                (locate_df['model'] == model) & 
                (locate_df['compute_function'] == compute_func) &
                (locate_df['max_num_tokens'].notna())
            ]
            if len(model_compute_df) > 1:
                # Sort by max_num_tokens
                model_compute_df = model_compute_df.sort_values('max_num_tokens')
                key = f"{model}_{compute_func}"
                ax.plot(model_compute_df['mean_precision'], model_compute_df['mean_recall'],
                       color=color_map.get(key, 'gray'), linestyle='--', alpha=0.5, linewidth=1.5)
    
    # Connect dots for llm results (same model)
    llm_df = task_df[task_df['result_type'] == 'llm']
    for model in llm_df['model'].unique():
        model_df = llm_df[llm_df['model'] == model]
        if len(model_df) > 1:
            key = f"{model}_llm"
            ax.plot(model_df['mean_precision'], model_df['mean_recall'],
                   color=color_map.get(key, 'gray'), linestyle='--', alpha=0.5, linewidth=1.5)
    
    ax.set_xlabel('Precision', fontsize=12)
    ax.set_ylabel('Recall', fontsize=12)
    ax.set_title(f'Precision vs Recall - {task.capitalize()} Spans', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(f'/home/hyeryung/data/mucoco/laser_edit/locate/analyze_results/precision_vs_recall_{task}.png', dpi=300, bbox_inches='tight')
    plt.close()

# Debug: Check for records with 0 precision and 0 recall for inconsistent spans
inconsistent_df = df[df['task'] == 'inconsistent']
zero_precision_recall = inconsistent_df[
    (inconsistent_df['mean_precision'] == 0.0) & (inconsistent_df['mean_recall'] == 0.0)
]

if len(zero_precision_recall) > 0:
    print("\n" + "="*80)
    print("DEBUG: Records with 0.0 precision and 0.0 recall for inconsistent spans:")
    print("="*80)
    for idx, row in zero_precision_recall.iterrows():
        print(f"\nRow {idx}:")
        print(f"  Model: {row['model']}")
        print(f"  Method: {row['method']}")
        print(f"  Prompt Type: {row.get('prompt_type', 'N/A')}")
        print(f"  Precision: {row['mean_precision']}")
        print(f"  Recall: {row['mean_recall']}")
        print(f"  F1: {row['mean_f1']}")
        print(f"  Num Examples: {row['num_examples']}")
        print(f"  Prediction File: {row['prediction_file']}")
        print(f"  Execution Seconds: {row['execution_seconds']}")
    print("="*80 + "\n")

# Save the processed dataframe
output_csv_path = '/home/hyeryung/data/mucoco/laser_edit/locate/analyze_results/processed_results.csv'
df.to_csv(output_csv_path, index=False)
print(f"Processed dataframe saved to: {output_csv_path}")
print(f"Columns saved: {list(df.columns)}")

print("Plots saved successfully!")
print(f"Total rows after filtering: {len(df)}")
print(f"Tasks: {df['task'].unique()}")
print(f"Result types: {df['result_type'].value_counts()}")

