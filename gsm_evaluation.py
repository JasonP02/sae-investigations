#%% Imports
"""Evaluation script for GSM8K (Grade School Math) dataset."""
import os
import gc
import time
import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import torch
import re
from typing import List, Dict, Optional
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
# Download both punkt and punkt_tab
nltk.download('punkt')
nltk.download('punkt_tab')

from config import GenerationConfig
from setup import setup_model_and_sae
from experiment import run_multiple_experiments
from visualization import visualize_experiment_results
from models import ExperimentResults

# Model and SAE setup
os.environ['TORCH_USE_CUDA_DSA']='1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# %% # Utility Functions
def load_gsm_dataset(split='test') -> List[Dict]:
    """Load GSM8K dataset from jsonl files."""
    dataset = []
    file_path = f"grade-school-math/grade_school_math/data/{split}.jsonl"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            example = json.loads(line)
            dataset.append(example)
    
    return dataset

def evaluate_on_gsm(model_state, num_samples=None, store_mode="off", config=None):
    """Run evaluation on GSM dataset."""
    dataset = load_gsm_dataset(split='test')
    if num_samples:
        dataset = dataset[:num_samples]
    
    results = []
    
    results_dir = Path("gsm_results")
    results_dir.mkdir(exist_ok=True)
    
    for i, example in enumerate(tqdm(dataset, desc="Evaluating")):
        try:
            start_time = time.time()
            prompt = format_problem_prompt(example['question'])
            print(f"\nProcessing problem {i+1}/{len(dataset)}")
            
            exp_results = run_multiple_experiments(
                prompt=prompt,
                num_runs=1,
                model_state=model_state,
                config=config
            )
            gen_time = time.time() - start_time
            
            for run_idx, generated_text in enumerate(exp_results.all_texts):
                generated_answer = extract_answer(generated_text)
                correct_answer = extract_answer(example['answer'])
                
                results.append({
                    'question': example['question'],
                    'sample_num': run_idx + 1,
                    'generated_text': generated_text,
                    'generated_answer': generated_answer,
                    'correct_answer': correct_answer,
                    'is_correct': generated_answer == correct_answer if generated_answer and correct_answer else False,
                    'stopping_reason': list(exp_results.stopping_reasons.elements())[run_idx],
                    'generation_time': gen_time
                })
            
            pd.DataFrame(results).to_csv(results_dir / 'gsm_results.csv', index=False)
            
            if store_mode != "off":
                exp_results.save_step_internals(
                    run_idx=i,
                    step_idx=0,
                    internals=exp_results.generation_internals[0][0].to_numpy()
                )
            
            torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            print(f"Error processing example {i}: {str(e)}")
            continue
    
    return results

def extract_answer(text: str) -> Optional[float]:
    """Extract the final numeric answer from the model's response."""
    match = re.search(r'####\s*(-?\d*\.?\d+)', text)
    if match:
        return float(match.group(1))
    return None

def extract_answer_boxed(text: str) -> Optional[float]:
    """Extract answer from LaTeX boxed format."""
    match = re.search(r'\\boxed{(-?\d*\.?\d+)}', text)
    if match:
        return float(match.group(1))
    return None

#%% Model Setup and Dataset Info
print("Initializing model and SAE...")
start = time.time()
model_state = setup_model_and_sae()
print(f"Model setup took {time.time() - start:.2f} seconds")

# Clear CUDA cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()
    
# Load dataset to check size
print("Loading dataset to check size...")
dataset = load_gsm_dataset(split='test')
print(f"\nDataset Info:")
print(f"Total number of problems: {len(dataset)}")

# Estimate runtime
total_minutes = len(dataset) * 0.5  # Using 0.5 minutes as average
print(f"\nEstimated runtime for full dataset:")
print(f"Hours: {total_minutes / 60:.1f}")
print(f"Days: {total_minutes / (60 * 24):.1f}")

print("\nSuggested subset sizes:")
print(f"Quick test (10 problems): ~{5/60:.1f} hours")
print(f"Small sample (50 problems): ~{25/60:.1f} hours")
print(f"Medium sample (100 problems): ~{50/60:.1f} hours")

#%% Evaluation
# Configure evaluation parameters
store_mode = "off"
subset_size = 15

def format_problem_prompt(question: str) -> str:
    """Format the math problem into a clear prompt."""
    base_prompt = \
        '''
You are an AI that solves math and word problems clearly and concisely. 
Break up the problem into brief steps and solve each step only once.
Do not repeat or second-guess steps you have already solved.
You should solve the problem in a minimum number of steps.
Provide the single numeric answer, preceded by '####' and end the response immediately.

Problem:
{question}

'''
    
    return base_prompt.format(question=question)

# Run evaluation
print(f"\nStarting GSM8K evaluation with {subset_size} problems...")
config = GenerationConfig(
    store_mode=store_mode,
    num_runs=1,
    max_new_tokens=1500,
    temperature=0.8,
    save_every_n_steps=1 if store_mode == "disk" else None,
    do_sample=True,
    top_p=0.9,
    eos_token_id=model_state.tokenizer.eos_token_id,
    pad_token_id=model_state.tokenizer.pad_token_id,
    repetition_window=10,
    max_repetition_ratio=0.2,
    min_unique_ratio=0.1,
    phrase_context_window=25,
    semantic_similarity_threshold=1.0,
    max_consecutive_fillers=15
)

results = evaluate_on_gsm(
    model_state=model_state,
    num_samples=subset_size,
    store_mode=store_mode,
    config=config,
)

# Create DataFrame and save results
results_df = pd.DataFrame(results)
results_df.to_csv('gsm_results/gsm_results_detailed.csv', index=False)

#%% Plots
def analyze_results(results_df: pd.DataFrame):
    """Create comprehensive analysis of results."""
    
    # 1. Termination Statistics
    term_stats = results_df['stopping_reason'].value_counts()
    fig_term = go.Figure(data=[
        go.Bar(x=term_stats.index, y=term_stats.values)
    ])
    fig_term.update_layout(
        title="Generation Termination Reasons",
        xaxis_title="Reason",
        yaxis_title="Count"
    )
    fig_term.show()
    
    # 2. Word Analysis
    def get_word_freq(text_series, top_k=20):
        all_words = []
        for text in text_series:
            words = word_tokenize(text.lower())
            all_words.extend([w for w in words if w.isalnum()])
        return Counter(all_words).most_common(top_k)
    
    # Individual answer word frequencies
    word_freq = get_word_freq(results_df['generated_text'])
    fig_words = go.Figure(data=[
        go.Bar(x=[w[0] for w in word_freq], y=[w[1] for w in word_freq])
    ])
    fig_words.update_layout(
        title="Top Words in Generated Answers",
        xaxis_title="Word",
        yaxis_title="Frequency"
    )
    fig_words.show()
    
    # 3. Accuracy Analysis
    # Try both #### and \boxed{} formats
    results_df['boxed_answer'] = results_df['generated_text'].apply(extract_answer_boxed)
    results_df['final_answer'] = results_df.apply(
        lambda x: x['generated_answer'] if x['generated_answer'] is not None else x['boxed_answer'], 
        axis=1
    )
    results_df['is_correct_final'] = results_df.apply(
        lambda x: x['final_answer'] == x['correct_answer'] if x['final_answer'] is not None else False,
        axis=1
    )
    
    accuracy = results_df['is_correct_final'].mean()
    fig_acc = go.Figure(data=[
        go.Indicator(
            mode="gauge+number",
            value=accuracy * 100,
            title={'text': "Accuracy (%)"},
            gauge={'axis': {'range': [0, 100]}}
        )
    ])
    fig_acc.show()
    
    # Print detailed statistics
    print("\nDetailed Statistics:")
    print(f"Total samples: {len(results_df)}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Average generation time: {results_df['generation_time'].mean():.2f} seconds")
    print("\nAnswer format distribution:")
    print(f"#### format: {results_df['generated_answer'].notna().sum()}")
    print(f"\\boxed{{}} format: {results_df['boxed_answer'].notna().sum()}")
    print(f"No recognized format: {results_df['final_answer'].isna().sum()}")

# Display example responses
from IPython.display import display, Markdown

print("Example Responses:")
print("-" * 80)
for i in range(min(10, subset_size)):  # Show first 3 examples
    display(Markdown(f"**Question {i+1}:** {results_df['question'].iloc[i]}"))
    display(Markdown(f"**Answer:** {results_df['correct_answer'].iloc[i]}"))
    display(Markdown(f"**Stopping Reason:** {results_df['stopping_reason'].iloc[i]}"))
    display(Markdown(f"**Model Response:** {results_df['generated_text'].iloc[i]}"))

    print("-" * 80)

# Run full analysis
analyze_results(results_df)

# Clean up
torch.cuda.empty_cache()
gc.collect()

# %%
