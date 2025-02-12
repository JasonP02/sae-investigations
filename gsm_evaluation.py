#%%
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

from config import GenerationConfig
from setup import setup_model_and_sae
from experiment import run_multiple_experiments
from visualization import visualize_experiment_results
from models import ExperimentResults

# Model and SAE setup
os.environ['TORCH_USE_CUDA_DSA']='1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def load_gsm_dataset(split='test') -> List[Dict]:
    """Load GSM8K dataset from jsonl files.
    
    Args:
        split: Dataset split to load ('train' or 'test')
    """
    dataset = []
    file_path = f"grade-school-math/grade_school_math/data/{split}.jsonl"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            example = json.loads(line)
            dataset.append(example)
    
    return dataset

def format_problem_prompt(question: str) -> str:
    """Format the math problem into a clear prompt."""
    base_prompt = \
        '''
        "You are an AI that solves problems using detailed reasoning. 
You should try to evaluate your thought process as you go along, 
and revise it if there are inconsistencies using phrases like (so, I said; wait, let me think; etc).

After solving, provide your final numeric answer preceded by ####. You should end your response immediately after giving the final answer.

Here is the problem:

{question}

Let's solve this step by step:
'''
    
    return base_prompt.format(question=question)

def extract_answer(text: str) -> Optional[float]:
    """Extract the final numeric answer from the model's response."""
    # Look for #### followed by a number
    match = re.search(r'####\s*(-?\d*\.?\d+)', text)
    if match:
        return float(match.group(1))
    return None

def evaluate_on_gsm(model_state, num_samples=None, store_mode="off"):
    """Evaluate model on GSM8K dataset.
    
    Args:
        model_state: Initialized model state
        num_samples: Number of samples to evaluate (None for all)
        store_mode: Storage mode for generation results
    """
    # Load dataset
    dataset = load_gsm_dataset(split='test')
    if num_samples:
        dataset = dataset[:num_samples]
    
    results = []
    config = GenerationConfig(
        store_mode=store_mode,
        num_runs=1,  # We only need one run per question
        max_new_tokens=100,  # Math problems need enough tokens for step-by-step reasoning
        temperature=0.6,  # Slightly lower temperature for more focused reasoning
        save_every_n_steps=1 if store_mode == "disk" else None,
        do_sample=True,  # Enable sampling
        top_p=0.85,  # Nucleus sampling
        # Add special tokens
        eos_token_id=model_state.tokenizer.eos_token_id,
        pad_token_id=model_state.tokenizer.pad_token_id,
        # Disable semantic similarity checks
        phrase_context_window=0,
        max_recent_phrases=0,
        semantic_similarity_threshold=1.0,
        # Disable filler phrase checks
        max_consecutive_fillers=1000,
        filler_patterns=[]
    )
    
    
    # Create results directory
    results_dir = Path("gsm_results")
    results_dir.mkdir(exist_ok=True)
    
    # Run evaluation
    for i, example in enumerate(tqdm(dataset, desc="Evaluating")):
        try:
            # Time each step
            start_time = time.time()
            
            # Format prompt
            prompt = format_problem_prompt(example['question'])
            print(f"\nProcessing problem {i+1}/{len(dataset)}")
            print(f"Prompt length: {len(prompt)} chars")
            
            # Generate answer
            gen_start = time.time()
            exp_results = run_multiple_experiments(
                prompt=prompt,
                num_runs=1,
                model_state=model_state,
                config=config
            )
            gen_time = time.time() - gen_start
            print(f"Generation took {gen_time:.2f} seconds")
            
            # Process all generated samples
            for run_idx, generated_text in enumerate(exp_results.all_texts):
                print(f"Sample {run_idx + 1} generated {len(generated_text.split())} words")
                
                generated_answer = extract_answer(generated_text)
                correct_answer = extract_answer(example['answer'])
                
                # Store results
                result = {
                    'question': example['question'],
                    'sample_num': run_idx + 1,
                    'generated_text': generated_text,
                    'generated_answer': generated_answer,
                    'correct_answer': correct_answer,
                    'is_correct': generated_answer == correct_answer if generated_answer and correct_answer else False,
                    'stopping_reason': list(exp_results.stopping_reasons.elements())[run_idx],
                    'generation_time': gen_time / len(exp_results.all_texts)  # Average time per sample
                }
                results.append(result)
            
            # Save intermediate results
            results_df = pd.DataFrame(results)
            results_df.to_csv(results_dir / 'gsm_results.csv', index=False)
            
            # Optional: Save full experiment results
            if store_mode != "off":
                exp_results.save_step_internals(
                    run_idx=i,
                    step_idx=0,
                    internals=exp_results.generation_internals[0][0].to_numpy()
                )
            
            # Cleanup
            torch.cuda.empty_cache()
            gc.collect()
            
            total_time = time.time() - start_time
            print(f"Total processing time: {total_time:.2f} seconds")
            print("-" * 80)
            
        except Exception as e:
            print(f"Error processing example {i}: {str(e)}")
            continue
    
    return results

#%%
print("Initializing model and SAE...")
start = time.time()
model_state = setup_model_and_sae()
print(f"Model setup took {time.time() - start:.2f} seconds")

# Clear CUDA cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()

#%%
# Load dataset to check size
print("Loading dataset to check size...")
dataset = load_gsm_dataset(split='test')
print(f"\nDataset Info:")
print(f"Total number of problems: {len(dataset)}")

# Estimate runtime
# Assuming ~30 seconds per problem on a 2060 (conservative estimate)
total_minutes = len(dataset) * 0.5  # Using 0.5 minutes as average
print(f"\nEstimated runtime for full dataset:")
print(f"Hours: {total_minutes / 60:.1f}")
print(f"Days: {total_minutes / (60 * 24):.1f}")

# Suggest subsets
print("\nSuggested subset sizes:")
print(f"Quick test (10 problems): ~{5/60:.1f} hours")
print(f"Small sample (50 problems): ~{25/60:.1f} hours")
print(f"Medium sample (100 problems): ~{50/60:.1f} hours")

# Start with a small subset
subset_size = 10
print(f"\nStarting GSM8K evaluation with {subset_size} problems...")
results = evaluate_on_gsm(
    model_state=model_state,
    num_samples=subset_size,
    store_mode="off"
)

# Analyze results
results_df = pd.DataFrame(results)
print("\nEvaluation Summary:")
print(f"Total problems evaluated: {len(results_df)}")
print(f"Accuracy: {results_df['is_correct'].mean():.2%}")
print("\nStopping reasons distribution:")
print(results_df['stopping_reason'].value_counts())
print("\nAverage generation time: {:.2f} seconds".format(results_df['generation_time'].mean()))

# Save detailed results
results_df.to_csv('gsm_results/gsm_results_detailed.csv', index=False)

# Clean up
torch.cuda.empty_cache()
gc.collect() 
# %%

# Display the question, answer, and model response with IPython
from IPython.display import display, Markdown

display(Markdown(f"**Question:** {results_df['question'].iloc[0]}"))
display(Markdown(f"**Answer:** {results_df['correct_answer'].iloc[0]}"))
display(Markdown(f"**Stopping Reason:** {results_df['stopping_reason'].iloc[0]}"))
display(Markdown(f"**Model Response:** {results_df['generated_text'].iloc[0]}"))

# %%
