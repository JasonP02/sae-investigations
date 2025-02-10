"""
Module for analyzing reasoning patterns in language model outputs.
"""

from typing import Dict, List, Tuple
from collections import Counter
import re
from config import GenerationConfig, ReasoningPatterns
from experiment import run_multiple_experiments

def analyze_reasoning_patterns(text: str, patterns: ReasoningPatterns) -> Dict:
    """
    Analyze a text for reasoning patterns.
    
    Args:
        text: The text to analyze
        patterns: ReasoningPatterns configuration
    
    Returns:
        Dictionary containing:
        - marker_counts: Counter of reasoning markers found
        - weighted_score: Overall reasoning score weighted by marker importance
        - reasoning_flow: List of (position, marker) tuples showing reasoning progression
    """
    marker_counts = Counter()
    reasoning_flow = []
    
    # Find all occurrences of reasoning markers
    for marker, weight in patterns.reasoning_markers.items():
        # Find all occurrences with their positions
        for match in re.finditer(re.escape(marker), text.lower()):
            pos = match.start()
            marker_counts[marker] += 1
            reasoning_flow.append((pos, marker))
    
    # Calculate weighted score
    weighted_score = sum(
        count * patterns.reasoning_markers[marker]
        for marker, count in marker_counts.items()
    )
    
    # Sort reasoning flow by position
    reasoning_flow.sort()
    
    return {
        'marker_counts': marker_counts,
        'weighted_score': weighted_score,
        'reasoning_flow': reasoning_flow
    }

def evaluate_prompt_effectiveness(
    base_question: str,
    config: GenerationConfig,
    num_variants: int = 5,
    runs_per_variant: int = 3,
) -> Dict:
    """
    Evaluate different prompt variants for their effectiveness in triggering reasoning.
    
    Args:
        base_question: The core question to test
        config: Generation configuration
        num_variants: Number of prompt variants to test
        runs_per_variant: Number of runs per variant
    
    Returns:
        Dictionary containing:
        - prompt_scores: Dict mapping prompt variants to their scores
        - best_prompt: The prompt variant that produced the best reasoning
        - pattern_analysis: Detailed analysis of reasoning patterns
    """
    prompt_results = {}
    
    # Generate prompt variants
    prompt_variants = [
        f"{config.prompt_template.prefix}{base_question}{config.prompt_template.suffix}",
        f"I need help understanding: {base_question}\nCan you explain step by step?",
        f"Let's approach this systematically. {base_question}",
        f"Question: {base_question}\nThink carefully and explain your reasoning:",
        f"Help me understand the logic behind this: {base_question}"
    ][:num_variants]
    
    # Test each variant
    for prompt in prompt_variants:
        # Run experiments
        results = run_multiple_experiments(
            prompt=prompt,
            num_runs=runs_per_variant,
            config=config
        )
        
        # Analyze reasoning patterns in each generation
        pattern_scores = []
        for text in results['all_texts']:
            analysis = analyze_reasoning_patterns(text, config.reasoning_patterns)
            pattern_scores.append(analysis['weighted_score'])
        
        # Store results
        avg_score = sum(pattern_scores) / len(pattern_scores)
        prompt_results[prompt] = {
            'avg_score': avg_score,
            'pattern_scores': pattern_scores,
            'generations': results['all_texts']
        }
    
    # Find best prompt
    best_prompt = max(prompt_results.items(), key=lambda x: x[1]['avg_score'])
    
    return {
        'prompt_scores': {p: r['avg_score'] for p, r in prompt_results.items()},
        'best_prompt': best_prompt[0],
        'pattern_analysis': prompt_results
    }

def suggest_prompt_improvements(analysis_results: Dict) -> List[str]:
    """
    Suggest improvements for prompts based on analysis results.
    
    Args:
        analysis_results: Results from evaluate_prompt_effectiveness
    
    Returns:
        List of suggestions for improving prompts
    """
    suggestions = []
    
    # Analyze what worked in the best prompt
    best_prompt = analysis_results['best_prompt']
    best_score = analysis_results['prompt_scores'][best_prompt]
    
    # Look for patterns in successful prompts
    high_scoring_prompts = [
        p for p, score in analysis_results['prompt_scores'].items()
        if score > best_score * 0.8  # Within 80% of best score
    ]
    
    # Common elements in successful prompts
    if "step by step" in best_prompt.lower():
        suggestions.append("Include explicit 'step by step' instruction")
    if "explain" in best_prompt.lower():
        suggestions.append("Ask for explicit explanation")
    if "think" in best_prompt.lower():
        suggestions.append("Encourage thinking/reflection")
    if "systematically" in best_prompt.lower():
        suggestions.append("Request systematic approach")
    
    return suggestions 