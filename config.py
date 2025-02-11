"""
Configuration classes for generation parameters.
"""

from dataclasses import dataclass
from typing import Optional, List, Set

@dataclass
class GenerationConfig:
    """
    Configuration class for text generation analysis.
    
    Attributes:
        Generation Parameters:
            max_new_tokens (int): Maximum number of tokens to generate
            temperature (float): Controls randomness in generation (higher = more random)
            top_p (float): Nucleus sampling parameter (higher = more diverse)
            do_sample (bool): Whether to use sampling vs greedy decoding
            min_confidence (float): Minimum confidence threshold for token generation
        
        Repetition Control:
            repetition_window (int): Number of recent tokens to check for repetition
            max_repetition_ratio (float): Maximum allowed ratio of repeated tokens
            ngram_size (int): Size of ngrams to check for repetition
            max_ngram_repeats (int): Maximum times an ngram can repeat
            min_unique_ratio (float): Minimum ratio of unique tokens required
            semantic_similarity_threshold (float): Threshold for semantic similarity between phrases
            max_consecutive_fillers (int): Maximum number of consecutive filler phrases allowed
            max_recent_phrases (int): Maximum number of recent phrases to keep for similarity checks
            phrase_context_window (int): Number of characters to look back for filler patterns
        
        Special Tokens:
            pad_token_id (Optional[int]): ID of padding token
            eos_token_id (Optional[int]): ID of end-of-sequence token
            filler_patterns (List[str]): Common filler patterns to detect
            phrase_end_tokens (Set[str]): Tokens that indicate phrase boundaries
    """
    num_runs: int = 10
    max_new_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.92
    do_sample: bool = True
    min_confidence: float = 0.2
    repetition_window: int = 16
    max_repetition_ratio: float = 0.5
    ngram_size: int = 6
    max_ngram_repeats: int = 1
    min_unique_ratio: float = 0.5
    semantic_similarity_threshold: float = 0.8
    max_consecutive_fillers: int = 2
    max_recent_phrases: int = 3
    phrase_context_window: int = 20
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = 151643
    filler_patterns: List[str] = None
    phrase_end_tokens: Set[str] = None
    
    def __post_init__(self):
        """Initialize default values for complex types after instance creation."""
        if self.filler_patterns is None:
            self.filler_patterns = [
                "is a", "is an", "is the", "and the", "then the",
                "to the", "of the", "in the", "on the", "at the",
                "and then", "so the", "which is", "that is"
            ]
        if self.phrase_end_tokens is None:
            self.phrase_end_tokens = {'.', '!', '?', ',', ';', ':'}
    
    @classmethod
    def default(cls) -> 'GenerationConfig':
        """Creates a default configuration optimized for general use."""
        return cls()
    
    @classmethod
    def creative(cls) -> 'GenerationConfig':
        """Creates a configuration optimized for creative, diverse outputs."""
        return cls(
            temperature=1.0,
            top_p=0.95,
            min_confidence=0.05,
            repetition_window=8,
            max_ngram_repeats=3,
            min_unique_ratio=0.3
        )
    
    @classmethod
    def precise(cls) -> 'GenerationConfig':
        """Creates a configuration optimized for precise, focused outputs."""
        return cls(
            temperature=0.7,
            top_p=0.85,
            min_confidence=0.2,
            repetition_window=16,
            max_ngram_repeats=5,
            min_unique_ratio=0.5
        )
    
    @classmethod
    def normal(cls) -> 'GenerationConfig':
        return cls(
            max_new_tokens=1000,          # Longer generation length
            do_sample=True,               # Keep sampling for variety
            temperature=1.0,              # Standard temperature
            top_p=1.0,                   # No filtering of token distribution
            min_confidence=0.0,          # No confidence threshold
            repetition_window=100,        # Large window
            min_unique_ratio=0.0,        # No uniqueness constraint
            max_recent_phrases=10,        # More phrases allowed
            semantic_similarity_threshold=1.0,  # No similarity filtering
            max_consecutive_fillers=1000,      # Many fillers allowed
            phrase_context_window=100,         # Large context window
            eos_token_id=151643
        )
    
    @classmethod
    def balanced(cls) -> 'GenerationConfig':
        """Creates a configuration that allows natural completion while preventing degeneration."""
        return cls(
            max_new_tokens=250,          # Keep max length but allow early stopping
            num_runs=5,
            do_sample=True,               # Use sampling for natural text
            temperature=0.8,              # Slightly reduced randomness
            top_p=0.95,                  # Light filtering of unlikely tokens
            min_confidence=0.1,          # Very light confidence threshold
            repetition_window=50,         # Medium window for repetition
            min_unique_ratio=0.15,       # Allow some repetition but catch loops
            max_recent_phrases=10,         # Track reasonable number of phrases
            semantic_similarity_threshold=0.85,  # Catch near-identical phrases
            max_consecutive_fillers=10,   # Limit filler phrases
            phrase_context_window=30,     # Medium context window
            eos_token_id=151643          # Qwen's EOS token ID
        ) 