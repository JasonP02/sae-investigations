"""
Configuration classes for generation parameters.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Set, Dict

@dataclass
class GenerationConfig:
    """Configuration class for text generation analysis.
    
    This class provides a single, well-documented configuration for text generation
    with parameters tuned for balanced, natural text generation while preventing
    degeneration.
    
    Attributes:
        Generation Control:
            num_runs (int): Number of generation attempts to run (default: 5)
            max_new_tokens (int): Maximum number of tokens to generate (default: 15)
            temperature (float): Controls randomness in generation. Higher values (>1.0) 
                increase randomness, lower values (<1.0) make text more focused and 
                deterministic (default: 0.9)
            top_p (float): Nucleus sampling parameter. Controls cumulative probability
                threshold for token selection. Higher values allow more diverse tokens
                (default: 0.9)
            do_sample (bool): Whether to use sampling vs greedy decoding. True enables
                sampling for more natural text (default: True)
            min_confidence (float): Minimum confidence threshold for token generation.
                Higher values force the model to be more certain (default: 0.05)
        
        Repetition Control:
            repetition_window (int): Number of recent tokens to check for repetition.
                Larger windows catch long-range repetition (default: 30)
            max_repetition_ratio (float): Maximum allowed ratio of repeated tokens
                within the repetition window (default: 0.7)
            ngram_size (int): Size of ngrams to check for repetition (default: 4)
            max_ngram_repeats (int): Maximum times an ngram can repeat (default: 2)
            min_unique_ratio (float): Minimum ratio of unique tokens required in the
                repetition window. Lower values allow more repetition (default: 0.1)
            
        Semantic Control:
            semantic_similarity_threshold (float): Maximum allowed semantic similarity
                between phrases. Higher values allow more similar content (default: 0.95)
            max_consecutive_fillers (int): Maximum allowed consecutive filler phrases
                before stopping generation (default: 15)
            max_recent_phrases (int): Maximum number of recent phrases to track for
                similarity checks (default: 5)
            phrase_context_window (int): Number of characters to look back for detecting
                filler patterns (default: 50)
        
        Special Tokens:
            pad_token_id (Optional[int]): ID of padding token, if needed (default: None)
            eos_token_id (Optional[int]): ID of end-of-sequence token (default: None)
            filler_patterns (List[str]): Common filler patterns to detect and limit.
                Initialized in __post_init__ with common English filler phrases
            phrase_end_tokens (Set[str]): Tokens that indicate phrase boundaries.
                Initialized in __post_init__ with common punctuation
        
        Monitoring:
            save_layers (Set[int]): Layer numbers to save during generation (default: {10})
            save_frequency (int): Save internal states every nth step (default: 1)
            debug_logging (bool): Enable detailed debug logging (default: False)
            log_frequency (int): Log progress every n steps (default: 10)
            save_every_n_steps (int): Save model states every n steps (default: 100)
        Analysis Mode:
            store_mode (str): Storage mode for results ("disk", "memory", or "off")
    """
    
    # Generation Control
    num_runs: int = 5
    max_new_tokens: int = 200
    temperature: float = 0.9
    top_p: float = 0.9
    do_sample: bool = True
    min_confidence: float = 0.05
    
    # Repetition Control
    repetition_window: int = 30
    max_repetition_ratio: float = 0.7
    ngram_size: int = 4
    max_ngram_repeats: int = 2
    min_unique_ratio: float = 0.05
    
    # Semantic Control
    semantic_similarity_threshold: float = 0.95
    max_consecutive_fillers: int = 15
    max_recent_phrases: int = 5
    phrase_context_window: int = 15
    
    # Special Tokens
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    filler_patterns: List[str] = None
    phrase_end_tokens: Set[str] = None
    
    # Monitoring
    save_layers: Set[int] = field(default_factory=lambda: {10})
    save_frequency: int = 1
    debug_logging: bool = False
    log_frequency: int = 10
    save_every_n_steps: int = 200
    
    # Analysis Mode
    store_mode: str = "off"  # Options: "disk", "memory", or "off"

    def __post_init__(self):
        """Initialize default values for complex types after instance creation."""
        if self.filler_patterns is None:
            self.filler_patterns = []
        if self.phrase_end_tokens is None:
            self.phrase_end_tokens = {'.', '!', '?', ',', ';', ':'}
        if not isinstance(self.save_layers, set):
            self.save_layers = set(self.save_layers)
    
    def to_dict(self) -> Dict:
        """Convert config to JSON-serializable dict"""
        return {
            **{k: v for k, v in self.__dict__.items() 
               if k not in {'filler_patterns', 'phrase_end_tokens', 'save_layers'}},
            'filler_patterns': list(self.filler_patterns),
            'phrase_end_tokens': list(self.phrase_end_tokens),
            'save_layers': list(self.save_layers)
        } 