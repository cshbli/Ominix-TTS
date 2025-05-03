from typing import Dict, Any

from .base_synthesizer import BaseSynthesizer
from .vits_synthesizer import VitsSynthesizer
from .vits_v3_synthesizer import VitsV3Synthesizer

def create_synthesizer(model, configs, prompt_cache=None) -> BaseSynthesizer:
    """
    Factory function to create the appropriate synthesizer
    
    Args:
        model: The model to use for synthesis
        configs: Configuration object
        prompt_cache: Cache for prompts and reference audio
        
    Returns:
        BaseSynthesizer: The appropriate synthesizer implementation
    """
    if configs.is_v3_synthesizer:
        return VitsV3Synthesizer(model, configs, prompt_cache)
    else:
        return VitsSynthesizer(model, configs)