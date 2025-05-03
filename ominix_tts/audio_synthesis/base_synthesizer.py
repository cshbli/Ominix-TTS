import torch
from typing import List

class BaseSynthesizer:
    """Base class for audio synthesizers"""
    
    def __init__(self, model, configs):
        self.model = model
        self.configs = configs
        self.device = configs.device
        self.precision = torch.float16 if configs.is_half else torch.float32
        
    def synthesize(self, semantic_tokens: List[torch.Tensor], 
                  phones: List[torch.Tensor], 
                  reference_spec: List[torch.Tensor],
                  idx_list: List[int],
                  speed_factor: float = 1.0,
                  **kwargs) -> List[torch.Tensor]:
        """
        Base method for synthesizing audio from semantic tokens
        
        Args:
            semantic_tokens: List of semantic token tensors
            phones: List of phone tensors
            reference_spec: Reference spectrograms for voice characteristics
            idx_list: List of indices for semantic token slicing
            speed_factor: Speed control factor
            
        Returns:
            List of audio fragments as tensors
        """
        raise NotImplementedError("Subclasses must implement synthesize()")