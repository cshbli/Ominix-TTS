import torch
import math
from typing import List
from tqdm import tqdm

from .base_synthesizer import BaseSynthesizer

class VitsSynthesizer(BaseSynthesizer):
    """VITS model synthesizer implementation"""
    
    def synthesize(self, 
                  semantic_tokens: List[torch.Tensor], 
                  phones: List[torch.Tensor], 
                  reference_spec: List[torch.Tensor],
                  idx_list: List[int],
                  speed_factor: float = 1.0,
                  parallel_synthesis: bool = True,
                  **kwargs) -> List[torch.Tensor]:
        """
        Synthesize audio using the VITS model
        
        Args:
            semantic_tokens: List of semantic token tensors
            phones: List of phone tensors
            reference_spec: Reference spectrograms
            idx_list: List of indices
            speed_factor: Speed control factor
            parallel_synthesis: Whether to use parallel synthesis
            
        Returns:
            List of audio fragments
        """
        batch_audio_fragment = []
        
        if parallel_synthesis and speed_factor == 1.0:
            # Parallel synthesis mode
            pred_semantic_list = [item[-idx:] for item, idx in zip(semantic_tokens, idx_list)]
            upsample_rate = math.prod(self.model.upsample_rates)
            
            # Calculate fragment indices for splitting
            audio_frag_idx = [pred_semantic_list[i].shape[0]*2*upsample_rate for i in range(len(pred_semantic_list))]
            audio_frag_end_idx = [sum(audio_frag_idx[:i+1]) for i in range(len(audio_frag_idx))]
            
            # Concatenate all semantic tokens for batch processing
            all_pred_semantic = torch.cat(pred_semantic_list).unsqueeze(0).unsqueeze(0).to(self.device)
            _batch_phones = torch.cat(phones).unsqueeze(0).to(self.device)
            
            # Generate audio in one pass
            _batch_audio_fragment = (self.model.decode(
                    all_pred_semantic, _batch_phones, reference_spec, speed=speed_factor
                ).detach()[0, 0, :])
            
            # Split the output into fragments
            audio_frag_end_idx.insert(0, 0)
            batch_audio_fragment = [_batch_audio_fragment[audio_frag_end_idx[i-1]:audio_frag_end_idx[i]] 
                                  for i in range(1, len(audio_frag_end_idx))]
        else:
            # Sequential synthesis mode
            for i, idx in enumerate(tqdm(idx_list)):
                phones_tensor = phones[i].unsqueeze(0).to(self.device)
                _pred_semantic = (semantic_tokens[i][-idx:].unsqueeze(0).unsqueeze(0))
                
                audio_fragment = (self.model.decode(
                        _pred_semantic, phones_tensor, reference_spec, speed=speed_factor
                    ).detach()[0, 0, :])
                    
                batch_audio_fragment.append(audio_fragment)
                
        return batch_audio_fragment