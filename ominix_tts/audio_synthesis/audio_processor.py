import torch
import numpy as np
from typing import List, Tuple

class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")


class AudioProcessor:
    """Handle audio post-processing and super-resolution operations"""
    
    def __init__(self, configs, sr_model=None):
        self.configs = configs
        self.sr_model = sr_model
        self.sr_model_not_exist = sr_model is None
        self.precision = torch.float16 if configs.is_half else torch.float32
        
    def initialize_sr_model(self):
        """Initialize the super-resolution model if needed"""
        if self.sr_model is not None:
            return
            
        try:
            from ..tools.audio_sr import AP_BWE            
            self.sr_model = AP_BWE(self.configs.device, DictToAttrRecursive)
            self.sr_model_not_exist = False
        except FileNotFoundError:
            print("Super-resolution model not found. Audio will not be super-sampled.")
            self.sr_model_not_exist = True
            
    def apply_super_sampling(self, audio: torch.Tensor, sr: int) -> Tuple[torch.Tensor, int]:
        """
        Apply super-sampling to audio to increase quality
        
        Args:
            audio: Audio tensor
            sr: Current sample rate
            
        Returns:
            Tuple of (super-sampled audio, new sample rate)
        """
        self.initialize_sr_model()
        
        if self.sr_model_not_exist:
            return audio, sr
            
        audio, new_sr = self.sr_model(audio.unsqueeze(0), sr)
        max_audio = np.abs(audio).max()
        if max_audio > 1: 
            audio /= max_audio
            
        return audio, new_sr
    
    def process_audio_batches(self,
                        audio: List[List[torch.Tensor]],
                        sr: int,
                        batch_index_list: List[List[int]] = None,
                        speed_factor: float = 1.0,
                        split_bucket: bool = True,
                        fragment_interval: float = 0.3,
                        super_sampling: bool = False) -> Tuple[int, np.ndarray]:
        """
        Process audio batches for final output
        
        Args:
            audio: List of lists of audio tensors
            sr: Sample rate
            batch_index_list: List of batch indices for reordering
            speed_factor: Speed factor
            split_bucket: Whether audio was split into buckets
            fragment_interval: Time interval to add between fragments
            super_sampling: Whether to apply super-sampling
            
        Returns:
            Tuple of (sample rate, final audio array)
        """
        # Add a silence interval between fragments
        zero_wav = torch.zeros(
                        int(self.configs.sampling_rate * fragment_interval),
                        dtype=self.precision,
                        device=self.configs.device
                    )

        # Add silence to each fragment and normalize
        for i, batch in enumerate(audio):
            for j, audio_fragment in enumerate(batch):
                max_audio = torch.abs(audio_fragment).max()
                if max_audio > 1: 
                    audio_fragment /= max_audio
                audio_fragment = torch.cat([audio_fragment, zero_wav], dim=0)
                audio[i][j] = audio_fragment

        # Reorder audio if needed
        if split_bucket:
            audio = self._recovery_order(audio, batch_index_list)
        else:
            audio = sum(audio, [])

        # Concatenate all fragments
        audio = torch.cat(audio, dim=0)

        # Apply super-sampling if requested
        if super_sampling:
            print("Applying audio super-sampling...")
            import time
            t1 = time.perf_counter()
            audio, sr = self.apply_super_sampling(audio, sr)
            t2 = time.perf_counter()
            print(f"Super-sampling time: {t2-t1:.3f}s")
        else:
            audio = audio.cpu().numpy()

        # Convert to 16-bit PCM
        audio = (audio * 32768).astype(np.int16)

        return sr, audio
        
    def _recovery_order(self, data: List[List[torch.Tensor]], batch_index_list: List[List[int]]) -> List[torch.Tensor]:
        """
        Recover the original order of fragments from batches
        
        Args:
            data: List of batches of audio fragments
            batch_index_list: List of indices for reordering
            
        Returns:
            Ordered list of audio fragments
        """
        length = len(sum(batch_index_list, []))
        ordered_data = [None] * length
        
        for i, index_list in enumerate(batch_index_list):
            for j, index in enumerate(index_list):
                ordered_data[index] = data[i][j]
                
        return ordered_data