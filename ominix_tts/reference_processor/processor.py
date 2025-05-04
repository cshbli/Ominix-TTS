import os
import torch
import torchaudio
import numpy as np
import librosa
from typing import Dict, List, Tuple, Optional, Any, Union

from ..tools.my_utils import load_audio
from ..module.mel_processing import spectrogram_torch
from ..feature_extractor.cnhubert import CNHubert
from ..module.models import SynthesizerTrn, SynthesizerTrnV3
from ..text_processor.processor import TextPreprocessor

class ReferenceProcessor:
    """
    Handles processing of reference audio for voice cloning.
    
    This module extracts features from reference audio that are needed for 
    voice cloning in the TTS system, including semantic tokens and spectrograms.
    """
    
    def __init__(self, 
                 text_preprocessor: TextPreprocessor,
                 cnhubert_model: CNHubert, 
                 vits_model: Union[SynthesizerTrn, SynthesizerTrnV3], 
                 device: torch.device, 
                 config: Any):
        """
        Initialize the reference processor
        
        Args:
            cnhubert_model: CNHuBERT model for feature extraction
            vits_model: The VITS Synthesizer model
            device: Device to run processing on
            config: Configuration object with audio processing parameters
        """
        self.text_preprocessor = text_preprocessor
        self.cnhubert_model = cnhubert_model
        self.vits_model = vits_model
        self.device = device
        self.config = config
        self.is_half = config.is_half
        
        # Audio processing parameters
        self.filter_length = config.filter_length
        self.sampling_rate = config.sampling_rate
        self.hop_length = config.hop_length
        self.win_length = config.win_length
        
        # Cache for processed reference audio
        self._cache = {
            "ref_audio_path": None,
            "prompt_semantic": None,
            "refer_spec": [],
            "aux_ref_audio_paths": [],
            "raw_audio": None,
            "raw_sr": None,
            "prompt_text": None,
            "prompt_lang": None,
            "phones": None,
            "bert_features": None,
            "norm_text": None,
        }
        
    def process_reference_audio(self, 
                         ref_audio_path: str,
                         aux_ref_audio_paths: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Process reference audio for voice cloning
        
        Args:
            ref_audio_path: Path to main reference audio file
            aux_ref_audio_paths: Optional paths to auxiliary reference audio files
                                for multi-speaker tone fusion
            
        Returns:
            Dictionary with processed reference features
        """
        if ref_audio_path in [None, ""]:
            if (self._cache["prompt_semantic"] is None or self._cache["refer_spec"] in [None, []]):
                raise ValueError("Reference audio path cannot be empty when reference features haven't been previously set")
            # Skip processing if ref_audio_path is empty but we have cached reference data
            return self._cache

        if not ref_audio_path or not os.path.exists(ref_audio_path):
            raise ValueError(f"Reference audio path does not exist: {ref_audio_path}")
        
        # Set up auxiliary paths
        aux_ref_audio_paths = aux_ref_audio_paths or []
        
        # Check if we need to reprocess the primary reference audio
        if ref_audio_path != self._cache["ref_audio_path"]:
            self._extract_primary_reference(ref_audio_path)
            self._cache["ref_audio_path"] = ref_audio_path
            
        # Process auxiliary references if needed
        self._process_auxiliary_references(aux_ref_audio_paths)
        
        return self._cache
    
    def _extract_primary_reference(self, ref_audio_path: str) -> None:
        """
        Extract features from the primary reference audio
        
        Args:
            ref_audio_path: Path to reference audio file
        """
        print(f"Extracting semantic features from reference audio: {ref_audio_path}")
        
        # Extract semantic tokens
        self._cache["prompt_semantic"] = self._extract_semantic(ref_audio_path)
        
        # Extract spectrogram
        spec, raw_audio, raw_sr = self._extract_spectrogram(ref_audio_path)
        
        # Store in cache
        self._cache["refer_spec"] = [spec]
        self._cache["raw_audio"] = raw_audio
        self._cache["raw_sr"] = raw_sr
        
        return None
    
    def _process_auxiliary_references(self, aux_ref_audio_paths: List[str]) -> None:
        """
        Process auxiliary reference audio files
        
        Args:
            aux_ref_audio_paths: List of paths to auxiliary reference audio files
        """
        # Check if aux paths have changed
        paths_set = set(aux_ref_audio_paths)
        cache_paths_set = set(self._cache["aux_ref_audio_paths"])
        
        if paths_set == cache_paths_set and len(aux_ref_audio_paths) == len(self._cache["aux_ref_audio_paths"]):
            # No changes needed
            return
            
        # Reset aux references and keep only the primary reference
        self._cache["aux_ref_audio_paths"] = aux_ref_audio_paths
        self._cache["refer_spec"] = [self._cache["refer_spec"][0]] if self._cache["refer_spec"] else []
        
        # Process each auxiliary reference
        for path in aux_ref_audio_paths:
            if not path or not os.path.exists(path):
                print(f"Auxiliary audio file not found, skipping: {path}")
                continue
                
            # Extract spectrogram
            spec, _, _, = self._extract_spectrogram(path)
            self._cache["refer_spec"].append(spec)

    """This function is responsible for extracting semantic features from a reference audio file for voice cloning.

    1. Prepares the reference audio:
        1.1 Takes a reference audio file path as input
        1.2 Loads the audio and verifies it's between 3-10 seconds long
        1.3 Adds a short silence (0.3 seconds) to the end
    2. Extracts semantic tokens:
        2.1 Processes the audio through the CNHuBERT model to extract hidden representations
        2.2 Passes these representations through the VITS model's extract_latent() method to get semantic tokens
        2.3 These tokens capture the voice characteristics, timbre, and speaking style

    These tokens will later be used as a reference for voice cloning during speech synthesis.
    The function is critical for voice cloning, as it creates a compact representation of the speaker's voice characteristics that the model can use when generating new speech. This enables the synthesized speech to mimic the voice quality, accent, and speaking style of the reference audio.
    """
    def _extract_semantic(self, ref_wav_path: str) -> torch.Tensor:
        """
        Extract reference semantic features from a reference audio file.
        
        Args:
            ref_wav_path: Path to the reference audio file
            
        Returns:
            reference_semantic: The extracted reference semantic tensor
        
        Raises:
            OSError: If the reference audio is outside the 3-10 second range
        """
        zero_wav = np.zeros(
            int(self.sampling_rate * 0.3),
            dtype=np.float16 if self.is_half else np.float32,
        )
        
        with torch.no_grad():
            wav16k, sr = librosa.load(ref_wav_path, sr=16000)
            if wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000:
                raise OSError("Reference audio must be between 3-10 seconds")
            
            wav16k = torch.from_numpy(wav16k)
            zero_wav_torch = torch.from_numpy(zero_wav)
            
            wav16k = wav16k.to(self.device)
            zero_wav_torch = zero_wav_torch.to(self.device)
            
            if self.is_half:
                wav16k = wav16k.half()
                zero_wav_torch = zero_wav_torch.half()
    
            # Concatenate audio with silence
            wav16k = torch.cat([wav16k, zero_wav_torch])
            
            # Extract HuBERT features
            hubert_feature = self.cnhubert_model.model(wav16k.unsqueeze(0))[
                "last_hidden_state"
            ].transpose(1, 2)
            
            # Extract latent codes
            codes = self.vits_model.extract_latent(hubert_feature)
    
            # Get reference semantic
            reference_semantic = codes[0, 0].to(self.device)
            
        return reference_semantic
            
    
    def _extract_spectrogram(self, ref_audio_path: str) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Extract spectrogram from reference audio for voice cloning.
        
        Args:
            ref_audio_path: Path to the reference audio file
            
        Returns:
            Tuple containing:
            - spec: The extracted spectrogram tensor
            - raw_audio: The raw audio tensor
            - raw_sr: The original sample rate
        """
        # Load raw audio data for caching
        raw_audio, raw_sr = torchaudio.load(ref_audio_path)
        raw_audio = raw_audio.to(self.device).float()
        
        # Load and normalize audio for spectrogram generation
        audio = load_audio(ref_audio_path, int(self.sampling_rate))
        audio = torch.FloatTensor(audio)
        
        # Normalize the audio
        maxx = audio.abs().max()
        if maxx > 1:
            audio /= min(2, maxx)
            
        # Prepare for spectrogram generation
        audio_norm = audio.unsqueeze(0)    
        
        # It converts the audio waveform to a spectrogram representation using the Short-Time Fourier Transform
        spec = spectrogram_torch(
            audio_norm,
            self.filter_length,
            self.sampling_rate,
            self.hop_length,
            self.win_length,
            center=False,
        )
        
        # Transfer to device and adjust precision
        spec = spec.to(self.device)
        if self.is_half:
            spec = spec.half()
            
        return spec, raw_audio, raw_sr
        
    def get_cached_reference(self) -> Dict[str, Any]:
        """
        Get the currently cached reference audio features
        
        Returns:
            Dictionary with cached reference features
        """
        return self._cache
        
    def process_reference_text(self, 
                          prompt_text: str, 
                          prompt_lang: str, 
                          text_preprocessor: TextPreprocessor, 
                          model_version: str) -> Dict[str, Any]:
        """
        Process prompt text and combine it with reference audio features
        
        Args:
            prompt_text: Prompt text for voice cloning
            prompt_lang: Language of prompt text
            text_processor: Text processor instance
            model_version: Model version string
            
        Returns:
            Dictionary with processed prompt data
        """
        if not prompt_text:
            return self._cache
            
        from ..text_processor.text_segmentation import splits
        
        # Add sentence terminator if needed
        prompt_text = prompt_text.strip("\n")
        if prompt_text and prompt_text[-1] not in splits:
            prompt_text += "。" if prompt_lang != "en" else "."
            
        # Only reprocess if prompt has changed
        if prompt_text != self._cache.get("prompt_text"):
            print(f"Processing prompt text: {prompt_text}")
            phones, bert_features, norm_text = text_preprocessor.segment_and_extract_feature_for_text(
                prompt_text,
                prompt_lang,
                model_version
            )
            
            # Update cache with new prompt data
            self._cache["prompt_text"] = prompt_text
            self._cache["prompt_lang"] = prompt_lang
            self._cache["phones"] = phones
            self._cache["bert_features"] = bert_features
            self._cache["norm_text"] = norm_text
            
        return self._cache
    
    def process_reference(self,
                     ref_audio_path: str,
                     prompt_text: Optional[str] = None,
                     prompt_lang: Optional[str] = None,
                     model_version: Optional[str] = None,
                     aux_ref_audio_paths: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Process reference audio and text for voice cloning
        
        This unified function handles both the reference audio processing
        and the prompt text processing in a single call, which improves
        efficiency and simplifies the pipeline code.
        
        Args:
            ref_audio_path: Path to main reference audio file
            prompt_text: Optional prompt text for voice cloning guidance
            prompt_lang: Language of the prompt text
            model_version: Model version (required if prompt_text is provided)
            aux_ref_audio_paths: Optional paths to auxiliary reference audio files
                            for multi-speaker tone fusion
            
        Returns:
            Dictionary with processed reference features and prompt data
            
        Raises:
            ValueError: If reference path is empty and no cached data exists
                    or if prompt_text is provided without text_processor/model_version
        """
        # Process reference audio if provided or check if we have cached features
        if ref_audio_path in [None, ""]:
            if (self._cache["prompt_semantic"] is None or self._cache["refer_spec"] in [None, []]):
                raise ValueError("Reference audio path cannot be empty when reference features haven't been previously set")
            # Skip processing if ref_audio_path is empty but we have cached reference data
        else:
            if not ref_audio_path or not os.path.exists(ref_audio_path):
                raise ValueError(f"Reference audio path does not exist: {ref_audio_path}")            

            # Check if we need to reprocess the primary reference audio
            if ref_audio_path != self._cache["ref_audio_path"]:
                self._extract_primary_reference(ref_audio_path)
                self._cache["ref_audio_path"] = ref_audio_path
            
            # Process auxiliary references if needed
            aux_ref_audio_paths = aux_ref_audio_paths or []
            self._process_auxiliary_references(aux_ref_audio_paths)
        
        # Process prompt text if provided
        if prompt_text and prompt_text.strip():
            if not model_version:
                raise ValueError("Model version must be provided when processing prompt text")
                
            from ..text_processor.text_segmentation import splits
            
            # Add sentence terminator if needed
            prompt_text = prompt_text.strip("\n")
            if prompt_text and prompt_text[-1] not in splits:
                prompt_text += "。" if prompt_lang != "en" else "."
                
            # Only reprocess if prompt has changed
            if prompt_text != self._cache.get("prompt_text") or prompt_lang != self._cache.get("prompt_lang"):
                print(f"Processing prompt text: {prompt_text}")
                phones, bert_features, norm_text = self.text_preprocessor.segment_and_extract_feature_for_text(
                    prompt_text,
                    prompt_lang,
                    model_version
                )
                
                # Update cache with new prompt data
                self._cache["prompt_text"] = prompt_text
                self._cache["prompt_lang"] = prompt_lang
                self._cache["phones"] = phones
                self._cache["bert_features"] = bert_features
                self._cache["norm_text"] = norm_text
        
        return self._cache