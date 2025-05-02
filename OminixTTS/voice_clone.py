import librosa
import numpy as np
import torch
import torchaudio
from typing import Union

from feature_extractor.cnhubert import CNHubert
from module.models import SynthesizerTrn, SynthesizerTrnV3
from module.mel_processing import spectrogram_torch
from tools.my_utils import load_audio

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
def extract_reference_semantic(ref_wav_path: str, 
                           device: torch.device,
                           sampling_rate: int,
                           cnhubert_model: CNHubert,
                           vits_model: Union[SynthesizerTrn, SynthesizerTrnV3],
                           is_half: bool = False) -> torch.Tensor:
    """
    Extract reference semantic features from a reference audio file.
    
    Args:
        ref_wav_path: Path to the reference audio file
        device: The device to run the model on (CPU or GPU)
        sampling_rate: The sampling rate of the audio
        cnhubert_model: The CNHubert model
        vits_model: The VITS model
        is_half: Whether to use half precision
        
    Returns:
        reference_semantic: The extracted reference semantic tensor
    
    Raises:
        OSError: If the reference audio is outside the 3-10 second range
    """
    zero_wav = np.zeros(
        int(sampling_rate * 0.3),
        dtype=np.float16 if is_half else np.float32,
    )
    
    with torch.no_grad():
        wav16k, sr = librosa.load(ref_wav_path, sr=16000)
        if (wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000):
            raise OSError("Reference audio must be between 3-10 seconds")
        
        wav16k = torch.from_numpy(wav16k)
        zero_wav_torch = torch.from_numpy(zero_wav)
        
        wav16k = wav16k.to(device)
        zero_wav_torch = zero_wav_torch.to(device)
        
        if is_half:
            wav16k = wav16k.half()
            zero_wav_torch = zero_wav_torch.half()

        # Concatenate audio with silence
        wav16k = torch.cat([wav16k, zero_wav_torch])
        
        # Extract HuBERT features
        hubert_feature = cnhubert_model.model(wav16k.unsqueeze(0))[
            "last_hidden_state"
        ].transpose(1, 2)
        
        # Extract latent codes
        codes = vits_model.extract_latent(hubert_feature)

        # Get reference semantic
        reference_semantic = codes[0, 0].to(device)
        
    return reference_semantic


def extract_reference_spectrogram(ref_audio_path: str, 
                                 device: torch.device,
                                 filter_length: int,
                                 sampling_rate: int,
                                 hop_length: int, 
                                 win_length: int,
                                 is_half: bool = False) -> torch.Tensor:
    """
    Extract spectrogram from reference audio for voice cloning.
    
    Args:
        ref_audio_path: Path to the reference audio file
        device: The device to run processing on (CPU or GPU)
        filter_length: Filter length for spectrogram generation
        sampling_rate: The sampling rate of the audio
        hop_length: Hop length for spectrogram generation
        win_length: Window length for spectrogram generation
        is_half: Whether to use half precision
        
    Returns:
        spec: The extracted spectrogram tensor
    """
    # Load raw audio data for caching
    raw_audio, raw_sr = torchaudio.load(ref_audio_path)
    raw_audio = raw_audio.to(device).float()
    
    # Load and normalize audio for spectrogram generation
    audio = load_audio(ref_audio_path, int(sampling_rate))
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
        filter_length,
        sampling_rate,
        hop_length,
        win_length,
        center=False,
    )
    
    # Transfer to device and adjust precision
    spec = spec.to(device)
    if is_half:
        spec = spec.half()
        
    return spec, raw_audio, raw_sr
