import torch
import torch.nn.functional as F
from typing import List
from tqdm import tqdm
import torchaudio

from .base_synthesizer import BaseSynthesizer

class VitsV3Synthesizer(BaseSynthesizer):
    """VITS V3 model synthesizer implementation with diffusion model"""
    
    def __init__(self, model, configs, prompt_cache):
        super().__init__(model, configs)
        self.prompt_cache = prompt_cache
        
        # Initialize utility functions
        from ..module.mel_processing import mel_spectrogram_torch        
        self.mel_fn = lambda x: mel_spectrogram_torch(x, **{
            "n_fft": 1024,
            "win_size": 1024,
            "hop_size": 256,
            "num_mels": 100,
            "sampling_rate": 24000,
            "fmin": 0,
            "fmax": None,
            "center": False
        })
        self.spec_min = -12
        self.spec_max = 2
        
    def norm_spec(self, x):
        return (x - self.spec_min) / (self.spec_max - self.spec_min) * 2 - 1
        
    def denorm_spec(self, x):
        return (x + 1) / 2 * (self.spec_max - self.spec_min) + self.spec_min
        
    def resample(self, audio_tensor, sr0):
        """Resample audio to target sample rate"""
        global resample_transform_dict
        if sr0 not in resample_transform_dict:
            resample_transform_dict[sr0] = torchaudio.transforms.Resample(
                sr0, 24000
            ).to(self.device)
        return resample_transform_dict[sr0](audio_tensor)
    
    def prepare_reference(self, reference_spec):
        """Prepare reference audio features for voice cloning"""
        prompt_semantic_tokens = self.prompt_cache["prompt_semantic"].unsqueeze(0).unsqueeze(0).to(self.device)
        prompt_phones = torch.LongTensor(self.prompt_cache["phones"]).unsqueeze(0).to(self.device)
        refer_audio_spec = reference_spec[0].to(dtype=self.precision, device=self.device)

        # Extract features from prompt
        fea_ref, ge = self.model.decode_encp(prompt_semantic_tokens, prompt_phones, refer_audio_spec)
        
        # Process reference audio
        ref_audio = self.prompt_cache["raw_audio"]
        ref_sr = self.prompt_cache["raw_sr"]
        ref_audio = ref_audio.to(self.device).float()
        
        if ref_audio.shape[0] == 2:
            ref_audio = ref_audio.mean(0).unsqueeze(0)
        if ref_sr != 24000:
            ref_audio = self.resample(ref_audio, ref_sr)

        # Generate mel spectrogram and normalize
        mel2 = self.mel_fn(ref_audio)
        mel2 = self.norm_spec(mel2)
        
        # Adjust dimensions for processing
        T_min = min(mel2.shape[2], fea_ref.shape[2])
        mel2 = mel2[:, :, :T_min]
        fea_ref = fea_ref[:, :, :T_min]
        
        if T_min > 468:
            mel2 = mel2[:, :, -468:]
            fea_ref = fea_ref[:, :, -468:]
            T_min = 468
            
        chunk_len = 934 - T_min
        mel2 = mel2.to(self.precision)
        
        return fea_ref, ge, mel2, T_min, chunk_len
    
    def sola_algorithm(self, audio_fragments: List[torch.Tensor], overlap_len: int):
        """Implement SOLA (Synchronous Overlap Add) algorithm for smoother concatenation"""
        for i in range(len(audio_fragments)-1):
            f1 = audio_fragments[i]
            f2 = audio_fragments[i+1]
            w1 = f1[-overlap_len:]
            w2 = f2[:overlap_len]
            
            assert w1.shape == w2.shape
            
            # Find optimal alignment by cross-correlation
            corr = F.conv1d(w1.view(1,1,-1), w2.view(1,1,-1), padding=w2.shape[-1]//2).view(-1)[:-1]
            idx = corr.argmax()
            
            # Adjust fragments based on optimal alignment
            f1_ = f1[:-(overlap_len-idx)]
            audio_fragments[i] = f1_

            f2_ = f2[idx:]
            window = torch.hann_window((overlap_len-idx)*2, device=f1.device, dtype=f1.dtype)
            f2_[:(overlap_len-idx)] = window[:(overlap_len-idx)]*f2_[:(overlap_len-idx)] + window[(overlap_len-idx):]*f1[-(overlap_len-idx):]
            audio_fragments[i+1] = f2_

        return torch.cat(audio_fragments, 0)
        
    def synthesize_single(self, 
                     semantic_tokens: torch.Tensor, 
                     phones: torch.Tensor, 
                     speed: float = 1.0,
                     sample_steps: int = 32) -> torch.Tensor:
        """Synthesize a single audio segment using VITS V3"""
        reference_spec = self.prompt_cache["refer_spec"][0].to(dtype=self.precision, device=self.device)
        fea_ref, ge, mel2, T_min, chunk_len = self.prepare_reference([reference_spec])
            
        # Generate features for this segment
        fea_todo, ge = self.model.decode_encp(semantic_tokens, phones, reference_spec, ge, speed)

        # Process in chunks to handle long sequences
        cfm_resss = []
        idx = 0
        while True:
            fea_todo_chunk = fea_todo[:, :, idx:idx + chunk_len]
            if fea_todo_chunk.shape[-1] == 0:
                break
                
            idx += chunk_len
            fea = torch.cat([fea_ref, fea_todo_chunk], 2).transpose(2, 1)

            # Run diffusion model
            cfm_res = self.model.cfm.inference(
                fea, 
                torch.LongTensor([fea.size(1)]).to(fea.device), 
                mel2, 
                sample_steps, 
                inference_cfg_rate=0
            )
            cfm_res = cfm_res[:, :, mel2.shape[2]:]

            # Update reference for next chunk
            mel2 = cfm_res[:, :, -T_min:]
            fea_ref = fea_todo_chunk[:, :, -T_min:]

            cfm_resss.append(cfm_res)
            
        # Concatenate and denormalize
        cfm_res = torch.cat(cfm_resss, 2)
        cfm_res = self.denorm_spec(cfm_res)

        # Generate waveform
        with torch.inference_mode():
            wav_gen = self.model.bigvgan(cfm_res)
            audio = wav_gen[0][0]
    
        return audio
    
    def synthesize(self, 
                  semantic_tokens: List[torch.Tensor], 
                  phones: List[torch.Tensor], 
                  reference_spec: List[torch.Tensor],
                  idx_list: List[int],
                  speed_factor: float = 1.0,
                  parallel_synthesis: bool = True,
                  sample_steps: int = 32,
                  **kwargs) -> List[torch.Tensor]:
        """
        Synthesize audio using the VITS V3 model
        
        Args:
            semantic_tokens: List of semantic token tensors
            phones: List of phone tensors
            reference_spec: Reference spectrograms
            idx_list: List of indices
            speed_factor: Speed control factor
            parallel_synthesis: Whether to use parallel synthesis
            sample_steps: Number of diffusion sampling steps
            
        Returns:
            List of audio fragments
        """
        if not parallel_synthesis:
            # Sequential synthesis
            batch_audio_fragment = []
            for i, idx in enumerate(tqdm(idx_list)):
                phones_tensor = phones[i].unsqueeze(0).to(self.device)
                _pred_semantic = semantic_tokens[i][-idx:].unsqueeze(0).unsqueeze(0)
                
                audio_fragment = self.synthesize_single(
                    _pred_semantic, phones_tensor, speed_factor, sample_steps
                )
                batch_audio_fragment.append(audio_fragment)
                
            return batch_audio_fragment
        
        else:
            # Parallel synthesis
            reference_spec_tensor = reference_spec[0].to(dtype=self.precision, device=self.device)
            fea_ref, ge, mel2, T_min, chunk_len = self.prepare_reference(reference_spec)
            
            # Prepare features for all fragments
            overlapped_len = 12
            feat_chunks = []
            feat_lens = []
            feat_list = []

            # Process each segment
            for i, idx in enumerate(idx_list):
                phones_tensor = phones[i].unsqueeze(0).to(self.device)
                semantic_tokens_tensor = semantic_tokens[i][-idx:].unsqueeze(0).unsqueeze(0)
                feat, _ = self.model.decode_encp(semantic_tokens_tensor, phones_tensor, reference_spec_tensor, ge, speed_factor)
                feat_list.append(feat)
                feat_lens.append(feat.shape[2])

            # Concatenate features
            feats = torch.cat(feat_list, 2)
            feats_padded = F.pad(feats, (overlapped_len, 0), "constant", 0)
            
            # Process in chunks
            pos = 0
            padding_len = 0
            while True:
                if pos == 0:
                    chunk = feats_padded[:, :, pos:pos + chunk_len]
                else:
                    pos = pos - overlapped_len
                    chunk = feats_padded[:, :, pos:pos + chunk_len]
                pos += chunk_len
                if chunk.shape[-1] == 0:
                    break

                # Padding for the last chunk
                padding_len = chunk_len - chunk.shape[2]
                if padding_len != 0:
                    chunk = F.pad(chunk, (0, padding_len), "constant", 0)
                feat_chunks.append(chunk)
            
            # Batch process all chunks
            feat_chunks = torch.cat(feat_chunks, 0)
            bs = feat_chunks.shape[0]
            fea_ref_repeated = fea_ref.repeat(bs, 1, 1)
            fea = torch.cat([fea_ref_repeated, feat_chunks], 2).transpose(2, 1)
            
            # Generate spectrograms
            pred_spec = self.model.cfm.inference(
                fea, 
                torch.LongTensor([fea.size(1)]).to(fea.device), 
                mel2, 
                sample_steps, 
                inference_cfg_rate=0
            )
            pred_spec = pred_spec[:, :, -chunk_len:]
            dd = pred_spec.shape[1]
            pred_spec = pred_spec.permute(1, 0, 2).contiguous().view(dd, -1).unsqueeze(0)
            pred_spec = self.denorm_spec(pred_spec)
        
            # Generate waveform
            with torch.no_grad():
                wav_gen = self.model.bigvgan(pred_spec)
                audio = wav_gen[0][0]

            # Split audio into fragments
            audio_fragments = []
            upsample_rate = 256
            pos = 0

            while pos < audio.shape[-1]:
                audio_fragment = audio[pos:pos+chunk_len*upsample_rate]
                audio_fragments.append(audio_fragment)
                pos += chunk_len*upsample_rate

            # Apply SOLA algorithm to smooth transitions
            audio = self.sola_algorithm(audio_fragments, overlapped_len*upsample_rate)
            audio = audio[overlapped_len*upsample_rate:-padding_len*upsample_rate]

            # Split by original token lengths
            audio_fragments = []
            for feat_len in feat_lens:
                audio_fragment = audio[:feat_len*upsample_rate]
                audio_fragments.append(audio_fragment)
                audio = audio[feat_len*upsample_rate:]

            return audio_fragments