import os
import sys
import threading
import re
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

from .LangSegmenter import LangSegmenter
from . import chinese
from .cleaner import clean_text
from . import cleaned_text_to_sequence
from .text_segmentation import split_big_text, splits, get_method as get_seg_method
from ..tools.i18n.i18n import I18nAuto, scan_language_list

# Initialize internationalization
language = os.environ.get("language", "Auto")
language = sys.argv[-1] if sys.argv[-1] in scan_language_list() else language
i18n = I18nAuto(language=language)

# Constants
PUNCTUATION: Set[str] = {'!', '?', '…', ',', '.', '-'}
MIN_PHONES_LENGTH: int = 6
MAX_TEXT_LENGTH: int = 510
MIN_MERGE_THRESHOLD: int = 5

@dataclass
class ProcessedText:
    """Data container for processed text results"""
    phones: List[int]
    bert_features: torch.Tensor
    norm_text: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "phones": self.phones,
            "bert_features": self.bert_features,
            "norm_text": self.norm_text,
        }

class TextProcessor:
    """
    Text processing for TTS systems, handling segmentation,
    phoneme conversion, and BERT feature extraction.
    """

    def __init__(self, bert_model: AutoModelForMaskedLM,
                 tokenizer: AutoTokenizer, device: torch.device):
        """
        Initialize TextProcessor with BERT model and tokenizer
        
        Args:
            bert_model: BERT model for feature extraction
            tokenizer: Tokenizer for the BERT model
            device: Device to run model inference on
        """
        self.bert_model = bert_model
        self.tokenizer = tokenizer
        self.device = device
        self.bert_lock = threading.RLock()

    def process(self, text: str, lang: str, text_split_method: str, version: str = "v2") -> List[Dict]:
        """
        Process text for TTS: segment, extract features, and convert to phonemes
        
        Args:
            text: Input text to process
            lang: Language code
            text_split_method: Method to use for text splitting
            version: Model version
            
        Returns:
            List of dictionaries with phones, bert_features, and norm_text
        """
        print(f'############ {i18n("切分文本")} ############')
        text = self._replace_consecutive_punctuation(text)
        texts = self.pre_seg_text(text, lang, text_split_method)
        result = []
        
        print(f'############ {i18n("提取文本Bert特征")} ############')
        for text in tqdm(texts):
            processed = self._process_single_text(text, lang, version)
            if processed:
                result.append(processed.to_dict())
                
        return result
    
    def _process_single_text(self, text: str, lang: str, version: str) -> Optional[ProcessedText]:
        """Process a single text segment"""
        phones, bert_features, norm_text = self.segment_and_extract_feature_for_text(text, lang, version)
        if phones is None or norm_text == "":
            return None
            
        return ProcessedText(
            phones=phones,
            bert_features=bert_features,
            norm_text=norm_text
        )

    def pre_seg_text(self, text: str, lang: str, text_split_method: str) -> List[str]:
        """
        Segment text into processable chunks
        
        Args:
            text: Input text
            lang: Language code
            text_split_method: Method for splitting text
            
        Returns:
            List of text segments
        """
        # Clean up text
        text = text.strip("\n")
        if not text:
            return []
            
        # Add initial sentence mark if needed
        if text[0] not in splits and len(self._get_first_segment(text)) < 4:
            text = "。" + text if lang != "en" else "." + text
            
        print(i18n("实际输入的目标文本:"))
        print(text)

        # Apply segmentation method
        seg_method = get_seg_method(text_split_method)
        text = seg_method(text)

        # Clean up newlines
        while "\n\n" in text:
            text = text.replace("\n\n", "\n")

        # Split into segments and filter
        _texts = text.split("\n")
        _texts = self._filter_text(_texts)
        _texts = self._merge_short_texts(_texts, MIN_MERGE_THRESHOLD)
        
        # Process each segment
        texts = []
        for segment in _texts:
            # Skip empty or symbol-only segments
            if not segment.strip() or not re.sub(r"\W+", "", segment):
                continue
                
            # Add sentence terminator if needed
            if segment[-1] not in splits:
                segment += "。" if lang != "en" else "."

            # Split long text to avoid BERT limitations
            if len(segment) > MAX_TEXT_LENGTH:
                texts.extend(split_big_text(segment))
            else:
                texts.append(segment)

        print(i18n("实际输入的目标文本(切句后):"))
        print(texts)
        return texts

    def segment_and_extract_feature_for_text(self, text: str, language: str, version: str = "v1") -> Tuple[List[int], torch.Tensor, str]:
        """
        Extract phoneme representation and BERT features from text
        
        This is the main interface for feature extraction from a text segment,
        providing phonemes, BERT embeddings, and normalized text.
        
        Args:
            text: Input text segment
            language: Language code
            version: Model version
            
        Returns:
            Tuple of (phoneme IDs, BERT features, normalized text)
        """
        return self.get_phones_and_bert(text, language, version)

    def get_phones_and_bert(self, text: str, language: str, version: str, final: bool = False) -> Tuple[List[int], torch.Tensor, str]:
        """
        Get phonemes and BERT features for text
        
        Args:
            text: Input text
            language: Language code
            version: Model version
            final: Whether this is a final attempt (for short text handling)
            
        Returns:
            Tuple of (phones, bert features, normalized text)
        """
        with self.bert_lock:
            # Process based on language type
            if language in {"en", "all_zh", "all_ja", "all_ko", "all_yue"}:
                # Handle single language processing
                phones, bert, norm_text = self._process_single_language(text, language, version)
            elif language in {"zh", "ja", "ko", "yue", "auto", "auto_yue"}:
                # Handle mixed language processing
                phones, bert, norm_text = self._process_mixed_language(text, language, version)
            else:
                # Default empty result for unsupported languages
                phones = []
                bert = torch.zeros((1024, 0), dtype=torch.float32).to(self.device)
                norm_text = ""

            # Handle short text by adding period prefix
            if not final and len(phones) < MIN_PHONES_LENGTH:
                return self.get_phones_and_bert("." + text, language, version, final=True)

            return phones, bert, norm_text

    def _process_single_language(self, text: str, language: str, version: str) -> Tuple[List[int], torch.Tensor, str]:
        """
        Process text in a single language
        
        Args:
            text: Input text
            language: Language code
            version: Model version
            
        Returns:
            Tuple of (phones, bert features, normalized text)
        """
        # Normalize text by removing extra spaces
        formattext = self._normalize_text(text, language)
            
        # Special handling for Chinese with English mix
        if language == "all_zh":
            if re.search(r'[A-Za-z]', formattext):
                formattext = re.sub(r'[a-z]', lambda x: x.group(0).upper(), formattext)
                formattext = chinese.mix_text_normalize(formattext)
                return self.get_phones_and_bert(formattext, "zh", version)
            else:
                phones, word2ph, norm_text = self.clean_text_inf(formattext, language, version)
                bert = self.get_bert_feature(norm_text, word2ph).to(self.device)
                return phones, bert, norm_text
                
        # Special handling for Cantonese with English mix
        elif language == "all_yue" and re.search(r'[A-Za-z]', formattext):
            formattext = re.sub(r'[a-z]', lambda x: x.group(0).upper(), formattext)
            formattext = chinese.mix_text_normalize(formattext)
            return self.get_phones_and_bert(formattext, "yue", version)
            
        # Default handling for other languages
        else:
            phones, word2ph, norm_text = self.clean_text_inf(formattext, language, version)
            bert = torch.zeros(
                (1024, len(phones)),
                dtype=torch.float32,
            ).to(self.device)
            return phones, bert, norm_text

    def _process_mixed_language(self, text: str, language: str, version: str) -> Tuple[List[int], torch.Tensor, str]:
        """
        Process text with mixed languages
        
        Args:
            text: Input text
            language: Language code
            version: Model version
            
        Returns:
            Tuple of (phones, bert features, normalized text)
        """
        textlist = []
        langlist = []
        
        # Language detection and segmentation
        if language == "auto":
            for tmp in LangSegmenter.getTexts(text):
                langlist.append(tmp["lang"])
                textlist.append(tmp["text"])
        elif language == "auto_yue":
            for tmp in LangSegmenter.getTexts(text):
                if tmp["lang"] == "zh":
                    tmp["lang"] = "yue"
                langlist.append(tmp["lang"])
                textlist.append(tmp["text"])
        else:
            for tmp in LangSegmenter.getTexts(text):
                if tmp["lang"] == "en":
                    langlist.append(tmp["lang"])
                else:
                    # Use user input language for non-English text
                    langlist.append(language)
                textlist.append(tmp["text"])
                
        # Process text segments
        phones_list = []
        bert_list = []
        norm_text_list = []
        
        for i in range(len(textlist)):
            lang = langlist[i]
            phones, word2ph, norm_text = self.clean_text_inf(textlist[i], lang, version)
            bert = self._get_bert_for_lang(phones, word2ph, norm_text, lang)
            phones_list.append(phones)
            norm_text_list.append(norm_text)
            bert_list.append(bert)
            
        # Combine results
        if bert_list:
            bert = torch.cat(bert_list, dim=1)
        else:
            bert = torch.zeros((1024, 0), dtype=torch.float32).to(self.device)
            
        phones = sum(phones_list, [])
        norm_text = ''.join(norm_text_list)
        
        return phones, bert, norm_text

    def get_bert_feature(self, text: str, word2ph: list) -> torch.Tensor:
        """Extract BERT features from text"""
        with torch.no_grad():
            # Tokenize and get hidden states
            inputs = self.tokenizer(text, return_tensors="pt")
            for i in inputs:
                inputs[i] = inputs[i].to(self.device)
                
            res = self.bert_model(**inputs, output_hidden_states=True)
            # Use the third-to-last hidden layer, skip [CLS] and [SEP] tokens
            res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
            
        # Verify alignment
        assert len(word2ph) == len(text), f"Mismatch in word2ph length: {len(word2ph)} vs text length: {len(text)}"
        
        # Convert word-level features to phone-level by repeating
        phone_level_feature = []
        for i in range(len(word2ph)):
            repeat_feature = res[i].repeat(word2ph[i], 1)
            phone_level_feature.append(repeat_feature)
            
        phone_level_feature = torch.cat(phone_level_feature, dim=0)
        return phone_level_feature.T

    def clean_text_inf(self, text: str, language: str, version: str = "v2") -> Tuple[List[str], List[int], str]:
        """Clean and normalize text, converting to phonemes"""
        language = language.replace("all_", "")
        phones, word2ph, norm_text = clean_text(text, language, version)
        phones = cleaned_text_to_sequence(phones, version)
        return phones, word2ph, norm_text

    def _get_bert_for_lang(self, phones: list, word2ph: list, norm_text: str, language: str) -> torch.Tensor:
        """Get BERT features based on language"""
        language = language.replace("all_", "")
        
        if language == "zh":
            return self.get_bert_feature(norm_text, word2ph).to(self.device)
        else:
            return torch.zeros((1024, len(phones)), dtype=torch.float32).to(self.device)

    def _filter_text(self, texts: List[str]) -> List[str]:
        """Filter out empty or whitespace-only texts"""
        if all(text in [None, " ", "\n", ""] for text in texts):
            raise ValueError(i18n("请输入有效文本"))
            
        return [text for text in texts if text not in [None, " ", ""]]

    def _replace_consecutive_punctuation(self, text: str) -> str:
        """Replace consecutive punctuation with a single punctuation mark"""
        punctuations = ''.join(re.escape(p) for p in PUNCTUATION)
        pattern = f'([{punctuations}])([{punctuations}])+'
        result = re.sub(pattern, r'\1', text)
        return result

    def _normalize_text(self, text: str, language: str) -> str:
        """Normalize text by removing extra spaces"""
        formattext = text
        while "  " in formattext:
            formattext = formattext.replace("  ", " ")
        return formattext
    
    def _get_first_segment(self, text: str) -> str:
        """Get first segment of text before any split character"""
        pattern = "[" + "".join(re.escape(sep) for sep in splits) + "]"
        text = re.split(pattern, text)[0].strip()
        return text
        
    def _merge_short_texts(self, texts: List[str], threshold: int) -> List[str]:
        """Merge short text segments"""
        if len(texts) < 2:
            return texts
            
        result = []
        text = ""
        
        for ele in texts:
            text += ele
            if len(text) >= threshold:
                result.append(text)
                text = ""
                
        if text:
            if not result:
                result.append(text)
            else:
                result[-1] += text
                
        return result