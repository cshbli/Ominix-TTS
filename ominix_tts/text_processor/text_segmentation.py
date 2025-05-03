"""
Text segmentation utilities for TTS preprocessing.
This module provides various methods to split text into smaller, processable segments.
    - "cut0": No splitting, return text as is
    - "cut1": Split text into segments of 4 sentences
    - "cut2": Split into segments of 50 words maximum
    - "cut3": Split at Chinese periods "。"
    - "cut4": Split at English periods "."
    - "cut5": Auto segmentation based on language detection
"""

import re
from typing import Callable, List, Dict, Set, Optional

# Punctuation sets
punctuation = set(['!', '?', '…', ',', '.', '-', " "])
splits: Set[str] = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…"}

# Registry for segmentation methods
METHODS: Dict[str, Callable] = {}

def register_method(name: str) -> Callable:
    """Decorator to register a segmentation method"""
    def decorator(func: Callable) -> Callable:
        METHODS[name] = func
        return func
    return decorator

def get_method(name: str) -> Callable:
    """Get a segmentation method by name"""
    method = METHODS.get(name)
    if method is None:
        raise ValueError(f"Segmentation method '{name}' not found. Available methods: {', '.join(METHODS.keys())}")
    return method

def get_method_names() -> List[str]:
    """Get a list of all available segmentation method names"""
    return list(METHODS.keys())

# Main entry point for text segmentation
def seg_text(text: str, language: str, method_name: str) -> List[str]:
    """
    Segment text using the specified method
    
    Args:
        text: Text to segment
        language: Language of the text
        method_name: Name of segmentation method to use
        
    Returns:
        List of segmented text parts
    """
    segmentation_method = get_method(method_name)
    segmented_text = segmentation_method(text)
    
    # Handle the case where segmentation returns a single string with newlines
    if isinstance(segmented_text, str):
        return segmented_text.split('\n')
    
    return segmented_text

def split_big_text(text: str, max_len: int = 510) -> List[str]:
    """
    Split a long text into smaller segments by punctuation marks
    
    Args:
        text: Long text to split
        max_len: Maximum length for each segment
        
    Returns:
        List of text segments
    """
    # Convert splits set to regex pattern
    punctuation_pattern = '([' + re.escape(''.join(splits)) + '])'
    
    # Split text at punctuation marks but keep the marks
    segments = re.split(punctuation_pattern, text)
    
    result = []
    current_segment = ''
    
    for segment in segments:
        # If adding this segment would exceed max_len, start a new segment
        if len(current_segment + segment) > max_len:
            if current_segment:
                result.append(current_segment)
            current_segment = segment
        else:
            current_segment += segment
    
    # Add the last segment
    if current_segment:
        result.append(current_segment)
    
    return result

def split_sentences(text: str) -> List[str]:
    """
    Split text into sentences based on punctuation
    
    Args:
        text: Text to split
        
    Returns:
        List of sentences
    """
    # Replace common multi-character punctuation
    text = text.replace("……", "。").replace("——", "，")
    
    # Ensure text ends with a sentence delimiter
    if text and text[-1] not in splits:
        text += "。"
    
    i_split_head = i_split_tail = 0
    len_text = len(text)
    sentences = []
    
    while i_split_head < len_text:
        if text[i_split_head] in splits:
            i_split_head += 1
            sentences.append(text[i_split_tail:i_split_head])
            i_split_tail = i_split_head
        else:
            i_split_head += 1
            
    return sentences

def is_only_punctuation(text: str) -> bool:
    """Check if a text contains only punctuation and whitespace"""
    return set(text).issubset(punctuation)

# ========================
# Segmentation Methods
# ========================

@register_method("cut0")
def cut0(text: str) -> str:
    """
    No segmentation - return text as is or filter if only punctuation
    """
    if not is_only_punctuation(text):
        return text
    return "\n"


@register_method("cut1")
def cut1(text: str) -> str:
    """
    Group every 4 sentences together
    """
    text = text.strip("\n")
    sentences = split_sentences(text)
    
    # Group sentences in batches of 4
    grouped_sentences = []
    for i in range(0, len(sentences), 4):
        group = sentences[i:i+4]
        combined = ''.join(group)
        if not is_only_punctuation(combined):
            grouped_sentences.append(combined)
            
    return "\n".join(grouped_sentences) if grouped_sentences else text

@register_method("cut2")
def cut2(text: str) -> str:
    """
    Group sentences until they reach ~50 characters
    """
    text = text.strip("\n")
    sentences = split_sentences(text)
    
    if len(sentences) < 2:
        return text
        
    segments = []
    current_segment = ""
    current_length = 0
    
    for sentence in sentences:
        sentence_len = len(sentence)
        current_length += sentence_len
        current_segment += sentence
        
        if current_length > 50:
            if not is_only_punctuation(current_segment):
                segments.append(current_segment)
            current_segment = ""
            current_length = 0
            
    # Handle any remaining text
    if current_segment and not is_only_punctuation(current_segment):
        # If the last segment is very short, merge with previous
        if segments and len(current_segment) < 50:
            segments[-1] += current_segment
        else:
            segments.append(current_segment)
            
    return "\n".join(segments)

@register_method("cut3")
def cut3(text: str) -> str:
    """
    Split text by Chinese period (。)
    """
    text = text.strip("\n")
    segments = []
    
    for segment in text.strip("。").split("。"):
        if segment and not is_only_punctuation(segment):
            segments.append(segment)
            
    return "\n".join(segments)

@register_method("cut4")
def cut4(text: str) -> str:
    """
    Split text by English period (.) but not decimal points
    """
    text = text.strip("\n")
    
    # Split on periods that aren't part of numbers
    segments = re.split(r'(?<!\d)\.(?!\d)', text.strip("."))
    
    # Filter out segments with only punctuation
    segments = [s for s in segments if s and not is_only_punctuation(s)]
    
    return "\n".join(segments)

@register_method("cut5")
def cut5(text: str) -> str:
    """
    Split text by all punctuation marks, preserving decimal points
    """
    text = text.strip("\n")
    punctuation_marks = {',', '.', ';', '?', '!', '、', '，', '。', '？', '！', ';', '：', '…'}
    
    segments = []
    current_segment = []
    
    for i, char in enumerate(text):
        if char in punctuation_marks:
            # Don't split on decimal points (e.g., 3.14)
            if (char == '.' and 
                i > 0 and i < len(text) - 1 and 
                text[i - 1].isdigit() and text[i + 1].isdigit()):
                current_segment.append(char)
            else:
                current_segment.append(char)
                segment = ''.join(current_segment)
                segments.append(segment)
                current_segment = []
        else:
            current_segment.append(char)
    
    # Add any remaining text
    if current_segment:
        segments.append(''.join(current_segment))
    
    # Filter out segments with only punctuation
    segments = [s for s in segments if not is_only_punctuation(s)]
    
    return "\n".join(segments)
