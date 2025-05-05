# Ominix-TTS: Advanced Multilingual Text-to-Speech with Voice Cloning

Ominix-TTS is a cutting-edge text-to-speech synthesis framework that transforms input text into natural-sounding speech using a sophisticated two-stage pipeline. The system excels in producing high-quality audio across multiple languages with voice cloning capabilities.

## Key Features

- **Two-Stage Synthesis Pipeline**: First converts text to semantic tokens, then transforms these tokens into audio waveforms
- **Multilingual Support**: Handles Chinese, English, Japanese, Korean, and Cantonese with both pure and mixed-language modes
- **Voice Cloning**: Replicates voice characteristics from a short reference audio sample
- **Voice Fusion**: Combines multiple reference voices for custom voice creation
- **High-Quality Output**: Produces natural-sounding speech with proper prosody and intonation
- **Configurable Parameters**: Offers control over speed, temperature, and other synthesis qualities

## Language Codes in Ominix-TTS

Here's a comprehensive table of all language codes supported by the Ominix-TTS system:

| Language Code | Description | Recognition Type |
|---------------|-------------|------------------|
| `"en"`        | Pure English | English only processing |
| `"zh"`        | Mixed Chinese-English | Chinese-English hybrid processing |
| `"all_zh"`    | Pure Chinese | Chinese only processing |
| `"yue"`       | Mixed Cantonese-English | Cantonese-English hybrid processing |
| `"all_yue"`   | Pure Cantonese | Cantonese only processing |
| `"ja"`        | Mixed Japanese-English | Japanese-English hybrid processing |
| `"all_ja"`    | Pure Japanese | Japanese only processing |
| `"ko"`        | Mixed Korean-English | Korean-English hybrid processing |
| `"all_ko"`    | Pure Korean | Korean only processing |
| `"auto"`      | Auto-detect language | Multi-language detection and processing |
| `"auto_yue"`  | Auto-detect with Cantonese support | Multi-language detection including Cantonese |

## Technical Architecture

Ominix-TTS operates through coordinated specialized models:
- **BERT Models**: Extract linguistic features from input text
- **CNHuBERT**: Processes reference audio to capture voice characteristics
- **Text2Semantic Model**: Converts text features into semantic tokens
- **SoVITS Model**: Transforms semantic tokens into audio waveforms

The system supports different model versions (v1, v2, v3) with increasing capabilities and language support, allowing users to balance between quality, speed, and resource requirements.

## Applications

Ideal for creating audiobooks, virtual assistants, accessibility tools, content localization, and any application requiring high-quality speech synthesis with the ability to match specific voice characteristics.

## Usage

1. Please install `ffmpeg`. ffmpeg is used to decode the reference audio file. 

    - For MacOS:
    ```
    brew install ffmpeg 
    ```

2. Recommend to create one virtual environment to run tests and examples

```
conda create -n TTS python=3.9
conda activate TTS
```
