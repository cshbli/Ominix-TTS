# Ominix-TTS Pipeline Introduction
## TTS Pipeline

Ominix-TTS pipeline transforms input text into synthesized speech using a two-stage pipeline: text-to-semantic conversion followed by semantic-to-speech synthesis.

- **Reference Voice Setup**:
    - Loads and processes reference audio to capture the target voice characteristics
    - Handles prompt text if provided to guide the synthesis

- **Text Preprocessing**:
   - Segments the input text using the specified method
   - Converts text to phonetic and linguistic features
   - Organizes the data into batches for efficient processing

- **Semantic Token Generation**:
   - Uses the Text2Semantic model to convert text features into semantic tokens
   - Applies sampling techniques (temperature, top-k, top-p) to control generation

- **Audio Synthesis**:
   - Feeds semantic tokens into the VITS model to generate audio waveforms
   - Handles both standard and V3 model variants with different synthesis approaches
   - Supports parallel or sequential processing based on configuration

- **Audio Post-processing**:
   - Concatenates audio fragments with appropriate intervals
   - Applies speed modification if requested
   - Optionally performs super-sampling for higher audio quality
   - Formats the audio for output

The pipeline operates as a generator, yielding audio samples either:
- All at once when `return_fragment=False`
- Fragment by fragment when `return_fragment=True`

Each yield returns a tuple containing the sample rate and audio waveform data.      

```mermaid
flowchart TB
    A[Start run] --> D[ReferenceProcessor]
    D --> E[TextPreprocessor]
    
    E --> F{Return fragment mode?}
    F -->|Yes| G[Process text as fragments]
    F -->|No| H[Process complete text]
    
    G --> I[text-to-semantic conversion]
    H --> J[text-to-semantic conversion]    
    
    D --> K[semantic-to-speech synthesis]
    I --> K
    J --> L[semantic-to-speech synthesis]
    D --> L
    
    K --> M[AudioProcessor<br><sub>Postprocess audio fragment</sub>]
    L --> N[AudioProcessor<br><sub>Postprocess all audio</sub>]
    
    M -->|Yield| Q[End run]
    N -->|Yield| Q

    classDef CoreProcess fill:#f9f,stroke:#333,stroke-width:2px;
    class I,J,K,L CoreProcess
```    

## Reference Voice Setup

Reference processing handles both audio and text reference material for voice cloning, with careful attention to caching for better performance.
- Loads and processes reference audio to capture the target voice characteristics
- Handles prompt text if provided to guide the synthesis

```mermaid
flowchart TD
    A[Start process_reference] --> B{Is ref_audio_path empty?}
    
    B -->|Yes| C{Are cached features available?}
    C -->|No| D[Raise ValueError: Reference path cannot be empty]
    C -->|Yes| E[Skip audio processing]
    
    B -->|No| H{Is path different from cached path?}
    
    H -->|Yes| I[Extract primary reference]
    I --> I1[Extract semantic tokens]
    I1 --> I2[Extract spectrogram]
    I2 --> L[Process auxiliary references]   
    
    H -->|No| K[Skip primary extraction]
    K --> L
    
    L --> Q{Is prompt_text provided?}
    
    E --> Q    
    
    Q -->|Yes| W[Process prompt text]
    W --> W1[Extract phones, bert features]
    W1 --> R[Return cache]

    

    Q -->|No| R[Return cache]    
    
    R --> Z[End process_reference]
```    
 
## Clean Reinstall package during development
```
# 1. Uninstall the package
pip uninstall -y ominix-tts

# 2. Remove build artifacts
rm -rf build dist *.egg-info
# unnecessary to remove __pycache__
# find . -type d -name __pycache__ -exec rm -rf {} +;

# 3. Clear pip's cache
pip cache purge

# 4. Reinstall
pip install .
```