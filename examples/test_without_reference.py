import soundfile as sf

from ominix_tts import MPipeline

# Initialize the pipeline and load the models
pipeline = MPipeline()

text = "This is a sample text for testing Ominix TTS voice synthesis."

# Start the TTS pipeline inference with the default reference audio and text
result_generator = pipeline(text=text, text_language="en")

# Process the generated audio
results = []
for item in result_generator:
    results.append(item)

# Write output
sf.write('output.wav', results[0][1], samplerate=results[0][0], subtype='PCM_16')
