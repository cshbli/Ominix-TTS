import soundfile as sf

from ominix_tts import MPipeline

# Initialize the pipeline and load the models
pipeline = MPipeline()

text = "Well, you know what the say , Families are like fudge, mostly sweet, but sometimes nuts. My family is doing great, thanks for asking!  My son is growing up to be a smart and handsome young man, just like his mom. He's currently working on his own talker show, which I'm sure will be even more hilarious than mine." 
ref_audio_path = "./ominix_tts/dataset/doubao-ref-ours.wav"   
ref_text = "我叫豆包呀，能陪你聊天解闷，不管是聊生活趣事，知识科普还是帮你出主意，我都在行哦。" 

# Get results from pipeline
result_generator = pipeline(
    text=text, 
    text_language="en", 
    ref_audio_path=ref_audio_path, 
    ref_text=ref_text, 
    ref_language="all_zh"
)

# Process the generated audio
results = []
for item in result_generator:
    results.append(item)

# Write output
sf.write('output.wav', results[0][1], samplerate=results[0][0], subtype='PCM_16')
