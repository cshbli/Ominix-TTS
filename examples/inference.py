import random
import os, sys
import soundfile as sf
from huggingface_hub import hf_hub_download

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../OminixTTS")))

from TTS import TTS, TTS_Config
from tools.i18n.i18n import I18nAuto
from model_download import download_folder_from_repo

#is_half = True
is_half = False
# gpt_path = './T2S_weights/txdb-e15.ckpt'
gpt_path = hf_hub_download(
    repo_id="cshbli/MoTTS",
    filename="models/T2S/txdb-e15.ckpt",
    revision="main"
)
print("gpt_path: " + gpt_path)
#sovits_path = './VITS_weights/txdb_e12_s204.pth'
sovits_path = hf_hub_download(
    repo_id="cshbli/MoTTS",
    filename="models/VITS/txdb_e12_s204.pth",
    revision="main"
)
print("sovits_path: " + sovits_path)
#bert_path = './MOTTS/pretrained_models/chinese-roberta-wwm-ext-large'
bert_path = download_folder_from_repo(
    repo_id="cshbli/MoTTS",
    folder_path="models/BERT/chinese-roberta-wwm-ext-large")
print("bert_path: " + bert_path)
# cnhubert_base_path = './MOTTS/pretrained_models/chinese-hubert-base'
cnhubert_base_path = download_folder_from_repo(
    repo_id="cshbli/MoTTS",
    folder_path="models/HuBERT/chinese-hubert-base")
print("cnhuert_base_path: " + cnhubert_base_path)

i18n = I18nAuto()
#device = "cuda"
device = "cpu"

# tts_config = TTS_Config("./MOTTS/configs/tts_infer.yaml")
tts_config = TTS_Config()
tts_config.device = device
tts_config.is_half = is_half
if gpt_path is not None:
    tts_config.t2s_weights_path = gpt_path
if sovits_path is not None:
    tts_config.vits_weights_path = sovits_path
if cnhubert_base_path is not None:
    tts_config.cnhuhbert_base_path = cnhubert_base_path
if bert_path is not None:
    tts_config.bert_base_path = bert_path

tts_pipline = TTS(tts_config)

text = "Well, you know what the say , Families are like fudge, mostly sweet, but sometimes nuts. My family is doing great, thanks for asking!  My son is growing up to be a smart and handsome young man, just like his mom. He's currently working on his own talker show, which I'm sure will be even more hilarious than mine."    #input text
text_language = "en"           #select "en","all_zh","all_ja"
inp_ref = "./dataset/doubao-ref-ours.wav"   #path of reference speaker
prompt_text = "我叫豆包呀，能陪你聊天解闷，不管是聊生活趣事，知识科普还是帮你出主意，我都在行哦。"               #text of reference speech
prompt_language = "all_zh"         #reference speech language
batch_size = 100              #inference batch size
speed_factor = 1.0             #control speed of output audio
top_k = 5                      #gpt
top_p = 1
temperature = 1
how_to_cut = "cut4"            #"cut0": not cut   "cut1": 4 sentences a cut   "cut2": 50 words a cut   "cut3": cut at chinese '。'  "cut4": cut at english '.'   "cut5": auto cut
ref_text_free = False
split_bucket = True            #suggest on
fragment_interval = 0.07     #interval between every sentence
seed = 233333               #seed

actual_seed = seed if seed not in [-1, "", None] else random.randrange(1 << 32)
inputs = {
    "text": text,
    "text_lang": text_language,
    "ref_audio_path": inp_ref,
    "prompt_text": prompt_text if not ref_text_free else "",
    "prompt_lang": prompt_language,
    "top_k": top_k,
    "top_p": top_p,
    "temperature": temperature,
    "text_split_method": how_to_cut,
    "batch_size": int(batch_size),
    "speed_factor": float(speed_factor),
    "split_bucket": split_bucket,
    "return_fragment": False,
    "fragment_interval": fragment_interval,
    "seed": actual_seed,
}
print(inputs)

# Get results from pipeline
result_generator = tts_pipline.run(inputs)
results = []
for item in result_generator:
    results.append((item, actual_seed))

# Write output
output = results[0]
sf.write('./output.wav', output[0][1], samplerate=output[0][0], subtype='PCM_16')
