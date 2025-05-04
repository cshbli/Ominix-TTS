from copy import deepcopy
import os, sys, gc
import random
import traceback
import time
from tqdm import tqdm
import ffmpeg
import os
from typing import List, Union
import numpy as np
from peft import LoraConfig, get_peft_model
import torch
import yaml
from huggingface_hub import hf_hub_download
from transformers import AutoModelForMaskedLM, AutoTokenizer

from .tools.audio_sr import AP_BWE
from .AR.models.t2s_lightning_module import Text2SemanticLightningModule
from .feature_extractor.cnhubert import CNHubert
from .module.models import SynthesizerTrn, SynthesizerTrnV3
from .tools.i18n.i18n import I18nAuto, scan_language_list
from .text_processor.text_segmentation import splits
from .text_processor.processor import TextPreprocessor
from .reference_processor.processor import ReferenceProcessor
from .process_ckpt import get_sovits_version_from_path_fast, load_sovits_new
from .model_download import download_folder_from_repo

language=os.environ.get("language","Auto")
language=sys.argv[-1] if sys.argv[-1] in scan_language_list() else language
i18n = I18nAuto(language=language)

def speed_change(input_audio:np.ndarray, speed:float, sr:int):
    # 将 NumPy 数组转换为原始 PCM 流
    raw_audio = input_audio.astype(np.int16).tobytes()

    # 设置 ffmpeg 输入流
    input_stream = ffmpeg.input('pipe:', format='s16le', acodec='pcm_s16le', ar=str(sr), ac=1)

    # 变速处理
    output_stream = input_stream.filter('atempo', speed)

    # 输出流到管道
    out, _ = (
        output_stream.output('pipe:', format='s16le', acodec='pcm_s16le')
        .run(input=raw_audio, capture_stdout=True, capture_stderr=True)
    )

    # 将管道输出解码为 NumPy 数组
    processed_audio = np.frombuffer(out, np.int16)

    return processed_audio


class NO_PROMPT_ERROR(Exception):
    pass


def set_seed(seed:int):
    seed = int(seed)
    seed = seed if seed != -1 else random.randint(0, 2**32 - 1)
    print(f"Set seed to {seed}")
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = False
            # torch.backends.cudnn.enabled = True
            # 开启后会影响精度
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
    except:
        pass
    return seed

class TTS_Config:
    default_configs={
        "v1":{
                "device": "cpu",
                "is_half": False,
                "version": "v1",
                "t2s_weights_path": "MOTTS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt",
                "vits_weights_path": "MOTTS/pretrained_models/s2G488k.pth",
                "cnhuhbert_base_path": "MOTTS/pretrained_models/chinese-hubert-base",
                "bert_base_path": "MOTTS/pretrained_models/chinese-roberta-wwm-ext-large",
            },
        "v2":{
                "device": "cpu",
                "is_half": False,
                "version": "v2",
                "t2s_weights_path": "MOTTS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
                "vits_weights_path": "MOTTS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth",
                "cnhuhbert_base_path": "MOTTS/pretrained_models/chinese-hubert-base",
                "bert_base_path": "MOTTS/pretrained_models/chinese-roberta-wwm-ext-large",
            },
        "v3":{
                "device": "cpu",
                "is_half": False,
                "version": "v3",
                "t2s_weights_path": "MOTTS/pretrained_models/s1v3.ckpt",
                "vits_weights_path": "MOTTS/pretrained_models/s2Gv3.pth",
                "cnhuhbert_base_path": "MOTTS/pretrained_models/chinese-hubert-base",
                "bert_base_path": "MOTTS/pretrained_models/chinese-roberta-wwm-ext-large",
            },
    }
    configs:dict = None
    v1_languages:list = ["auto", "en", "zh", "ja",  "all_zh", "all_ja"]
    v2_languages:list = ["auto", "auto_yue", "en", "zh", "ja", "yue", "ko", "all_zh", "all_ja", "all_yue", "all_ko"]
    languages:list = v2_languages
    # "all_zh",#全部按中文识别
    # "en",#全部按英文识别#######不变
    # "all_ja",#全部按日文识别
    # "all_yue",#全部按中文识别
    # "all_ko",#全部按韩文识别
    # "zh",#按中英混合识别####不变
    # "ja",#按日英混合识别####不变
    # "yue",#按粤英混合识别####不变
    # "ko",#按韩英混合识别####不变
    # "auto",#多语种启动切分识别语种
    # "auto_yue",#多语种启动切分识别语种

    def __init__(self, configs: Union[dict, str]=None):

        # 设置默认配置文件路径
        # configs_base_path:str = "MOTTS/configs/"
        # os.makedirs(configs_base_path, exist_ok=True)
        # self.configs_path:str = os.path.join(configs_base_path, "tts_infer.yaml")

        if configs in ["", None]:
            # if not os.path.exists(self.configs_path):
            #     self.save_configs()
            #     print(f"Create default config file at {self.configs_path}")
            configs:dict = deepcopy(self.default_configs)

        if isinstance(configs, str):
            self.configs_path = configs
            configs:dict = self._load_configs(self.configs_path)

        assert isinstance(configs, dict)
        version = configs.get("version", "v2").lower()
        assert version in ["v1", "v2", "v3"]
        self.default_configs[version] = configs.get(version, self.default_configs[version])
        self.configs:dict = configs.get("custom", deepcopy(self.default_configs[version]))

        self.device = self.configs.get("device", torch.device("cpu"))
        if "cuda" in str(self.device) and not torch.cuda.is_available():
            print(f"Warning: CUDA is not available, set device to CPU.")
            self.device = torch.device("cpu")

        self.is_half = self.configs.get("is_half", False)
        # if str(self.device) == "cpu" and self.is_half:
        #     print(f"Warning: Half precision is not supported on CPU, set is_half to False.")
        #     self.is_half = False

        self.version = version
        self.t2s_weights_path = self.configs.get("t2s_weights_path", None)
        self.vits_weights_path = self.configs.get("vits_weights_path", None)
        self.bert_base_path = self.configs.get("bert_base_path", None)
        self.cnhuhbert_base_path = self.configs.get("cnhuhbert_base_path", None)
        self.languages = self.v1_languages if self.version=="v1" else self.v2_languages

        self.is_v3_synthesizer:bool = False


        if (self.t2s_weights_path in [None, ""]) or (not os.path.exists(self.t2s_weights_path)):
            self.t2s_weights_path = gpt_path = hf_hub_download(
                repo_id="cshbli/MoTTS",
                filename="models/T2S/txdb-e15.ckpt",
                revision="main"
            )
            print(f"fall back to default t2s_weights_path: {self.t2s_weights_path}")
        if (self.vits_weights_path in [None, ""]) or (not os.path.exists(self.vits_weights_path)):
            self.vits_weights_path = hf_hub_download(
                repo_id="cshbli/MoTTS",
                filename="models/VITS/txdb_e12_s204.pth",
                revision="main"
            )
            print(f"fall back to default vits_weights_path: {self.vits_weights_path}")
        if (self.bert_base_path in [None, ""]) or (not os.path.exists(self.bert_base_path)):
            self.bert_base_path = download_folder_from_repo(    
                repo_id="cshbli/MoTTS",
                folder_path="models/BERT/chinese-roberta-wwm-ext-large"
            )
            print(f"fall back to default bert_base_path: {self.bert_base_path}")
        if (self.cnhuhbert_base_path in [None, ""]) or (not os.path.exists(self.cnhuhbert_base_path)):
            self.cnhuhbert_base_path = download_folder_from_repo(
                repo_id="cshbli/MoTTS",
                folder_path="models/HuBERT/chinese-hubert-base"
            )
            print(f"fall back to default cnhuhbert_base_path: {self.cnhuhbert_base_path}")
        self.update_configs()

        self.max_sec = None
        self.hz:int = 50
        self.semantic_frame_rate:str = "25hz"
        self.segment_size:int = 20480
        self.filter_length:int = 2048
        self.sampling_rate:int = 32000
        self.hop_length:int = 640
        self.win_length:int = 2048
        self.n_speakers:int = 300



    def _load_configs(self, configs_path: str)->dict:
        if os.path.exists(configs_path):
            ...
        else:
            print(i18n("路径不存在,使用默认配置"))
            self.save_configs(configs_path)
        with open(configs_path, 'r', encoding='utf-8') as f:
            configs = yaml.load(f, Loader=yaml.FullLoader)

        return configs

    def save_configs(self, configs_path:str=None)->None:
        configs=deepcopy(self.default_configs)
        if self.configs is not None:
            configs["custom"] = self.update_configs()

        if configs_path is None:
            configs_path = self.configs_path
        with open(configs_path, 'w') as f:
            yaml.dump(configs, f)

    def update_configs(self):
        self.config = {
            "device"             : str(self.device),
            "is_half"            : self.is_half,
            "version"            : self.version,
            "t2s_weights_path"   : self.t2s_weights_path,
            "vits_weights_path"  : self.vits_weights_path,
            "bert_base_path"     : self.bert_base_path,
            "cnhuhbert_base_path": self.cnhuhbert_base_path,
        }
        return self.config

    def update_version(self, version:str)->None:
        self.version = version
        self.languages = self.v1_languages if self.version=="v1" else self.v2_languages

    def __str__(self):
        self.configs = self.update_configs()
        string = "TTS Config".center(100, '-') + '\n'
        for k, v in self.configs.items():
            string += f"{str(k).ljust(20)}: {str(v)}\n"
        string += "-" * 100 + '\n'
        return string

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(self.configs_path)

    def __eq__(self, other):
        return isinstance(other, TTS_Config) and self.configs_path == other.configs_path


class MPipeline:
    def __init__(self, configs: Union[dict, str, TTS_Config] = None):
        if isinstance(configs, TTS_Config):
            self.configs = configs
        else:
            self.configs:TTS_Config = TTS_Config(configs)

        self.t2s_model:Text2SemanticLightningModule = None
        self.vits_model:Union[SynthesizerTrn, SynthesizerTrnV3] = None
        self.bert_tokenizer:AutoTokenizer = None
        self.bert_model:AutoModelForMaskedLM = None
        self.cnhuhbert_model:CNHubert = None
        self.sr_model:AP_BWE = None
        self.sr_model_not_exist:bool = False

        self.stop_flag:bool = False
        self.precision:torch.dtype = torch.float16 if self.configs.is_half else torch.float32

        self._init_models()

        self.text_preprocessor:TextPreprocessor = TextPreprocessor(
            self.bert_model,
            self.bert_tokenizer,
            self.configs.device,
            self.precision
        )
        
        self.reference_processor:ReferenceProcessor = ReferenceProcessor(
                self.text_preprocessor,
                self.cnhuhbert_model, 
                self.vits_model, 
                self.configs.device, 
                self.configs
        )

        self.prompt_cache:dict = {
            "ref_audio_path" : None,
            "prompt_semantic": None,
            "refer_spec"     : [],
            "prompt_text"    : None,
            "prompt_lang"    : None,
            "phones"         : None,
            "bert_features"  : None,
            "norm_text"      : None,
            "aux_ref_audio_paths": [],
        }

    def _init_models(self,):
        self.init_t2s_weights(self.configs.t2s_weights_path)        
        self.init_vits_weights(self.configs.vits_weights_path)        
        self.init_bert_weights(self.configs.bert_base_path)        
        self.init_cnhuhbert_weights(self.configs.cnhuhbert_base_path)
        

    def init_cnhuhbert_weights(self, base_path: str):
        print(f"Loading CNHuBERT weights from {base_path}")
        self.cnhuhbert_model = CNHubert(base_path)
        self.cnhuhbert_model=self.cnhuhbert_model.eval()
        self.cnhuhbert_model = self.cnhuhbert_model.to(self.configs.device)
        if self.configs.is_half and str(self.configs.device)!="cpu":
            self.cnhuhbert_model = self.cnhuhbert_model.half()



    def init_bert_weights(self, base_path: str):
        print(f"Loading BERT weights from {base_path}")
        self.bert_tokenizer = AutoTokenizer.from_pretrained(base_path)
        self.bert_model = AutoModelForMaskedLM.from_pretrained(base_path)
        self.bert_model=self.bert_model.eval()
        self.bert_model = self.bert_model.to(self.configs.device)
        if self.configs.is_half and str(self.configs.device)!="cpu":
            self.bert_model = self.bert_model.half()

    def init_vits_weights(self, weights_path: str):
        
        self.configs.vits_weights_path = weights_path
        version, model_version, if_lora_v3=get_sovits_version_from_path_fast(weights_path)
        path_sovits_v3=self.configs.default_configs["v3"]["vits_weights_path"]

        if if_lora_v3==True and os.path.exists(path_sovits_v3)==False:
            info= path_sovits_v3 + i18n("SoVITS V3 底模缺失，无法加载相应 LoRA 权重")
            raise FileExistsError(info)

        # dict_s2 = torch.load(weights_path, map_location=self.configs.device,weights_only=False)
        dict_s2 = load_sovits_new(weights_path)        

        hps = dict_s2["config"]

        hps["model"]["semantic_frame_rate"] = "25hz"
        if 'enc_p.text_embedding.weight'not in dict_s2['weight']:
            hps["model"]["version"] = "v2"#v3model,v2sybomls
        elif dict_s2['weight']['enc_p.text_embedding.weight'].shape[0] == 322:
            hps["model"]["version"] = "v1"
        else:
            hps["model"]["version"] = "v2"
        # version = hps["model"]["version"]

        self.configs.filter_length = hps["data"]["filter_length"]
        self.configs.segment_size = hps["train"]["segment_size"]
        self.configs.sampling_rate = hps["data"]["sampling_rate"]
        self.configs.hop_length = hps["data"]["hop_length"]
        self.configs.win_length = hps["data"]["win_length"]
        self.configs.n_speakers = hps["data"]["n_speakers"]
        self.configs.semantic_frame_rate = hps["model"]["semantic_frame_rate"]
        kwargs = hps["model"]
        # print(f"self.configs.sampling_rate:{self.configs.sampling_rate}")

        self.configs.update_version(model_version)

        print(f"model_version:{model_version}")
        # print(f'hps["model"]["version"]:{hps["model"]["version"]}')
        if model_version!="v3":
            vits_model = SynthesizerTrn(
                self.configs.filter_length // 2 + 1,
                self.configs.segment_size // self.configs.hop_length,
                n_speakers=self.configs.n_speakers,
                **kwargs
            )
            self.configs.is_v3_synthesizer = False
        else:
            vits_model = SynthesizerTrnV3(
                self.configs.filter_length // 2 + 1,
                self.configs.segment_size // self.configs.hop_length,
                n_speakers=self.configs.n_speakers,
                **kwargs
            )
            self.configs.is_v3_synthesizer = True
            # self.init_bigvgan()
            if "pretrained" not in weights_path and hasattr(vits_model, "enc_q"):
                del vits_model.enc_q

        if if_lora_v3==False:
            print(f"Loading VITS weights from {weights_path}. {vits_model.load_state_dict(dict_s2['weight'], strict=False)}")
        else:
            print(f"Loading VITS pretrained weights from {weights_path}. {vits_model.load_state_dict(load_sovits_new(path_sovits_v3)['weight'], strict=False)}")
            lora_rank=dict_s2["lora_rank"]
            lora_config = LoraConfig(
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],
                r=lora_rank,
                lora_alpha=lora_rank,
                init_lora_weights=True,
            )
            vits_model.cfm = get_peft_model(vits_model.cfm, lora_config)
            print(f"Loading LoRA weights from {weights_path}. {vits_model.load_state_dict(dict_s2['weight'], strict=False)}")
            
            vits_model.cfm = vits_model.cfm.merge_and_unload()


        vits_model = vits_model.to(self.configs.device)
        vits_model = vits_model.eval()

        self.vits_model = vits_model
        if self.configs.is_half and str(self.configs.device)!="cpu":
            self.vits_model = self.vits_model.half()


    def init_t2s_weights(self, weights_path: str):
        print(f"Loading Text2Semantic weights from {weights_path}")
        self.configs.t2s_weights_path = weights_path
        # self.configs.save_configs()
        self.configs.hz = 50
        dict_s1 = torch.load(weights_path, map_location=self.configs.device, weights_only=False)
        config = dict_s1["config"]
        self.configs.max_sec = config["data"]["max_sec"]
        t2s_model = Text2SemanticLightningModule(config, "****", is_train=False)
        t2s_model.load_state_dict(dict_s1["weight"])
        t2s_model = t2s_model.to(self.configs.device)
        t2s_model = t2s_model.eval()
        self.t2s_model = t2s_model
        if self.configs.is_half and str(self.configs.device)!="cpu":
            self.t2s_model = self.t2s_model.half()
    
    def enable_half_precision(self, enable: bool = True, save: bool = True):
        '''
            To enable half precision for the TTS model.
            Args:
                enable: bool, whether to enable half precision.

        '''
        if str(self.configs.device) == "cpu" and enable:
            print("Half precision is not supported on CPU.")
            return

        self.configs.is_half = enable
        self.precision = torch.float16 if enable else torch.float32
        if save:
            self.configs.save_configs()
        if enable:
            if self.t2s_model is not None:
                self.t2s_model =self.t2s_model.half()
            if self.vits_model is not None:
                self.vits_model = self.vits_model.half()
            if self.bert_model is not None:
                self.bert_model =self.bert_model.half()
            if self.cnhuhbert_model is not None:
                self.cnhuhbert_model = self.cnhuhbert_model.half()
            if self.bigvgan_model is not None:
                self.bigvgan_model = self.bigvgan_model.half()
        else:
            if self.t2s_model is not None:
                self.t2s_model = self.t2s_model.float()
            if self.vits_model is not None:
                self.vits_model = self.vits_model.float()
            if self.bert_model is not None:
                self.bert_model = self.bert_model.float()
            if self.cnhuhbert_model is not None:
                self.cnhuhbert_model = self.cnhuhbert_model.float()
            if self.bigvgan_model is not None:
                self.bigvgan_model = self.bigvgan_model.float()

    def set_device(self, device: torch.device, save: bool = True):
        '''
            To set the device for all models.
            Args:
                device: torch.device, the device to use for all models.
        '''
        self.configs.device = device
        if save:
            self.configs.save_configs()
        if self.t2s_model is not None:
            self.t2s_model = self.t2s_model.to(device)
        if self.vits_model is not None:
            self.vits_model = self.vits_model.to(device)
        if self.bert_model is not None:
            self.bert_model = self.bert_model.to(device)
        if self.cnhuhbert_model is not None:
            self.cnhuhbert_model = self.cnhuhbert_model.to(device)
        if self.bigvgan_model is not None:
            self.bigvgan_model = self.bigvgan_model.to(device)
        if self.sr_model is not None:
            self.sr_model = self.sr_model.to(device)    

    def batch_sequences(self, sequences: List[torch.Tensor], axis: int = 0, pad_value: int = 0, max_length:int=None):
        seq = sequences[0]
        ndim = seq.dim()
        if axis < 0:
            axis += ndim
        dtype:torch.dtype = seq.dtype
        pad_value = torch.tensor(pad_value, dtype=dtype)
        seq_lengths = [seq.shape[axis] for seq in sequences]
        if max_length is None:
            max_length = max(seq_lengths)
        else:
            max_length = max(seq_lengths) if max_length < max(seq_lengths) else max_length

        padded_sequences = []
        for seq, length in zip(sequences, seq_lengths):
            padding = [0] * axis + [0, max_length - length] + [0] * (ndim - axis - 1)
            padded_seq = torch.nn.functional.pad(seq, padding, value=pad_value)
            padded_sequences.append(padded_seq)
        batch = torch.stack(padded_sequences)
        return batch
    
    def stop(self,):
        '''
        Stop the inference process.
        '''
        self.stop_flag = True


    @torch.no_grad()
    def run(self, inputs:dict):
        """
        Text to speech inference.

        Args:
            inputs (dict):
                {
                    "text": "",                   # str.(required) text to be synthesized
                    "text_lang: "",               # str.(required) language of the text to be synthesized
                    "ref_audio_path": "",         # str.(required) reference audio path
                    "aux_ref_audio_paths": [],    # list.(optional) auxiliary reference audio paths for multi-speaker tone fusion
                    "prompt_text": "",            # str.(optional) prompt text for the reference audio
                    "prompt_lang": "",            # str.(required) language of the prompt text for the reference audio
                    "top_k": 5,                   # int. top k sampling
                    "top_p": 1,                   # float. top p sampling
                    "temperature": 1,             # float. temperature for sampling
                    "text_split_method": "cut0",  # str. text split method, see text_segmentation_method.py for details.
                    "batch_size": 1,              # int. batch size for inference
                    "batch_threshold": 0.75,      # float. similarity_threshold for batch splitting.
                    "split_bucket: True,          # bool. whether to split the batch into multiple buckets.
                    "return_fragment": False,     # bool. step by step return the audio fragment.
                    "speed_factor":1.0,           # float. control the speed of the synthesized audio.
                    "fragment_interval":0.3,      # float. to control the interval of the audio fragment.
                    "seed": -1,                   # int. random seed for reproducibility.
                    "parallel_infer": True,       # bool. whether to use parallel inference.
                    "repetition_penalty": 1.35    # float. repetition penalty for T2S model.
                    "sample_steps": 32,           # int. number of sampling steps for VITS model V3.
                    "super_sampling": False,       # bool. whether to use super-sampling for audio when using VITS model V3.
                }
        returns:
            Tuple[int, np.ndarray]: sampling rate and audio data.
        """
        ########## variables initialization ###########
        self.stop_flag:bool = False
        text:str = inputs.get("text", "")
        text_lang:str = inputs.get("text_lang", "")
        assert text_lang in self.configs.languages
        ref_audio_path:str = inputs.get("ref_audio_path", "")
        aux_ref_audio_paths:list = inputs.get("aux_ref_audio_paths", [])
        prompt_text:str = inputs.get("prompt_text", "")
        prompt_lang:str = inputs.get("prompt_lang", "")
        top_k:int = inputs.get("top_k", 5)
        top_p:float = inputs.get("top_p", 1)
        temperature:float = inputs.get("temperature", 1)
        text_split_method:str = inputs.get("text_split_method", "cut0")
        batch_size = inputs.get("batch_size", 1)
        batch_threshold = inputs.get("batch_threshold", 0.75)
        speed_factor = inputs.get("speed_factor", 1.0)
        split_bucket = inputs.get("split_bucket", True)
        return_fragment = inputs.get("return_fragment", False)
        fragment_interval = inputs.get("fragment_interval", 0.3)
        seed = inputs.get("seed", -1)
        seed = -1 if seed in ["", None] else seed
        actual_seed = set_seed(seed)
        parallel_infer = inputs.get("parallel_infer", True)
        repetition_penalty = inputs.get("repetition_penalty", 1.35)
        sample_steps = inputs.get("sample_steps", 32)
        super_sampling = inputs.get("super_sampling", False)

        if parallel_infer:
            print(i18n("并行推理模式已开启"))
            self.t2s_model.model.infer_panel = self.t2s_model.model.infer_panel_batch_infer
        else:
            print(i18n("并行推理模式已关闭"))
            self.t2s_model.model.infer_panel = self.t2s_model.model.infer_panel_naive_batched

        if return_fragment:
            print(i18n("分段返回模式已开启"))
            if split_bucket:
                split_bucket = False
                print(i18n("分段返回模式不支持分桶处理，已自动关闭分桶处理"))

        if split_bucket and speed_factor==1.0 and not (self.configs.is_v3_synthesizer and parallel_infer):
            print(i18n("分桶处理模式已开启"))
        elif speed_factor!=1.0:
            print(i18n("语速调节不支持分桶处理，已自动关闭分桶处理"))
            split_bucket = False
        elif self.configs.is_v3_synthesizer and parallel_infer:
            print(i18n("当开启并行推理模式时，SoVits V3模型不支持分桶处理，已自动关闭分桶处理"))
            split_bucket = False
        else:
            print(i18n("分桶处理模式已关闭"))

        if fragment_interval<0.01:
            fragment_interval = 0.01
            print(i18n("分段间隔过小，已自动设置为0.01"))        
        
        print("############ Reference Audio/Text Processing ############")
        t_ref_start = time.perf_counter()
        try:
            no_prompt_text = prompt_text in [None, ""]
            if not no_prompt_text:
                assert prompt_lang in self.configs.languages
            if no_prompt_text and self.configs.is_v3_synthesizer:
                raise NO_PROMPT_ERROR("prompt_text cannot be empty when using SoVITS_V3")
            
            self.prompt_cache = self.reference_processor.process_reference(
                ref_audio_path=ref_audio_path,
                prompt_text=prompt_text,
                prompt_lang=prompt_lang,                
                model_version=self.configs.version,
                aux_ref_audio_paths=aux_ref_audio_paths
            )
        except Exception as e:
            print(f"Error processing reference audio: {str(e)}")
            raise e            
        
        t_ref_end = time.perf_counter()
        t_reference = t_ref_end - t_ref_start
        print(f"Reference audio processing time: {t_reference:.3f} seconds")

        ###### text preprocessing ########        
        data:list = None
        if not return_fragment:
            data = self.text_preprocessor.process(text, text_lang, text_split_method, self.configs.version)
            if len(data) == 0:
                yield 16000, np.zeros(int(16000), dtype=np.int16)
                return

            batch_index_list:list = None
            data, batch_index_list = self.text_preprocessor.create_inference_batches(data,
                                prompt_data=self.prompt_cache if not no_prompt_text else None,
                                batch_size=batch_size,
                                similarity_threshold=batch_threshold,
                                split_bucket=split_bucket)
        else:
            print(f'############ {i18n("切分文本")} ############')
            texts = self.text_preprocessor.pre_seg_text(text, text_lang, text_split_method)
            data = []
            for i in range(len(texts)):
                if i%batch_size == 0:
                    data.append([])
                data[-1].append(texts[i])

        t2 = time.perf_counter()
        try:
            print("############ 推理 ############")
            ###### inference ######
            t_34 = 0.0
            t_45 = 0.0
            audio = []
            output_sr = self.configs.sampling_rate if not self.configs.is_v3_synthesizer else 24000

            # Import synthesizer components
            from .audio_synthesis.synthesizer_factory import create_synthesizer
            from .audio_synthesis.audio_processor import AudioProcessor
        
            # Create synthesizer and audio processor
            synthesizer = create_synthesizer(self.vits_model, self.configs, self.prompt_cache)
            audio_processor = AudioProcessor(self.configs, self.sr_model)

            for item in data:
                t3 = time.perf_counter()
                if return_fragment:
                    item = self.text_preprocessor.process_text_fragments(
                        item,
                        text_lang,
                        batch_size,
                        batch_threshold,
                        version=self.configs.version,
                        no_prompt_text=no_prompt_text
                    )
                    if item is None:
                        continue

                batch_phones:List[torch.LongTensor] = item["phones"]
                # batch_phones:torch.LongTensor = item["phones"]
                batch_phones_len:torch.LongTensor = item["phones_len"]
                all_phoneme_ids:torch.LongTensor = item["all_phones"]
                all_phoneme_lens:torch.LongTensor  = item["all_phones_len"]
                all_bert_features:torch.LongTensor = item["all_bert_features"]
                norm_text:str = item["norm_text"]
                max_len = item["max_len"]

                print(i18n("前端处理后的文本(每句):"), norm_text)
                if no_prompt_text :
                    prompt = None
                else:
                    prompt = self.prompt_cache["prompt_semantic"].expand(len(all_phoneme_ids), -1).to(self.configs.device)

                print(f"############ {i18n('预测语义Token')} ############")
                pred_semantic_list, idx_list = self.t2s_model.model.infer_panel(
                    all_phoneme_ids,
                    all_phoneme_lens,
                    prompt,
                    all_bert_features,
                    # prompt_phone_len=ph_offset,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    early_stop_num=self.configs.hz * self.configs.max_sec,
                    max_len=max_len,
                    repetition_penalty=repetition_penalty,
                )
                t4 = time.perf_counter()
                t_34 += t4 - t3

                refer_audio_spec:torch.Tensor = [item.to(dtype=self.precision, device=self.configs.device) for item in self.prompt_cache["refer_spec"]]                
                
                print(f"############ {i18n('合成音频')} ############")
                batch_audio_fragment = synthesizer.synthesize(
                    pred_semantic_list,
                    batch_phones,
                    refer_audio_spec,
                    idx_list,
                    speed_factor=speed_factor,
                    parallel_synthesis=parallel_infer,
                    sample_steps=sample_steps
                )

                t5 = time.perf_counter()
                t_45 += t5 - t4
                if return_fragment:
                    print("%.3f\t%.3f\t%.3f\t%.3f" % (t_ref_end - t_ref_start, t2 - t_ref_end, t4 - t3, t5 - t4))
                    yield audio_processor.process_audio_batches(
                        [batch_audio_fragment],
                        output_sr,
                        None,
                        speed_factor,
                        False,
                        fragment_interval,
                        super_sampling and self.configs.is_v3_synthesizer
                    )
                else:
                    audio.append(batch_audio_fragment)

                if self.stop_flag:
                    yield 16000, np.zeros(int(16000), dtype=np.int16)
                    return

            if not return_fragment:
                print("%.3f\t%.3f\t%.3f\t%.3f" % (t_ref_end - t_ref_start, t2 - t_ref_end, t_34, t_45))
                if len(audio) == 0:
                    yield 16000, np.zeros(int(16000), dtype=np.int16)
                    return
                
                yield audio_processor.process_audio_batches(
                    audio,
                    output_sr,
                    batch_index_list,
                    speed_factor,
                    split_bucket,
                    fragment_interval,
                    super_sampling and self.configs.is_v3_synthesizer
                )
        except Exception as e:
            traceback.print_exc()
            # 必须返回一个空音频, 否则会导致显存不释放。
            yield 16000, np.zeros(int(16000), dtype=np.int16)
            # 重置模型, 否则会导致显存释放不完全。
            del self.t2s_model
            del self.vits_model
            self.t2s_model = None
            self.vits_model = None
            self.init_t2s_weights(self.configs.t2s_weights_path)
            self.init_vits_weights(self.configs.vits_weights_path)
            raise e
        finally:
            self.empty_cache()

    def empty_cache(self):
        try:
            gc.collect() # 触发gc的垃圾回收。避免内存一直增长。
            if "cuda" in str(self.configs.device):
                torch.cuda.empty_cache()
            elif str(self.configs.device) == "mps":
                torch.mps.empty_cache()
        except:
            pass
    
    def __call__(self, 
            text:str,   # input text
            text_language:str,  # select "en", "all_zh", "all_ja"
            ref_audio_path:str,  # reference audio path          
            ref_text:str="",     # reference text
            ref_language:str="all_zh",  # reference text language
            ref_text_free:bool=False, # whether to use reference text
            aux_ref_audio_paths:list=[],
            batch_size:int=100,             # inference batch size
            speed_factor:float=1.0, # control speed of output audio
            top_k:int=5,
            top_p:float=1,
            temperature:float=1,
            text_split_method:str="cut4", #"cut0": not cut   "cut1": 4 sentences a cut   "cut2": 50 words a cut   "cut3": cut at chinese '。'  "cut4": cut at english '.'   "cut5": auto cut
            split_bucket:bool=True,
            return_fragment:bool=False,
            fragment_interval:float=0.07,   # interval between every sentence
            seed:int=233333
            ):
        
        actual_seed = seed if seed not in [-1, "", None] else random.randrange(1 << 32)
        inputs = {
            "text": text,
            "text_lang": text_language,
            "ref_audio_path": ref_audio_path,
            "prompt_text": ref_text if not ref_text_free else "",
            "prompt_lang": ref_language,
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "text_split_method": text_split_method,
            "batch_size": batch_size,
            "speed_factor": speed_factor,
            "split_bucket": split_bucket,
            "return_fragment": return_fragment,
            "fragment_interval": fragment_interval,
            "seed": actual_seed,
        }
        print(inputs)

        return self.run(inputs)

