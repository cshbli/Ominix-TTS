from copy import deepcopy
import os, sys, gc
import random
import traceback
import time
import ffmpeg
import os
from typing import List
import numpy as np
from peft import LoraConfig, get_peft_model
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForMaskedLM, AutoTokenizer

from .tools.audio_sr import AP_BWE
from .AR.models.t2s_lightning_module import Text2SemanticLightningModule
from .feature_extractor.cnhubert import CNHubert
from .module.models import SynthesizerTrn, SynthesizerTrnV3
from .tools.i18n.i18n import I18nAuto, scan_language_list
from .text_processor.processor import TextPreprocessor
from .reference_processor.processor import ReferenceProcessor
from .process_ckpt import get_sovits_version_from_path_fast, load_sovits_new
from .model_download import download_folder_from_repo

language=os.environ.get("language","Auto")
language=sys.argv[-1] if sys.argv[-1] in scan_language_list() else language
i18n = I18nAuto(language=language)

def get_text_split_method(text_language: str, provided_method: str = "") -> str:
    """
    Determine the appropriate text split method based on language and user preference.
    
    This function selects a text splitting strategy based on the input language when
    no specific method is provided. For English, it defaults to sentence-based splitting,
    while for other languages it uses automatic splitting.
    
    Args:
        text_language: The language of the text ("en", "all_zh", "all_ja", etc.)
        provided_method: User-provided split method (empty string if none provided)
        
    Returns:
        The text split method to use
        
    Note:
        Available methods:
        - "cut0": No splitting
        - "cut1": Split after every 4 sentences
        - "cut2": Split after every 50 words
        - "cut3": Split at Chinese period ("。")
        - "cut4": Split at English period (".")
        - "cut5": Automatic language-specific splitting

        Supported languages:
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
    """
    # If a method is provided, use it
    if provided_method is not None and provided_method != "":
        return provided_method
        
    # Otherwise, select based on language
    text_language = text_language.lower()
    
    if "en" in text_language:
        return "cut4"  # English period-based splitting
    elif "zh" in text_language or "yue" in text_language:
        return "cut3"  # Chinese period-based splitting
    else:
        return "cut5"  # Automatic language-specific splitting


def load_text_file(file_path):
    """
    Load the contents of a text file as a string.
    
    Args:
        file_path: Path to the text file
        
    Returns:
        String containing the file contents
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read().strip()
        return content
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return ""
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return ""


def get_reference_path(relative_path):
    """
    Get the absolute path to a reference file, either from the installed package
    or from the relative path.
    
    Args:
        relative_path: Relative path within the package (e.g., "dataset/doubao-ref-ours.wav")
        
    Returns:
        Absolute path to the reference file
    """
    try:
        # Try to import ominix_tts to check if it's installed
        import ominix_tts
        import os.path
        
        # Get the package installation directory
        package_dir = os.path.dirname(ominix_tts.__file__)
        
        # Construct absolute path to the reference file
        absolute_path = os.path.join(package_dir, relative_path)
        
        # Verify the file exists at the package location
        if os.path.exists(absolute_path):
            return absolute_path
        else:
            print(f"Fall back to relative path if file doesn't exist in package")
            return relative_path
            
    except ImportError:
        print(f"Package not installed, use relative path")
        return relative_path
    

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


def set_seed(seed: int) -> int:
    """
    Set random seeds for reproducibility across all random number generators.
    
    Args:
        seed: Integer seed value. If -1, a random seed will be generated.
        
    Returns:
        The actual seed used (either the provided seed or a generated one)
    
    Note:
        This sets seeds for Python's random module, NumPy, and PyTorch
        (including CUDA if available).
    """
    # Convert to integer and handle special case
    try:
        seed = int(seed)
    except (ValueError, TypeError):
        print(f"Warning: Invalid seed value '{seed}', using random seed instead")
        seed = random.randint(0, 2**32 - 1)
        
    # Generate random seed if requested
    if seed == -1:
        seed = random.randint(0, 2**32 - 1)
        
    print(f"Setting seed to {seed}")
    
    # Set environment variable for potential subprocesses
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Set seeds for core random number generators
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Handle CUDA-specific seeding
    if torch.cuda.is_available():
        try:
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            
            # Control precision settings
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            
            # Note: We keep these commented as they impact performance
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = False
        except Exception as e:
            print(f"Warning: CUDA seed setting failed with error: {str(e)}")
    
    return seed

class TTSConfiguration:
    """
    Configuration manager for Ominix-TTS system.
    
    Handles all configuration settings for model versions, paths, device settings,
    audio parameters, and language support.
    """
    
    # Class constants for better maintainability
    REPO_ID = "cshbli/MoTTS"
    REPO_REVISION = "main"
    
    # Default model paths organized by version
    DEFAULT_CONFIGS = {
        "v1": {
            "device": "cpu",
            "is_half": False,
            "version": "v1",
            "t2s_weights_path": "./pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt",
            "vits_weights_path": "./pretrained_models/s2G488k.pth",
            "cnhuhbert_base_path": "./pretrained_models/chinese-hubert-base",
            "bert_base_path": "./pretrained_models/chinese-roberta-wwm-ext-large",
        },
        "v2": {
            "device": "cpu",
            "is_half": False,
            "version": "v2",
            "t2s_weights_path": "./pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
            "vits_weights_path": "./pretrained_models/gsv-v2final-pretrained/s2G2333k.pth",
            "cnhuhbert_base_path": "./pretrained_models/chinese-hubert-base",
            "bert_base_path": "./pretrained_models/chinese-roberta-wwm-ext-large",
        },
        "v3": {
            "device": "cpu",
            "is_half": False,
            "version": "v3",
            "t2s_weights_path": "./pretrained_models/s1v3.ckpt",
            "vits_weights_path": "./pretrained_models/s2Gv3.pth",
            "cnhuhbert_base_path": "./pretrained_models/chinese-hubert-base",
            "bert_base_path": "./pretrained_models/chinese-roberta-wwm-ext-large",
        },
    }
    
    # Language support by version
    LANGUAGES = {
        "v1": ["auto", "en", "zh", "ja", "all_zh", "all_ja"],
        "v2": ["auto", "auto_yue", "en", "zh", "ja", "yue", "ko", "all_zh", "all_ja", "all_yue", "all_ko"]
    }
    
    # Fallback model files when downloading from repository
    FALLBACK_FILES = {
        "t2s_weights_path": {"filename": "models/T2S/txdb-e15.ckpt"},
        "vits_weights_path": {"filename": "models/VITS/txdb_e12_s204.pth"},
        "bert_base_path": {"folder_path": "models/BERT/chinese-roberta-wwm-ext-large"},
        "cnhuhbert_base_path": {"folder_path": "models/HuBERT/chinese-hubert-base"}
    }
    
    def __init__(self, version: str = "v2"):
        """
        Initialize the TTS configuration.
        
        Args:
            version: Model version - one of "v1", "v2", or "v3"
        """
        # Validate version
        if version not in self.DEFAULT_CONFIGS:
            raise ValueError(f"Unsupported version: {version}, must be one of {list(self.DEFAULT_CONFIGS.keys())}")
        
        # Set version and basic config from defaults
        self.version = version
        self.configs = deepcopy(self.DEFAULT_CONFIGS[version])
        
        # Set device configuration
        self._setup_device()
        
        # Set paths
        self._setup_paths()
        
        # Setup language support
        self.languages = self.LANGUAGES["v1"] if self.version == "v1" else self.LANGUAGES["v2"]
        
        # Setup synthesizer flags
        self.is_v3_synthesizer = False
        
        # Set audio processing parameters
        self._setup_audio_params()
        
    def _setup_device(self):
        """Configure device and precision settings"""
        self.device = self.configs.get("device", torch.device("cpu"))
        if "cuda" in str(self.device) and not torch.cuda.is_available():
            print("Warning: CUDA is not available, setting device to CPU.")
            self.device = torch.device("cpu")
            
        self.is_half = self.configs.get("is_half", False)
        
    def _setup_paths(self):
        """Set up and validate model paths"""
        self.t2s_weights_path = self._get_valid_path("t2s_weights_path")
        self.vits_weights_path = self._get_valid_path("vits_weights_path")
        self.bert_base_path = self._get_valid_path("bert_base_path")
        self.cnhuhbert_base_path = self._get_valid_path("cnhuhbert_base_path")
        
        # Update configs with validated paths
        self.update_configs()
        
    def _get_valid_path(self, path_key):
        """
        Get a valid path for a model file or download if not available
        
        Args:
            path_key: The key for the model path
            
        Returns:
            Valid path to the model file
        """
        path = self.configs.get(path_key)
        
        # Check if path is valid
        if path in [None, ""] or not os.path.exists(path):
            # Download model if needed
            if "filename" in self.FALLBACK_FILES[path_key]:
                path = hf_hub_download(
                    repo_id=self.REPO_ID,
                    filename=self.FALLBACK_FILES[path_key]["filename"],
                    revision=self.REPO_REVISION
                )
            else:
                path = download_folder_from_repo(
                    repo_id=self.REPO_ID,
                    folder_path=self.FALLBACK_FILES[path_key]["folder_path"]
                )
            print(f"Downloaded {path_key}: {path}")
            
        return path
        
    def _setup_audio_params(self):
        """Set up audio processing parameters with defaults"""
        self.max_sec = None
        self.hz = 50
        self.semantic_frame_rate = "25hz"
        self.segment_size = 20480
        self.filter_length = 2048
        self.sampling_rate = 32000
        self.hop_length = 640
        self.win_length = 2048
        self.n_speakers = 300
        
    def update_configs(self):
        """
        Update the configuration dictionary with current settings
        
        Returns:
            Updated configuration dictionary
        """
        self.config = {
            "device": str(self.device),
            "is_half": self.is_half,
            "version": self.version,
            "t2s_weights_path": self.t2s_weights_path,
            "vits_weights_path": self.vits_weights_path,
            "bert_base_path": self.bert_base_path,
            "cnhuhbert_base_path": self.cnhuhbert_base_path,
        }
        return self.config

    def update_version(self, version: str) -> None:
        """
        Update the configuration version
        
        Args:
            version: New version to use
        """
        if version not in self.DEFAULT_CONFIGS:
            raise ValueError(f"Unsupported version: {version}, must be one of {list(self.DEFAULT_CONFIGS.keys())}")
            
        self.version = version
        self.languages = self.LANGUAGES["v1"] if version == "v1" else self.LANGUAGES["v2"]

    def __str__(self):
        """String representation showing the configuration"""
        config = self.update_configs()
        string = "TTS Configuration".center(100, '-') + '\n'
        for k, v in config.items():
            string += f"{str(k).ljust(20)}: {str(v)}\n"
        string += "-" * 100 + '\n'
        return string

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        # Use consistent attribute for hash
        return hash((self.version, self.device, self.is_half))

    def __eq__(self, other):
        if not isinstance(other, TTSConfiguration):
            return False
        return (self.version == other.version and
                str(self.device) == str(other.device) and
                self.is_half == other.is_half)


class MPipeline:
    def __init__(self, version: str = "v2"):
        """
        Initialize the TTS pipeline with all required models
        
        Sets up the configuration and loads all necessary models for text-to-speech synthesis
        with voice cloning capabilities.
        
        Args:
            version: Model version to use ("v1", "v2", or "v3")
        
        Raises:
            ValueError: If an invalid version is provided
            FileNotFoundError: If required model files cannot be found
        """
        # Initialize configuration
        self.configs = TTSConfiguration(version)
        
        # Initialize model placeholders
        self.t2s_model = None
        self.vits_model = None
        self.bert_tokenizer = None
        self.bert_model = None 
        self.cnhuhbert_model = None
        self.sr_model = None
        self.sr_model_not_exist = False
        
        # Set inference parameters
        self.stop_flag = False
        self.precision = torch.float16 if self.configs.is_half else torch.float32
        
        # Load all models
        print("Initializing TTS pipeline models...")
        
        # Load core models in order of dependency
        self.init_bert_weights(self.configs.bert_base_path)
        self.init_cnhuhbert_weights(self.configs.cnhuhbert_base_path)
        self.init_t2s_weights(self.configs.t2s_weights_path)
        self.init_vits_weights(self.configs.vits_weights_path)
        
        # Initialize text preprocessor with loaded models
        self.text_preprocessor = TextPreprocessor(
            self.bert_model,
            self.bert_tokenizer,
            self.configs.device,
            self.precision
        )
        
        # Initialize reference processor with loaded models
        self.reference_processor = ReferenceProcessor(
            self.text_preprocessor,
            self.cnhuhbert_model, 
            self.vits_model, 
            self.configs.device, 
            self.configs
        )
        
        # Initialize empty prompt cache
        self.prompt_cache = {
            "ref_audio_path": None,
            "prompt_semantic": None,
            "refer_spec": [],
            "prompt_text": None,
            "prompt_lang": None,
            "phones": None,
            "bert_features": None,
            "norm_text": None,
            "aux_ref_audio_paths": [],
        }
        
        print(f"Pipeline initialized with {version} models on {self.configs.device}")    
    
    
    def init_t2s_weights(self, weights_path: str) -> None:
        """
        Initialize Text-to-Semantic model weights from checkpoint file
        
        This function loads Text2Semantic model weights from the specified path,
        configures the model parameters based on the checkpoint, and sets up the
        model on the appropriate device with proper precision settings.
        
        Args:
            weights_path: Path to the Text2Semantic model checkpoint file
            
        Raises:
            FileNotFoundError: If the weights file doesn't exist
            RuntimeError: If there's an issue loading the model weights
        """
        # Validate input path
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Text2Semantic weights not found at: {weights_path}")
            
        print(f"Loading Text2Semantic weights from {weights_path}")
        self.configs.t2s_weights_path = weights_path
        
        try:
            # Load checkpoint with appropriate device mapping
            dict_s1 = torch.load(
                weights_path, 
                map_location=self.configs.device, 
                weights_only=False
            )
            
            # Extract configuration and update settings
            config = dict_s1["config"]
            if not config:
                raise RuntimeError("Invalid checkpoint: missing configuration data")
                
            # Update configuration parameters
            self.configs.hz = 50  # Fixed semantic token rate
            self.configs.max_sec = config["data"]["max_sec"]
            
            # Initialize model with loaded configuration
            t2s_model = Text2SemanticLightningModule(config, "****", is_train=False)
            
            # Load model weights
            load_result = t2s_model.load_state_dict(dict_s1["weight"])
            if load_result and hasattr(load_result, "missing_keys") and load_result.missing_keys:
                print(f"Warning: Missing keys when loading T2S model: {load_result.missing_keys}")
            
            # Move model to device and set evaluation mode
            t2s_model = t2s_model.to(self.configs.device)
            t2s_model.eval()
            
            # Apply half-precision if requested (for GPU only)
            if self.configs.is_half and str(self.configs.device) != "cpu":
                t2s_model = t2s_model.half()
                
            # Store model in instance
            self.t2s_model = t2s_model
        
        except Exception as e:
            self.t2s_model = None
            raise RuntimeError(f"Failed to load Text2Semantic model: {str(e)}") from e
        
    def init_vits_weights(self, weights_path: str) -> None:
        """
        Initialize VITS model weights from checkpoint file
        
        This function loads SoVITS model weights from the specified path,
        determines the model version, configures audio parameters, and sets up
        the appropriate synthesizer model with the weights.
        
        Args:
            weights_path: Path to the SoVITS model checkpoint file
            
        Raises:
            FileNotFoundError: If weights file or required base model is missing
            RuntimeError: If there's an issue loading the model weights
        """
        # Validate path exists
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"VITS weights not found at: {weights_path}")
            
        print(f"Loading VITS weights from {weights_path}")
        self.configs.vits_weights_path = weights_path
        
        try:
            # Get model version and LoRA status
            version, model_version, is_lora_v3 = get_sovits_version_from_path_fast(weights_path)

            # For LoRA models, verify base model exists
            path_sovits_v3 = self.configs.DEFAULT_CONFIGS["v3"]["vits_weights_path"]
            if is_lora_v3 and not os.path.exists(path_sovits_v3):
                raise FileNotFoundError(
                    f"{path_sovits_v3} - {i18n('SoVITS V3 底模缺失，无法加载相应 LoRA 权重')}"
                )
                
            # Load model weights
            dict_s2 = load_sovits_new(weights_path)
            hps = dict_s2["config"]
            
            # Set semantic frame rate
            hps["model"]["semantic_frame_rate"] = "25hz"
            
            # Determine model version from weights
            if 'enc_p.text_embedding.weight' not in dict_s2['weight']:
                hps["model"]["version"] = "v2"  # v3 model, v2 symbols
            elif dict_s2['weight']['enc_p.text_embedding.weight'].shape[0] == 322:
                hps["model"]["version"] = "v1"
            else:
                hps["model"]["version"] = "v2"
                
            # Update configuration with model parameters
            self._update_audio_config_from_hps(hps)
            self.configs.update_version(model_version)

            kwargs = hps["model"]
            
            print(f"model_version:{model_version}")
            # Initialize appropriate model based on version
            if model_version != "v3":
                """Create a V2 SynthesizerTrn model"""
                self.configs.is_v3_synthesizer = False
                vits_model = SynthesizerTrn(
                    self.configs.filter_length // 2 + 1,
                    self.configs.segment_size // self.configs.hop_length,
                    n_speakers=self.configs.n_speakers,
                    **kwargs
                )
            else:
                vits_model = self._create_v3_model(hps["model"], weights_path)
                """Create a V3 SynthesizerTrnV3 model"""
                self.configs.is_v3_synthesizer = True
                vits_model = SynthesizerTrnV3(
                    self.configs.filter_length // 2 + 1,
                    self.configs.segment_size // self.configs.hop_length,
                    n_speakers=self.configs.n_speakers,
                    **kwargs
                )                
                self.init_bigvgan()
                if "pretrained" not in weights_path and hasattr(vits_model, "enc_q"):
                    del vits_model.enc_q
                
            # Load weights into model
            self._load_weights_into_model(vits_model, dict_s2, is_lora_v3, path_sovits_v3)
            
            # Finalize model setup
            vits_model = vits_model.to(self.configs.device).eval()
            self.vits_model = vits_model
            
            # Apply half precision if needed
            if self.configs.is_half and str(self.configs.device) != "cpu":
                self.vits_model = self.vits_model.half()
                
        except Exception as e:
            self.vits_model = None
            raise RuntimeError(f"Failed to load VITS model: {str(e)}") from e
            
    def _update_audio_config_from_hps(self, hps):
        """Update audio configuration parameters from model hyperparameters"""
        self.configs.filter_length = hps["data"]["filter_length"]
        self.configs.segment_size = hps["train"]["segment_size"]
        self.configs.sampling_rate = hps["data"]["sampling_rate"]
        self.configs.hop_length = hps["data"]["hop_length"]
        self.configs.win_length = hps["data"]["win_length"]
        self.configs.n_speakers = hps["data"]["n_speakers"]
        self.configs.semantic_frame_rate = hps["model"]["semantic_frame_rate"]

    def _load_weights_into_model(self, vits_model, dict_s2, is_lora_v3, path_sovits_v3):
        """Load weights into the model, handling LoRA if necessary"""
        if not is_lora_v3:
            # Standard weights loading
            result = vits_model.load_state_dict(dict_s2['weight'], strict=False)
            # print(f"Loaded VITS weights with result: {result}")
        else:
            # LoRA weights loading
            # First load base model
            base_weights = load_sovits_new(path_sovits_v3)['weight']
            result = vits_model.load_state_dict(base_weights, strict=False)
            print(f"Loaded VITS base model with result: {result}")
            
            # Then apply LoRA
            lora_rank = dict_s2["lora_rank"]
            lora_config = LoraConfig(
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],
                r=lora_rank,
                lora_alpha=lora_rank,
                init_lora_weights=True,
            )
            
            # Apply LoRA to the CFM module
            vits_model.cfm = get_peft_model(vits_model.cfm, lora_config)
            result = vits_model.load_state_dict(dict_s2['weight'], strict=False)
            print(f"Applied LoRA weights with result: {result}")
            
            # Merge LoRA weights back into the model
            vits_model.cfm = vits_model.cfm.merge_and_unload()

    def init_bert_weights(self, base_path: str) -> None:
        """
        Initialize BERT model weights and tokenizer
        
        This function loads a pre-trained BERT model and tokenizer from the specified path,
        configures them for inference, and sets them up on the appropriate device with
        proper precision settings.
        
        Args:
            base_path: Path to the pre-trained BERT model directory
            
        Raises:
            FileNotFoundError: If the BERT model directory doesn't exist
            RuntimeError: If there's an issue loading the model or tokenizer
        """
        # Validate path exists
        if not os.path.exists(base_path):
            raise FileNotFoundError(f"BERT model not found at: {base_path}")
            
        print(f"Loading BERT weights from {base_path}")
        
        try:
            # Load tokenizer with optimized settings
            self.bert_tokenizer = AutoTokenizer.from_pretrained(base_path)
            
            # Load model with optimized settings for inference
            self.bert_model = AutoModelForMaskedLM.from_pretrained(base_path)
            
            # Set model to evaluation mode
            self.bert_model = self.bert_model.eval()
            
            # Move model to appropriate device
            self.bert_model = self.bert_model.to(self.configs.device)
            
            # Apply half precision if requested (for GPU only)
            if self.configs.is_half and str(self.configs.device) != "cpu":
                self.bert_model = self.bert_model.half()
                
        except Exception as e:
            # Clean up resources in case of failure
            self.bert_model = None
            self.bert_tokenizer = None
            raise RuntimeError(f"Failed to load BERT model: {str(e)}") from e
        
    def init_cnhuhbert_weights(self, base_path: str) -> None:
        """
        Initialize CNHuBERT model weights for language feature extraction
        
        This function loads the CNHuBERT model from the specified path and sets it up
        for inference on the appropriate device with the correct precision settings.
        CNHuBERT is used for extracting acoustic features from reference audio.
        
        Args:
            base_path: Path to the CNHuBERT model directory
            
        Raises:
            FileNotFoundError: If the model directory doesn't exist
            RuntimeError: If there's an issue loading the model
        """
        # Validate path exists
        if not os.path.exists(base_path):
            raise FileNotFoundError(f"CNHuBERT model not found at: {base_path}")
            
        print(f"Loading CNHuBERT weights from {base_path}")
        
        try:
            # Initialize model
            self.cnhuhbert_model = CNHubert(base_path)
            
            # Set model to evaluation mode
            self.cnhuhbert_model = self.cnhuhbert_model.eval()
            
            # Move model to appropriate device
            self.cnhuhbert_model = self.cnhuhbert_model.to(self.configs.device)
            
            # Apply half precision if requested (for GPU only)
            if self.configs.is_half and str(self.configs.device) != "cpu":
                self.cnhuhbert_model = self.cnhuhbert_model.half()
                
        except Exception as e:
            # Clean up resources in case of failure
            self.cnhuhbert_model = None
            raise RuntimeError(f"Failed to load CNHuBERT model: {str(e)}") from e
    
    def stop(self,):
        '''
        Stop the inference process.
        '''
        self.stop_flag = True

    def empty_cache(self):
        try:
            gc.collect() # 触发gc的垃圾回收。避免内存一直增长。
            if "cuda" in str(self.configs.device):
                torch.cuda.empty_cache()
            elif str(self.configs.device) == "mps":
                torch.mps.empty_cache()
        except:
            pass


    """ This decorator is used to optimize the performance of the inference function by disabling gradient calculation during inference. 
    When using this decorator: 
        1. Forward pass operations don't track gradients 
        2. Memory usage is reduced significantly 
        3. Computation is faster since PyTorch doesn't build the computational graph needed for backpropagation
    """
    @torch.no_grad()
    def __call__(self, 
        text: str,   # input text
        text_language: str,  # select "en", "all_zh", "all_ja"
        ref_audio_path: str = None,  # reference audio path          
        ref_text: str = None,     # reference text
        ref_language: str = "all_zh",  # reference text language
        aux_ref_audio_paths: list = [],
        batch_size: int = 100,             # inference batch size
        speed_factor: float = 1.0, # control speed of output audio
        top_k: int = 5,
        top_p: float = 1,
        temperature: float = 1,
        text_split_method: str = "", 
        split_bucket: bool = True,
        return_fragment: bool = False,
        fragment_interval: float = 0.07,   # interval between every sentence
        seed: int = 233333,
        repetition_penalty: float = 1.35,  # repetition penalty for T2S model
        sample_steps: int = 32,            # number of sampling steps for VITS model V3
        super_sampling: bool = False,       # whether to use super-sampling for audio
        parallel_infer: bool = True,       # whether to use parallel inference
        batch_threshold: float = 0.75     # threshold for batch splitting
        ):
        """
        Text to speech synthesis with voice cloning
        
        Args:
            text: Text to be synthesized
            text_language: Language of the text ("en", "all_zh", "all_ja", etc.)
            ref_audio_path: Path to reference audio for voice cloning
            ref_text: Optional text corresponding to reference audio
            ref_language: Language of reference text
            aux_ref_audio_paths: Additional reference audio paths for voice fusion
            batch_size: Inference batch size
            speed_factor: Control speed of output audio
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            temperature: Temperature for sampling
            text_split_method: Method for text segmentation (if empty, will be determined by language)
            split_bucket: Whether to use similar-length text bucketing
            return_fragment: Whether to return audio fragments sequentially
            fragment_interval: Interval between fragments in seconds
            seed: Random seed for reproducibility
            repetition_penalty: Penalty for repeated tokens in semantic generation
            sample_steps: Sampling steps for V3 model generation
            super_sampling: Whether to use super-sampling for V3 models
            parallel_infer: Whether to use parallel inference
            batch_threshold: Similarity threshold for batch splitting
            
        Yields:
            Tuple[int, np.ndarray]: Sample rate and audio data
        """
        # Initialize stop flag
        self.stop_flag = False

        # If no reference audio is provided, use default
        if ref_audio_path in [None, ""]:
            # Use installed package path if available
            default_audio_path = get_reference_path("dataset/doubao-ref-ours.wav")
            default_text_path = get_reference_path("dataset/doubao-ref.txt")            
            ref_audio_path = default_audio_path
            ref_text = load_text_file(default_text_path)
            ref_language = "all_zh"
            print(f"Warning: No reference audio provided, using default reference audio: {ref_audio_path}")

        # Set text_split_method based on language if not provided
        text_split_method = get_text_split_method(text_language, text_split_method)
        
        # Set random seed for reproducibility
        actual_seed = seed if seed not in [-1, "", None] else random.randrange(1 << 32)
        set_seed(actual_seed)
        
        # Validate language
        assert text_language in self.configs.languages

        # Configure inference mode
        if parallel_infer:
            print(i18n("并行推理模式已开启"))
            self.t2s_model.model.infer_panel = self.t2s_model.model.infer_panel_batch_infer
        else:
            print(i18n("并行推理模式已关闭"))
            self.t2s_model.model.infer_panel = self.t2s_model.model.infer_panel_naive_batched

        # Configure fragment mode
        if return_fragment:
            print(i18n("分段返回模式已开启"))
            if split_bucket:
                split_bucket = False
                print(i18n("分段返回模式不支持分桶处理，已自动关闭分桶处理"))

        # Configure bucketing
        if split_bucket and speed_factor == 1.0 and not (self.configs.is_v3_synthesizer and parallel_infer):
            print(i18n("分桶处理模式已开启"))
        elif speed_factor != 1.0:
            print(i18n("语速调节不支持分桶处理，已自动关闭分桶处理"))
            split_bucket = False
        elif self.configs.is_v3_synthesizer and parallel_infer:
            print(i18n("当开启并行推理模式时，SoVits V3模型不支持分桶处理，已自动关闭分桶处理"))
            split_bucket = False
        else:
            print(i18n("分桶处理模式已关闭"))

        # Validate fragment interval
        if fragment_interval < 0.01:
            fragment_interval = 0.01
            print(i18n("分段间隔过小，已自动设置为0.01"))
        
        try:
            # Process reference audio and text
            print("############ Reference Audio/Text Processing ############")
            t_ref_start = time.perf_counter()
            
            # Check if reference audio path is provided            
            no_ref_text = ref_text in [None, ""]
            if not no_ref_text:
                assert ref_language in self.configs.languages
            if no_ref_text and self.configs.is_v3_synthesizer:
                raise NO_PROMPT_ERROR("ref_text cannot be empty when using SoVITS_V3")
            
            # Process reference audio and prompt text
            self.prompt_cache = self.reference_processor.process_reference(
                ref_audio_path=ref_audio_path,
                prompt_text=ref_text,
                prompt_lang=ref_language,
                model_version=self.configs.version,
                aux_ref_audio_paths=aux_ref_audio_paths
            )
            
            t_ref_end = time.perf_counter()
            t_reference = t_ref_end - t_ref_start
            print(f"Reference audio processing time: {t_reference:.3f} seconds")

            # Text preprocessing
            print("############ Text Preprocessing ############")
            t_text_start = time.perf_counter()
            
            data = None
            batch_index_list = None
            
            if not return_fragment:
                # Process complete text
                data = self.text_preprocessor.process(text, text_language, text_split_method, self.configs.version)
                if len(data) == 0:
                    yield 16000, np.zeros(int(16000), dtype=np.int16)
                    return

                # Create inference batches
                data, batch_index_list = self.text_preprocessor.create_inference_batches(
                    data,
                    prompt_data=self.prompt_cache if not no_ref_text else None,
                    batch_size=batch_size,
                    similarity_threshold=batch_threshold,
                    split_bucket=split_bucket
                )
            else:
                # Prepare text for fragment mode
                print(f'############ {i18n("切分文本")} ############')
                texts = self.text_preprocessor.pre_seg_text(text, text_language, text_split_method)
                data = []
                for i in range(len(texts)):
                    if i % batch_size == 0:
                        data.append([])
                    data[-1].append(texts[i])
            
            t_text_end = time.perf_counter()
            print(f"Text preprocessing time: {t_text_end - t_text_start:.3f} seconds")

            # semantic and synthesis
            print("############ 推理 ############")
            t_semantic = 0.0
            t_synthesis = 0.0
            audio = []
            output_sr = self.configs.sampling_rate if not self.configs.is_v3_synthesizer else 24000

            # Import synthesizer components
            from .audio_synthesis.synthesizer_factory import create_synthesizer
            from .audio_synthesis.audio_processor import AudioProcessor
        
            # Create synthesizer and audio processor
            synthesizer = create_synthesizer(self.vits_model, self.configs, self.prompt_cache)
            audio_processor = AudioProcessor(self.configs, self.sr_model)

            # Process each batch
            for item in data:
                t_sem_start = time.perf_counter()
                
                # Process text fragments in fragment mode
                if return_fragment:
                    item = self.text_preprocessor.process_text_fragments(
                        item,
                        text_language,
                        batch_size,
                        batch_threshold,
                        version=self.configs.version,
                        no_prompt_text=no_ref_text
                    )
                    if item is None:
                        continue

                # Extract batch data
                batch_phones = item["phones"]
                batch_phones_len = item["phones_len"]
                all_phoneme_ids = item["all_phones"]
                all_phoneme_lens = item["all_phones_len"]
                all_bert_features = item["all_bert_features"]
                norm_text = item["norm_text"]
                max_len = item["max_len"]

                print(i18n("前端处理后的文本(每句):"), norm_text)
                
                # Prepare prompt for semantic generation
                if no_ref_text:
                    prompt = None
                else:
                    prompt = self.prompt_cache["prompt_semantic"].expand(len(all_phoneme_ids), -1).to(self.configs.device)

                # Generate semantic tokens
                print(f"############ {i18n('预测语义Token')} ############")
                pred_semantic_list, idx_list = self.t2s_model.model.infer_panel(
                    all_phoneme_ids,
                    all_phoneme_lens,
                    prompt,
                    all_bert_features,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    early_stop_num=self.configs.hz * self.configs.max_sec,
                    max_len=max_len,
                    repetition_penalty=repetition_penalty,
                )
                
                t_sem_end = time.perf_counter()
                t_semantic += t_sem_end - t_sem_start

                # Prepare reference spectrogram for synthesis
                refer_audio_spec = [
                    item.to(dtype=self.precision, device=self.configs.device) 
                    for item in self.prompt_cache["refer_spec"]
                ]
                
                # Synthesize audio
                t_synth_start = time.perf_counter()
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

                t_synth_end = time.perf_counter()
                t_synthesis += t_synth_end - t_synth_start
                
                # Process audio output
                if return_fragment:
                    # Return audio fragments individually
                    print(f"Time breakdown: Ref={t_reference:.3f}s, Text={t_text_end - t_text_start:.3f}s, " +
                        f"Semantic={t_semantic:.3f}s, Synthesis={t_synthesis:.3f}s")
                        
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
                    # Collect audio for complete processing
                    audio.append(batch_audio_fragment)

                # Check if process should be stopped
                if self.stop_flag:
                    yield 16000, np.zeros(int(16000), dtype=np.int16)
                    return

            # Process complete audio in non-fragment mode
            if not return_fragment:
                print(f"Time breakdown: Ref={t_reference:.3f}s, Text={t_text_end - t_text_start:.3f}s, " +
                    f"Semantic={t_semantic:.3f}s, Synthesis={t_synthesis:.3f}s")
                    
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
            # Return empty audio to prevent memory leak
            yield 16000, np.zeros(int(16000), dtype=np.int16)
            # Reset models to avoid incomplete memory cleanup
            del self.t2s_model
            del self.vits_model
            self.t2s_model = None
            self.vits_model = None
            raise e
            
        finally:
            # Clean up resources
            self.empty_cache()
