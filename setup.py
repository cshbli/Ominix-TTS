from setuptools import setup, find_packages

setup(
    name="ominix-tts",  # Package name (use hyphens, not underscores)
    version="0.1.0",  # Initial version
    packages=find_packages(),  # Automatically find your_package/
    install_requires=[  # List dependencies                
        "cn2an",
        "fast_langdetect>=0.3.0",
        "ffmpeg-python",
        "g2p_en",
        "gradio>=4.0,<=4.24.0",
        "huggingface_hub>=0.13",
        "jieba",
        "jieba_fast",
        "librosa>=0.9.2",
        "matplotlib",
        "numpy>=1.23.4,<2.0.0",
        "peft",
        "pypinyin",
        "pytorch-lightning>2.0",
        "split-lang",
        "torchaudio",
        "tqdm",
        "transformers>=4.43",
        "wordsegment",
        "x_transformers",
    ],
    author="Hongbing Li",
    author_email="cshbli@hotmail.com",
    description="A short description of your package",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/cshbli/Ominix-TTS",  # GitHub URL
    classifiers=[  # Metadata for PyPI
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",  # Minimum Python version
)
