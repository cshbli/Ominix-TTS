from setuptools import setup, find_packages

setup(
    name="ominix-tts",  # Package name (use hyphens, not underscores)
    version="0.1.0",  # Initial version
    packages=find_packages(),  # Automatically find your_package/
    install_requires=[  # List dependencies                
        "ffmpeg-python",
        "huggingface_hub>=0.13",
        "librosa>=0.9.2",
        "numpy>=1.23.4",
        "torchaudio",
        "tqdm",
        "transformers>=4.43",
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
