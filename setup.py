from setuptools import setup, find_packages

setup(
    name="Ominix-TTS",  # Package name (use hyphens, not underscores)
    version="0.1.0",  # Initial version
    packages=find_packages(),  # Automatically find your_package/
    install_requires=[  # List dependencies        
        "numpy>=1.21.0",
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
