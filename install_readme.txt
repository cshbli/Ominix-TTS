环境设置
conda create -n TTS python=3.9
conda activate TTS

安装依赖
方法一
bash install.sh
方法二
pip install -r requirements.txt
pip install git+https://github.com/r9y9/pyopenjtalk.git@fix-cmake4
下载weights
https://drive.google.com/drive/folders/1yKs1B5CHgGB0XBDegAlLa7MFjy4emOCh?usp=sharing
从链接下载weights
pretrained_model放在MOTTS/pretrained_models
G2PWModel放在MOTTS/text/G2PWModel
uvr5_weights放在tools/uvr5/uvr5_weights
使用
INFERENCE:python MOTTS/inference.py