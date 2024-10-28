FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

RUN pip install jupyter pandas matplotlib numpy tqdm joblib
RUN pip install pydicom 
RUN pip install dicomsdl 
RUN pip install opencv-python pillow scipy
RUN pip install timm timm-3d
RUN pip install torchmetrics pycocotools
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y