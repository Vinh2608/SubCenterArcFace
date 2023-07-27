from typing import Any
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

class Align_Detect_Face:
    def __init__(self) -> None:
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    def __call__(self, img_path):
        mtcnn = MTCNN(image_size=112, margin=0, min_face_size=20,
            thresholds=[0.5, 0.6, 0.6], factor=0.709, post_process=True,
            device=self.device)
        img = Image.open(img_path)
        align_img = mtcnn(img)
        if align_img == None:
          return None
        else:
          return align_img