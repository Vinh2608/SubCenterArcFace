import faiss
from typing import Any
import matplotlib.pyplot as plt
from models import *
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training
from PIL import Image
import tqdm
import numpy as np
from torchvision import transforms
from align_detect_face import Align_Detect_Face

img_list = []
with open("label_test.txt", "r") as file:
   lines = file.readlines()
   for line in lines:
      img_list.append('VN-celeb_align_frontal_full/' + line.split()[0])

img_dict = dict(enumerate(img_list))
mean=[0.60746885, 0.47471561, 0.41313071]
std = [0.2621818,0.23118107,0.2242216]

class Model:
    def __init__(self, name, checkpoint, fp16=False,load_checkpoint=True):
        self.model = get_model(name)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if load_checkpoint:
          self.model.load_state_dict(checkpoint['model_state_dict'])
    def __call__(self, img):
        if img == None:
          return None
        else: 
           return self.model(img)
    def eval(self):
        self.model.eval()

# It takes a retrieval function and a database of images, and returns a faiss index
class my_faiss():
    def __call__(self, model,DB):
        num_index = 512
        self.index = faiss.IndexFlatL2(num_index)
        for img_index, img in DB.items():
        # try:
            align_detect_face = Align_Detect_Face()
            img = align_detect_face(img)
            if img == None:
              continue
            else:
              img = img.unsqueeze(0)
              embedded = model(img).detach().numpy()
              self.index.add(embedded)
            # except Exception as e:
            #   print('error: ', e)
            #   continue
        return self.index

checkpoint = torch.load('/content/SubCenterArcFace/iresnet34_model_s=30_m=0.5_0.023411080241203308_99_acc666_subcenter_train_k1.pt')

model = Model("r34", checkpoint,False,True)
model.eval()
faiss_search = my_faiss()
faiss_index = faiss_search(model, img_dict)

faiss.write_index(faiss_index, 'faiss_normal.bin')
#### load Index by bin file ####
normal_idx = faiss.read_index('faiss_normal.bin')

torch.save('img_list':img_list, 'image_list/')