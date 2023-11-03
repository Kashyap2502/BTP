from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import torchvision.datasets as datasets
import torchvision
from torch.utils.data import DataLoader
import PIL.Image as Image
import cv2
import os
import glob
import numpy as np


data=torch.load('data.pt')
faces,names=data[0],data[1]
mtcnn=MTCNN()
res=InceptionResnetV1(pretrained='vggface2').eval()
# res=torch.load("flayer_model.pt",map_location=torch.device('cpu'))
device=torch.device("cuda") if torch.cuda.is_available() else "cpu"
res=res.to(device)
device=torch.device("cuda") if torch.cuda.is_available() else "cpu"
class Facerecognition:
    def __init__(self):
        
        self.encodings = faces
        self.names = names

  
        self.frame_resizing = 1

    def load(self, images_path):
       
        
        images_path = glob.glob(os.path.join(images_path, "*.*"))
        
        device=torch.device("cuda") if torch.cuda.is_available() else "cpu"

     
        for img_path in images_path:
            
            img = cv2.imread(img_path)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

           
            basename = os.path.basename(img_path)
            (filename, ext) = os.path.splitext(basename)
            print(filename)
            print(type(res))
            img,p=mtcnn(rgb_img,return_prob=True)
            if type(img) != type(None):
                img=img.to(device)
                img_1=res(img.unsqueeze(0))
                if len(self.encodings)>=1:
                    dist_list = [] 
    
                    for idx, emb in enumerate(self.encodings):
                        dist = torch.dist(img_1, emb).item()
                        dist_list.append(dist)
                    idx_min = dist_list.index(min(dist_list))
                    if min(dist_list)<0.2:
                        print(f"This face is already recorded as {self.names[idx_min]} no need of {filename}")
                    else:
                        self.encodings.append(img_1.detach())
                        self.names.append(filename)
                else:
                    self.encodings.append(img_1.detach())
                    self.names.append(filename)
            else:
                print(f"Unable to detect the {filename}")
        data=[self.encodings,self.names]
        torch.save(data,'data.pt')

             