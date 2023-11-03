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

p1="Facerecognition"
data=torch.load(os.path.join(p1,'data.pt'))
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
    def detect(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
       
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations=mtcnn.detect(rgb_small_frame)
        
        if type(face_locations[0])!=type(None):
           
            face = mtcnn(rgb_small_frame)
            face.to(device)
            face_1 =res(face.unsqueeze(0))
            face_locations=face_locations[0][0]
            face_names = []
            face_acc=[]
            if len(self.known_face_encodings)==0:
                return "Unknown"
            else:
                dist_list=[]
                for idx, emb in enumerate(self.known_face_encodings):
                    dist = torch.dist(face_1, emb).item()
                    dist_list.append(dist)
                idx_min = dist_list.index(min(dist_list))
                if min(dist_list)<0.8:
                    return self.known_face_names[idx_min]
                else:
                    return "Unknown"
        else:
            return "Unknown"
       