import torch 
import pickle
from model import Net
import os
from PIL import Image
import numpy as np

model = Net()
model.load_state_dict(torch.load('export_model.pth'))
model = model.eval()

with open('../data/this_is_gallery/normalize_para.pickle', 'rb') as f:
    prices_normalize = pickle.load(f)
    
mean = prices_normalize[0]
std = prices_normalize[1]

for input_fig in os.listdir('../data/predict'):
    image_path = "../data/predict/{}".format(input_fig)
    im = Image.open(image_path)
    
    resize_h = 300
    resize_w = 300
    im = im.resize([resize_h, resize_w])
    
    im = np.array(im).astype(np.float32).transpose(2,1,0)
    im = torch.from_numpy(im.astype(np.float32))
    # print(input_fig)   
    with torch.no_grad():
        output = model(im)
        output = output*std + mean
    print(input_fig)
    print(output)

a = np.random.randint(0, 10, (3, 300, 300))
a = torch.from_numpy(a.astype(np.float32))

print(model(a)*std+mean)
print(std)
print(mean)
