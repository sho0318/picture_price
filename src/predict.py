import torch 
import pickle
from model import Net
import os
from PIL import Image
import numpy as np

model = Net()
model.load_state_dict(torch.load('export_model.pth'))
model = model.eval()

with open('../data/label/price_normalize.pickle', 'rb') as f:
    prices_normalize = pickle.load(f)
    
label_mean_std = prices_normalize['mean_std']
mean = label_mean_std[0]
std = label_mean_std[1]

for input_fig in os.listdir('../data/predict'):
    image_path = "../data/predict/{}".format(input_fig)
    im = Image.open(image_path)
    
    resize_h = 400
    resize_w = 400
    im = im.resize([resize_h, resize_w])
    
    im = np.array(im).astype(np.float32).transpose(2,1,0)
    im = torch.from_numpy(im.astype(np.float32))
    print(input_fig)   
    output = model(im)
    # print(output)
    output = output*std + mean

    print(output)

a = np.random.randint(0, 10, (3, 400, 400))
print(a)
a = torch.from_numpy(a.astype(np.float32))

print(model(a)*std+mean)
