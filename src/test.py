import os
import pandas as pd
import pickle
import numpy as np
from PIL import Image

image_path = "../data/reshape_fig/0.jpg"
image = Image.open(image_path)

image = np.array(image).astype(np.float32).transpose(2,1,0)
image = image/255
print(image)