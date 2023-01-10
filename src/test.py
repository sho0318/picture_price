import os
import pandas as pd
import pickle
import numpy as np
from PIL import Image
import re

with open('../data/this_is_gallery/df.pickle', 'rb') as f:
    df = pickle.load(f)

labels = df['price']
labels = labels.astype(int)
labels = labels
tmp = labels.mean() 
tmp = labels.std()
print(labels)
print(tmp)