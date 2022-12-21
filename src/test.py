import os
import pandas as pd
import pickle
import numpy as np
from PIL import Image


with open('../data/yahoo_shop/df.pickle', 'rb') as f:
    df = pickle.load(f)

label = list(df['label'])

print(label)
with open('../data/yahoo_shop/label.pickle', 'wb') as f:
    pickle.dump(label, f)