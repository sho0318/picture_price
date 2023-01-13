import os
import pandas as pd
import pickle
import numpy as np
from PIL import Image
import re

with open('../data/this_is_gallery/preprocess_df.pickle', 'rb') as f:
    df = pickle.load(f)

print(df)