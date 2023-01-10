import os
import pandas as pd
import pickle
import numpy as np
from PIL import Image
import re

with open('../data/this_is_gallery/preprocess_df.pickle', 'rb') as f:
    df = pickle.load(f)

df_src = (df.loc[:10030]).reset_index(drop=True)
df_label = (df.loc[10030:]).reset_index(drop=True)


print(df_src)
print(df_label)

df = pd.concat([df_src, df_label], axis=1, ignore_index=True)
df.columns = ['a','b']
print(df)