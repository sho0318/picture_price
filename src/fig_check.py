import glob
import os
import pickle

from PIL import Image

with open('../data/this_is_gallery/preprocess_df.pickle', 'rb') as f:
    df = pickle.load(f)

for path  in glob.glob('../data/this_is_gallery/fig/*'):
    try:
        image = Image.open(path).convert('RGB')
    except:
        df = df[df['paths'] != path]
        print(path)

df = df.reset_index(drop=True)

with open('../data/this_is_gallery/preprocess_df.pickle', 'wb') as f:
    pickle.dump(df, f)

print(df)