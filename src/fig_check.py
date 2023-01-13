import glob
import os
import pickle

from PIL import Image

for path  in glob.glob('../data/this_is_gallery/fig/*'):
    try:
        image = Image.open(path).convert('RGB')
    except:
        print(path)

with open('../data/this_is_gallery/preprocess_df.pickle', 'rb') as f:
    df = pickle.load(f)

print(df)

# image_path = "../data/this_is_gallery/fig/{}.jpg".format(index)
#         image = Image.open(image_path).convert('RGB')