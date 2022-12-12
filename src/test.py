import os

DIR = "../data/fig/"
print(sum(os.path.isfile(os.path.join(DIR, name)) for name in os.listdir(DIR)))
print(len(os.listdir(DIR)))