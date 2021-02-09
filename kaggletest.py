import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


import os
import skimage.io

df = pd.read_csv('G:/panda/d/train.csv')
df = df[df['data_provider'] == 'karolinska'].reset_index(drop=True)
print(df)
ids = df['image_id'].values
df.head()
TRAIN = 'G:/panda/d/train_images_sent/'
MASKS = 'G:/panda/d/train_label_masks/'

img = skimage.io.MultiImage(os.path.join(TRAIN,ids[0]+'.tiff'))
print(img[0].shape, img[1].shape, img[2].shape)