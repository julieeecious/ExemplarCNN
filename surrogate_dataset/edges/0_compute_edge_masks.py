import os
import numpy as np
from PIL import Image
from skimage import color
from skimage.filters import sobel
from tqdm import tqdm

# Converting the class images into "edge masks"
#out_path = './class_folder_edge/'
#in_path = '../../surrogate_dataset_imagenet/classes_folder/'

out_path = './ExemplarCNN/surrogate_dataset/class_folder_16000_set_edge/'
in_path = './ExemplarCNN/surrogate_dataset/classes_folder_16000_set/'

nb_classes = 16000

classes = os.listdir(in_path)
classes.sort()

try:
    os.mkdir(out_path)
except:
    pass

for img in tqdm(classes):
    image = np.asarray(Image.open(in_path + img))
    edge_mask = sobel(color.rgb2gray(image))
    im = Image.fromarray(np.uint8(edge_mask*255))
    im.save(out_path + str(img[:-5]).zfill(len(str(nb_classes))) + '_edge.png')

