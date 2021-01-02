import os
import shutil
from PIL import Image
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns


depth_folder = 'different_depths/raw_images'
csv_folder = 'different_depths/csv'

#save_ad = csv_folder+'/'+images+'.csv', im, delimiter=","

eff_example_folder = 'efficientpose_example/raw_images'

def heatmap2d(arr: np.ndarray,titel):
    plt.imshow(arr, cmap='viridis')
    plt.title(titel)
    plt.colorbar()
    #plt.show()
    plt.savefig('different_depths/{}.png'.format(titel))
    plt.close()

for images in os.listdir(depth_folder):
    im = np.array(Image.open(depth_folder +'/'+ images))
    #df = pd.DataFrame(data=im.astype(float))
    #df.to_csv(csv_folder+'/'+images, sep=' ', header=False, float_format='%.2f', index=False)
    try:
        print(im)
        #plt.imshow(im, cmap='hot', interpolation='nearest')
        heatmap2d(im,images)
        #plt.title(images)
        #plt.show()
        #ax = sns.heatmap(im, linewidth=0.5)
        #plt.show()
        #np.savetxt(eff_example_folder+'/'+images+'.csv', im, delimiter=",")
    except ValueError:
        print(images)

'''
for images in os.listdir(depth_folder):
    im = np.array(Image.open(depth_folder +'/'+ images))
    print(images)
    print(im)
    #df = pd.DataFrame(data=im.astype(float))
    #df.to_csv(csv_folder+'/'+images, sep=' ', header=False, float_format='%.2f', index=False)

    with open(csv_folder+'/'+images+'.txt',"w+") as f:
        for line in im:
            np.savetxt(f, line, fmt='%.2f')
    #np.savetxt(csv_folder+'/'+images, im, delimiter=",")
'''