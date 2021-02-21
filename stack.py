import cv2 
import numpy as np
import matplotlib.pyplot as plt
import os
import pywt
from copy import copy   
from skimage.measure import compare_ssim 
from math import floor
from copy import copy
plt.rcParams["figure.figsize"] = [10,10]

print("Importing Dependencies and Helper Functions")

scene = 2
images = []                                                                    #read images
image_files = sorted(os.listdir("Dataset/scene{}/".format(scene)))
for img in image_files:
    if img.split(".")[-1].lower() not in ["jpg", "jpeg", "png", "PNG"]:
        image_files.remove(img)

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] =[8,8]
print ("Fetching Images",end = "")
for i in range(len(image_files)):
    print ("...",end = "")
    img = image_files[i]
    img = cv2.imread("Dataset/scene{}/{}".format(scene, img))
    bgr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    images.append(img)
print()
print ("Images Obtained")

print ("Showing Images")
for i in images:
    plt.imshow(cv2.cvtColor(i, cv2.COLOR_BGR2RGB))
    plt.show()
    
print ("Downsampling Images.....")
downsample=1    #change to improve throughput
down= [i[0::downsample,0::downsample,:] for i in images]                          #downsample images
out = copy(down)
zeros = np.zeros_like(out[0])



scale = 0.54   #Set scale
print ("Creating Mask.....")

out[-1] = cv2.resize(out[-1],(int(zeros.shape[1]*scale),int(zeros.shape[0]*scale)))

h,w,c = zeros.shape
h_,w_,c_ = out[-1].shape
zeros[(h - h_)//2: (h + h_)//2,(w - w_)//2: (w + w_)//2,: ] = out[-1]
out[-1] = zeros


im = copy(out[0])

obj= copy(out[-1])

src_mask = np.zeros(obj.shape, obj.dtype)

mask = (obj[:,:,:] != np.array([0,0,0]))
src_mask[mask] = 255
gray = cv2.cvtColor(src_mask, cv2.COLOR_RGB2GRAY)
gray[gray> 0] = 255
final_mask = np.stack([gray,gray,gray], axis = -1)
plt.imsave("results_demo/mask.png",final_mask)
center = (1992,1524)
print ("Done......")
print ("Computing Area Under Mask.......")

normal_clone = cv2.seamlessClone(obj, im, final_mask, center, cv2.NORMAL_CLONE)   #clone
# plt.imshow(cv2.cvtColor(normal_clone, cv2.COLOR_RGB2BGR))
plt.imsave("results_demo/results.png", cv2.cvtColor(normal_clone, cv2.COLOR_RGB2BGR))
plt.show()
print ("Results saved at results_demo/results.png")
