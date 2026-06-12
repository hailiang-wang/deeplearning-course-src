import torch
import numpy as np

torch.set_printoptions(edgeitems=2, threshold=50)

import imageio

dir_path = "data/p1ch4/volumetric-dicom/2-LUNG 3.0  B70f-04083"
vol_arr = imageio.volread(dir_path, "DICOM")

print(vol_arr.shape)

'''
PyTorch Tensor
'''
vol = torch.from_numpy(vol_arr).float()

import matplotlib.pyplot as plt

plt.imshow(vol_arr[50])
plt.show()
