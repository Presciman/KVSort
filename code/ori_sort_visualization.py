import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


array = np.fromfile('iter372_layer20_token1500_key.npy',dtype=np.float32)
array = array.reshape(6, 2048, 8, 128)

data_fp32 = array.transpose(1,0,2,3).reshape(2048,6*8*128)
print("Shape: {}".format(data_fp32.shape))
plt.imshow(data_fp32[:512,:1024],interpolation='none',cmap='seismic')
plt.colorbar()
plt.savefig('llama3_gsm8k_key_ori.png')
plt.close()
data_fp32 = np.sort(data_fp32,axis=1)

plt.imshow(data_fp32[:-512,:-1024],interpolation='none',cmap='seismic')
plt.colorbar()
plt.savefig('llama3_gsm8k_key_sorted.png')
plt.close()

rel_eb=1E-2
abs_eb=(data_fp32.max()-data_fp32.min())*rel_eb
#Use FZ-GPu compress
import os
dimensions = ' '.join([str(x) for x in data_fp32.shape])
data_fp32.tofile('data_temp.bin')
os.system('./fz-gpu data_temp.bin {} 1 {}'.format(dimensions,abs_eb))
os.system('rm -rf data_temp.bin')
