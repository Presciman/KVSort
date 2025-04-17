# This code studies the relationship between the compression ratio and the  ratio of sorted data to original data.

import numpy as np
import os
import subprocess
import matplotlib.pyplot as plt
from tqdm import tqdm


def fz_compress(data_fp32,REL_EB):
    data_fp32.tofile('data_temp.bin')
    ABS_EB = (data_fp32.max()-data_fp32.min())*REL_EB
    fz_command =  ['./fz-gpu', 'data_temp.bin', str(data_fp32.shape[0]), str(data_fp32.shape[1]), '1', str(ABS_EB)]
    os.system('rm -rf data_temp.bin')
    return subprocess.run(fz_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, check=True)

def sz_compress(data_fp32,REL_EB,input_data_path,output_data_path):
    os.system('rm -rf {}'.format(input_data_path))
    data_fp32.tofile(input_data_path)
    ABS_EB = (data_fp32.max()-data_fp32.min())*REL_EB
    sz_command = [
    '[Your SZ3 install dir]/SZ3/install/bin/sz3',
    '-f',
    '-i', input_data_path,
    '-o', output_data_path,
    '-2', str(data_fp32.shape[0]), str(data_fp32.shape[1]),
    '-c', '/N/u/sunbaix/BigRed200/SZ3/tools/sz3/sz3.config',
    '-M', 'ABS', str(ABS_EB),
    '-a'
    ]
    
    return subprocess.run(sz_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, check=True)

def partial_sort(data_fp32, ratio):
    sorted_data = np.sort(data_fp32, axis=1)  # Fully sorted data
    sorted_index = np.argsort(data_fp32, axis=1)  # Indices of sorted order

    num_sorted = int(data_fp32.shape[1] * ratio)  # Number of elements to sort
    partially_sorted = np.zeros_like(data_fp32)

    # Keep part of the row sorted, the rest stays as original
    partially_sorted[:, :num_sorted] = sorted_data[:, :num_sorted]
    partially_sorted[:, num_sorted:] = data_fp32[:, num_sorted:]

    return partially_sorted


def compress_one_layer(layer_no):
    
    file_name = 'iter_48_layer{}_value.npy'.format(layer_no) # Llama3 gsm8k
    # file_name = 'iter_0_layer{}_value.npy'.format(layer_no) # GPT-j-6b coqa
    # file_name = 'iter_33_layer{}_value.npy'.format(layer_no) # opt-13b-gsm8k

    data_fp32 = np.load(os.path.join(file_dir, model_and_dataset, file_name))
    # print(data_fp32.shape)

    data_fp32 = data_fp32[0,:,:,:]
    data_fp32 = data_fp32.transpose(1,0,2).reshape(data_fp32.shape[1],data_fp32.shape[0]*data_fp32.shape[2])
    # data_fp32 = data_fp32.transpose(0,2,1,3).reshape(data_fp32.shape[0]*data_fp32.shape[2],data_fp32.shape[1]*data_fp32.shape[3])
    # Compress using fzGPU
    REL_EB  = 1E-2
    




    # Compress using SZ3

    CR_list = []
    
    # for ratio in tqdm(np.arange(0, 1.01, 0.01)):
    #     sorted_data = partial_sort(data_fp32,ratio)
    #     result = sz_compress(sorted_data,REL_EB,'data_temp.bin','data_temp.bin.sz')
    #     flag = 0
    #     for i in result.stdout.split('\n'):
    #         if 'compression ratio' in i and flag == 0:
    #             CR_list.append(round(float(i.split('= ')[1].replace('\n','')),2))
    #             flag+= 1
    # sorted_index = np.argsort(data_fp32, axis=1)
    # Is token-wised sorting able?
    sorted_data = np.sort(data_fp32, axis=1)
    print('data_fp32 shape:', data_fp32.shape)
    result = sz_compress(sorted_data,REL_EB,'data_temp.bin','data_temp.bin.sz')
    flag = 0
    for i in result.stdout.split('\n'):
        if 'compression ratio' in i and flag == 0:
            print(round(float(i.split('= ')[1].replace('\n','')),2))
            flag+= 1

    # for i in range(10):
    #     print('sorted_index[{}]:'.format(i), [format(val, "08b") for val in sorted_index[i]])
    


    # sorted_data = partial_sort(data_fp32,0.99)
    # result = sz_compress(sorted_data,REL_EB,'data_temp.bin','data_temp.bin.sz')
    # flag = 0
    # for i in result.stdout.split('\n'):
    #     if 'compression ratio' in i and flag == 0:
    #         CR_list.append(round(float(i.split('= ')[1].replace('\n','')),2))
    #         flag+= 1
    
    # sorted_data = np.sort(data_fp32, axis=1)
    # result = sz_compress(sorted_data,REL_EB,'data_temp.bin','data_temp.bin.sz')
    # flag = 0
    # for i in result.stdout.split('\n'):
    #     if 'compression ratio' in i and flag == 0:
    #         CR_list.append(round(float(i.split('= ')[1].replace('\n','')),2))
    #         flag+= 1

    return CR_list

# Read data
file_dir = '[Your saved Key and vlaue path]'
model_and_dataset = 'llama3_8b_gsm8k'
# model_and_dataset = 'gpt-j-6b_coqa'
# model_and_dataset = 'opt_13b_gsm8k'

layer_list = range(2,4) #llama3 gsm8k
# layer_list = range(0,32) #llama3 gsm8k
# layer_list = range(0,16) #gpt-j-6b coqa
# layer_list = range(0,40) #opt-13b-gsm8k
layer_lists=[]
for layer_no in layer_list:
    CR_list = compress_one_layer(layer_no)
    layer_lists.append(CR_list)

print(layer_lists)



