#1. Share between each token
#2. Share between key and value
#3. Share between each layer

import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess

file_dir = '/your/file/dir'
model_and_dataset = 'llama3_8b_gsm8k_batch1'

layer_list = range(0,32) #llama3 gsm8k
REL_EB = 1e-2
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
    '-c', '[Your SZ3 install dir]/SZ3/tools/sz3/sz3.config',
    '-M', 'ABS', str(ABS_EB),
    '-a'
    ]
    
    return subprocess.run(sz_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, check=True)


group_size = 4
total= []
for i in layer_list:
    # print('---{}---'.format(i))
    file_name_key = 'iter_1118_layer{}_key.npy'.format(i)
    data_key = np.load(os.path.join(file_dir, model_and_dataset, file_name_key))
    data_key = data_key.transpose(2,0,1,3).reshape(data_key.shape[2],data_key.shape[0]*data_key.shape[1]*data_key.shape[3])
    ABS_EB_KEY = (data_key.max()-data_key.min())*REL_EB
    file_name_value = 'iter_1118_layer{}_value.npy'.format(i)
    data_value = np.load(os.path.join(file_dir, model_and_dataset, file_name_value))
    data_value = data_value.transpose(2,0,1,3).reshape(data_value.shape[2],data_value.shape[0]*data_value.shape[1]*data_value.shape[3])
    ABS_EB_VALUE = (data_value.max()-data_value.min())*REL_EB
    CR_per_layer_key = []
    CR_per_layer_value = []
    means = data_value[:,0]
    idx_tk = np.argsort(means)
    data_value_sorted = np.take_along_axis(data_value, idx_tk[:, None], axis=0)

    # data_value_sorted = np.sort(data_value, axis=1)
    result = sz_compress(data_value_sorted, ABS_EB_VALUE, 'data_temp.bin', 'data_temp.bin.sz')
    flag = 0
    for i in result.stdout.split('\n'):
        if 'compression ratio' in i and flag == 0:
            CR_layer = round(float(i.split('= ')[1].replace('\n',''))/2,2)
            total.append(CR_layer)
            # CR_per_layer_value.append(round(float(i.split('= ')[1].replace('\n',''))/2,2))
            flag+= 1
    # print(CR_per_layer_value[0],end=',')
    print(CR_layer,end=',')
    


    #Without key sorting - o1
    '''for j in range(0, data_value.shape[1]-group_size, group_size):
        # data_key_grouped = data_key[j:j+group_size,:]
        # data_key_grouped_sorted = np.sort(data_key_grouped, axis=0)
        # idx_key_grouped_sorted = np.argsort(data_key_grouped, axis=0)
        # result = sz_compress(data_key_grouped_sorted, ABS_EB_KEY, 'data_temp.bin', 'data_temp.bin.sz')
        # flag = 0
        # for i in result.stdout.split('\n'):
        #     if 'compression ratio' in i and flag == 0:
        #         CR_per_layer_key.append(round(float(i.split('= ')[1].replace('\n','')),2))
        #         flag+= 1

        data_value_grouped = data_value[:,j:j+group_size]
        # data_value_grouped_sorted = data_value_grouped[idx_key_grouped_sorted]
        # data_value_grouped_sorted = np.take_along_axis(data_value_grouped, idx_key_grouped_sorted, axis=0)
        data_value_grouped_sorted = np.sort(data_value_grouped, axis=1)
        result = sz_compress(data_value_grouped_sorted, ABS_EB_VALUE, 'data_temp.bin', 'data_temp.bin.sz')
        flag = 0
        for i in result.stdout.split('\n'):
            if 'compression ratio' in i and flag == 0:
                CR_per_layer_value.append(round(float(i.split('= ')[1].replace('\n','')),2))
                flag+= 1

    # average_cr_key = len(CR_per_layer_key) / sum([1/x for x in CR_per_layer_key])
    average_cr_value = len(CR_per_layer_value) / sum([1/x for x in CR_per_layer_value])
    total.append(average_cr_value)
    print(average_cr_value,end=',')
    # print(CR_per_layer_value)
    # print('Key: {}, Value:{}'.format(average_cr_key, average_cr_value))
    # print('Key_list: {}, Value_list:{}'.format(CR_per_layer_key, CR_per_layer_value))
    # file_name_value = 'iter_1118_layer{}_value.npy'.format(i)
    # data_value = np.load(os.path.join(file_dir, model_and_dataset, file_name_value))
    # data_value = data_value.transpose(2,0,1,3).reshape(data_value.shape[2],data_value.shape[0]*data_value.shape[1]*data_value.shape[3])
    # ABS_EB_VALUE = (data_value.max()-data_value.min())*REL_EB
    # sorted_value = np.sort(data_value, axis=1)

    # result = sz_compress(data_key, ABS_EB_KEY, 'data_temp.bin', 'data_temp.bin.sz')
    # flag = 0
    # for i in result.stdout.split('\n'):
    #     if 'compression ratio' in i and flag == 0:
    #         print('Key:',end=' ')
    #         print(round(float(i.split('= ')[1].replace('\n','')),2),end=',')
    #         flag+= 1
    # result = sz_compress(data_value, ABS_EB_VALUE, 'data_temp.bin', 'data_temp.bin.sz')
    # flag = 0
    # for i in result.stdout.split('\n'):
    #     if 'compression ratio' in i and flag == 0:
    #         print('Value:',end=' ')
    #         print(round(float(i.split('= ')[1].replace('\n','')),2))
    #         flag+= 1'''
print()
print(len(total)/sum([1/x for x in total]))