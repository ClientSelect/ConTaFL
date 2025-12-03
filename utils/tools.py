import numpy as np
import matplotlib.pyplot as plt
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"



def draw_plt(acc_list, loss_list):
    '''
    做出训练中的精度和损失曲线
    :param acc_list:
    :param loss_list:
    :return:
    '''
    acc_list = np.array(acc_list)
    loss_list = np.array(loss_list)
    plt.plot(acc_list)
    plt.legend(['acc'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)

    figure_save_path = "file_fig"
    if not os.path.exists(figure_save_path):
        os.makedirs(figure_save_path)  # 如果不存在目录figure_save_path，则创建
    plt.savefig(os.path.join(figure_save_path, 'acc_noniid_0.7_beta.png'))  # 第一个是指存储路径，第二个是图片名字
    # plt.savefig('acc')
    plt.cla()
    plt.plot(loss_list)
    plt.legend(['loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    # plt.savefig('loss')
    plt.savefig(os.path.join(figure_save_path, 'loss_test_avg_bad.png'))#第一个是指存储路径，第二个是图片名字


import torch

def get_set_gpus(gpu_ids):
    # If running on a CPU-only machine, return an empty list or keep CPU mode
    if not torch.cuda.is_available():
        print("CUDA is not available. Running on CPU.")
        return []  # No GPU available, return empty list or None
    
    # Otherwise, proceed with GPU setup
    gpus = [int(id) for id in gpu_ids.split(',') if int(id) >= 0] if len(gpu_ids) > 1 else [int(gpu_ids)]
    
    if gpus:
        torch.cuda.set_device(gpus[0])  # Set the first available GPU
    
    return gpus