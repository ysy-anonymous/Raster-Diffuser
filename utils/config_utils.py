import torch
import numpy as np

def get_norm_stat(ori_map_size, data_size):
    # normalization factor for inference
    if ori_map_size == (8, 8) and data_size == 100:
        norm_stat = {
            'min': [0.0335, 0.0060],
            'max': [7.9933, 7.9287]
        }
    elif ori_map_size == (8, 8) and data_size == 2000:
        norm_stat = {
            'min': [0.0035, 0.0007],
            'max': [7.9995, 8.0000]
        }
    elif ori_map_size == (16, 16) and data_size == 6257: # horizon length 32
        norm_stat = {
            'min': [8.1102e-05, 3.1935e-05],
            'max': [15.9999, 15.9989]
        }
    elif ori_map_size == (32, 32) and data_size == 11210:
        norm_stat = {
            'min': [0.0010, 0.0040],
            'max': [31.9999, 31.9988]
        }
    elif ori_map_size == (8, 8) and data_size == 38345:
        norm_stat = {
            'min': [5.8019e-05, 7.5751e-05],
            'max': [8.0000, 8.0000]
        }
    elif ori_map_size == (8, 8) and data_size == 95792:
        norm_stat = {
            'min': [3.4133e-05, 4.6726e-05],
            'max': [8.0000, 8.0000]
        }
    elif ori_map_size == (8, 8) and data_size == 1053418:
        norm_stat =  {
            'min': [3.2796e-06, 5.5942e-06],
            'max': [8.0000, 8.0000]
        }
    elif ori_map_size == (16, 16) and data_size == 97529: # horizon length 64
        norm_stat = {
            'min': [0.0002, 0.0002],
            'max': [16.0000, 15.9999]
        }
    elif ori_map_size == (32, 32) and data_size == 95035:
        norm_stat = {
            'min': [1.4727e-05, 6.0066e-04],
            'max': [31.9999, 32.0000]
        }
    elif ori_map_size == (8, 8) and data_size == 8000:
        norm_stat = {
            'min': [9.4140e-05, 1.1851e-03],
            'max': [7.9995, 8.0000]
        }
    else:
        raise Exception("Normalization stat not found for map size {} and data size {}".format(ori_map_size, data_size))

    return norm_stat