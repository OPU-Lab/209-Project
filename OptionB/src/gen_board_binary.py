import math
import numpy as np
from src.ddr_generation import ker_re_arrange
from src.Config import Config
from src.conv_hw import load_ker


def gen_layer_4board(ker, bias, configs, out_dir, first_layer=False):
    re_ker, re_bias = ker_re_arrange(ker, bias, configs, first_layer=first_layer)
    re_ker = re_ker * (2 ** configs['weights_fix_lo'])
    re_bias = re_bias * (2 ** configs['bias_fix_lo'])

    arg = 'w' if first_layer else 'a'
    ker_file = out_dir + '/weights/weight_all.txt'
    bias_file = out_dir + '/weights/bias_all.txt'

    with open(ker_file, arg) as f_ker:
        for w in range(np.shape(re_ker)[0]):
            short = re_ker[w, :]
            for t in range(16):
                sub_short = short[t * 64: (t + 1) * 64]
                for sub in range(64):
                    data_hex = hex(int(sub_short[sub]) & int('FF', 16))[2:]
                    f_ker.write(str(int(sub_short[sub])) + ' ')

    bias_c_num = configs['output_size'][2]
    with open(bias_file, arg) as f_bias:
        for w in range(math.ceil(bias_c_num / 64)):
            for t in range(64):
                data_hex = hex(int(re_bias[w * 64 + t]) & int('FFFF', 16))[2:]
                f_bias.write(str(int(re_bias[w * 64 + t])) + ' ')


def write_bin_4board(file_in_dir, file_out_dir, end_layer_idx, network_name):
    ker_file = file_in_dir + '/weights/weight_all.txt'
    bias_file = file_in_dir + '/weights/bias_all.txt'
    ins_file = file_in_dir + '/onboard_ins/{}_0_to_{}.txt'.format(network_name, end_layer_idx)

    ker_bin_file = file_out_dir + '/onboard_binary/weights.bin'
    bias_bin_file = '{}/onboard_binary/bias_{}.bin'.format(file_out_dir, end_layer_idx)
    ins_bin_file = '{}/onboard_binary/ins_{}.bin'.format(file_out_dir, end_layer_idx)

    with open(ker_file, 'r') as f_ker:
        ker = f_ker.readline().split(' ')[:-1]

    ker_bin = np.array(ker, dtype=np.int8).tobytes()
    with open(ker_bin_file, 'wb') as f_ker_bin:
        f_ker_bin.write(ker_bin)

    with open(bias_file, 'r') as f_bias:
        bias = f_bias.readline().split(' ')[:-1]

    bias_split = []
    for item in bias:
        bias_split = bias_split + [int(item) // 256] + [int(item) % 256]

    bias_bin = np.array(bias_split, dtype=np.int8).tobytes()
    with open(bias_bin_file, 'wb') as f_bias_bin:
        f_bias_bin.write(bias_bin)

    with open(ins_file, 'r') as f_ins:
        ins = [line[:-1] for line in f_ins.readlines()]

    ins_split = []
    for item in ins:
        ins_split = ins_split + [item[0:8]] + [item[8:16]] + [item[16:24]] + [item[24:32]]
    ins_dec = [int(i, 2) for i in ins_split]
    ins_bin = np.array(ins_dec, dtype=np.int8).tobytes()
    with open(bias_bin_file, 'ab') as f_bias_bin:
        f_bias_bin.write(ins_bin)

    ins_2bram = ins[0:1024] if len(ins) > 1024 else ins + ['0'*32] * (1024 - len(ins))
    ins_tmp = []
    for idx in range(0, 1024, 16):
        ins_tmp = ins_tmp + ins_2bram[idx:idx+16][::-1]
    ins_2bram_split = []
    for item in ins_tmp:
        ins_2bram_split = ins_2bram_split + [item[0:8]] + [item[8:16]] + [item[16:24]] + [item[24:32]]

    ins_2bram_dec = [int(i, 2) for i in ins_2bram_split]
    ins_2bram_bin = np.array(ins_2bram_dec, dtype=np.int8).tobytes()
    with open(ins_bin_file, 'wb') as f_ins_bin:
        f_ins_bin.write(ins_2bram_bin)


def gen_network_4board(network_name):
    """

    :param network_name:
    :return:
    """
    network_configs = Config(network_name)

    file_in_dir = network_configs.father_dir + '/data/' + network_name
    file_out_dir = network_configs.father_dir + '/results/' + network_name

    for layer_id in range(network_configs.layer_num):
        print('Running for layer: ' + str(layer_id))
        layer_configs = network_configs.generate_layer_config(layer_id)

        ker, bias = load_ker(layer_configs)

        gen_layer_4board(ker, bias, layer_configs, file_in_dir, first_layer=(layer_id == 0))

    # for end_layer_idx in range(network_configs.layer_num):
    #     write_bin_4board(file_in_dir, file_out_dir, end_layer_idx, network_name)
    write_bin_4board(file_in_dir, file_out_dir, network_configs.layer_num-1, network_name)

