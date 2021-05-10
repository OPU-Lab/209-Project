import numpy as np
import os
import math
import re


class DdrSimu(object):

    def __init__(self, ddr_addr_len=25, create_file=False):
        """
        create a zero binary file to simulate ddr
        :param ddr_addr_len: ddr's address's bit length
        :param create_file: whether a new file should be created (ddr)
        :return:
        """
        self.father_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        line_num = 1 << ddr_addr_len
        if create_file:
            file = open(self.father_dir + '/results/ddr','wb+')
            for i in range(line_num):
                b_data = bytearray(64)
                file.write(b_data)
            file.close()

    def ddr_write(self, ddr_store_result, blk_idx, configs, base_addr=0):
        """
        write the result of ddr_store function into ddr
        :param ddr_store_result: the ofm of ddr_store function
        :param blk_idx: the index of current block
        :param configs: the configuration dict
        :param base_addr: the initial address of ddr for the current block
        :return:
        """
        # there are in total 2^25 numbers and each number contains 64 channels
        # each channel is 8 bits ( a byte) in length
        h, w, _ = np.shape(ddr_store_result)
        ofm_h, ofm_w, _ = configs['output_size']
        addr_file = open(configs['out_dir'] + 'ddr_addr.txt', 'a+')
        ddr_file = open(self.father_dir + '/results/ddr', 'ab')
        ddr_store_result *= (2 ** configs['output_fm_fix_lo'])
        ddr_store_result = ddr_store_result.astype('int8')
        for hh in range(h):
            for ww in range(w):
                real_h = blk_idx[0] * h + hh
                real_w = blk_idx[1] * w + ww
                ins_addr = (real_h * ofm_w + real_w) + base_addr + blk_idx[2] * ofm_w * ofm_h
                ddr_file.seek(ins_addr * 64)
                data_to_be_write = []
                for i in range(64):
                    write_res = ddr_store_result[hh, ww, i]
                    if write_res < 0:
                        write_res += (1 << 8)
                    data_to_be_write.append(int(write_res).to_bytes(length=1, byteorder='big'))
                for b in data_to_be_write:
                    ddr_file.write(b)
                addr_file.write(str(ins_addr) + '\n')
                ddr_file.flush()
        ddr_file.close()
        addr_file.close()

    def ddr_load(self, configs, ifm_base=0, ker_base=500, bias_base=100, ins_base=200, res_base=300):
        """
        load the feature map/kernel/bias/instruction data from the ddr
        :param ddr_file_addr: the address of ddr file
        :param configs: the configuration dict
        :return:
        """
        h, w, c = configs['input_size']
        k_h, k_w, _ = configs['ker_size']
        output_size = configs['output_size']
        blk_h, blk_w, blk_c = configs['dma_blk_size']

        ddr_file = open(self.father_dir + '/results/ddr', 'rb')
        all_data = ddr_file.read()
        print(len(all_data))
        ddr_content = ddr_file.readlines()
        ddr_file.close()

        # load ifm part from the ddr file
        line_num_file = open(configs['4hw'] + 'line_num.txt', 'r')
        line_n = line_num_file.readlines()
        line_num_range = dict()
        for line in line_n:
            name = re.findall('.+:', line)[0][:-1]
            tmp = re.findall(':.+', line)[0]
            tmp_beg = int(re.findall('.+?--', tmp)[0][2:-3]) if re.findall('.+?--', tmp) != [] else None
            tmp_end = int(re.findall('--.+', tmp)[0][3:]) if re.findall('--.+', tmp) != [] else None
            line_num_range[name] = (tmp_beg, tmp_end)



        ifm_buffer = open(self.father_dir + '/results/ifm_buffer', 'wb+')
        blk_range = []

        for hh in range(math.ceil(output_size[0] / blk_h)):
            for ww in range(math.ceil(output_size[1] / blk_w)):
                for cc in range(math.ceil(output_size[2] / blk_c)):
                    blk_range.append([hh, ww, cc])
        f = open(configs['out_dir'] + 'ddr_load_addr.txt', 'w+')
        for blk_idx in blk_range:
            for hh in range(blk_h):
                for ww in range(blk_w):
                    real_h = blk_idx[0] * blk_h + hh
                    real_w = blk_idx[1] * blk_w + ww
                    index = real_h * w + real_w + blk_idx[2] * h * w
                    f.write(str(index) + '\n')
                    real_addr = ifm_base + index * 64
                    cur_line = all_data[real_addr:real_addr + 64]
                    byte_res = bytes(cur_line)
                    ifm_buffer.write(byte_res)
            ifm_buffer.seek(0)
        ifm_buffer.close()
        f.close()

        ker_buffer = open(self.father_dir + '/results/ker_buffer', 'wb+')
        ker_len = line_num_range['ker'][1] - line_num_range['ker'][0] + 1
        for i in range(ker_len):
            real_addr = i * 64 + ker_base
            cur_line = all_data[real_addr:real_addr + 64]
            byte_res = bytes(cur_line)
            ker_buffer.write(byte_res)
        ker_buffer.seek(0)
        ker_buffer.close()

        bias_buffer = open(self.father_dir + '/results/bias_buffer', 'wb+')
        bias_len = line_num_range['bias'][1] - line_num_range['bias'][0] + 1
        for i in range(bias_len):
            real_addr = i * 64 + bias_base
            cur_line = all_data[real_addr:real_addr + 64]
            byte_res = bytes(cur_line)
            bias_buffer.write(byte_res)
        bias_buffer.seek(0)
        bias_buffer.close()

        res_buffer = open(self.father_dir + '/results/res_buffer', 'wb+')
        if line_num_range['res'][0] == None:
            pass
        else:
            res_len = line_num_range['res'][1] - line_num_range['res'][0] + 1
            for i in range(res_len):
                real_addr = i * 64 + res_base
                cur_line = all_data[real_addr:real_addr + 64]
                byte_res = bytes(cur_line)
                res_buffer.write(byte_res)
            res_buffer.seek(0)
            res_buffer.close()

        ins_buffer = open(self.father_dir + '/results/ins_buffer', 'wb+')
        ins_len = line_num_range['ins'][1] - line_num_range['ins'][0] + 1
        for i in range(ins_len):
            real_addr = i * 64 + ins_base
            cur_line = all_data[real_addr:real_addr + 64]
            byte_res = bytes(cur_line)
            ins_buffer.write(byte_res)
        ins_buffer.seek(0)
        ins_buffer.close()












# needs to be removed later
if __name__ == '__main__':
    b = 'b2d4660aa60'.encode('utf8')
    print(bytes(b))
    exit()
    ddr_store_result = np.random.rand(35, 35, 64)
    configs = dict()
    configs['ofm_blk_size'] = (35, 35, 64)
    configs['ofm_size'] = (210, 210, 256)
    configs['output_fm_fix_lo'] = 5

    a = DdrSimu(ddr_addr_len=25, create_file=False)
    # the third dimension of blk_idx redundant in ddr_write
    a.ddr_write(ddr_store_result, (0, 0, 0), configs, base_addr=0)




