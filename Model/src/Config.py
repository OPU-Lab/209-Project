import re
import os
import math


class Config(object):
    config_dict = dict()

    def __init__(self, network_name):
        # set parameters
        self.father_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        self.ir_path = self.father_dir + '/data/{}/ir.txt'.format(network_name)

        # parse IR
        self.parse_ir(self.ir_path)

        # set global configurations for one network
        self.layer_num = self.config_dict['num'][0]
        self.board_file_dir = self.father_dir + '/results/' + network_name

        self.config_dict['mac_num'] = [32 for i in range(self.layer_num)]
        self.config_dict['pe_num'] = [32 for i in range(self.layer_num)]
        self.config_dict['shift_num_bias'] = [0 for i in range(self.layer_num)]
        self.config_dict['shift_num_temp'] = [0 for i in range(self.layer_num)]
        self.config_dict['write_internal'] = [True for i in range(self.layer_num)]
        self.config_dict['write_final'] = [True for i in range(self.layer_num)]
        self.config_dict['ipa_bit_len'] = [26 for i in range(self.layer_num)]
        self.config_dict['temp_bit_len'] = [16 for i in range(self.layer_num)]
        self.config_dict['ofm_bit_len'] = [8 for i in range(self.layer_num)]
        self.config_dict['father_dir'] = [self.father_dir for i in range(self.layer_num)]

        # reformat the IR data to make it consistent with the hw representative
        self.post_processing(network_name)

    def parse_ir(self, filename):
        """
        Parse configs from the IR file
        :param filename:
        :return:
        """
        with open(filename, 'r') as f:
            configs_data = f.readlines()
            for line in configs_data:
                # remove the \n char
                cur_line = line[:-1]
                title = re.findall('.+?:', cur_line)
                # remove the : char
                cur_title = title[0][:-1]
                content = re.findall(':.+', cur_line)
                cur_content = content[0][1:]
                exec('self.config_dict[cur_title]=' + cur_content)

    def post_processing(self, network_name):
        # because the 'strides' in IR file has 3 channels while we only need the first two in the sw
        self.config_dict['out_dir'] = ['' for i in range(self.layer_num)]
        self.config_dict['4hw'] = ['' for i in range(self.layer_num)]

        self.config_dict['ifm_file'] = ['' for i in range(self.layer_num)]
        self.config_dict['ker_file'] = ['' for i in range(self.layer_num)]
        self.config_dict['bias_file'] = ['' for i in range(self.layer_num)]
        self.config_dict['ins_file'] = ['' for i in range(self.layer_num)]
        self.config_dict['res_file'] = ['' for i in range(self.layer_num)]

        self.config_dict['ipa_frac_len'] = [0 for i in range(self.layer_num)]
        self.config_dict['temp_frac_len'] = [0 for i in range(self.layer_num)]
        self.config_dict['res_frac_len'] = self.config_dict['output_fm_fix_lo']
        self.config_dict['append_remove_padding'] = [[1, 1, 1, 1] for i in range(self.layer_num)]
        self.config_dict['append_size'] = [None for i in range(self.layer_num)]

        self.config_dict['res_remove_padding'] = [[0, 0, 0, 0] for i in range(self.layer_num)]
        self.config_dict['res_input_size'] = [[0, 0, 0] for i in range(self.layer_num)]
        self.config_dict['conv_result_size'] = [(0, 0, 0) for i in range(self.layer_num)]
        self.config_dict['conv_result_blk_size'] = [(0, 0, 0) for i in range(self.layer_num)]
        if not os.path.exists(self.father_dir + '/results/'):
            os.mkdir(self.father_dir + '/results/')
        for i in range(self.layer_num):
            if not os.path.exists(self.father_dir + '/results/' + network_name + '/'):
                os.mkdir(self.father_dir + '/results/' + network_name + '/')

            self.config_dict['out_dir'][i] = self.father_dir + '/results/' + network_name + '/layer_' + str(i) + '/'
            if not os.path.exists(self.config_dict['out_dir'][i]):
                os.mkdir(self.config_dict['out_dir'][i])

            self.config_dict['4hw'][i] = self.config_dict['out_dir'][i] + '4hw/'
            if not os.path.exists(self.config_dict['4hw'][i]):
                os.mkdir(self.config_dict['4hw'][i])

            if i == 0:
                self.config_dict['ifm_file'][i] = '{}/data/{}/{}'.format(self.father_dir, network_name,
                                                                         self.config_dict['input_file_name'])
            else:
                self.config_dict['ifm_file'][i] = ['{}/results/{}/layer_{}/ofm.txt'.format(
                    self.father_dir, network_name, idx) for idx in self.config_dict['input_from'][i]]

            self.config_dict['ker_file'][i] = '{}/data/{}/weights/weight_{}.mat'.format(
                self.father_dir, network_name, i)
            self.config_dict['bias_file'][i] = '{}/data/{}/weights/bias_{}.mat'.format(self.father_dir, network_name, i)
            self.config_dict['ins_file'][i] = '{}/data/{}/ins/{}_{}.txt'.format(
                self.father_dir, network_name, network_name, i)

            int_len = self.config_dict['ofm_bit_len'][i] - self.config_dict['output_fm_fix_lo'][i]
            self.config_dict['ipa_frac_len'][i] = self.config_dict['ipa_bit_len'][i] - int_len
            self.config_dict['temp_frac_len'][i] = self.config_dict['temp_bit_len'][i] - int_len

            if not self.config_dict['shortcut_source'][i]:
                self.config_dict['shortcut_source'][i] = None
            else:
                self.config_dict['shortcut_source'][i] = self.config_dict['shortcut_source'][i][0] - 1
                self.config_dict['res_file'][i] = self.father_dir + '/results/' + network_name + '/layer_' + str(
                    self.config_dict['shortcut_source'][i]) + '/ofm.txt'
                self.config_dict['res_remove_padding'][i] = \
                    self.config_dict['post_padding_size'][self.config_dict['shortcut_source'][i]]
                self.config_dict['res_input_size'][i] = \
                    self.config_dict['output_size'][self.config_dict['shortcut_source'][i]]
            h, w, c = self.config_dict['output_size'][i]
            if self.config_dict['post_padding'][i]:
                pad_u, pad_d, pad_l, pad_r = self.config_dict['post_padding_size'][i]
                self.config_dict['output_size'][i] = [h - pad_u - pad_d, w - pad_l - pad_r, c]
            self.config_dict['ker_stride'][i] = self.config_dict['ker_stride'][i][:2]
            self.config_dict['post_padding'][i] = bool(self.config_dict['post_padding'][i])
            self.config_dict['pooling'][i] = bool(self.config_dict['pooling'][i])
            ifm_h, ifm_w, ifm_c = self.config_dict['input_size'][i]
            ker_h, ker_w, _ = self.config_dict['ker_size'][i]
            s_h, s_w = self.config_dict['ker_stride'][i]
            _, _, ofm_c = self.config_dict['output_size'][i]
            p_u, p_d, p_l, p_r = self.config_dict['pre_remove_padding'][i]
            ifm_h -= p_u + p_d
            ifm_w -= p_l + p_r
            self.config_dict['conv_result_size'][i] = (math.ceil((ifm_h - ker_h) / s_h) + 1,
                                                       math.ceil((ifm_w - ker_w) / s_w) + 1, ofm_c)
            if i == 0:
                self.config_dict['conv_result_blk_size'][i] = (self.config_dict['dma_blk_size'][i][0],
                                                               self.config_dict['dma_blk_size'][i][1],
                                                               min(self.config_dict['ofm_blk_size'][i][2],
                                                                   self.config_dict['output_size'][i][2]))
            else:
                h, w, _ = self.config_dict['dma_blk_size'][i]
                c = min(self.config_dict['ofm_blk_size'][i][2], self.config_dict['output_size'][i][2])
                k_h, k_w, _ = self.config_dict['ker_size'][i]
                s_h, s_w = self.config_dict['ker_stride'][i]
                self.config_dict['conv_result_blk_size'][i] = (math.floor((h - k_h) / s_h) + 1, math.floor((w - k_w) / s_w) + 1, c)
            self.config_dict['cout_unit'][i] = min(self.config_dict['cout_unit'][i], self.config_dict['output_size'][i][2])

    def generate_layer_config(self, layer_id):
        layer_config_dict = dict()
        for key in self.config_dict.keys():
            if len(self.config_dict[key]) == self.layer_num:
                layer_config_dict[key] = self.config_dict[key][layer_id]
        return layer_config_dict
