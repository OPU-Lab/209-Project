import numpy as np
import scipy.io as scio
import math
from src.DdrSimu import DdrSimu
import copy


def write2file(filename, data, frac_len, bit_len, line_num=32, arg='a'):
    """
    Write data to file in hardware pattern
    :param filename: str, the name of the file
    :param data: numpy array, (h, w, c)
    :param frac_len:
    :param bit_len:
    :param line_num:
    :param arg: argument used when opening file for write
    :return:
    """
    hex_len = math.ceil(bit_len * 1.0 / 4)
    if data.ndim == 3:
        h, w, c = data.shape
        with open(filename, arg) as fs:
            for hh in range(h):
                for ww in range(w):
                    if c <= line_num:
                        # this is for yolo_v3 layer_0 only
                        for cc in range(line_num - c):
                            fs.write('0' * hex_len + ' ')
                        for cc in range(c):
                            data_hex = hex(int(data[hh, ww, cc] * (2 ** frac_len)) & int('1' * bit_len, 2))[2:]
                            data_hex = '0' * (hex_len - len(data_hex)) + data_hex
                            fs.write(data_hex + ' ')
                        fs.write('\n')
                        """
                        if cc == c - 1:
                            fs.write(data_hex + '\n')
                        else:
                            fs.write(data_hex + ' ')
                        """
                    else:
                        for cc in range(c):
                            data_hex = hex(int(data[hh, ww, cc] * (2 ** frac_len)) & int('1' * bit_len, 2))[2:]
                            data_hex = '0' * (hex_len - len(data_hex)) + data_hex
                            fs.write(data_hex + ' ')
                            if cc % line_num == line_num - 1:
                                fs.write('\n')
                            """
                            if cc % line_num == line_num - 1:
                                fs.write(data_hex + '\n')
                            else:
                                fs.write(data_hex + ' ')
                            """

    elif data.ndim == 1:
        c = data.shape[0]
        with open(filename, arg) as fs:
            for cc in range(c):
                data_hex = hex(int(data[cc] * (2 ** frac_len)) & int('1' * bit_len, 2))[2:]
                data_hex = '0' * (hex_len - len(data_hex)) + data_hex
                fs.write(data_hex + ' ')
            fs.write('\n')
    else:
        print('Supported dimension of data is 1 or 3, this dimension is: {},'.format(data.ndim))


def fetch_block(ifm, ker_size, stride, blk_idx, blk_size, configs):
    """
    Fetch a block from original input feature map according to the index of the block
    :param ifm: input feature map, numpy array
    :param ker_size: kernel size tuple, (h, w)
    :param stride: strides in tuple, (h, w)
    :param blk_idx: index of current block to fetch, (h, w, c)
    :param blk_size: the size of required block, (h, w, c)
    :return:
    """
    ifm_h, ifm_w, ifm_c = ifm.shape
    blk_h, blk_w, blk_c = blk_size
    blk_h_idx, blk_w_idx, blk_c_idx = blk_idx
    real_s_h, real_s_w = configs['real_stride']

    # w first, h second, c third
    start_h = real_s_h * blk_h_idx
    start_w = real_s_w * blk_w_idx
    start_c = blk_c * blk_c_idx
    end_h = min(start_h + blk_h, ifm_h)
    end_w = min(start_w + blk_w, ifm_w)
    end_c = min(start_c + blk_c, ifm_c)

    block = ifm[start_h:end_h, start_w:end_w, start_c:end_c]
    if block.shape[2] != blk_size[2]:
        cur_shape = np.shape(block)
        tmp = np.zeros((cur_shape[0], cur_shape[1], blk_size[2]))
        tmp[:cur_shape[0], :cur_shape[1], :cur_shape[2]] = block
        block = tmp
    return block


def dma_copy(blk, cout_unit):
    """
    Copy the block according to cout_unit, this is same as using dma_copy_mode
    :param blk: block to be copied
    :param cout_unit: indicate the copy time = cout_unit // 2
    :return: blk_cpy: the copied block
    """
    cpy_time = cout_unit // 2
    w, h, c = blk.shape
    blk_cpy = np.zeros([w, h, c * cpy_time])
    for i in range(cpy_time):
        blk_cpy[..., i * c:(i + 1) * c] = blk

    return blk_cpy


def dma_ifm(ifm, ker_size, stride, blk_idx, blk_size, cout_unit, configs):
    """
    Get one block from the original input feature map, and copy it according to the copy mode
    :param ifm: original input feature map, numpy array, (h, w, c)
    :param ker_size: (h, w)
    :param stride: (h, w)
    :param blk_idx: (h, w, c)
    :param blk_size: (h, w, c)
    :param cout_unit:
    :return:
    """
    blk = fetch_block(ifm, ker_size, stride, blk_idx, blk_size, configs)

    blk_cpy = dma_copy(blk, cout_unit)

    return blk_cpy


def dma_ker(ker, blk_ci, ci_start, blk_co, co_start):
    """
    Get one block from the original kernels, and rearrange it
    :param ker: original kernel, numpy array, (h, w, ci, co)
    :param blk_ci: number of input channel
    :param ci_start: start index of input channel for this block
    :param blk_co: number of output channel
    :param co_start: start index of output channel for this block
    :return:
    """
    kh, kw, kci, kco = ker.shape
    # for output layer
    blk_co = math.ceil(blk_co / 16) * 16
    ker_ori = np.zeros([kh, kw, blk_ci, blk_co])
    ker_ori[..., blk_ci - kci:, :kco] = ker
    ker_cpy = np.zeros([kh, kw, blk_ci * blk_co])
    for i in range(blk_co // 2 - 1, -1, -1):
        for j in range(blk_ci - 1, -1, -1):
            ker_cpy[..., 2 * (blk_co // 2 - 1 - i) * blk_ci + 2 * (blk_ci - 1 - j)] = ker_ori[..., ci_start + j,
                                                                                              co_start + i * 2 + 1]
            ker_cpy[..., 2 * (blk_co // 2 - 1 - i) * blk_ci + 2 * (blk_ci - 1 - j) + 1] = ker_ori[..., ci_start + j,
                                                                                                  co_start + i * 2]
    return ker_cpy


def pe(ifm, ker):
    """
    Perform dot-products of input feature maps and kernels, sharing input feature map.
    :param ifm: input feature map in numpy array, w x h x mac_num
    :param ker: kernels in numpy array, 2 x mac_num
    :return: dot product in numpy array, two numpy arrays, w x h
    """
    # expand ifm in channel
    h, w, c = ifm.shape
    ifm_tmp = np.zeros([h, w, 2 * c])
    for cc in range(c):
        ifm_tmp[..., 2 * cc] = ifm[..., cc]
        ifm_tmp[..., 2 * cc + 1] = ifm[..., cc]

    # expand kernel in w and c
    ker_tmp = np.zeros([h, w, 2 * c])
    for hh in range(h):
        for ww in range(w):
            ker_tmp[hh, ww, :] = ker

    product = ifm_tmp * ker_tmp
    dot_product_0 = np.sum(product[:, :, 1::2], axis=2)
    dot_product_1 = np.sum(product[:, :, 0::2], axis=2)
    return dot_product_0, dot_product_1


def pe_group(ifm, ker, mac_num, cin_unit):
    """
    :param ifm: w x h x cin_unit
    :param ker: cin_unit x 2
    :param mac_num: the number of macs in each PE
    :param cin_unit:
    :return:
    """
    w, h, _ = ifm.shape
    # in case the cin_unit is smaller than the mac_num
    group_num = math.ceil(cin_unit / mac_num)
    dot_product_0 = np.zeros([w, h])
    dot_product_1 = np.zeros([w, h])
    for i in range(group_num):
        tmp_0, tmp_1 = pe(ifm[..., mac_num * i:mac_num * (i + 1)], ker[mac_num * 2 * i:mac_num * 2 * (i + 1)])
        dot_product_0 += tmp_0
        dot_product_1 += tmp_1

    return dot_product_0, dot_product_1


# pass the parameter configs into ipa to deal with the problem of unable to divide blk exactly
def ipa(ifm, ker, mac_num, cin_unit, cout_unit, configs):
    """
    Simulate IPA in hardware, calculate cout_unit output channels in parallel.
    :param ifm: Re-arranged input feature map, w x h x cin_unit * pe_num
    :param ker: Re-arranged kernels, cin_unit * pe_num * 2
    :param mac_num:
    :param cin_unit:
    :param cout_unit:
    :param configs:
    :return:
    """
    h, w, _ = ifm.shape
    if np.shape(ifm)[:2] != configs['conv_result_blk_size'][:2]:
        tmp = np.zeros((configs['conv_result_blk_size'][0], configs['conv_result_blk_size'][1], np.shape(ifm)[2]))
        tmp[:np.shape(ifm)[0], :np.shape(ifm)[1], :] = ifm
        # If the actual fetched blk's size is smaller than the dma_blk_size, then complete it with zero
        ifm = tmp
    h, w, _ = ifm.shape
    dot_product_0 = np.zeros([h, w, cout_unit // 2])
    dot_product_1 = np.zeros([h, w, cout_unit // 2])
    for i in range(cout_unit // 2):
        dot_product_0[..., i], dot_product_1[..., i] = \
            pe_group(ifm[:, :, cin_unit * i:cin_unit * (i + 1)], ker[cin_unit * 2 * i:cin_unit * 2 * (i + 1)], mac_num,
                     cin_unit)

    return dot_product_0, dot_product_1


def output_ctrl(ipa_out, bias, temp, configs, first_round, last_round):
    """
    Calculate the output by adding bias, intermediate results, the steps are:
        1. re-arrange and concatenate channels, considering valid channel number
        2. expand and shift bias or temp results (save in last round)
        3. add bias if first round else add temp
        4. cut sum
        5. round if last round
    :param ipa_out: output of ipa modules, numpy array, (h, w, c)
    :param bias:
    :param temp: intermediate results from last round of output control
    :param configs:
    :param first_round: whether this is the first round, indicate when to add bias
    :param last_round: whether this is the last round, indicate when to round to 8 bits
    :return:
    """
    h, w, c = ipa_out.shape
    cout_unit = configs['cout_unit']
    ofm_blk = np.zeros((h, w, 64))
    # if the output is made up of multiple blocks, 'w' will clear the previously written blks
    # 1. re-arrange ipa_out channels
    valid_cout_num = int(configs['mac_num'] * configs['pe_num'] / configs['cin_unit'])
    for cc in range(cout_unit // 2):
        ipa_blk_idx = 2 * cc // valid_cout_num
        ofm_blk[..., 64 - 1 - 2 * cc] = ipa_out[..., 64 * ipa_blk_idx + 64 - 1 -
                                                (cc - valid_cout_num // 2 * ipa_blk_idx)]
        ofm_blk[..., 64 - 2 - 2 * cc] = ipa_out[..., 64 * ipa_blk_idx + 64 // 2 - 1 -
                                                (cc - valid_cout_num // 2 * ipa_blk_idx)]

    if configs['write_internal']:
        write2file(configs['out_dir'] + 'out_adder_a.txt', ofm_blk, configs['ipa_frac_len'], 26, line_num=64)

    # 2. expand and shift bias or temp results
    bias = bias * (2 ** configs['shift_num_bias'])
    bias = float2fix(bias, frac_len=configs['ipa_frac_len'], word_len=26, round_method='floor')
    bias_blk = np.zeros([h, w, 64])
    for hh in range(h):
        for ww in range(w):
            # only for the last layer of cr
            # bias_blk[hh, ww, :] = bias
            bias_blk[hh, ww, 64 - cout_unit:] = bias

    temp = temp * (2 ** configs['shift_num_temp'])
    temp = float2fix(temp, frac_len=configs['ipa_frac_len'], word_len=26, round_method='floor')

    # 3. add bias or temp
    if first_round:
        if configs['write_internal'] is True and first_round is True:
            write2file(configs['out_dir'] + 'out_adder_b.txt', bias_blk, configs['ipa_frac_len'], 26, line_num=64)
        ofm_blk = ofm_blk + bias_blk
    else:
        if configs['write_internal'] is True and first_round is False:
            write2file(configs['out_dir'] + 'out_adder_b.txt', temp, configs['ipa_frac_len'], 26, line_num=64)
        ofm_blk = ofm_blk + temp

    ofm_blk = float2fix(ofm_blk, frac_len=configs['ipa_frac_len'], word_len=27, round_method='floor')
    if configs['write_internal'] is True:
        write2file(configs['out_dir'] + 'out_adder_result.txt', ofm_blk, configs['ipa_frac_len'], 27, line_num=64)

    # 4. cut sum
    ofm_blk = float2fix(ofm_blk, frac_len=configs['temp_frac_len'], word_len=16, round_method='floor')
    if configs['write_internal'] is True:
        write2file(configs['out_dir'] + 'cut_data_16.txt', ofm_blk, configs['temp_frac_len'], 16, line_num=64)

    # 5. round
    if last_round:
        ofm_blk = float2fix(ofm_blk, frac_len=configs['output_fm_fix_lo'], word_len=8, round_method='round')
        if configs['write_final'] is True:
            write2file(configs['out_dir'] + 'cut_data_8.txt', ofm_blk, configs['output_fm_fix_lo'], 8, line_num=64)
    else:
        # write for comparison with hardware
        ofm_blk_tmp = float2fix(ofm_blk, frac_len=configs['output_fm_fix_lo'], word_len=8, round_method='floor')
        if configs['write_final'] is True:
            write2file(configs['out_dir'] + 'cut_data_8.txt', ofm_blk_tmp, configs['output_fm_fix_lo'], 8, line_num=64)

    return ofm_blk


def conv_blk(ifm_blk, ker_blk, bias_blk, ofm_blk, configs, first_blk=False, last_blk=False):
    """
    Calculate convolution of one block, the input are rearranged input feature map, kernel and bias from dma
    :param ifm_blk:
    :param ker_blk:
    :param bias_blk:
    :param ofm_blk:
    :param configs:
    :param first_blk:
    :param last_blk:
    :return:
    """
    ifm_h, ifm_w, ifm_c = ifm_blk.shape
    kh, kw, _ = ker_blk.shape
    ofm_c = configs['cout_unit']
    mac_num = configs['mac_num']
    cin_unit = configs['cin_unit']
    s_h, s_w = configs['ker_stride']
    valid_cout_num = int(configs['mac_num'] * configs['pe_num'] / configs['cin_unit'])
    valid_cout_num = min(valid_cout_num, ofm_c)
    dma_output_fm = None
    dma_output_ker = None
    for khh in range(kh):
        for kww in range(kw):
            ipa_out = np.zeros((configs['conv_result_blk_size'][0], configs['conv_result_blk_size'][1],
                                64 * math.ceil(ofm_c / valid_cout_num)))
            for cc in range(0, math.ceil(ofm_c / valid_cout_num)):
                co_start = 64 * cc + 64 - valid_cout_num // 2
                co_end = 64 * cc + 64
                # only for the last layer of cr
                # if ofm_c == 64:
                #     ker_start = valid_cout_num * cin_unit * (math.ceil(ofm_c / valid_cout_num) - 1 - cc)
                #     ker_end = ker_start + valid_cout_num * cin_unit
                # else:
                #     ker_start = valid_cout_num * cin_unit * (math.ceil(64 / valid_cout_num) - 1 - cc)
                #     ker_end = ker_start + valid_cout_num * cin_unit

                ker_start = valid_cout_num * cin_unit * (math.ceil(ofm_c / valid_cout_num) - 1 - cc)
                ker_end = ker_start + valid_cout_num * cin_unit
                # put strides into consideration
                ipa_out[..., co_start:co_end], ipa_out[..., co_start - 64 // 2:co_end - 64 // 2] = \
                    ipa(ifm_blk[khh:ifm_h - kh + khh + 1:s_h, kww:ifm_w - kw + kww + 1:s_w, :],
                        ker_blk[khh:khh + 1, kww:kww + 1, ker_start:ker_end].flatten(),
                        mac_num, cin_unit, valid_cout_num, configs)

                # format for comparison with hardware
                if cc == 0:
                    dma_output_fm = ifm_blk[khh:ifm_h - kh + khh + 1:s_h, kww:ifm_w - kw + kww + 1:s_w, :]
                    dma_output_ker = np.zeros((dma_output_fm.shape[0], dma_output_fm.shape[1], ker_end - ker_start))
                    for ii in range(dma_output_fm.shape[0]):
                        for jj in range(dma_output_fm.shape[1]):
                            dma_output_ker[ii, jj, :] = ker_blk[khh:khh + 1, kww:kww + 1, ker_start:ker_end]

                else:
                    dma_output_fm = np.concatenate((dma_output_fm,
                                                    ifm_blk[khh:ifm_h - kh + khh + 1:s_h,
                                                    kww:ifm_w - kw + kww + 1:s_w, :]), axis=2)

                    dma_output_ker_tmp = np.zeros((dma_output_fm.shape[0], dma_output_fm.shape[1], ker_end - ker_start))
                    for ii in range(dma_output_fm.shape[0]):
                        for jj in range(dma_output_fm.shape[1]):
                            dma_output_ker_tmp[ii, jj, :] = ker_blk[khh:khh + 1, kww:kww + 1, ker_start:ker_end]

                    dma_output_ker = np.concatenate((dma_output_ker, dma_output_ker_tmp), axis=2)

            # cut the valid part of the ipa_out out
            v_h = int((ifm_h - kh) / s_h + 1)
            v_w = int((ifm_w - kw) / s_w + 1)
            ipa_out = ipa_out[:v_h, :v_w, :]
            ofm_blk = ofm_blk[:np.shape(ipa_out)[0], :np.shape(ipa_out)[1], :]
            if configs['write_internal'] is True:
                write2file(configs['out_dir'] + 'dma_output_fm.txt', dma_output_fm, configs['input_fm_fix_lo'], 8,
                           line_num=ifm_blk.shape[2])
                write2file(configs['out_dir'] + 'dma_output_ker.txt', dma_output_ker, configs['weights_fix_lo'], 8,
                           line_num=ifm_blk.shape[2] * 2)
                write2file(configs['out_dir'] + 'dma_output_result.txt', ipa_out,
                           configs['input_fm_fix_lo'] + configs['weights_fix_lo'], configs['ipa_bit_len'], line_num=64)

            first_round = first_blk and khh == 0 and kww == 0
            last_round = last_blk and khh == kh - 1 and kww == kw - 1

            ipa_out = float2fix(ipa_out, frac_len=configs['ipa_frac_len'], word_len=26, round_method='floor')
            ofm_blk = output_ctrl(ipa_out, bias_blk, ofm_blk, configs, first_round, last_round)

    return ofm_blk


def conv(ifm, ker, bias, configs):
    """
    The main function of calculating a convolution layer
    :param ifm: input feature map, (h, w, c), channel is arranged in hardware way
    :param ker: kernel
    :param bias: bias
    :param configs:
    :return:
    """
    ifm_h, ifm_w, ifm_c = ifm.shape
    kh, kw, _, _ = ker.shape
    # To deal with the situation that the ofm_blk channel number is greater than the ofm channel number
    if configs['output_size'][2] < configs['ofm_blk_size'][2]:
        configs['ofm_blk_size'] = list(configs['ofm_blk_size'])
        configs['ofm_blk_size'][2] = configs['output_size'][2]
        configs['ofm_blk_size'] = tuple(configs['ofm_blk_size'])

    ofm_h, ofm_w, ofm_c = configs['conv_result_size']
    ceil_ofm_c = math.ceil(ofm_c / 64) * 64
    ofm_blk_h, ofm_blk_w, ofm_blk_c = configs['ofm_blk_size']
    ceil_ofm_blk_c = math.ceil(ofm_blk_c / 64) * 64
    ofm = np.zeros((ofm_h, ofm_w, ceil_ofm_c))

    cout_unit = configs['cout_unit']
    strides = configs['ker_stride']
    dma_blk_h, dma_blk_w, _ = configs['dma_blk_size']
    cin_unit = configs['cin_unit']

    pad_u, pad_d, pad_l, pad_r = (0, 0, 0, 0)
    ofm_w_pad = np.zeros(
        (configs['output_size'][0] + pad_u + pad_d, configs['output_size'][1] + pad_l + pad_r, ceil_ofm_c))
    if configs['post_padding']:
        pad_u, pad_d, pad_l, pad_r = configs['post_padding_size']
        ofm_w_pad = np.zeros(
            (configs['output_size'][0] + pad_u + pad_d, configs['output_size'][1] + pad_l + pad_r, ceil_ofm_c))

    blk_range = []
    pool_h, pool_w = configs['pooling_size'][:-1]
    pool_s_h, pool_s_w = configs['pooling_stride'][:-1]
    out_h_overlap, out_w_overlap = (pool_h - pool_s_h, pool_w - pool_s_w)
    k_h, k_w = configs['ker_size'][:-1]
    k_s_h, k_s_w = configs['ker_stride']
    in_h_overlap = k_s_h * out_h_overlap + k_h - k_s_h
    in_w_overlap = k_s_w * out_w_overlap + k_w - k_s_w
    real_s_h, real_s_w = (dma_blk_h - in_h_overlap, dma_blk_w - in_w_overlap)
    end_h = math.ceil((ifm_h - dma_blk_h) / real_s_h) + 1
    end_w = math.ceil((ifm_w - dma_blk_w) / real_s_w) + 1
    for hh in range(end_h):
        for ww in range(end_w):
            blk_range.append([hh, ww])
    configs['real_stride'] = (real_s_h, real_s_w)
    configs['ofm_stride'] = (ofm_blk_h - out_h_overlap, ofm_blk_w - out_w_overlap)

    valid_cout_num = int(configs['mac_num'] * configs['pe_num'] / configs['cin_unit'])
    ofm_sh, ofm_eh = 0, 0
    ofm_sw, ofm_ew = 0, 0
    co_start = 0
    real_ofm_h, real_ofm_w, _ = configs['output_size']
    ddr_obj = DdrSimu(ddr_addr_len=25, create_file=False)
    for blk_hw_idx in blk_range:
        ofm_blk = np.zeros((configs['conv_result_blk_size'][0], configs['conv_result_blk_size'][1], 64))
        for co_idx in range(math.ceil(ofm_c / cout_unit)):
            # only for the last layer of cr
            # if co_idx == 1:
            #     configs['cout_unit'] = 32
            for ci_idx in range(math.ceil(ifm_c / cin_unit)):
                for co_start in range(co_idx * cout_unit, (co_idx + 1) * cout_unit, ofm_blk_c):
                    blk_idx = blk_hw_idx + [ci_idx]
                    ifm_blk = dma_ifm(ifm, (kh, kw), strides, blk_idx, [dma_blk_h, dma_blk_w, cin_unit], valid_cout_num,
                                      configs)
                    ker_blk = dma_ker(ker[:, :, blk_idx[2] * cin_unit: (blk_idx[2] + 1) * cin_unit,
                                      co_start: co_start + ofm_blk_c], cin_unit, cin_unit * blk_idx[2] % cin_unit,
                                      ofm_blk_c, co_start % ofm_blk_c)

                    co_end = co_start + ofm_blk_c if co_start + ofm_blk_c < ofm_c else ofm_c
                    bias_blk = np.zeros(ofm_blk_c)
                    bias_blk[ofm_blk_c - (co_end - co_start):] = bias[co_start:co_end][::-1]

                    first_blk = True if blk_idx[2] == 0 else False
                    last_blk = True if blk_idx[2] == math.ceil(ifm_c / cin_unit) - 1 else False
                    conv_ofm_blk_h, conv_ofm_blk_w, conv_ofm_blk_c = configs['conv_result_blk_size']
                    ofm_sh = blk_idx[0] * (conv_ofm_blk_h - k_s_h * out_h_overlap)
                    ofm_sw = blk_idx[1] * (conv_ofm_blk_w - k_s_w * out_w_overlap)
                    ofm_sc = co_start
                    ofm_eh = ofm_sh + conv_ofm_blk_h if ofm_sh + conv_ofm_blk_h < ofm_h else ofm_h
                    ofm_ew = ofm_sw + conv_ofm_blk_w if ofm_sw + conv_ofm_blk_w < ofm_w else ofm_w
                    ofm_ec = co_start + ceil_ofm_blk_c if co_start + ceil_ofm_blk_c < ceil_ofm_c else ceil_ofm_c
                    # fit the output_ctrl_ref output into ofm
                    # adapted to the cases that the ofm_c_num cannot be exactly divided by the ofm_blk_c_num
                    ofm_blk_valid_h, ofm_blk_valid_w, ofm_blk_valid_c = \
                        ofm[ofm_sh:ofm_eh, ofm_sw:ofm_ew, ofm_sc:ofm_ec].shape
                    ofm_blk[:ofm_blk_valid_h, :ofm_blk_valid_w, :ofm_blk_valid_c] = \
                        ofm[ofm_sh:ofm_eh, ofm_sw:ofm_ew, ofm_sc:ofm_ec]

                    ofm_blk = conv_blk(ifm_blk, ker_blk, bias_blk, ofm_blk, configs, first_blk, last_blk)
                    ofm[ofm_sh:ofm_eh, ofm_sw:ofm_ew, ofm_sc:ofm_ec] = ofm_blk

            ddr_data = ddr_store(ofm_blk, blk_hw_idx + [co_idx], configs)
            blk_idx = blk_hw_idx + [co_idx]
            if configs['upsampling_scale'] != 0:
                ddr_data = upsample_nearest(ddr_data, configs['upsampling_scale'])
                up_ofm_sh = blk_idx[0] * ofm_blk_h * configs['upsampling_scale']
                up_ofm_sw = blk_idx[1] * ofm_blk_w * configs['upsampling_scale']
                up_ofm_sc = co_start
                up_ofm_eh = ofm_sh + ofm_blk_h * configs[
                    'upsampling_scale'] if ofm_sh + ofm_blk_h < real_ofm_h else real_ofm_h
                up_ofm_ew = ofm_sw + ofm_blk_w * configs[
                    'upsampling_scale'] if ofm_sw + ofm_blk_w < real_ofm_w else real_ofm_w
                up_ofm_ec = co_start + ceil_ofm_blk_c if co_start + ceil_ofm_blk_c < ceil_ofm_c else ceil_ofm_c
                pad_u, pad_d, pad_l, pad_r = configs['post_padding_size'] if configs['post_padding'] else [0, 0, 0, 0]
                # fit the output_ctrl_ref output into ofm
                ofm_w_pad[pad_u + up_ofm_sh:pad_u + up_ofm_eh, pad_l + up_ofm_sw:pad_l + up_ofm_ew,
                up_ofm_sc:up_ofm_ec] = ddr_data

                ofm_blk_range = []
                for hh in range(math.ceil(ofm_h / configs['upsampling_scale'])):
                    for ww in range(math.ceil(ofm_w / configs['upsampling_scale'])):
                        ofm_blk_range.append([hh, ww])
                for ofm_blk_idx in ofm_blk_range:
                    tmp_ofm_sc = 0
                    tmp_ofm_sh = ofm_blk_idx[0] * configs['upsampling_scale']
                    tmp_ofm_sw = ofm_blk_idx[1] * configs['upsampling_scale']
                    tmp_ofm_ec = 64
                    tmp_ofm_eh = tmp_ofm_sh + configs['upsampling_scale'] if tmp_ofm_sh + \
                                                                             configs[
                                                                                 'upsampling_scale'] < ofm_h else ofm_h
                    tmp_ofm_ew = tmp_ofm_sw + configs['upsampling_scale'] if tmp_ofm_sw + \
                                                                             configs[
                                                                                 'upsampling_scale'] < ofm_w else ofm_w

                    if configs['write_final']:
                        write2file(configs['out_dir'] + 'final_output.txt',
                                   ddr_data[tmp_ofm_sh:tmp_ofm_eh, tmp_ofm_sw:tmp_ofm_ew, tmp_ofm_sc:tmp_ofm_ec],
                                   configs['output_fm_fix_lo'], 8,
                                   line_num=64)
                        ddr_obj.ddr_write(copy.deepcopy(ddr_data), blk_idx, configs, base_addr=0)
            else:
                if configs['write_final']:
                    write2file(configs['out_dir'] + 'final_output.txt', ddr_data, configs['output_fm_fix_lo'], 8,
                               line_num=64)
                    ddr_obj.ddr_write(copy.deepcopy(ddr_data), blk_idx, configs, base_addr=0)

                ofm_sh = blk_idx[0] * ofm_blk_h
                ofm_sw = blk_idx[1] * ofm_blk_w
                ofm_sc = co_start
                ofm_eh = ofm_sh + ofm_blk_h if ofm_sh + ofm_blk_h < real_ofm_h else real_ofm_h
                ofm_ew = ofm_sw + ofm_blk_w if ofm_sw + ofm_blk_w < real_ofm_w else real_ofm_w
                ofm_ec = co_start + ceil_ofm_blk_c if co_start + ceil_ofm_blk_c < ceil_ofm_c else ceil_ofm_c

                pad_u, pad_d, pad_l, pad_r = configs['post_padding_size'] if configs['post_padding'] else [0, 0, 0, 0]
                ofm_w_pad[pad_u + ofm_sh:pad_u + ofm_eh, pad_l + ofm_sw:pad_l + ofm_ew, ofm_sc:ofm_ec] = ddr_data

    if configs['post_padding'] and configs['write_final']:
        hw_pad_u, hw_pad_d, hw_pad_l, hw_pad_r = configs['post_padding_size']
        pad_data_num = (hw_pad_u + hw_pad_d) * ofm_w + (hw_pad_l + hw_pad_r) * ofm_h + \
                       (hw_pad_u + hw_pad_d) * (hw_pad_l + hw_pad_r)
        pad_data = np.zeros((1, pad_data_num, ofm_c))
        write2file(configs['out_dir'] + 'final_output.txt', pad_data, 0, 8, line_num=64, arg='a')
    # ddr_obj.ddr_load(configs)
    return ofm_w_pad, ofm_w_pad


def saturation(x, frac_len, bit_width):
    """
    Check saturation for the inputs, this is only used for reading input from files
    :param x:
    :param frac_len:
    :param bit_width:
    :return:
    """
    sat_max = 2 ** (bit_width - 1) - 1
    sat_min = -2 ** (bit_width - 1)
    x = x * (2 ** frac_len)
    x[x > sat_max] = sat_max
    x[x < sat_min] = sat_min
    x = x / (2 ** frac_len)

    return x


def load_ifm(configs, first_layer=False):
    """
    This is used for loading input feature map
    :param configs:
    :param first_layer:
    :return:
    """
    if first_layer:
        if configs['ifm_file'].endswith('npy'):
            ifm = np.load(configs['ifm_file'])
        elif configs['ifm_file'].endswith('mat'):
            ifm = scio.loadmat(configs['ifm_file'])['value']
        ifm = float2fix(ifm, frac_len=configs['input_fm_fix_lo'], word_len=8, round_method='round')



    else:
        pad_u_r, pad_d_r, pad_l_r, pad_r_r = configs['pre_remove_padding']
        h, w, c = configs['input_size']
        ifm = np.zeros((h, w, c))
        content = []
        for filename in configs['ifm_file']:
            with open(filename, 'r') as fs:
                content += fs.readlines()

        line_c = min(c, 64)
        for cc in range(c // line_c):
            for hh in range(h):
                for ww in range(w):
                    idx = cc * h * w + hh * w + ww
                    content[idx] = content[idx].split(' ')[:-1]
                    ifm[hh, ww, cc * line_c:(cc + 1) * line_c] = \
                        np.array([int(x, 16) if int(x, 16) < 128 else int(x, 16) - 256 for x in
                                  content[idx][64 - line_c:]])
        ifm = ifm / (2 ** configs['input_fm_fix_lo'])
        h, w, c = configs['input_size']
        ifm = ifm[pad_u_r: h - pad_d_r, pad_l_r: w - pad_r_r, :]

    return ifm


def load_ker(configs):
    """
    This is used for loading kernel and bias files
    :param configs:
    :return:
    """
    # load bias and kernels
    ker = scio.loadmat(configs['ker_file'])['value']
    bias = scio.loadmat(configs['bias_file'])['value']
    # address overflow problems
    ker = saturation(ker, configs['weights_fix_lo'], 8)
    bias = saturation(bias.flatten(), configs['bias_fix_lo'], 16)

    return ker, bias


def padding(ifm, padding_mode):
    """
    Add paddings to ifm according to the padding_mode
    :param ifm: the original ifm to add paddings
    :param padding_mode: a 4-element tuple. The four numbers represent the number of
    paddings to be added in the direction of up, down, left, right separately.
    :return:
    """
    w_in, h_in, c_in = ifm.shape
    tmp = np.zeros((w_in + padding_mode[0] + padding_mode[1], h_in + padding_mode[2] + padding_mode[3], c_in),
                   dtype=np.float64)
    tmp[padding_mode[0]:padding_mode[0] + w_in, padding_mode[2]:padding_mode[2] + h_in, :] = ifm[:, :, :]
    return tmp


def ifm_reshape_to_27c(ifm):
    """
    Scan the 3-channel ifm with a 3*3 kernel and reshape it into (a-2, b-2, 27) when calculating layer_0
    :param ifm: the original ifm with a size like (a, b, 3)
    :return:
    """
    w, h, c = np.shape(ifm)
    new_ifm = np.zeros((w - 2, h - 2, c * 9))
    for i in range(w - 2):
        for j in range(h - 2):
            new_ifm[i, j, :] = np.reshape(np.transpose(ifm[i:i + 3, j:j + 3, :], (1, 0, 2)), 27, order='F')
    return new_ifm


def ker_reshpae_to_27c(ker):
    """
    Reshape the kernel with a shape (3, 3, 3, 32) into (1, 1, 27, 32) when calculating layer_0
    :param ker: the original ker with a size like (3, 3, 3, 32)
    :return:
    """
    # Assume the input kernel is arranged in RGB order
    k_h, k_w, k_in, k_out = np.shape(ker)
    ker_bgr = ker[..., ::-1, :]
    ker_reshape = np.reshape(np.transpose(ker_bgr, (1, 0, 2, 3)), [1, 1, k_h * k_w * k_in, -1], order='F')
    ker_reshape = ker_reshape[..., ::-1, :]

    return ker_reshape


def create_blank_file(out_dir, configs):
    """
    Create the blank middle result files before start testing a layer
    :param out_dir: the index of a layer to be tested
    :param configs:
    :return:
    """
    if configs['write_internal']:
        f = open(out_dir + 'dma_output_fm.txt', 'w+')
        f.close()
        f = open(out_dir + 'dma_output_ker.txt', 'w+')
        f.close()
        f = open(out_dir + 'dma_output_result.txt', 'w+')
        f.close()
        f = open(out_dir + 'out_adder_a.txt', 'w+')
        f.close()
        f = open(out_dir + 'out_adder_b.txt', 'w+')
        f.close()
        f = open(out_dir + 'out_adder_result.txt', 'w+')
        f.close()
        f = open(out_dir + 'cut_data_16.txt', 'w+')
        f.close()
        f = open(out_dir + 'ddr_addr.txt', 'w+')
        f.close()
        f = open(out_dir + 'activation_in.txt', 'w+')
        f.close()
        f = open(out_dir + 'res_adder_a.txt', 'w+')
        f.close()
        f = open(out_dir + 'res_adder_b.txt', 'w+')
        f.close()
        f = open(out_dir + 'pool_in.txt', 'w+')
        f.close()
        f = open(out_dir + 'pool_out.txt', 'w+')
        f.close()
        f = open(out_dir + 'res_adder_out_cut.txt', 'w+')
        f.close()
        f = open(out_dir + 'res_adder_out.txt', 'w+')
        f.close()


    if configs['write_final']:
        f = open(out_dir + 'cut_data_8.txt', 'w+')
        f.close()
        f = open(out_dir + 'final_output.txt', 'w+')
        f.close()
    f = open(configs['father_dir'] + '/results/ddr', 'wb')
    f.close()
    f = open(out_dir + 'ddr_addr.txt', 'w+')
    f.close()
    f = open(out_dir + 'ofm.txt', 'w+')
    f.close()


def leaky_relu(ofm):
    """
    Implement the leaky_relu function. For each ofm's element x<0, replace it with 0.125 * x
    :param ofm: the result of conv on the entire ifm
    :return:
    """
    ofm[ofm < 0] *= 0.125
    return ofm


def relu(ofm):
    """
    Implement the relu function. For each ofm's element x<0, replace it with 0
    :param ofm: the result of conv on the entire ifm
    :return:
    """
    ofm[ofm < 0] = 0
    return ofm


def create_final_output(ofm, configs):
    """
    Write the fina result into a file. This file can be used as the input of next layer.
    :param ofm: the result of conv on the entire ifm
    :param configs: the dict of configuration
    :return:
    """
    for idx in range(math.ceil(ofm.shape[2] / 64)):
        start = idx * 64
        end = (idx + 1) * 64
        write2file(configs['out_dir'] + 'ofm.txt', ofm[:, :, start:end], configs['output_fm_fix_lo'], 8, line_num=64)


def write_binary(ofm, configs, layer_id):
    """

    :param ofm:
    :param configs:
    :param layer_id:
    :return:
    """
    for idx in range(math.ceil(ofm.shape[2] / 64)):
        start = idx * 64
        end = (idx + 1) * 64
        data_bin = np.array(ofm[:, :, start:end] * (2 ** configs['output_fm_fix_lo']), np.int8)
        data_bin = data_bin.tobytes()
        if idx == 0:
            arg = 'wb'
        else:
            arg = 'ab'
        with open('{}layer{}.bin'.format(configs['out_dir'], layer_id), arg) as fs:
            fs.write(data_bin)


def create_final_output_without_padding(ofm, configs):
    """
    Write the final result into a file. This file can be used as the input of next layer.
    :param ofm: the result of conv on the entire ifm
    :param configs: the dict of configuration
    :return:
    """
    start = 0
    end = configs['cout_unit']
    # adapted to the case of layer_58, which has a channel number of 255
    # that cannot divided exactly by the blk_c number 64
    c_num = math.ceil(np.shape(ofm)[2] / configs['ofm_blk_size'][2]) * configs['ofm_blk_size'][2]
    temp = np.zeros((np.shape(ofm)[0], np.shape(ofm)[1], c_num))
    temp[:np.shape(ofm)[0], :np.shape(ofm)[1], :np.shape(ofm)[2]] = ofm[:, :, :]
    ofm = temp
    while end <= c_num:
        write2file(configs['out_dir'] + 'final_output_no_padding.txt', ofm[:, :, start:end],
                   configs['output_fm_fix_lo'], 8,
                   line_num=64)
        start += configs['cout_unit']
        end += configs['cout_unit']


def upsample_nearest(ofm, scale):
    """
    Upsampling the ofm accroding to scale with the interpolation of nearest.
    :param ofm: the ofm to be upsampled
    :param scale: the number of times the height and width need to be magnified.
    :return:
    """
    ofm_size = np.shape(ofm)
    scaled_ofm_size = (ofm_size[0] * scale, ofm_size[1] * scale, ofm_size[2])
    scaled_ofm = np.zeros(scaled_ofm_size)
    for h in range(scaled_ofm_size[0]):
        for w in range(scaled_ofm_size[1]):
            for c in range(scaled_ofm_size[2]):
                scaled_ofm[h, w, c] = ofm[h // scale, w // scale, c]
    return scaled_ofm


def ddr_store(ofm, blk_idx, configs):
    """
    Carry out the operation of res, pooling and activation on the ofm.
    activation_type 0 stands for relu and 1 stands for leaky_relu.
    :param blk_idx: The index of the current ofm block. Used to implement the res function.
    :param ofm: the ofm to be processed.
    :param configs: the configuration data.
    :return:
    """
    res_pos = configs['res_position']
    if res_pos == 0:
        ofm = res(ofm, blk_idx, configs)
        ofm = activation(ofm, configs)
        ofm = pooling(ofm, configs)
    elif res_pos == 1:
        ofm = activation(ofm, configs)
        ofm = res(ofm, blk_idx, configs)
        ofm = pooling(ofm, configs)
    elif res_pos == 2:
        ofm = activation(ofm, configs)
        ofm = pooling(ofm, configs)
        ofm = res(ofm, blk_idx, configs)
    elif res_pos == 3:
        ofm = activation(ofm, configs)
        ofm = pooling(ofm, configs)
    return ofm


def concatenate(ofm, configs):
    """
    Implement the concatenate layer.
    :param ofm: the ofm to be processed.
    :param configs: the configuration data.
    :return:
    """
    append_frac_len = configs['concatenate_frac_len']
    append_in_h, append_in_w, append_in_c = configs['append_size']
    append_in_h += 2
    append_in_w += 2
    pad_u_a, pad_d_a, pad_l_a, pad_r_a = configs['append_remove_padding']

    append_in = np.zeros((append_in_h, append_in_w, append_in_c))
    with open(configs['concatenate_file'], "r") as fs:
        content = fs.readlines()
        line_c = min(append_in_c, 64)
        for cc in range(append_in_c // line_c):
            for hh in range(append_in_h):
                for ww in range(append_in_w):
                    idx = cc * append_in_h * append_in_w + hh * append_in_w + ww
                    content[idx] = content[idx].split(' ')[:-1]
                    append_in[hh, ww, cc * line_c:(cc + 1) * line_c] = \
                        np.array([int(x, 16) if int(x, 16) < 128 else int(x, 16) - 256 for x in content[idx]])
    append_in = append_in / (2 ** append_frac_len)
    append_in = append_in[pad_u_a:append_in_h - pad_d_a, pad_l_a:append_in_w - pad_r_a, :]
    ofm = np.concatenate((ofm, append_in), axis=2)
    return ofm


def activation(ofm, configs):
    """
    Implement the activation layer.
    :param ofm: the ofm to be processed.
    :param configs: the configuration data.
    :return:
    """
    activation_type = configs['activation_type']
    if configs['write_internal']:
        write2file(configs['out_dir'] + 'activation_in.txt', ofm,
               configs['output_fm_fix_lo'], 8,
               line_num=64)
    if activation_type == 0:
        ofm = ofm
    elif activation_type == 1:
        ofm = relu(ofm)
    elif activation_type == 2:
        ofm = leaky_relu(ofm)
        ofm = float2fix(ofm, frac_len=configs['output_fm_fix_lo'], word_len=8, round_method='round')
    else:
        # activation_type == 3 ==> H-sigmoid
        # activation_type == 4 ==> H-swish
        # activation_type == 5 ==> LUT based
        # to be completed
        pass

    return ofm


def pooling(ofm, configs):
    """
    Implement the pooling layer. Disabled for yolo v3
    :param ofm: the ofm to be processed.
    :param configs: the configuration data.
    :return:
    """
    if configs['write_internal']:
        write2file(configs['out_dir'] + 'pool_in.txt', ofm,
               configs['output_fm_fix_lo'], 8,
               line_num=64)
    if configs['pooling'] == 1:
        pooling_size = configs['pooling_size'][:2]
        pooling_stride = configs['pooling_stride'][:2]
        # 1 stands for max pooling while 2 stands for average pooling
        pooling_type = configs['pooling_type']
        pool_pad = configs['pool_pad']
        pool_pad_size = configs['pool_pad_size'][:4]
        if pool_pad == 1:
            ofm = padding(ofm, pool_pad_size)
        h, w, c = np.shape(ofm)
        s_h, s_w = pooling_stride
        blk_h, blk_w = pooling_size
        pool_res = np.zeros((math.ceil((h - blk_h) / s_h) + 1, math.ceil((w - blk_w) / s_w) + 1, c))
        for hh in range(0, h - blk_h + s_h, s_h):
            for ww in range(0, w - blk_w + s_w, s_w):
                blk = ofm[hh:hh + blk_h, ww:ww + blk_w, :]
                if pooling_type == 1:
                    pool_res[math.ceil(hh / s_h), math.ceil(ww / s_w), :] = np.max(np.reshape(blk, (-1, c), order='C'),
                                                                                   axis=0)
                elif pooling_type == 2:
                    reshaped_blk = np.reshape(blk, (-1, c), order='C')
                    div_num, _ = np.shape(reshaped_blk)
                    below, above = math.floor(math.log2(div_num)), math.ceil(math.log2(div_num))
                    if abs(2 ** below - div_num) > abs(2 ** above - div_num):
                        div_num = 2 ** above
                    else:
                        div_num = 2 ** below
                    pool_res[math.ceil(hh / s_h), math.ceil(ww / s_w), :] = np.sum(reshaped_blk, axis=0) / div_num
    else:
        pool_res = ofm
    if configs['write_internal']:
        write2file(configs['out_dir'] + 'pool_out.txt', pool_res,
                   configs['output_fm_fix_lo'], 8,
                   line_num=64)
    return pool_res


def res(ofm, blk_idx, configs):
    """
    Implement the residual layer.
    :param ofm: the ofm to be processed.
    :param blk_idx:
    :param configs: the configuration data.
    :return:
    """
    res_frac_len = configs['res_frac_len']
    res_add_h, res_add_w, res_add_c = configs['output_size'] if configs['res_position'] == 2 \
        else configs['conv_result_size']
    pad_u_r, pad_d_r, pad_l_r, pad_r_r = configs['res_remove_padding']
    res_in_h, res_in_w, _ = configs['res_input_size']

    res_in = np.zeros((res_add_h, res_add_w, res_add_c))
    with open(configs['res_file'], "r") as fs:
        content = fs.readlines()
        line_c = min(res_add_c, 64)
        for cc in range(res_add_c // line_c):
            for hh in range(res_add_h):
                for ww in range(res_add_w):
                    idx = cc * res_add_h * res_add_w + hh * res_add_w + ww
                    content[idx] = content[idx].split(' ')[:-1]
                    res_in[hh, ww, cc * line_c:(cc + 1) * line_c] = \
                        np.array([int(x, 16) if int(x, 16) < 128 else int(x, 16) - 256 for x in
                                  content[idx][64 - line_c:]])
    res_in = res_in / (2 ** res_frac_len)

    res_adder = res_in[pad_u_r:res_add_h - pad_d_r, pad_l_r: res_add_w - pad_r_r, :]

    blk_h_idx, blk_w_idx, blk_c_idx = blk_idx
    blk_h, blk_w, blk_c = configs['ofm_blk_size'] if configs['res_position'] == 2 \
        else configs['conv_result_blk_size']
    start_h = blk_h * blk_h_idx
    start_w = blk_w * blk_w_idx
    start_c = blk_c * blk_c_idx
    end_h = min(start_h + blk_h, res_add_h)
    end_w = min(start_w + blk_w, res_add_w)
    end_c = min(start_c + blk_c, res_add_c)

    if configs['write_internal']:
        write2file(configs['out_dir'] + 'res_adder_a.txt', ofm,
                   configs['output_fm_fix_lo'], 8,
                   line_num=64)
        write2file(configs['out_dir'] + 'res_adder_b.txt', res_adder[start_h: end_h, start_w: end_w, start_c: end_c],
                   configs['output_fm_fix_lo'], 8,
                   line_num=64)

    ofm = ofm + res_adder[start_h: end_h, start_w: end_w, start_c: end_c]
    write2file(configs['out_dir'] + 'res_adder_out.txt', ofm,
               configs['output_fm_fix_lo'], 8,
               line_num=64)

    ofm = float2fix(ofm, frac_len=configs['output_fm_fix_lo'], word_len=8, round_method='floor')
    write2file(configs['out_dir'] + 'res_adder_out_cut.txt', ofm,
               configs['output_fm_fix_lo'], 8,
               line_num=64)

    return ofm


def float2fix(value, frac_len=0, word_len=8, round_method='floor'):
    min_value = -2 ** (word_len - 1)
    max_value = 2 ** (word_len - 1) - 1

    if round_method == 'round':
        fix_value = np.floor(value * (2 ** frac_len) + 0.5)
    else:
        fix_value = np.floor(value * (2 ** frac_len))

    fix_value[fix_value < min_value] = min_value
    fix_value[fix_value > max_value] = max_value
    fix_value = fix_value / (2 ** frac_len)

    return fix_value
