from src.conv_hw import *


def generate_coe_file(ins_file, out_file):
    """
    Read a instruction file and generate a coe file with it.
    :param ins_file: The path of the instruction file
    :param out_file:
    :return:
    """
    output_string = "memory_initialization_radix=2;\nmemory_initialization_vector =\n"

    with open(ins_file, 'r') as f:
        ins_data = f.readlines()
        for i in range(len(ins_data)):
            if i == 1024 - 1:
                output_string += str(ins_data[i][:-1]) + ';\n'
            else:
                output_string += str(ins_data[i][:-1]) + ',\n'
        for i in range(len(ins_data), 1024):
            if i == 1024 - 1:
                output_string += '00000000000000000000000000000000;\n'
            else:
                output_string += '00000000000000000000000000000000,\n'

    with open(out_file, 'w+') as f:
        f.write(output_string)


def generate_ins_hex_string(ins_file):
    """
    Read a instruction file and return a hex string of the ins.
    :param ins_file: The path of the instruction file
    :return:
    """
    with open(ins_file, 'r') as f:
        ins_data = f.readlines()
        ins_string = ''
        for i in range(math.ceil(len(ins_data) / 16)):
            for k in range(16):
                if i * 16 + k >= len(ins_data):
                    ins_string += 8 * '0'
                else:
                    cur = ins_data[i * 16 + k][:-1]
                    cur_int = int('0b' + cur, base=2)
                    ins_string += "%(num)08X" % {'num': cur_int}
            ins_string += '\n'

    ins_line_num = math.ceil(len(ins_data) / 16)
    return ins_string, ins_line_num


def ker_re_arrange(ker, bias, configs, first_layer=False):
    """
    Check saturation for the inputs, this is only used for reading input from files
    :param ker: The path of the folder where the weight and bias mat files locate
    :param bias: e.g. 0, 1, 2.. starts from 0.
    :param configs: Configurations used in the generation.
    :param first_layer:
    :return:
    """
    # The processing of the bias data
    bias_c_num = configs['output_size'][2]
    bias_num_round = math.ceil(bias_c_num / 64) * 64
    bias_new = np.zeros(bias_num_round)
    bias_num_left = bias_c_num
    cnt = 0
    for i in range(math.ceil(bias_c_num / 64)):
        real_c = min(bias_num_left, 64)
        cnt = cnt + 64 - real_c
        for k in range(real_c - 1, -1, -1):
            bias_new[cnt] = bias[i * 64 + k]
            cnt += 1
        bias_num_left = bias_num_left - real_c

    # The processing of the kernel data
    input_c = configs['input_size'][2]
    output_c = configs['output_size'][2]
    kh, kw, _ = configs['ker_size']
    kernel_out = 0
    output_num = int(configs['mac_num'] * configs['pe_num'] / configs['cin_unit'])

    if output_c % output_num != 0:
        out_pad = output_num - (output_c % output_num)
        pad_zero = np.zeros((np.shape(ker)[0], np.shape(ker)[1], np.shape(ker)[2], out_pad))
        ker = np.append(ker, pad_zero, axis=3)

    output_c = math.ceil(output_c / output_num) * output_num
    if first_layer:
        for out_c in range(math.ceil(output_c / 64)):
            real_out_c = min(output_c, 64)
            for in_c in range(math.ceil(input_c / 64)):
                real_in_c = min(input_c, 64)
                for out_c_in_64 in range(math.ceil(real_out_c / output_num)):
                    complete_round = np.zeros((1, 1))
                    for out_c_in_out_num in range(1, output_num // 2 + 1):
                        cnt = 0
                        c = np.zeros((kh * kw * real_in_c * 2, 1))
                        a = np.zeros((kh * kw * real_in_c * 2, 1))
                        b = np.zeros((kh * kw * real_in_c * 2, 1))
                        for inc in range(real_in_c - 1, -1, -1):
                            for ii in range(kh):
                                for jj in range(kw):
                                    a[cnt] = ker[ii, jj, inc + in_c * real_in_c, out_c * 64 + out_c_in_64 * output_num +
                                                 out_c_in_out_num * 2 - 1]
                                    b[cnt] = ker[ii, jj, inc + in_c * real_in_c, out_c * 64 + out_c_in_64 * output_num +
                                                 out_c_in_out_num * 2 - 2]
                                    c[cnt] = a[cnt]
                                    c[cnt + 1] = b[cnt]
                                    cnt += 2
                        if (real_in_c * kh * kw) % 32 != 0:
                            extra_zeros = np.zeros((2 * (32 - (real_in_c * kh * kw) % 32), 1))
                        else:
                            extra_zeros = np.zeros((0, 0))
                        if np.shape(extra_zeros) != (0, 0):
                            two_channel = np.append(c, extra_zeros, axis=0)
                        else:
                            two_channel = c
                        complete_round = np.append(two_channel.T, complete_round, axis=1)
                    complete_round = complete_round[:, :-1]
                    if np.shape(kernel_out) == ():
                        kernel_out = complete_round
                    else:
                        kernel_out = np.append(kernel_out, complete_round, axis=0)
    else:
        for out_c in range(math.ceil(output_c / 64)):
            if output_c % 64 == 0:
                real_out_c = min(output_c, 64)
            else:
                real_out_c = min(output_c - out_c * 64, 64)
            for in_c in range(math.ceil(input_c / 64)):
                if input_c % 64 == 0:
                    real_in_c = min(input_c, 64)
                else:
                    real_in_c = min(input_c - in_c * 64, 64)
                for ii in range(kh):
                    for jj in range(kw):
                        for out_c_in_64 in range(math.ceil(real_out_c / output_num)):
                            complete_round = np.zeros((1, 1))
                            for out_c_in_out_num in range(1, output_num // 2 + 1):
                                cnt = 0
                                c = np.zeros((real_in_c * 2, 1))
                                a = np.zeros((real_in_c * 2, 1))
                                b = np.zeros((real_in_c * 2, 1))
                                for inc in range(real_in_c - 1, -1, -1):
                                    a[cnt] = ker[ii, jj, inc + in_c * real_in_c, out_c * 64 + out_c_in_64 * output_num +
                                                 out_c_in_out_num * 2 - 1]
                                    b[cnt] = ker[ii, jj, inc + in_c * real_in_c, out_c * 64 + out_c_in_64 * output_num +
                                                 out_c_in_out_num * 2 - 2]
                                    c[cnt] = a[cnt]
                                    c[cnt + 1] = b[cnt]
                                    cnt += 2
                                if real_in_c == 16:
                                    extra_zeros = np.zeros((0, 0))
                                elif real_in_c % 32 != 0:
                                    extra_zeros = np.zeros((2 * (32 - real_in_c % 32), 1))
                                else:
                                    extra_zeros = np.zeros((0, 0))
                                if np.shape(extra_zeros) != (0, 0):
                                    two_channel = np.append(c, extra_zeros, axis=0)
                                else:
                                    two_channel = c
                                complete_round = np.append(two_channel.T, complete_round, axis=1)
                            complete_round = complete_round[:, :-1]
                            if np.shape(kernel_out) == ():
                                kernel_out = complete_round
                            else:
                                kernel_out = np.append(kernel_out, complete_round, axis=0)

    return kernel_out, bias_new


def generate_ker_string(ker, bias, configs):
    """
    Generate the string for kernel and bias, used for simulation
    :param ker:
    :param bias:
    :param configs:
    :return:
    """

    # generate bias string
    bias_c_num = configs['output_size'][2]
    bias_output_string = ""
    for w in range(math.ceil(bias_c_num / 64)):
        for t in range(32):
            cur_int = int(bias[w * 64 + t] * (2 ** configs['bias_fix_lo']))
            data_hex = hex(cur_int & int('1' * 16, 2))[2:]
            data_hex = '0' * (4 - len(data_hex)) + data_hex
            bias_output_string += data_hex
        bias_output_string += '\n'
        for t in range(32, 64):
            cur_int = int(bias[w * 64 + t] * (2 ** configs['bias_fix_lo']))
            data_hex = hex(cur_int & int('1' * 16, 2))[2:]
            data_hex = '0' * (4 - len(data_hex)) + data_hex
            bias_output_string += data_hex
        bias_output_string += '\n'
    bias_line_num = math.ceil(bias_c_num / 64) * 2

    # generate kernel string
    kernel_output_string = ""
    for w in range(np.shape(ker)[0]):
        short = ker[w, :]
        for t in range(16):
            sub_short = short[t * 64: (t + 1) * 64]
            for sub in range(64):
                cur_int = int(sub_short[sub] * (2 ** configs['weights_fix_lo']))
                data_hex = hex(cur_int & int('1' * 8, 2))[2:]
                data_hex = '0' * (2 - len(data_hex)) + data_hex
                kernel_output_string += data_hex
            kernel_output_string += '\n'
    ker_line_num = np.shape(ker)[0] * 16

    return kernel_output_string, bias_output_string, ker_line_num, bias_line_num

    
def rearrange_fc_weight(ker):
    in_channel_round = math.ceil(128 / 64)
    blk_in_size = 116 * 64 * in_channel_round
    output_num = 4
    ker = ker.squeeze()
    co = ker.shape[-1]
    ci = ker.shape[-2]
    if co % output_num != 0:
        co_t = co + output_num - co % output_num
        ker_t = np.zeros((ci, co_t))
        ker_t[:,:co] = ker
        ker = ker_t
    kernel_out = []
    round = []
    for i in range(0, ci, blk_in_size):
        for icr in range(in_channel_round):
            for j in range(0, co, output_num):
                in_cnt = min(blk_in_size, ci - i)
                for ii in range(icr*64, in_cnt, 64*in_channel_round):
                    for jj in range(output_num-1, -1, -2):
                        b0 = ker[i+ii:i+ii+64, j+jj]
                        b1 = ker[i+ii:i+ii+64, j+jj-1]
                        bt = []
                        for t in range(64):
                            bt.append(b0[t])
                            bt.append(b1[t])
                        bt.reverse()
                        round += bt
                        if len(round) == 1024:
                            kernel_out.append(round)
                            round = []
                for p in range(8 - int(in_cnt/(64 * in_channel_round)) % 8):
                    for po in range(4):
                        for pp in range(64):
                            round.append(0)
                        if len(round) == 1024:
                            kernel_out.append(round)
                            round = []
                if len(round) > 0:
                    for t in range(1024 - len(round)):
                        round.append(0)
                    kernel_out.append(round)
                    round = []
    kernel_out = np.array(kernel_out)
    return kernel_out

def generate4hw(ker, bias, configs, first_layer=False):
    """
    Generated the ddr and coe file. Ifm and weight files' paths are already known.
    :param ker:
    :param bias:
    :param configs: The number of the layer to be added as residual
    :param first_layer:
    :return:
    """
    # write coe file
    generate_coe_file(configs['ins_file'], configs['4hw']+'/ins.coe')

    # write ddr file
    re_ker, re_bias = ker_re_arrange(ker, bias, configs, first_layer)
    if configs['type'] == 0:
        re_ker = rearrange_fc_weight(ker, configs)
    kernel_string, bias_string, ker_line_num, bias_line_num = generate_ker_string(re_ker, re_bias, configs)
    ins_string, ins_line_num = generate_ins_hex_string(configs['ins_file'])

    ifm_line_num = 0
    with open(configs['4hw']+'/ddr.txt', 'w+') as f_ddr:
        if configs['ifm_file'][-3:] != 'mat':
            ifm_data = ''
            for filename in configs['ifm_file']:
                with open(filename, 'r') as f_ifm:
                    ifm_data += f_ifm.read()

            for char in ifm_data:
                if char != ' ':
                    f_ddr.write(char)
                if char == '\n':
                    ifm_line_num += 1
        else:
            ifm = scio.loadmat(configs['ifm_file'])['value']
            ifm = float2fix(ifm, frac_len=configs['input_fm_fix_lo'], word_len=8, round_method='round')
            ifm = ifm_reshape_to_27c(padding(ifm[0], (1, 1, 1, 1)))
            h, w, c = np.shape(ifm)
            ifm_frac_len = configs['input_fm_fix_lo']
            bit_len = 8
            hex_len = math.ceil(bit_len * 1.0 / 4)
            for hh in range(h):
                for ww in range(w):
                    f_ddr.write('0' * hex_len * 32)
                    for cc in range(c):
                        data_hex = hex(int(ifm[hh, ww, cc] * (2 ** ifm_frac_len)) & int('1' * bit_len, 2))[2:]
                        data_hex = '0' * (hex_len - len(data_hex)) + data_hex
                        f_ddr.write(data_hex)
                    f_ddr.write('0' * hex_len * 5)
                    f_ddr.write('\n')
                    ifm_line_num += 1
        f_ddr.write(kernel_string)
        f_ddr.write(bias_string)

        res_line_num = 0
        if configs['shortcut_source'] is None:
            res_line_num = 0
        else:
            with open(configs['res_file'], 'r') as f_res:
                res_data = f_res.read()
                for char in res_data:
                    if char != ' ':
                        f_ddr.write(char)
                    if char == '\n':
                        res_line_num += 1

        f_ddr.write(ins_string)

    # write line number for debug
    cur_pos = 0
    with open(configs['4hw']+'/line_num.txt', 'w+') as f_line_num:
        f_line_num.write('ifm:\t' + str(cur_pos) + '\t--\t' + str(cur_pos + ifm_line_num - 1) + '\n')
        cur_pos = cur_pos + ifm_line_num
        f_line_num.write('ker:\t' + str(cur_pos) + '\t--\t' + str(cur_pos + ker_line_num - 1) + '\n')
        cur_pos = cur_pos + ker_line_num
        f_line_num.write('bias:\t' + str(cur_pos) + '\t--\t' + str(cur_pos + bias_line_num - 1) + '\n')
        cur_pos = cur_pos + bias_line_num

        if res_line_num >= 1:
            f_line_num.write('res:\t' + str(cur_pos) + '\t--\t' + str(cur_pos + res_line_num - 1) + '\n')
            cur_pos = cur_pos + res_line_num
        else:
            f_line_num.write('res:\t -1\n')

        f_line_num.write('ins:\t' + str(cur_pos) + "\t--\t" + str(cur_pos + ins_line_num - 1) + '\n')


