from src.conv_hw import *
from src.Config import Config
from src.ddr_generation import generate4hw
from src.gen_board_binary import gen_network_4board
from src.utils import pre_process, post_process
import logging
import time
import cv2


def run_layer(layer_id, configs):
    """
    Run convolution for one layer
    :param layer_id:
    :param configs:
    :return:
    """
    ifm = load_ifm(configs, first_layer=(layer_id == 0))
    ker, bias = load_ker(configs)

    # generate files for hardware simulation
    # generate4hw(ker, bias, configs, first_layer=(layer_id == 0))

    # run software reference
    if layer_id == 0:
        # rearrangement for the first layer
        ifm = padding(ifm[0, :, :, :], (1, 1, 1, 1))
        ifm = ifm_reshape_to_27c(ifm)
        ker = ker_reshpae_to_27c(ker)
        configs['ker_size'] = [1, 1, 1]
        write2file(configs['out_dir'] + 'reshaped_ifm.txt',
                   ifm,
                   configs['input_fm_fix_lo'], 8,
                   line_num=64)
        ofm, ofm_w_pad = conv(ifm, ker, bias, configs)
    else:
        ofm, ofm_w_pad = conv(ifm, ker, bias, configs)

    if configs['post_padding']:
        create_final_output(ofm_w_pad, configs)
        write_binary(ofm_w_pad, configs, layer_id)
    else:
        create_final_output(ofm, configs)
        write_binary(ofm, configs, layer_id)

    return ofm_w_pad


def run_network(layer_id, network_name):
    network_configs = Config(network_name)
    # When debugging, we can test layer by layer
    # for layer_id in range(network_configs.config_dict['num'][0]):
    print('Running for layer: ' + str(layer_id))
    layer_configs = network_configs.generate_layer_config(layer_id)
    create_blank_file(layer_configs['out_dir'], layer_configs)
    ofm = run_layer(layer_id, layer_configs)

    print('Done.')
    return ofm


def generate_det_box_for_yolov3(network_name='yolov3_leaky_relu=0.1'):
    # generate detection box
    fpga_res = []
    father_dir = Config(network_name).father_dir
    to_be_generate = [58, 66, 74]
    for item in to_be_generate:
        network_configs = Config(network_name)
        configs = network_configs.generate_layer_config(item)
        h, w, c = configs['output_size']
        ifm = np.zeros((h, w, c))
        content = []
        for filename in [father_dir + r'\results\%s\layer_' % network_name + str(item) + '\ofm.txt']:
            with open(filename, 'r') as fs:
                content += fs.readlines()
        blk_n = math.ceil(c / 64)
        line_c = min(c, 64)
        _fpga_res = np.zeros((1, len(content) * 64))
        for l_n in range(len(content) - 1, -1, -1):
            content[l_n] = content[l_n].split(' ')[:-1]
            _fpga_res[0, l_n * 64:(l_n + 1) * 64] = np.array(
                [int(x, 16) if int(x, 16) < 128 else int(x, 16) - 256 for x in
                 content[l_n][64 - line_c:]])
        _fpga_res = _fpga_res.reshape([h * w * 64, blk_n], order='F').reshape([h, w, 64, blk_n])
        _fpga_res = np.concatenate(tuple([_fpga_res[..., blk_n - 1 - i] for i in range(blk_n)]), axis=2)
        _fpga_res = _fpga_res[..., blk_n * 64 - c:][..., ::-1] / (2 ** configs['output_fm_fix_lo'])
        ofm = _fpga_res
        fpga_res.append(ofm)
    logger = logging.getLogger(__name__)
    config = dict()
    config['img_shape'] = (1080, 1920, 3)
    config['img_in_path'] = father_dir + r"\data\%s" % network_name
    config['img_out_path'] = father_dir + r"\results\%s" % network_name
    config['inst_file'] = '/models/vehicle_det_inst.bin'
    config['ifm_shape'] = (416, 416, 3)
    config['ifm_frac_len'] = 4
    config['ifm_base_addr'] = 0
    config['ofm_shape'] = [(13, 13, 255), (26, 26, 255), (52, 52, 255)]
    config['ofm_frac_len'] = [4, 4, 4]
    config['ofm_base_addr'] = [821156, 850600, 909080]
    with open(father_dir + r'\data\%s' % network_name + r'\vehicle_det_classes.txt', 'r') as fs:
        content = fs.readlines()
    config['class_names'] = [c.strip() for c in content]
    with open(father_dir + r'\data\%s' % network_name + r'\vehicle_det_anchors.txt', 'r') as fs:
        content = fs.readline()
    content = [float(x) for x in content.split(',')]
    config['anchors'] = np.array(content).reshape(-1, 2)
    config['box_n'] = 3
    config['cls_n'] = 80
    config['score_ths'] = 0.6
    config['nms_ths'] = 0.45
    config['correct_flag'] = True
    post_process_res = post_process(logger, 'vehicle_det', "input_image.jpg", img_shape, fpga_res, config,
                                    father_dir + r"\data\%s" % network_name)
    for item in post_process_res:
        print(item)


def pre_process_for_yolov3(network_name='yolov3_leaky_relu=0.1'):
    father_dir = Config(network_name).father_dir
    img_array, img_shape = pre_process('vehicle_det', 7,
                                       father_dir + r'\data\%s\inp_0.jpg' % network_name,
                                       (416, 416, 3))
    new_array = np.zeros((1, 416, 416, 3))
    new_array[0, ...] = img_array
    new_array = new_array / (2 ** 7)
    np.save(father_dir + r"\data\%s\inp.npy" % network_name, new_array)
    return img_shape


if __name__ == '__main__':
    # for i in range(10):
    #     run_network(i, network_name='cr')

    # The master branch cannot handle layer_0 of resnet_v1_50 because pool_pad op is not supported.
    # layer_0 can be run at the ddr_simulate_by_block branch

    img_shape = pre_process_for_yolov3('yolov3_leaky_relu=0.1')
    for i in range(52, 53):
        ofm = run_network(i, network_name='yolov3_leaky_relu=0.1')
    generate_det_box_for_yolov3('yolov3_leaky_relu=0.1')

    '''
    This is used for generating onboard test binary files
    '''
    # gen_network_4board('cr')
