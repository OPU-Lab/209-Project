"""
Copyright (c) 2019 Nunova All Rights Reserved.
NOTICE: All information contained herein is, and remains the property of
Nunova, if any. The intellectual and technical concepts
contained herein are proprietary to Nunova and may be
covered by U.S. and Foreign Patents, patents in process, and are protected
by trade secret or copyright law. Dissemination of this information or
reproduction of this material is strictly forbidden unless prior written
permission is obtained from Nunova
"""
import numpy as np
from PIL import Image
import cv2
import os


def logistic(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x):
    return np.exp(x) / np.expand_dims(np.sum(np.exp(x), axis=-1), axis=-1)


def post_process(log, model, img_inst, img_shape, fpga_res, config, img_in_path):
    """
    Post-process the output according to different models
    :param log:
    :param model: model name, currently support: vehicle_det, lp_det, lp_cls
    :param img_inst:
    :param img_shape:
    :param fpga_res:
    :param config:
    :param img_in_path:
    :return:
    """
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    boxes = []
    box_conf = []
    box_probs = []

    if model == 'vehicle_det' or model == 'lp_det':
        for idx in range(len(fpga_res)):
            if model == 'vehicle_det':
                _boxes, _box_conf, _box_probs = convert_res(model, config, config['anchors'][anchor_mask[idx]],
                                                            fpga_res[idx], img_shape)
            else:
                _boxes, _box_conf, _box_probs = convert_res(model, config, config['anchors'], fpga_res[idx], img_shape)
            boxes.append(_boxes)
            box_conf.append(_box_conf)
            box_probs.append(_box_probs)

        boxes = np.concatenate(boxes, axis=0)
        box_conf = np.concatenate(box_conf, axis=0)
        box_probs = np.concatenate(box_probs, axis=0)

        det_boxes, det_scores, det_classes = fast_region(config, boxes, box_conf, box_probs)
        # log.logger.debug
        log.debug('{} results: \nboxes: {}\nscores: {}\nclasses: {}\n.'.format(model, det_boxes,
                                                                                      det_scores, det_classes))

        if model == 'vehicle_det':
            result = convert_box(log, config, det_boxes, det_scores, det_classes, img_in_path + img_inst)
        else:
            result = []
            save_det_img(log, img_in_path, img_inst, config, det_boxes, det_classes)

        return result
    elif model == 'lp_cls' or model == 'lp_cls_new':
        confidence = 0.0
        pred = softmax(fpga_res[0][0, 2:, :])
        classes = np.argmax(pred, axis=-1)
        plate = ''
        for i, one in enumerate(classes):
            if one < len(config['class_names']) and (i == 0 or (one != classes[i-1])):
                plate += config['class_names'][one]
                confidence += pred[i][one]
        confidence /= len(plate)

        result = (plate, confidence)
        log.logger.debug('{} results: \nplates: {}\nconfidence: {}\n'.format(model, plate, confidence))

        return result
    else:
        log.logger.error('Do not support this model: {}'.format(model))


def letterbox_img(src_img, dst_img_shape):
    """
    Resize image with unchanged aspect ratio using padding
    :param src_img: PIL Image object
    :param dst_img_shape:
    :return:
    """

    src_w, src_h = src_img.size
    dst_h, dst_w, _ = dst_img_shape
    scale = min(dst_w / src_w, dst_h / src_h)
    rsz_w = int(src_w * scale)
    rsz_h = int(src_h * scale)

    rsz_img = src_img.resize((rsz_w, rsz_h), Image.BICUBIC)
    letterbox = Image.new('RGB', (dst_w, dst_h), (128, 128, 128))
    letterbox.paste(rsz_img, ((dst_w - rsz_w) // 2, (dst_h - rsz_h) // 2))

    return letterbox


def pre_process(model, frac_len, img_in_path, dst_img_shape):
    """
    Pre-process the input images according to different models
    :param model: model name, currently support: vehicle_det, lp_det, lp_cls
    :param frac_len:
    :param img_in_path:
    :param dst_img_shape:
    :return:
    """
    img = cv2.imread(img_in_path)
    rsz_h, rsz_w, _ = dst_img_shape
    img_array = cv2.resize(img, (rsz_w, rsz_h))
    img_array = img_array / 255.
    img_array = float2fix(img_array, frac_len, 8, round_method='round')
    
    if model == 'vehicle_det' or model == 'lp_cls':
        img_array = img_array[..., ::-1]

    return img_array, img.shape


def correct_box(ifm_shape, img_shape, box_xy, box_wh, correct_flag=False):
    ifm_h, ifm_w, _ = ifm_shape
    img_h, img_w, _ = img_shape

    if correct_flag:
        if ifm_w / img_w < ifm_h / img_h:
            cor_w = ifm_w
            cor_h = np.floor((img_h * ifm_w) / img_w + 0.5)
        else:
            cor_w = np.floor((img_w * ifm_h) / img_h + 0.5)
            cor_h = ifm_h

        box_xy[..., 0] = (box_xy[..., 0] - (ifm_w - cor_w) / 2. / ifm_w) / (cor_w / ifm_w)
        box_xy[..., 1] = (box_xy[..., 1] - (ifm_h - cor_h) / 2. / ifm_h) / (cor_h / ifm_h)
        box_wh[..., 0] = box_wh[..., 0] * (ifm_w / cor_w)
        box_wh[..., 1] = box_wh[..., 1] * (ifm_h / cor_h)

    box_xy[box_xy < 0] = 0
    box_xy_min = box_xy - (box_wh / 2.)
    box_xy_max = box_xy + (box_wh / 2.)

    boxes = np.concatenate([
        box_xy_min[..., 1].reshape(np.prod(box_xy_min[..., 1].shape), 1),
        box_xy_min[..., 0].reshape(np.prod(box_xy_min[..., 0].shape), 1),
        box_xy_max[..., 1].reshape(np.prod(box_xy_max[..., 1].shape), 1),
        box_xy_max[..., 0].reshape(np.prod(box_xy_max[..., 0].shape), 1)
        ], axis=1)
    boxes *= np.array([img_h, img_w, img_h, img_w])

    return boxes


def convert_res(model, config, anchor, fpga_res, img_shape):
    """
    Convert the results of last layer to bounding box parameters
    :param model:
    :param config:
    :param anchor:
    :param fpga_res:
    :param img_shape:
    :return:
    """
    reg_h, reg_w, _ = fpga_res.shape
    ifm_h, ifm_w, _ = config['ifm_shape']

    fpga_res = fpga_res.reshape([reg_h, reg_w, config['box_n'], config['cls_n'] + 5])

    box_xy = logistic(fpga_res[..., :2])
    box_wh = np.exp(fpga_res[..., 2:4])
    box_conf = logistic(fpga_res[..., 4:5])
    box_probs = softmax(fpga_res[..., 5:])

    box_idx = np.zeros([reg_h, reg_w, 1, 2])
    box_idx[..., 0, 0] = np.array([[i for i in range(reg_w)]] * reg_h)
    box_idx[..., 0, 1] = np.transpose(box_idx[..., 0, 0], [1, 0])
    box_xy = (box_xy + box_idx) / (reg_h, reg_w)
    if model == 'vehicle_det':
        box_wh = box_wh * np.reshape(anchor, [1, 1, config['box_n'], 2]) / (ifm_h, ifm_w)
    else:
        box_wh = box_wh * np.reshape(anchor, [1, 1, config['box_n'], 2]) / (reg_h, reg_w)

    boxes = correct_box(config['ifm_shape'], img_shape, box_xy, box_wh, config['correct_flag'])

    box_conf = np.reshape(box_conf, [-1, box_conf.shape[-1]])
    box_probs = np.reshape(box_probs, [-1, box_probs.shape[-1]])

    return boxes, box_conf, box_probs


def nms(boxes, scores, thresh):
    order = scores.argsort()[::-1]
    h1 = boxes[:, 0]
    w1 = boxes[:, 1]
    h2 = boxes[:, 2]
    w2 = boxes[:, 3]
    areas = (h2 - h1 + 1) * (w2 - w1 + 1)

    nms_boxes = []
    nms_scores = []
    while order.size > 0:
        i = order[0]
        nms_boxes.append(boxes[i])
        nms_scores.append(scores[i])
        hh1 = np.maximum(h1[i], h1[order[1:]])
        ww1 = np.maximum(w1[i], w1[order[1:]])
        hh2 = np.minimum(h2[i], h2[order[1:]])
        ww2 = np.minimum(w2[i], w2[order[1:]])
        h = np.maximum(0.0, hh2 - hh1 + 1)
        w = np.maximum(0.0, ww2 - ww1 + 1)
        inter = h * w
        over = inter / (areas[i] + areas[order[1:]] - inter)
        idx = np.where(over <= thresh)[0]
        order = order[idx + 1]

    return nms_boxes, nms_scores


def fast_region(config, boxes, box_conf, box_probs):
    scores = box_conf * box_probs
    mask = scores >= config['score_ths']

    # do nms for each class
    nms_boxes = []
    nms_scores = []
    nms_classes = []
    for c in range(scores.shape[1]):
        class_boxes = boxes[mask[..., c].flatten()]
        class_box_scores = scores[mask[..., c], c]
        _nms_boxes, _nms_scores = nms(class_boxes, class_box_scores, config['nms_ths'])
        if _nms_boxes:
            nms_boxes.append(_nms_boxes)
            nms_scores.append(_nms_scores)
            nms_classes.append([c] * len(_nms_scores))
    
    if nms_boxes:
        nms_boxes = np.concatenate(nms_boxes, axis=0)
        nms_scores = np.concatenate(nms_scores, axis=0)
        nms_classes = np.concatenate(nms_classes, axis=0)

    return nms_boxes, nms_scores, nms_classes


def save_det_img(log, img_in_path, img_inst, config, det_boxes, det_classes):
    """
    Crop from the original image according to the bounding box, and save the cropped image
    Use PIL.Image.crop, (x0, y0) to (x1, y1) means (left, top) to (right, bottom)
    :param log:
    :param img_in_path:
    :param img_inst:
    :param config:
    :param det_boxes:
    :param det_classes:
    :return:
    """
    img = cv2.imread(img_in_path+img_inst)
    for i in range(len(det_boxes)):
        log.logger.debug('Class name is {}'.format(config['class_names'][det_classes[i]]))
        top, left, bottom, right = det_boxes[i]
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(config['img_shape'][0], np.floor(bottom + 0.5).astype('int32'))
        right = min(config['img_shape'][1], np.floor(right + 0.5).astype('int32'))
        if top < bottom and left < right:
            crop_img = img[top:bottom, left:right, :]
            crop_img_name = os.path.join(config['img_out_path'], img_inst.split('.jpg')[0] + '-lp_' + str(i) + '.jpg')
            cv2.imwrite(crop_img_name, crop_img)


def convert_box(log, config, det_boxes, det_scores, det_classes, image_path):
    # log.logger.debug
    log.debug('Found {} boxes for {}'.format(len(det_boxes), image_path.split('/')[-1]))
    box_result_bucket = []
    for i, c in reversed(list(enumerate(det_classes))):
        predicted_class = config['class_names'][c]
        box = det_boxes[i]
        score = det_scores[i]

        if box is None or predicted_class not in ['car']:
            continue

        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(config['img_shape'][0], np.floor(bottom + 0.5).astype('int32'))
        right = min(config['img_shape'][1], np.floor(right + 0.5).astype('int32'))

        box_result_bucket.append([image_path, [left, top, right, bottom], predicted_class, score])

    return box_result_bucket


def float2fix(value, frac_len=0, word_len=8, round_method='floor'):
    min_value = -2 ** (word_len - 1)
    max_value = 2 ** (word_len - 1) - 1

    if round_method == 'round':
        fix_value = np.floor(value * (2 ** frac_len) + 0.5)
    else:
        fix_value = np.floor(value * (2 ** frac_len))

    fix_value[fix_value < min_value] = min_value
    fix_value[fix_value > max_value] = max_value

    fix_value = fix_value.astype(np.int8)

    return fix_value
