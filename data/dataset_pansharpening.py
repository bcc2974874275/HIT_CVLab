import random
import numpy as np
import torch.utils.data as data
import utils.utils_image as util


class DatasetPansharpening(data.Dataset):
    def __init__(self, opt):
        super(DatasetPansharpening, self).__init__()
        self.opt = opt

        # ------------------------------------
        # get paths of L/H
        # ------------------------------------
        self.paths_gt = util.get_h5_paths(opt['dataroot_gt'])
        self.paths_ms = util.get_h5_paths(opt['dataroot_ms'])
        self.paths_lms = util.get_h5_paths(opt['dataroot_lms'])
        self.paths_pan = util.get_h5_paths(opt['dataroot_pan'])

    def __getitem__(self, index):
        # ------------------------------------
        # get image
        # ------------------------------------
        gt_path = self.paths_gt[index]
        img_gt = util.load_h5(gt_path, name='gt')
        img_gt = util.pansharpening2single(img_gt)

        ms_path = self.paths_ms[index]
        img_ms = util.load_h5(ms_path, name='ms')
        img_ms = util.pansharpening2single(img_ms)

        lms_path = self.paths_lms[index]
        img_lms = util.load_h5(lms_path, name='lms')
        img_lms = util.pansharpening2single(img_lms)

        pan_path = self.paths_pan[index]
        img_pan = util.load_h5(pan_path, name='pan')
        img_pan = util.pansharpening2single(img_pan)

        # ------------------------------------
        # if train, data augment
        # ------------------------------------
        if self.opt['phase'] == 'train':
            pass

            # --------------------------------
            # augmentation - flip and/or rotate
            # --------------------------------
            # mode = random.randint(0, 7)
            # img_L, img_H = util.augment_img(img_L, mode=mode), util.augment_img(img_H, mode=mode)

        # ------------------------------------
        # L/H pairs, HWC to CHW, numpy to tensor
        # ------------------------------------
        img_gt, img_ms, img_lms, img_pan = util.chw2tensor(img_gt), util.chw2tensor(img_ms), \
                                           util.chw2tensor(img_lms), util.chw2tensor(img_pan)



        return {'gt': img_gt, 'ms': img_ms, 'lms': img_lms, 'pan': img_pan, \
                 'gt_path': gt_path, 'ms_path': ms_path, 'pan_path': pan_path, 'lms_path': lms_path}

    def __len__(self):
        return len(self.paths_pan)
