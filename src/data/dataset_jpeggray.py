import random

import cv2
import numpy as np
import torch
import torch.utils.data as data
import utils.utils_image as util


class DatasetJPEG(data.Dataset):
    def __init__(self, opt):
        super(DatasetJPEG, self).__init__()
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.patch_size = self.opt['H_size'] if opt['H_size'] else 64

        # -------------------------------------
        # get the path of H, return None if input is None
        # -------------------------------------
        self.paths_H = util.get_image_paths(opt['dataroot_H'])

    def __getitem__(self, index):
        # -------------------------------------
        # get H image
        # -------------------------------------
        H_path = self.paths_H[index]
        img_H = util.imread_uint(H_path, self.n_channels)
        L_path = H_path

        if self.opt['phase'] == 'train':
            # --------------------------------
            # get L/H/M patch pairs
            # --------------------------------
            H, W = img_H.shape[:2]
            self.patch_size_plus8 = self.patch_size+8

            # ---------------------------------
            # randomly crop the patch
            # ---------------------------------
            rnd_h = random.randint(0, max(0, H - self.patch_size_plus8))
            rnd_w = random.randint(0, max(0, W - self.patch_size_plus8))
            patch_H = img_H[rnd_h:rnd_h + self.patch_size_plus8, rnd_w:rnd_w + self.patch_size_plus8, :]

            # ---------------------------------
            # augmentation - flip, rotate
            # ---------------------------------
            mode = random.randint(0, 7)
            patch_H = util.augment_img(patch_H, mode=mode)

            # ---------------------------------
            # HWC to CHW, numpy(uint) to tensor
            # ---------------------------------
            img_L = patch_H.copy()

            # get noise level

            if random.random() > 0.75:
                quality_factor = random.randint(5, 95)
            else:
                quality_factor = random.choice([10, 20, 30, 40, 50, 60])

            noise_level = (100-quality_factor)/100.0
            #img_L = cv2.cvtColor(img_L, cv2.COLOR_RGB2BGR)

            if random.random() > 0.25:
                img_L = util.rgb2ycbcr(img_L)
            else:
                img_L = cv2.cvtColor(img_L, cv2.COLOR_RGB2GRAY)

            img_H = img_L.copy()
            _, encimg = cv2.imencode('.jpg', img_L, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
            img_L = cv2.imdecode(encimg, 0)

            noise_level = torch.FloatTensor([noise_level])

            H, W = img_H.shape[:2]
            if random.random() > 0.5:
                rnd_h = random.randint(0, max(0, H - self.patch_size))
                rnd_w = random.randint(0, max(0, W - self.patch_size))
            else:
                rnd_h = 0
                rnd_w = 0
            img_H = img_H[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size]
            img_L = img_L[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size]

            img_L, img_H = util.uint2tensor3(img_L), util.uint2tensor3(img_H)

        else:
            # --------------------------------
            # get L/H/M image pairs
            # --------------------------------

            H_path = self.paths_H[index]
            img_H = cv2.imread(H_path, cv2.IMREAD_UNCHANGED) 
            L_path = H_path

            img_L = img_H.copy()

            quality_factor = 10
            noise_level = (100-quality_factor)/100.0
            img_H = img_L.copy()
            _, encimg = cv2.imencode(
                '.jpg',
                img_L,
                [int(cv2.IMWRITE_JPEG_QUALITY),
                quality_factor]
            )

            img_L = cv2.imdecode(encimg, 0)

            noise_level = torch.FloatTensor([noise_level])
            img_L, img_H = util.uint2tensor3(img_L[..., np.newaxis]), util.uint2tensor3(img_H[..., np.newaxis])

        return {'L': img_L, 'H': img_H, 'qf': noise_level, 'L_path': L_path, 'H_path': H_path}

    def __len__(self):
        return len(self.paths_H)
