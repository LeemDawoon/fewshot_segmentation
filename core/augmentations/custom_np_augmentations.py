import numpy as np
import numbers

import cv2
import random


class TransposeInput(object):
    def __init__(self):
        pass

    def __call__(self, img):
        return np.transpose(img, (1, 2, 0)) # x, y, z 로 바꾸기


class TransposeOutput(object):
    def __init__(self):
        pass

    def __call__(self, img):
        return np.transpose(img, (2, 0, 1)) # z, x, y 로 바꾸기


class NumpyScale(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        # return cv2.resize(np.transpose(img, (1, 2, 0)), (self.size, self.size)) # TODO: 이거 transpose 하는 이유 - opencv는 matrix가 x, y, z 순서이다.
        return cv2.resize(img, (self.size, self.size))


class NumpyZScale(object):
    def __init__(self, size, channel):
        self.size = size # TODO: 어떤 건 self.size = (int(size), int(size)) 이렇게 하고 ..... 통일
        self.channel = channel

    def __call__(self, img):
        img = np.transpose(img, (2, 0, 1))  # z, x, y 로 바꾸기
        img = cv2.resize(img, (img.shape[2], self.size * self.channel))
        img = np.transpose(img, (1, 2, 0))  # x, y, z 로 바꾸기
        return img


class NumpyCenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        shape = np.shape(img)
        w, h = shape[0], shape[1]
        th, tw = self.size  # TODO: tw, th = self.size  아닌지 물어보기

        if w == tw and h == th:
            return img

        sx = w // 2 - tw // 2
        sy = h // 2 - th // 2

        return img[sx: sx + tw, sy: sy + tw]


class NumpyRandomCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        shape = np.shape(img)
        w, h = shape[0], shape[1]
        th, tw = self.size
        if w == tw and h == th:
            return img

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return img[x1: x1 + tw, y1: y1 + tw]


class NumpyRandomZCrop(object):
    def __init__(self, size=16, channel=4):
        self.size = size
        self.channel = channel

    def _get_random_range(self, lesion_mask_sum):
        idx_arr, = np.where(lesion_mask_sum != 0)
        idx_arr = np.array(idx_arr)
        if idx_arr.shape[0] == 0:
            # print('infarct is not exist !!!!!!')
            center_start = 0
            center_end = self.n_z
        else:
            center_start = idx_arr.min()
            center_end = idx_arr.max()
        random_center = random.randint(center_start, center_end)
        start_idx = int(random_center - self.size / 2)
        end_idx = int(random_center + self.size / 2)

        if start_idx < 0:
            dz = 0 - start_idx
            start_idx += dz
            end_idx += dz
        elif end_idx >= self.n_z:
            dz = self.n_z - end_idx - 1
            start_idx += dz
            end_idx += dz

        return start_idx, end_idx

    def __call__(self, img):
        self.n_z = int(img.shape[2] / self.channel) # z-축 크기
        if self.n_z <= self.size:
            return img

        # self.n_xy = img.shape[0]
        lesion_mask_sum = np.sum(img[:, :, -self.n_z:], axis=(0, 1))
        start_idx, end_idx = self._get_random_range(lesion_mask_sum)

        crop = None
        for i in range(self.channel):
            if i == 0:
                crop = img[:, :, start_idx: end_idx]
            else:
                crop = np.concatenate([crop, img[:, :, start_idx + self.n_z * i: end_idx + self.n_z * i]], axis=-1)
        return crop

class NumpyRandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        (h, w) = img.shape[:2]
        center = (w / 2, h / 2)

        # Perform the rotation
        M = cv2.getRotationMatrix2D(center, rotate_degree, 1.0)
        img = cv2.warpAffine(img, M, (w, h))
        return img


class NumpyFiveTenCrop(object):
    def __init__(self, size, n_crop=5):
        if n_crop not in [5, 10]:
            raise Exception('Error: n_crop should be 5 or 10')

        self.n_crop = n_crop
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def _five_crop(self, img, sx, sy, tw, th):
        center = img[sx: sx + tw, sy: sy + th, :]
        left = img[0: tw, sy: sy + th, :]
        right = img[img.shape[0] - tw:, sy: sy + th, :]
        top = img[sx: sx + tw, 0: th, :]
        bottom = img[sx: sx + tw, img.shape[1] - th:, :]
        return [center, left, right, top, bottom]

    def __call__(self, img):
        shape = np.shape(img)
        w, h = shape[0], shape[1]
        tw, th = self.size
        if w == tw and h == th:
            return img

        sx = w // 2 - tw // 2
        sy = h // 2 - th // 2

        img_list = self._five_crop(img, sx, sy, tw, th)
        if self.n_crop == 10:
            img = np.fliplr(img)
            img_list.extend(self._five_crop(img, sx, sy, tw, th))
        img = np.concatenate(img_list, axis=2) # x축으로 concat
        return img


class NumpyInfarctROI(object):
    def __init__(self, size, n_roi=7, channel=4):
        self.channel = channel
        self.n_roi = n_roi
        self.size = size

    def loc_correct(self, loc):
        loc -= self.size / 2
        if loc < 0:
            loc = 0
        elif loc > self.n_xy - self.size - 1:
            loc = self.n_xy - self.size - 1
        return int(loc)

    def _get_grid_roi(self, img):
        n_axis_split = int(np.sqrt(self.n_roi))
        margin_random_i = random.randint(0, self.size)  # 약간 랜덤함을 주기위함
        # idx_list = np.linspace(0, self.n_xy - self.size, num=n_axis_split, endpoint=False, dtype=np.int).tolist()
        idx_list = np.linspace(margin_random_i, self.n_xy - self.size - margin_random_i, num=n_axis_split, endpoint=False, dtype=np.int).tolist()
        is_first = True
        for xs in idx_list:
            for ys in idx_list:
                if is_first:
                    roi = img[xs:xs + self.size, ys:ys + self.size, :]
                    is_first = False
                else:
                    roi = np.concatenate([roi, img[xs:xs + self.size, ys:ys + self.size, :]], axis=-1)
        return roi

    def _get_roi(self, img, lesion_mask):
        (x, y, z) = np.where(lesion_mask != 0)
        n_lesion = x.shape[0]
        if n_lesion < self.n_roi:
            return self._get_grid_roi(img)
        else:
            margin_random_i = random.randint(0, int(n_lesion/10))  # 약간 랜덤함을 주기위함
            # idx_list = np.linspace(0, n_lesion, num=self.n_roi, endpoint=False, dtype=np.int).tolist()
            idx_list = np.linspace(margin_random_i, n_lesion - margin_random_i, num=self.n_roi, endpoint=False, dtype=np.int).tolist()
            roi = None
            for i, idx in enumerate(idx_list):
                xs = self.loc_correct(x[idx])
                ys = self.loc_correct(y[idx])
                if i == 0:
                    roi = img[xs:xs + self.size, ys:ys + self.size, :]
                else:
                    roi = np.concatenate([roi, img[xs:xs + self.size, ys:ys + self.size, :]], axis=-1)
            return roi

    def __call__(self, img):
        self.n_z = int(img.shape[2] / self.channel)  # z-축 크기
        self.n_xy = img.shape[0]

        if self.n_xy == self.size:
            return img

        lesion_mask = img[:, :, -self.n_z:]
        roi_list = self._get_roi(img, lesion_mask)
        return roi_list



class NumpyRandomHorizontalFlip(object):
    def __init__(self):
        pass

    def __call__(self, img):
        if random.random() < 0.5:
            return np.fliplr(img).copy()
        return img

class NumpyRandomVerticalFlip(object):
    def __init__(self):
        pass

    def __call__(self, img):
        if random.random() < 0.5:
            return np.flipud(img).copy()
        return img

class NumpyVerticalFlip(object):
    def __init__(self):
        pass

    def __call__(self, img):
        return np.flipud(img).copy()


class NumpyNormalize(object):
    def __init__(self, mean_std):
        self.mean = mean_std[0]
        self.std = mean_std[1]

    def __call__(self, img):
        for i, (m, s) in enumerate(zip(self.mean, self.std)):
            img[:, :, i] = (img[:, :, i] - m)/s

        return img