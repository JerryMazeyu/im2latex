from os.path import join

from torch.utils.data import Dataset
import torch
import os
from torchvision import transforms as T
from PIL import Image
import numpy as np
import cv2
from scipy import misc


def __gray2RGB(img):  # 新增的灰度转RGB图像的预处理函数
    img = np.array(img)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)




transform = T.Compose([  # 所有的图像经过Resize成（32，128）大小 --> RGB --> Tensor
    T.Resize((32, 128)),
    T.Lambda(lambda img: __gray2RGB(img)),
    T.ToTensor()
])

class Im2LatexDataset(Dataset):
    def __init__(self, data_dir, split, max_len):
        """args:
        data_dir: root dir storing the prepoccessed data
        split: train, validate or test
        """
        assert split in ["train", "validate", "test"]
        self.data_dir = data_dir
        self.split = split
        self.max_len = max_len
        self.pairs = self._load_pairs()

    def _load_pairs(self):
        pairs = torch.load(join(self.data_dir, "{}.pkl".format(self.split)))
        for i, (img, formula) in enumerate(pairs):
            pair = (img, " ".join(formula.split()[:self.max_len]))
            pairs[i] = pair
        return pairs

    def __getitem__(self, index):
        return self.pairs[index]

    def __len__(self):
        return len(self.pairs)


class LoadTensorFromPath(Dataset):
    def __init__(self, path, transform = transform):
        """
        在inference中可以用到，从指定路径导入DataSet
        """
        self.imgList = filter(lambda x: x.endswith('jpg') or x.endswith('png'), os.listdir(path))
        self.imgList = [os.path.join(path, x) for x in self.imgList]
        self.transform = transform

    def __getitem__(self, index):
        tmp = self.transform(Image.open(self.imgList[index]))
        return tmp.unsqueeze(0), self.imgList[index]

    def __len__(self):
        return len(self.imgList)
