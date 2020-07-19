
from os.path import join
import argparse

from PIL import Image
from torchvision import transforms as T
import torch
import cv2
import numpy as np


# custom_file_lst = "file.lst"  # path id lst
# custom_formula_lst = "im2latex_formulas.norm.lst" # formular lst
# custom_img_path = "手写填空-分式-1" # img path

#
# custom_file_lst, custom_formula_lst, custom_img_path = "", "", ""
custom_img_path = 'data'  # valid下的data文件夹
custom_formula_lst = 'im2latex_formulas.norm.lst'  #
custom_file_lst = 'im2latex_valid.lst'






def preprocess(data_dir, split):
    assert split in ["train", "validate", "test"]

    print("Process {} dataset...".format(split))
    if len(custom_img_path) == 0:
        images_dir = join(data_dir, "formula_images_processed")
    else:
        images_dir = join(data_dir, custom_img_path)

    if len(custom_formula_lst) == 0:
        formulas_file = join(data_dir, "im2latex_formulas.norm.lst")
    else:
        formulas_file = join(data_dir, custom_formula_lst)

    with open(formulas_file, 'r') as f:
        formulas = [formula.strip('\n') for formula in f.readlines()]

    if len(custom_file_lst) == 0:
        split_file = join(data_dir, "im2latex_{}_filter.lst".format(split))
    else:
        split_file = join(data_dir, custom_file_lst)

    pairs = []


    # transform = transforms.ToTensor()

    def __gray2RGB(img):
        img = np.array(img)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


    transform = T.Compose([
        T.Resize((32, 128)),
        T.Lambda(lambda img: __gray2RGB(img)),
        T.ToTensor()
    ])



    with open(split_file, 'r') as f:
        for line in f:
            img_name, formula_id = line.strip('\n').split()
            # load img and its corresponding formula
            img_path = join(images_dir, img_name)
            img = Image.open(img_path)
            img_tensor = transform(img)
            formula = formulas[int(formula_id)]
            pair = (img_tensor, formula)
            pairs.append(pair)
        pairs.sort(key=img_size)

    out_file = join(data_dir, "{}.pkl".format(split))
    torch.save(pairs, out_file)
    print("Save {} dataset to {}".format(split, out_file))


def img_size(pair):
    img, formula = pair
    return tuple(img.size())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Im2Latex Data Preprocess Program")
    parser.add_argument("--data_path", type=str,
                        default="./data/", help="The dataset's dir")
    args = parser.parse_args()

    splits = ["validate", "test", "train"]
    # for s in splits:
    #     preprocess(args.data_path, s)
    preprocess('./experiment/valid', 'validate')