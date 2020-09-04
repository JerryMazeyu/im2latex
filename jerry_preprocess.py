import os
from os.path import join
import cv2
from torchvision import transforms as T
import numpy as np
from PIL import Image
import torch
import json
from shutil import copyfile
from tqdm import tqdm
from build_vocab import build_vocab
import shutil
from random import shuffle
import argparse

parser = argparse.ArgumentParser(description="Im2Latex Training Program")
parser.add_argument("--ROOT", default="Jerry/Jerry2018T11", help="存放的路径")
# parser.add_argument("--NAME2ID", default="", help="图像对应编号")
parser.add_argument("--NAME2ID", default="{'手写填空-分式-1': 0, '手写填空-数字序列-1':1, '手写填空-根式-1':2, '手写填空-集合-1':3, '手写填空-集合-2':4, '手写填空-根式-集合-分式-混合':5}", help="图像对应编号")
parser.add_argument("--ORIGIN_PATH", default='/Users/mazeyu/Downloads/dataset/output', help="平时不要动")
# parser.add_argument("--SUP_NAME2ID", default="", help="微调的数据")
parser.add_argument('--SUP_NAME2ID', default="{'/Users/mazeyu/PycharmProjects/autoscore/2018T10/true': 4, '/Users/mazeyu/PycharmProjects/autoscore/2018T10/false': 5, '/Users/mazeyu/PycharmProjects/autoscore/im2latex/results/exp1_2018T10/True2False': 6, '/Users/mazeyu/PycharmProjects/autoscore/im2latex/results/exp1_2018T10/False2True': 7}")
args = parser.parse_args()




# txts下的NAME2ID
# NAME2ID = {'dataset_手写填空-分式-1': 0, 'dataset_手写填空-数字序列-1':1, 'dataset_手写填空-根式-1':2, 'dataset_手写填空-集合-1':3}
NAME2ID = args.NAME2ID


# ROOT = 'Jerry/Jerry2018T10'
ROOT = args.ROOT


# ID2FORMULA从的json格式
with open(join(ROOT, 'ID2FORMULA.json'), 'r') as file:
    ID2FORMULA = json.load(file)



class JerryPreprocessFineTune(object):
    def __init__(self, finetuneImgPath, label, saveP='./finetuneRes', baseFormula='/Users/mazeyu/PycharmProjects/autoscore/im2latex/Jerry/JerryRealData/im2latex_formulas.norm.lst'):
        """
        FineTune代码：
        finetuneImgPath：里面全是jpg文件的文件夹
        label：文件夹对应的label（只能一个）
        saveP：所有的生成后的结果都存放在这里（im2latex_formulas.norm.lst、im2latex_train_filter.lst等）到时直接用这个训练就行了
        baseFormula：根据原来的formulas.norm.lst生成新的，这里写他的路径
        """
        self.finetuneImgPath = finetuneImgPath
        self.imgs = os.listdir(self.finetuneImgPath)
        self.imgs = filter(lambda x: True if x.endswith('png') else False, self.imgs)
        self.label = label
        self.saveP = saveP
        self.baseFormula = baseFormula

    def _preprocess(self, custom_formula_lst, custom_file_lst, data_dir, split):
        def img_size(pair):
            img, formula = pair
            return tuple(img.size())
        assert split in ["train", "validate", "test"]

        print("Process {} dataset...".format(split))

        images_dir = join(data_dir)

        if len(custom_formula_lst) == 0:
            formulas_file = join(data_dir, "im2latex_formulas.norm.lst")
        else:
            formulas_file = join(data_dir, custom_formula_lst)

        with open(formulas_file, 'r') as f:
            formulas = [formula.strip('\n') for formula in f.readlines()]


        split_file = custom_file_lst

        pairs = []


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

        out_file = join(self.saveP, "{}.pkl".format(split))
        torch.save(pairs, out_file)
        print("Save {} dataset to {}".format(split, out_file))

    def main(self):
        if not os.path.exists(self.saveP):
            os.mkdir(self.saveP)
        with open(self.baseFormula, 'a+') as file:
            formulas = open(self.baseFormula, 'rU').readlines()
            file.writelines(self.label + '\n')
            tarID = len(formulas)
        shutil.copyfile(self.baseFormula, os.path.join(self.saveP, 'im2latex_formulas.norm.lst'))
        with open(os.path.join(self.saveP, 'im2latex_train_filter.lst'), 'w') as f:
            for img in self.imgs:
                f.write(img + " " + str(tarID) + '\n')
        shutil.copyfile(os.path.join(self.saveP, 'im2latex_train_filter.lst'), os.path.join(self.saveP, 'im2latex_valid_filter.lst'))
        self._preprocess(data_dir=self.finetuneImgPath, custom_formula_lst=self.baseFormula, custom_file_lst=os.path.join(self.saveP, 'im2latex_train_filter.lst'), split='train')
        self._preprocess(data_dir=self.finetuneImgPath, custom_formula_lst=self.baseFormula, custom_file_lst=os.path.join(self.saveP, 'im2latex_valid_filter.lst'), split='validate')
        build_vocab(data_dir=self.saveP, min_count=0)

class JerryPreprocessBuild():
    def __init__(self, root=ROOT, name2id=eval(NAME2ID), id2formula=ID2FORMULA, originImgsPath=args.ORIGIN_PATH):
        """
        从头构建一个数据集
        root：根目录，最终将所有的训练所需要的数据都放在这个下面
        name2id：
        id2formula：
        originImgsPath：最初的图片路径 下面的目录结构是 originImgsPath--name--pngs
        """
        if os.path.exists(os.path.join(root, 'imgs')):
            shutil.rmtree(os.path.join(root, 'imgs'))
        os.mkdir(os.path.join(root, 'imgs'))
        try:
            os.remove(os.path.join(root, 'im2latex_formulas.norm.lst'))
            os.remove(os.path.join(root, 'im2latex_train_filter.lst'))
            os.remove(os.path.join(root, 'im2latex_valid_filter.lst'))
            os.remove(os.path.join(root, 'train.pkl'))
            os.remove(os.path.join(root, 'validate.pkl'))
            os.remove(os.path.join(root, 'vocab.pkl'))
        except:
            print("Nothing to delete. Start building...")
        self.imgPath = os.path.join(root, 'imgs')
        self.root = root
        self.name2id = name2id
        if args.SUP_NAME2ID != "":
            self.name2id.update(eval(args.SUP_NAME2ID))
        self.id2formula = {}
        for dic in id2formula:
            self.id2formula.update(dic)
        self.originImgsPath = originImgsPath
        self.names = list(name2id.keys())


    def build_im2latex_formulas(self):
        self.keyid2index = {}
        with open(os.path.join(self.root, 'im2latex_formulas.norm.lst'), 'w') as f:
            ind = 0
            for key, formula in self.id2formula.items():
                f.writelines(formula + '\n')
                self.keyid2index[key] = ind
                ind += 1

    def buildpkl(self, split, data_dir, formulas_file, split_file):
        def img_size(pair):
            img, formula = pair
            return tuple(img.size())

        assert split in ["train", "validate", "test"]
        print("Process {} dataset...".format(split))
        images_dir = join(data_dir)
        with open(formulas_file, 'r') as f:
            formulas = [formula.strip('\n') for formula in f.readlines()]
        pairs = []

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

        out_file = join(self.root, "{}.pkl".format(split))
        torch.save(pairs, out_file)
        print("Save {} dataset to {}".format(split, out_file))

    def main(self):

        self.build_im2latex_formulas()
        file = open(os.path.join(self.root, 'im2latex_all_filter.lst'), 'w')
        count = 0
        for name in self.names:
            if name.find("/" or "\\") != -1:  # 对fine-tune的部分进行不同的处理
                originP = name
                tarImgList = [os.path.join(name, x) for x in os.listdir(originP)]
            else:
                originP = os.path.join(self.originImgsPath, name[8:])
                tarImgList = [os.path.join(self.originImgsPath, name[8:], x) for x in os.listdir(originP)]
            tarImgList = list(filter(lambda x: True if x.endswith('png') else False, tarImgList))
            for img in tqdm(tarImgList, ncols=60):
                if name.find("/" or "\\") != -1:
                    tarID = '(%s,%s)' % (self.name2id[name], 0)
                else:
                    imgID = os.path.split(img)[-1].split('_')[0]
                    tarID = '(%s,%s)' % (self.name2id[name], imgID)
                tarPath = os.path.join(self.root, 'imgs', '%s_%s' % (self.name2id[name], os.path.split(img)[-1]))
                tarName = os.path.split(tarPath)[-1]
                copyfile(img, tarPath)
                file.writelines('%s %s\n' % (tarName, self.keyid2index[tarID]))
                count += 1
        file.close()
        file = open(os.path.join(self.root, 'im2latex_all_filter.lst'), 'r')
        file1 = open(os.path.join(self.root, 'im2latex_train_filter.lst'), 'w')
        file2 = open(os.path.join(self.root, 'im2latex_valid_filter.lst'), 'w')
        trainCount = int(0.9*count)  # 训练/验证 9:1
        tmp = file.readlines()
        shuffle(tmp)
        for ind, line in enumerate(tmp):
            if ind < trainCount:
                file1.writelines(line)
            else:
                file2.writelines(line)
        file1.close()
        file2.close()
        file.close()
        os.remove(os.path.join(self.root, 'im2latex_all_filter.lst'))
        self.buildpkl('train', self.imgPath, os.path.join(self.root,'im2latex_formulas.norm.lst'), os.path.join(self.root, 'im2latex_train_filter.lst'))
        self.buildpkl('validate', self.imgPath, os.path.join(self.root, 'im2latex_formulas.norm.lst'),os.path.join(self.root, 'im2latex_valid_filter.lst'))
        if os.path.exists(os.path.join(self.root, 'imgs')):  # 把原图像删去，可以注释
            shutil.rmtree(os.path.join(self.root, 'imgs'))
        build_vocab(data_dir=self.root, min_count=0)

if __name__ == '__main__':
    # NAME2ID.update({'/Users/mazeyu/PycharmProjects/autoscore/sqrt3/true': 4, '/Users/mazeyu/PycharmProjects/autoscore/sqrt3/false': 5, '/Users/mazeyu/PycharmProjects/autoscore/im2latex/results/exp6/False2True': 6, '/Users/mazeyu/PycharmProjects/autoscore/im2latex/results/exp6/True2False': 7})
    # NAME2ID.update({'/Users/mazeyu/PycharmProjects/autoscore/2018T10/true': 4, '/Users/mazeyu/PycharmProjects/autoscore/2018T10/false': 5, '/Users/mazeyu/PycharmProjects/autoscore/im2latex/results/exp1_2018T10/True2False': 6, '/Users/mazeyu/PycharmProjects/autoscore/im2latex/results/exp1_2018T10/False2True': 7})
    p = JerryPreprocessBuild()  # 写完ID2FORMULA就可以运行这个代码
    ####### p = JerryPreprocessFineTune('finetune', label='- frac { 1 } { 3 }') # 当finetune时运行这个代码，但是train.py参数修改一下
    p.main()

    


















