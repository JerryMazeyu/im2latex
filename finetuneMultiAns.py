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
import yaml


# 解析参数
parser = argparse.ArgumentParser(description="Im2Latex Training Program")

parser.add_argument("-d", "--dataPath", help="原始数据路径：<dataPath>/<right>、<wrong> ")
# parser.add_argument("-d", "--dataPath", default='/Users/mazeyu/Desktop/im2latex_assets/papers/T1_3', help="原始数据路径：<dataPath>/<right>、<wrong> ")

parser.add_argument("-e", "--expName", default="", help="实验名称，最后训练数据会存在：./finetune/<expName>")
# parser.add_argument("-e", "--expName", default="exp3", help="实验名称，最后训练数据会存在：./finetune/<expName>")

parser.add_argument("-a", "--answer", default="", help="如果是一个答案的，则写入答案")
#
parser.add_argument("-s", "--sub", default="{}", help="如果多个答案的，则写入字典")
# parser.add_argument("-s", "--sub", default="{0:'0',1:'1',2:'2',3:'3'}", help="如果多个答案的，则写入字典")

parser.add_argument("-i", "--info", default="./asset/fineTuneYaml.yaml", help="如果整体训练，需要传入一个yaml文件")
# parser.add_argument("-i", "--info", default="", help="如果整体训练，需要传入一个yaml文件")
args = parser.parse_args()

# 找到项目的根目录的绝对路径
import sys
a = (os.path.abspath(sys.argv[0])[0: os.path.abspath(sys.argv[0]).find('im2latex')])
b = 'im2latex'
root = os.path.join(a, b)  # /Users/mazeyu/GithubProjects/im2latex

if not os.path.exists(os.path.join(root, 'finetune')):
    os.mkdir(os.path.join(root, 'finetune'))

# 原始训练集，可以后期修改
NAME2ID = {'手写填空-分式-1': 0, '手写填空-数字序列-1':1, '手写填空-根式-1':2, '手写填空-集合-1':3, '手写填空-集合-2':4, '手写填空-根式-集合-分式-混合':5}

# 原始的ID2FORMULA.json
with open(os.path.join(root, 'ID2FORMULA.json'), 'r') as file:
    id2formula_list = json.load(file)

# 原始训练集路径，可以后期修改
ORIGIN_PATH = os.path.join(root, 'asset', 'output')  # /Users/mazeyu/GithubProjects/im2latex/asset/output

# 如果未指定实验名称，就选择和dataPath文件名相同
dataPath = args.dataPath
if args.info != "":
    expName = 'all'
elif args.expName == "":
    expName = os.path.split(dataPath)[-1]
else:
    expName = args.expName



class BuildTrainData():
    def __init__(self, dataPath=dataPath, expName=expName, subId2Formula=eval(args.sub), ans=args.answer):
        if args.info != "":
            self.all = True
            with open(args.info, 'r') as file:
                info = yaml.load(file, Loader=yaml.FullLoader)
            subId2Formula = info['Answers']
            files = info['Files']
        else:
            self.all = False
            files = None
        self.dataPath = dataPath
        self.expName = expName
        if self.all:
            self.resPath = os.path.join(root, 'finetune', 'all')
        else:
            self.resPath = os.path.join(root, 'finetune', self.expName)  # /Users/mazeyu/GithubProjects/im2latex/finetune/T1_1
        self.originId2Formula = id2formula_list  # [{"(0,0)": "xxx"}...]
        self.start = len(self.originId2Formula)  # 4
        self.subId2Formula = subId2Formula  # 如果是多个答案的情况，就给出一个dict
        self.ans = ans  # 如果是只有一个答案的情况下，就传入一个ans即可
        if self.all:
            self.rightImgPath = [os.path.join(x, 'right') for x in files]  # ['.../T1_1/right', '.../T1_2/right']
            self.wrongImgPath = [os.path.join(x, 'wrong') for x in files]  # ['.../T1_1/wrong', '.../T1_2/wrong']
        else:
            self.rightImgPath = os.path.join(self.dataPath, 'right')
            self.wrongImgPath = os.path.join(self.dataPath, 'wrong')
        self.imgPath = os.path.join(self.resPath, 'imgs')  # /Users/mazeyu/GithubProjects/im2latex/finetune/T1_1<all>/imgs
# =====================================================================================
        if os.path.exists(self.resPath):  # 如果已经有相应的实验，删除旧的
            print("%s Has Existed, Delete It..." % self.resPath)
            shutil.rmtree(self.resPath, ignore_errors=True)
        else:
            print("Nothing To Delete, Start Building...")
        os.mkdir(self.resPath)
        os.mkdir(self.imgPath)
# =====================================================================================
        print("Start Build ID2FORMULA.json...")
        if not self.all:
            multiFile = os.path.join(self.rightImgPath, '0')
            if os.path.exists(multiFile):  # 处理多个答案的情况
                assert self.__isAllFile(self.rightImgPath), "Keep Your File Clean!"
                self.multi = True
                answerDict = {}
                orderImgPaths = sorted(eval(x) for x in os.listdir(self.rightImgPath))  # 防止遍历文件夹不按顺序
                for ind, file in enumerate(orderImgPaths):
                    v = self.subId2Formula[file]
                    k = "(%s,%s)" % (self.start, ind)
                    answerDict[k] = v
            else:  # 处理单个情况
                self.multi = False
                answerDict = {"(%s,%s)" % (self.start, 0): self.ans}
        else:
            self.multi = []
            answerDict = {}
            ind = 0
            idx = 0
            for rightImgPath in self.rightImgPath:
                multiFile = os.path.join(rightImgPath, '0')
                if os.path.exists(multiFile):  # 处理多个答案的情况
                    assert self.__isAllFile(rightImgPath), "Keep Your File Clean!"
                    self.multi.append(True)
                    idx += 1
                    for file in os.listdir(rightImgPath):
                        v = self.subId2Formula[idx-1][eval(file)]
                        k = "(%s,%s)" % (self.start, ind)
                        ind += 1
                        answerDict[k] = v
                else:  # 处理单个情况
                    self.multi.append(False)
                    idx += 1
                    answerDict["(%s,%s)" % (self.start, ind)] = str(self.subId2Formula[idx-1])
                    ind += 1
        print("Right Ans Is ", answerDict)
        self.originId2Formula.append(answerDict)
        if not self.all:
            wrongAns = {"(%s,%s)" % (self.start+1, 0):'{ }'}
            self.originId2Formula.append(wrongAns)
        with open(os.path.join(self.resPath, 'ID2FORMULA.json'), 'w') as file:
            json.dump(self.originId2Formula, file)
        tmp = {}
        for i in self.originId2Formula:
            tmp.update(i)
        self.originId2Formula = tmp
        print("ID2Formula Build OK!")
# =====================================================================================
        self.build_im2latex_formulas()
# =====================================================================================
        print("Building /imgs And im2latex_all_filter.lst")
        file = open(os.path.join(self.resPath, 'im2latex_all_filter.lst'), 'w')
        if self.all:
            for xx in self.rightImgPath:
                NAME2ID.update({xx: self.start})
        else:
            NAME2ID.update({self.rightImgPath: self.start, self.wrongImgPath: self.start+1})
        filenameList = os.listdir(ORIGIN_PATH)
        if self.all:
            filenameList += self.rightImgPath
        else:
            filenameList.append(self.rightImgPath)
            filenameList.append(self.wrongImgPath)
        count = 0
        idx = 0  # 在一起训练时候记录有多少个right
        for filename in filenameList:  # filename: ['dataset_手写填空-分式-1', ...]
            className = NAME2ID.get(filename)  # 找到(x,y)中的x
            if className == None:
                continue
            if filename.find("/" or "\\") == -1:
                path = os.path.join(ORIGIN_PATH, filename)  # .../output/dataset_手写填空-分式-1
                for imgName in tqdm(self.filterpng(path), ncols=60):
                    ansId = imgName.split('_')[0]
                    tarID = '(%s,%s)' % (className, ansId)
                    tarIndex = self.keyid2index[tarID]  # 找到后面的id
                    oriImgPath = os.path.join(path, imgName)  # .../output/dataset_手写填空-分式-1/0_001.png
                    tarImgPath = os.path.join(self.imgPath, '_'.join([str(className), imgName]))  # .../finetune/imgs/0_0_001.jpg
                    copyfile(oriImgPath, tarImgPath)
                    file.writelines('%s %s\n' % ('_'.join([str(className), imgName]), tarIndex))
                    count += 1
            else:  # 新数据
                path = filename  # .../papers/T1_1/right/
                # TODO: self.multi 还没处理
                if self.all:
                    multi = self.multi[idx]
                    idx += 1
                    if not multi:
                        for imgName in tqdm(self.filterpng(path), ncols=60):
                            ansId = 0
                            tarID = '(%s,%s)' % (className, ansId)
                            tarIndex = self.keyid2index[tarID]  # 找到后面的id <path> <id>
                            oriImgPath = os.path.join(path, imgName)  # .../T1_1/right/1.png
                            tarImgPath = os.path.join(self.imgPath,
                                                      '_'.join([str(className), imgName]))  # .../finetune/imgs/4_1.png
                            copyfile(oriImgPath, tarImgPath)
                            file.writelines('%s %s\n' % ('_'.join([str(className), imgName]), tarIndex))
                            count += 1
                    else:
                        for subFile in tqdm(os.listdir(path), ncols=60):
                            ansId = int(subFile)
                            tarID = '(%s,%s)' % (className, ansId)
                            tarIndex = self.keyid2index[tarID]
                            for imgName in self.filterpng(os.path.join(path, subFile)):
                                oriImgPath = os.path.join(path, subFile, imgName)
                                tarImgPath = os.path.join(self.imgPath, '_'.join([str(className), subFile, imgName]))
                                copyfile(oriImgPath, tarImgPath)
                                file.writelines('%s %s\n' % ('_'.join([str(className), subFile, imgName]), tarIndex))
                                count += 1

                else:
                    if not self.multi or path.endswith('wrong'):
                        for imgName in tqdm(self.filterpng(path), ncols=60):
                            ansId = 0
                            tarID = '(%s,%s)' % (className, ansId)
                            tarIndex = self.keyid2index[tarID]  # 找到后面的id <path> <id>
                            oriImgPath = os.path.join(path, imgName)  # .../T1_1/right/1.png
                            tarImgPath = os.path.join(self.imgPath, '_'.join([str(className), imgName]))  # .../finetune/imgs/4_1.png
                            copyfile(oriImgPath, tarImgPath)
                            file.writelines('%s %s\n' % ('_'.join([str(className), imgName]), tarIndex))
                            count += 1
                    else:
                        for subFile in tqdm(os.listdir(path), ncols=60):
                            ansId = int(subFile)
                            tarID = '(%s,%s)' % (className, ansId)
                            tarIndex = self.keyid2index[tarID]
                            for imgName in self.filterpng(os.path.join(path, subFile)):
                                oriImgPath = os.path.join(path, subFile, imgName)
                                tarImgPath = os.path.join(self.imgPath, '_'.join([str(className), subFile, imgName]))
                                copyfile(oriImgPath, tarImgPath)
                                file.writelines('%s %s\n' % ('_'.join([str(className), subFile, imgName]), tarIndex))
                                count += 1

        file.close()
        print("Building /imgs And im2latex_all_filter.lst OK!")
# =====================================================================================
        print("Writing Filters...")
        file = open(os.path.join(self.resPath, 'im2latex_all_filter.lst'), 'r')
        file1 = open(os.path.join(self.resPath, 'im2latex_train_filter.lst'), 'w')
        file2 = open(os.path.join(self.resPath, 'im2latex_valid_filter.lst'), 'w')
        trainCount = int(0.9 * count)  # 训练/验证 9:1
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
        print("Writing OK!")
# =====================================================================================
        os.remove(os.path.join(self.resPath, 'im2latex_all_filter.lst'))
        self.buildpkl('train', self.imgPath, os.path.join(self.resPath, 'im2latex_formulas.norm.lst'),
                      os.path.join(self.resPath, 'im2latex_train_filter.lst'))
        self.buildpkl('validate', self.imgPath, os.path.join(self.resPath, 'im2latex_formulas.norm.lst'),
                      os.path.join(self.resPath, 'im2latex_valid_filter.lst'))
        if not self.all:  # 如果不是全部训练，就先不删除imgs，可以基于该文件增加数据
            if os.path.exists(os.path.join(self.resPath, 'imgs')):  # 把原图像删去，可以注释
                shutil.rmtree(os.path.join(self.resPath, 'imgs'))
        build_vocab(data_dir=self.resPath, min_count=0)
        print("Build Training Data OK!")


    def __isAllFile(self, path):
        """判断是否该路径下只有文件夹"""
        for file in os.listdir(path):
            if file == '.DS_Store':
                os.remove(os.path.join(path, file))
            if not os.path.isdir(os.path.join(path, file)) and not file.startswith('.'):
                return False
        return True


    def build_im2latex_formulas(self):
        """创建norm.list"""
        self.keyid2index = {}
        with open(os.path.join(self.resPath, 'im2latex_formulas.norm.lst'), 'w') as f:
            ind = 0
            for key, formula in self.originId2Formula.items():
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

        out_file = join(self.resPath, "{}.pkl".format(split))
        torch.save(pairs, out_file)
        print("Save {} dataset to {}".format(split, out_file))

    def filterpng(self, path):
        return list(filter(lambda x: True if x.endswith('png') else False, os.listdir(path)))



if __name__ == '__main__':
    build = BuildTrainData()


