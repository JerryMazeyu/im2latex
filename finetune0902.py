import os
import argparse
import shutil
import json
import sys
import torch


parser = argparse.ArgumentParser(description='config')
parser.add_argument('-r','--root', required=True)
parser.add_argument('-e', '--epoch', default=10)
args = parser.parse_args()

a = (os.path.abspath(sys.argv[0])[0: os.path.abspath(sys.argv[0]).find('im2latex')])
b = 'im2latex'
root = os.path.join(a, b)





with open(os.path.join(root, 'paper2017.json'), 'r') as file:
    info = json.load(file)

pythonPath = os.path.join(root, 'finetuneMultiAns.py')

for (fileName, (ans, type)) in info.items():
    dataPath = os.path.join(args.root, fileName)
    if type == 0:
        pythoncmd = "python %s -d \"%s\" -a \"%s\"" % (pythonPath, dataPath, ans)
    else:
        pythoncmd = "python %s -d \"%s\" -s \"%s\"" % (pythonPath, dataPath, ans)
    os.system(pythoncmd)

for (fileName, _) in info.items():
    dataPath = os.path.join(root, 'finetune', fileName)
    savedir = os.path.join(root, 'ckpt')
    pythoncmd = "python train.py --data_path=\"%s\" --save_dir=\"%s\" --dropout=0.4 --batch_size=16 --epoches=%s --exp=\"%s\"" % (dataPath, savedir, args.epoch, fileName)
    os.system(pythoncmd)