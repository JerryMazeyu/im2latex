# load checkpoint and inference
from os.path import join
from functools import partial
import argparse
import os
import sys
import torch
from tqdm import tqdm
from build_vocab import load_vocab
from model import LatexProducer, Im2LatexModel
from data import LoadTensorFromPath
import datetime
import json
from utils import stripNonsense

parser = argparse.ArgumentParser(description="Im2Latex Evaluating Program")
parser.add_argument('-d', '--dataPath', default="", help='Test File')
parser.add_argument('-i', '--info', default="", help="JsonByShiJiang")
parser.add_argument('-e', '--expName', default="", help="experiment name")
parser.add_argument('-c', '--csvPath', default="", help="experiment name")
parser.add_argument('-a', '--ans', default="", help="( 3 , + inf )  or ///['(3 , + inf )', 'x > 3']")
parser.add_argument('-cn', '--colName', default="")

args = parser.parse_args()


a = (os.path.abspath(sys.argv[0])[0: os.path.abspath(sys.argv[0]).find('im2latex')])
b = 'im2latex'
root = os.path.join(a, b)

expName = args.expName
vocabPath = os.path.join(root, 'finetune', expName)
dataPath = args.dataPath
modelPath = os.path.join(root, 'ckpt', '%s.pt' % expName)
cuda = True if torch.cuda.is_available() else False
beamSize = 5
resultPath = os.path.join(root, 'results', 'result.txt')
maxLen = 64

if not args.colName:
    colName = expName
else:
    colName = args.colName

answer = args.ans
if answer.startswith("///"):
    answer = eval(answer[3:])
else:
    answer = [answer]
csvPath = args.csvPath

try:
    with open(args.info, 'r') as file:
        params = json.load(file)
    for (k, v) in params.items():
        setattr(args, k, v)
except:
    pass

if not cuda:
    checkpoint = torch.load(join(modelPath), map_location=torch.device('cpu'))
else:
    checkpoint = torch.load(join(modelPath))
model_args = checkpoint['args']

# 读入词典模型
vocab = load_vocab(vocabPath)

model = Im2LatexModel(
        len(vocab), model_args.emb_dim, model_args.dec_rnn_h,
        add_pos_feat=model_args.add_position_features,
        dropout=model_args.dropout
    )

model.load_state_dict(checkpoint['model_state_dict'])

latex_producer = LatexProducer(
        model, vocab, max_len=maxLen,
        use_cuda=cuda, beam_size=beamSize)

# 加载图像数据
tensors_ = LoadTensorFromPath(dataPath)

if not os.path.exists(resultPath):
    os.mknod(resultPath)


time_info = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
file_true = open(os.path.join('./results', time_info + 'true.txt'), 'w')
file_false = open(os.path.join('./results', time_info + 'false.txt'), 'w')
for img, name in tqdm(tensors_, ncols=60):
    res = latex_producer(img)
    if stripNonsense(res[0]) in answer:
        file_true.writelines(name)
        file_true.writelines('\n')
    else:
        file_false.writelines(name + " " + res[0])
        file_false.writelines('\n')
file_false.close()
file_true.close()

