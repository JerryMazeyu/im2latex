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
from data import LoadTensorFromPath, LoadTensorFromList
import datetime
import json
# from utils import stripNonsense, toStandardLatex, splitPath
# import _thread
# import queue
import time
from tqdm import tqdm
import multiprocessing as mp


parser = argparse.ArgumentParser(description="Im2Latex Evaluating Program")
parser.add_argument('-d', '--dataPath', default="", help='Test File')
parser.add_argument('-i', '--info', default="", help="JsonByShiJiang")
parser.add_argument('-e', '--expName', default="", help="Experiment Name")
parser.add_argument('-v', '--evaluation', default="", help="If Evaluate")
parser.add_argument('-c', '--csvPath', default="", help="csv Path")
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
resultPath = os.path.join(root, 'results', '%s.txt' % expName)
if os.path.exists(resultPath):
    os.remove(resultPath)
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

tensors_ = LoadTensorFromPath(dataPath)

res = []
def setcallback(x):
    res.append({'name': x[1], 'predict':x[0]})

def multiplication(model, tensor, name):
    return model(tensor)[0], name

if __name__ == '__main__':
    pool = mp.Pool(10)
    for img, name in tqdm(tensors_, ncols=60):
        pool.apply_async(func=multiplication, args=(latex_producer, img, name), callback=setcallback)
    pool.close()
    pool.join()
    with open(resultPath, 'w') as file:
        file.writelines(json.dumps({"result": res}))




