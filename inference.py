# load checkpoint and inference
from os.path import join
import argparse
import os
import time
import torch
from tqdm import tqdm
from build_vocab import load_vocab
from model import LatexProducer, Im2LatexModel
from data import LoadTensorFromPath

import json
from utils import toStandardLatex
import multiprocessing as mp


parser = argparse.ArgumentParser(description="Im2Latex Evaluating Program")
parser.add_argument('--model_path', default="", help='path of the evaluated model')
parser.add_argument('--vocab_path', default="", help="where is your vocab.pkl")
parser.add_argument("--data_path", type=str, default="", help="The dataset's dir")
parser.add_argument("--result_path", type=str, default="", help="The file to store result")
parser.add_argument('-i', '--info', default="/Users/mazeyu/GithubProjects/im2latex/param.json", help="JsonByShiJiang")

args = parser.parse_args()


beamSize = 5
maxLen = 64

if args.info != "":
    try:
        with open(args.info, 'r') as file:
            params = json.load(file)
        for (k, v) in params.items():
            setattr(args, k, v)
    except:
        pass

# 读入词典模型
vocab = load_vocab(args.vocab_path)
use_cuda = True if torch.cuda.is_available() else False

# 加载模型
if not use_cuda:
    checkpoint = torch.load(join(args.model_path), map_location=torch.device('cpu'))
else:
    checkpoint = torch.load(join(args.model_path))
model_args = checkpoint['args']


model = Im2LatexModel(
    len(vocab), model_args.emb_dim, model_args.dec_rnn_h,
    add_pos_feat=model_args.add_position_features,
    dropout=model_args.dropout
)
model.load_state_dict(checkpoint['model_state_dict'])

latex_producer = LatexProducer(
    model, vocab, max_len=maxLen,
    use_cuda=use_cuda, beam_size=beamSize)

tensors_ = LoadTensorFromPath(args.data_path)

res = []
def setcallback(x):
    res.append({'name': x[1], 'predict':x[0]})

def multiplication(model, tensor, name):
    return toStandardLatex(model(tensor)[0]), name

if __name__ == '__main__':
    pool = mp.Pool(10)
    for img, name in tqdm(tensors_, ncols=60):
        pool.apply_async(func=multiplication, args=(latex_producer, img, name), callback=setcallback)
    pool.close()
    pool.join()
    with open(args.result_path, 'w') as file:
        file.writelines(json.dumps({"result": res}))
