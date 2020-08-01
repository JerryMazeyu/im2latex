# load checkpoint and inference
from os.path import join
from functools import partial
import argparse
import os


import torch
from tqdm import tqdm

from build_vocab import load_vocab

from model import LatexProducer, Im2LatexModel

from data import LoadTensorFromPath
import datetime

import json


def main():

    parser = argparse.ArgumentParser(description="Im2Latex Evaluating Program")
    parser.add_argument('--model_path', default='ckpt/best_ckpt.pt',
                        help='path of the evaluated model')
    # parser.add_argument('--vocab_path', default='./Jerry/JerryRealData', help="where is your vocab.pkl")
    parser.add_argument('--vocab_path', default='./Jerry/Jerry2018T10', help="where is your vocab.pkl")

    # model args
    parser.add_argument("--data_path", type=str,
                        default="/Users/mazeyu/PycharmProjects/autoscore/2018T10bak/true", help="The dataset's dir")
    parser.add_argument("--cuda", action='store_true',
                        default=False, help="Use cuda or not")
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--result_path", type=str,
                        default="./results/result.txt", help="The file to store result")
    parser.add_argument("--max_len", type=int,
                        default=64, help="Max step of decoding")
    parser.add_argument('-i', '--info', default="", help="JsonByShiJiang")
    parser.add_argument('--expname', default="", help="experiment name")
    parser.add_argument('--csvPath', default="/Users/mazeyu/PycharmProjects/autoscore/2018cksx.csv", help="experiment name")
    parser.add_argument('--ans', default='( sqrt { 3 } / 3 , + inf )')
    parser.add_argument('--colName', default='T11_1')

    args = parser.parse_args()


    if args.info != "":
        try:
            with open(args.info, 'r') as file:
                params = json.load(file)
            for (k, v) in params.items():
                setattr(args, k, v)
        except:
            pass




    # 加载模型
    if not args.cuda:
        checkpoint = torch.load(join(args.model_path), map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(join(args.model_path))
    model_args = checkpoint['args']

    # 读入词典模型
    vocab = load_vocab(args.vocab_path)
    use_cuda = True if args.cuda and torch.cuda.is_available() else False

    model = Im2LatexModel(
        len(vocab), model_args.emb_dim, model_args.dec_rnn_h,
        add_pos_feat=model_args.add_position_features,
        dropout=model_args.dropout
    )
    model.load_state_dict(checkpoint['model_state_dict'])

    latex_producer = LatexProducer(
        model, vocab, max_len=args.max_len,
        use_cuda=use_cuda, beam_size=args.beam_size)

    # 加载图像数据
    tensors_ = LoadTensorFromPath(args.data_path)

    if not os.path.exists(args.result_path):
        os.mknod(args.result_path)

    # with open(args.result_path, 'a') as file:
    #     file.writelines('\n')
    #     file.writelines(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    #     file.writelines('\n')
    #     file.writelines('='*80)
    #     file.writelines('\n')
    #     for img, name in tqdm(tensors_, ncols=60):
    #         res = latex_producer(img)
    #         file.writelines('\n')
    #         file.writelines(name + '-->' + res[0])
    #         file.writelines('\n')



# ===================================这里是为sqrt3定制的代码=================================================
    time_info = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    file_true = open(os.path.join('./results', time_info + 'true.txt'), 'w')
    file_false = open(os.path.join('./results', time_info + 'false.txt'), 'w')
    for img, name in tqdm(tensors_, ncols=60):
        res = latex_producer(img)
        if res[0] == args.ans:
            file_true.writelines(name)
            file_true.writelines('\n')
        else:
            file_false.writelines(name + " " + res[0])
            file_false.writelines('\n')
    file_false.close()
    file_true.close()


    from jerry_evaluation import jerryEvaluation
    jerryEvaluation(trueFile=os.path.join('./results', time_info + 'true.txt'), falseFile=os.path.join('./results', time_info + 'false.txt'), expName=args.expname, csvPath=args.csvPath, colName=args.colName)





if __name__ == "__main__":
    main()  # 一般来说直接执行 python inference.py 即可
