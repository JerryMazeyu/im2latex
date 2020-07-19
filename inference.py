# load checkpoint and inference
from os.path import join
from functools import partial
import argparse
import os

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import Im2LatexDataset
from build_vocab import Vocab, load_vocab
from utils import collate_fn
from model import LatexProducer, Im2LatexModel
from model.score import score_files
from data import LoadTensorFromPath
import datetime


def main():

    parser = argparse.ArgumentParser(description="Im2Latex Evaluating Program")
    parser.add_argument('--model_path', default='ckpt/best_ckpt.pt',
                        help='path of the evaluated model')
    parser.add_argument('--vocab_path', default='./Jerry/JerryRealData', help="where is your vocab.pkl")

    # model args
    parser.add_argument("--data_path", type=str,
                        default="./sample_data_1", help="The dataset's dir")
    parser.add_argument("--cuda", action='store_true',
                        default=False, help="Use cuda or not")
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--result_path", type=str,
                        default="./results/result.txt", help="The file to store result")
    parser.add_argument("--max_len", type=int,
                        default=64, help="Max step of decoding")


    args = parser.parse_args()

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

    with open(args.result_path, 'a') as file:
        file.writelines('\n')
        file.writelines(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        file.writelines('\n')
        file.writelines('='*80)
        file.writelines('\n')
        for img, name in tqdm(tensors_, ncols=60):
            res = latex_producer(img)
            file.writelines('\n')
            file.writelines(name + '-->' + res[0])
            file.writelines('\n')




if __name__ == "__main__":
    main()  # 一般来说直接执行 python inference.py 即可
