import os
import math

import torch
import torch.nn.functional as F
from torch.distributions.bernoulli import Bernoulli

from build_vocab import PAD_TOKEN, UNK_TOKEN
import cv2


def collate_fn(sign2id, batch):
    # filter the pictures that have different weight or height
    size = batch[0][0].size()  # 找出第一个图像的size example --> (224, 224)
    batch = [img_formula for img_formula in batch
             if img_formula[0].size() == size]  # batch是这个原始batch中的size和第一个一样大的那些图像
    # sort by the length of formula
    batch.sort(key=lambda img_formula: len(img_formula[1].split()),  # 根据公式的长短从长到短对batch数据进行排序
               reverse=True)

    imgs, formulas = zip(*batch)
    formulas = [formula.split() for formula in formulas]
    # targets for training , begin with START_TOKEN
    tgt4training = formulas2tensor(add_start_token(formulas), sign2id)
    # targets for calculating loss , end with END_TOKEN
    tgt4cal_loss = formulas2tensor(add_end_token(formulas), sign2id)
    imgs = torch.stack(imgs, dim=0)
    return imgs, tgt4training, tgt4cal_loss  # 一个dataloader返回三个东西


def formulas2tensor(formulas, sign2id):
    """convert formula to tensor"""

    batch_size = len(formulas)
    max_len = len(formulas[0])
    tensors = torch.ones(batch_size, max_len, dtype=torch.long) * PAD_TOKEN
    for i, formula in enumerate(formulas):
        for j, sign in enumerate(formula):
            tensors[i][j] = sign2id.get(sign, UNK_TOKEN)
    return tensors


def add_start_token(formulas):
    return [['<s>']+formula for formula in formulas]


def add_end_token(formulas):
    return [formula+['</s>'] for formula in formulas]


def count_parameters(model):
    """count model parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def tile(x, count, dim=0):
    """
    Tiles x on dimension dim count times.
    """
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1) \
         .transpose(0, 1) \
         .repeat(count, 1) \
         .transpose(0, 1) \
         .contiguous() \
         .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x


def load_formulas(filename):
    formulas = dict()
    with open(filename) as f:
        for idx, line in enumerate(f):
            formulas[idx] = line.strip()
    print("Loaded {} formulas from {}".format(len(formulas), filename))
    return formulas


def cal_loss(logits, targets):
    """args:
        logits: probability distribution return by model
                [B, MAX_LEN, voc_size]
        targets: target formulas
                [B, MAX_LEN]
    """
    padding = torch.ones_like(targets) * PAD_TOKEN
    mask = (targets != padding)

    targets = targets.masked_select(mask)
    logits = logits.masked_select(
        mask.unsqueeze(2).expand(-1, -1, logits.size(2))
    ).contiguous().view(-1, logits.size(2))
    logits = torch.log(logits)

    assert logits.size(0) == targets.size(0)

    loss = F.nll_loss(logits, targets)
    return loss


def get_checkpoint(ckpt_dir):
    """return full path if there is ckpt in ckpt_dir else None"""
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError("No checkpoint found in {}".format(ckpt_dir))

    ckpts = [f for f in os.listdir(ckpt_dir) if f.startswith('ckpt')]
    if not ckpts:
        raise FileNotFoundError("No checkpoint found in {}".format(ckpt_dir))

    last_ckpt, max_epoch = None, 0
    for ckpt in ckpts:
        epoch = int(ckpt.split('-')[1])
        if epoch > max_epoch:
            max_epoch = epoch
            last_ckpt = ckpt
    full_path = os.path.join(ckpt_dir, last_ckpt)
    print("Get checkpoint from {} for training".format(full_path))
    return full_path


def schedule_sample(prev_logit, prev_tgt, epsilon):
    prev_out = torch.argmax(prev_logit, dim=1, keepdim=True)
    prev_choices = torch.cat([prev_out, prev_tgt], dim=1)  # [B, 2]
    batch_size = prev_choices.size(0)
    prob = Bernoulli(torch.tensor([epsilon]*batch_size).unsqueeze(1))
    # sampling
    sample = prob.sample().long().to(prev_tgt.device)
    next_inp = torch.gather(prev_choices, 1, sample)
    return next_inp


def cal_epsilon(k, step, method):
    """
    Reference:
        Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks
        See details in https://arxiv.org/pdf/1506.03099.pdf
    """
    assert method in ['inv_sigmoid', 'exp', 'teacher_forcing']

    if method == 'exp':
        return k**step
    elif method == 'inv_sigmoid':
        return k/(k+math.exp(step/k))
    else:
        return 1.


def calConnectedComponent(img):
    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    num, _ = cv2.connectedComponents(img)
    return num - 1


def stripNonsense(ans:str):
    stack1, stack2 = [], []
    nonsense = []
    ansList = ans.split(" ")
    for ind, ele in enumerate(ansList):
        if ele in ["(", "["]:
            stack1.append(ele)
        if ele == "{{":
            stack2.append(ele)
        if ele in [")", "]"]:
            try:
                stack1.pop()
            except:
                nonsense.append(ind)
        if ele == "}}":
            try:
                stack2.pop()
            except:
                nonsense.append(ind)
    for i in nonsense:
        del ansList[i]
    return " ".join(ansList)

def toStandardLatex(ans:str):
    ans = stripNonsense(ans)
    mapping = {'{{': '\{', '}}': '\}', 'frac':'\\frac', 'sqrt':'\sqrt', 'E':'\in', 'U':'\\cup', 'inf':'\infty', 'pi':'\pi', 'p1': 'P_{1}', 'p2': 'P_{2}', 'p3': 'P_{3}', 'p4': 'P_{4}'}
    res = []
    for i in ans.split(" "):
        repSym = mapping.get(i, None)
        if repSym:
            res.append(repSym)
        else:
            res.append(i)
    return "".join(res)

def splitPath(path, numOfGroup=4):
    def list_of_groups(init_list, children_list_len):
        list_of_groups = zip(*(iter(init_list),) * children_list_len)
        end_list = [list(i) for i in list_of_groups]
        count = len(init_list) % children_list_len
        end_list.append(init_list[-count:]) if count != 0 else end_list
        return end_list
    allFile = list(filter(lambda x: x.endswith('jpg') or x.endswith('png'), os.listdir(path)))
    allFile = [os.path.join(path, x) for x in allFile]
    return list_of_groups(allFile, len(allFile)//numOfGroup+numOfGroup)


