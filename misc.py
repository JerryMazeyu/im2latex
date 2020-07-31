# 处理新的数据
import pandas as pd
import os
from shutil import copyfile
from tqdm import tqdm
from shutil import rmtree
from random import shuffle
import argparse

parser = argparse.ArgumentParser(description="Im2Latex Training Program")
parser.add_argument("--score_csv", default="/Users/mazeyu/PycharmProjects/autoscore/2018cksx.csv", help="CSV路径")
parser.add_argument("--img_root", default="/Users/mazeyu/PycharmProjects/autoscore/processed/T_10", help="原始图像路径")
parser.add_argument("--res_file", default="/Users/mazeyu/PycharmProjects/autoscore/2018T10", help="结果存放路径")
parser.add_argument("--col_name", default="T10_1", help="CSV列名")
parser.add_argument("--train_num", type=int, default=700, help="训练样本数据")
parser.add_argument("--tst_count", type=int, default=10000, help="训练样本数据")
args = parser.parse_args()


def main(score_csv=args.score_csv, img_root=args.img_root, res_file=args.res_file, colName=args.col_name, trainNum=args.train_num, tst_count=args.tst_count):
    # # score_csv = '/Users/mazeyu/PycharmProjects/autoscore/2017qksx.csv'  # csv路径
    # # img_root = '/Users/mazeyu/PycharmProjects/autoscore/T_5'
    # # res_file = '/Users/mazeyu/PycharmProjects/autoscore/sqrt3'
    # colName = 'T10_1'  # csv中的列名
    # trainNum = 700  # 训练的数据


    df = pd.read_csv(score_csv, sep=',')
    df = df.set_index('KSH')


    def getscore(pngname):
        id = pngname.strip('0')
        id = id.split('_')[0]
        return df[colName].loc[int(id)]


    try:
        rmtree(res_file)
        print("%s Has Been Deleted." % res_file)
    except:
        print("Nothing To Delete.")

    if not os.path.exists(res_file):
        os.mkdir(res_file)
        os.mkdir(os.path.join(res_file, 'true'))
        os.mkdir(os.path.join(res_file, 'false'))
        os.mkdir(os.path.join(res_file, 'tst'))
        print('Making Init File OK.')


    file_list_full = os.listdir(img_root)
    shuffle(file_list_full)
    file_list_test = file_list_full[0:tst_count]
    file_list_train = file_list_full[-trainNum:]
    print("Start Building Tst File.")
    for img in tqdm(file_list_test, ncols=80):
        copyfile(os.path.join(img_root, img), os.path.join(res_file, 'tst', img))

    print("Start Building Train File.")
    for img in tqdm(file_list_train, ncols=80):
        score = getscore(img)
        if score >= 4:
            copyfile(os.path.join(img_root, img), os.path.join(res_file, 'true', img))
        else:
            copyfile(os.path.join(img_root, img), os.path.join(res_file, 'false', img))




if __name__ == '__main__':
    main()