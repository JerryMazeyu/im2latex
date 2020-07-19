txt_file_path = '/Users/mazeyu/Downloads/dataset/dataset_手写填空-分式-1.txt'
img_file_path = '/Users/mazeyu/PycharmProjects/autoscore/im2latex/experiment/valid/data'
tar_txt = '/Users/mazeyu/PycharmProjects/autoscore/im2latex/experiment/valid/valid.txt'


import os

tar_imgs = os.listdir(img_file_path)

def helper(line):
    return True if eval(line)['file'] in tar_imgs else False


tar = open(tar_txt, 'w')
with open(txt_file_path, 'r') as file:
    res = filter(helper, file.readlines())
    for i in res:
        tar.write(i)
    tar.close()

