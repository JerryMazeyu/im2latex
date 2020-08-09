import pandas as pd
import os.path as path
import os
import shutil






def jerryEvaluation(trueFile, falseFile, csvPath='/Users/mazeyu/PycharmProjects/autoscore/2018cksx.csv', save=True, expName='exp1', colName='T11_1'):
    """
    给出一些指标
    :param csvPath:
    :param trueFile:
    :param falseFile:
    :return:
    """
    df = pd.read_csv(csvPath, sep=',')
    df = df.set_index('KSH')

    def getscore(pngname):
        id = pngname.strip('0')

        id = id.split('_')[0]
        return df[colName].loc[int(id)]

    fileTrue = open(trueFile, 'r')
    fileFalse = open(falseFile, 'r')

    originalRoot = path.split(fileTrue.readlines()[0])[0]
    fileTrue.close()
    fileTrue = open(trueFile, 'r')
    predTrueList = [path.split(x)[1].strip('\n') for x in fileTrue.readlines()]
    predFalseList = [path.split(x.split(" ")[0])[1] for x in fileFalse.readlines()]

    fileTrue.close()
    fileFalse.close()

    sumCount = len(predFalseList) + len(predTrueList)
    true2False = []  # 对的预测为错
    false2True = []  # 错的预测为对
    rightAns = []  # 没错的

    for i in predTrueList:
        if getscore(i) >= 4:
            rightAns.append(i)
        else:
            false2True.append(i)
    for i in predFalseList:
        if getscore(i) == 0:
            rightAns.append(i)
        else:
            true2False.append(i)

    A = 'Real True -> Pred False : %s in %s' % (len(true2False), sumCount)
    B = 'Real False -> Pred True : %s in %s' % (len(false2True), sumCount)
    C = 'Right Ans: %s in %s' % (len(rightAns), sumCount)
    print(A)
    print(B)
    print(C)
    if save:
        print('Starting Saving...')
        if not os.path.exists(path.join('result', expName)):
            root = path.join('./results', expName)
            os.mkdir(root)
            os.mkdir(path.join(root, 'True2False'))
            os.mkdir(path.join(root, 'False2True'))
        else:
            raise ValueError('ExpName Exist.')
        with open(path.join(root, 'summary.txt'), 'w') as file:
            file.writelines(A)
            file.writelines('\n')
            file.writelines(B)
            file.writelines('\n')
            file.writelines(C)
            file.writelines('\n')
        for img in true2False:
            shutil.copyfile(path.join(originalRoot, img), path.join(root, 'True2False', img))
        for img in false2True:
            shutil.copyfile(path.join(originalRoot, img), path.join(root, 'False2True', img))
        print("Info Saved.")



if __name__ == '__main__':
    jerryEvaluation(trueFile= 'results/2020-07-31 17:08:55true.txt',falseFile='results/2020-07-31 17:08:55false.txt', expName='exp1_2018T11')
