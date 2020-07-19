import os
#
# root = '/Users/mazeyu/Downloads/dataset/'  #
# txt_name = 'dataset_手写填空-分式-1.txt'
# img_root = '/Users/mazeyu/Downloads/dataset/output/手写填空-分式-1'
# save_dir = './experiment'

root = '/Users/mazeyu/PycharmProjects/autoscore/im2latex/experiment/valid'
txt_name = 'valid.txt'
img_root = '/Users/mazeyu/PycharmProjects/autoscore/im2latex/experiment/valid/data'
save_dir = '/Users/mazeyu/PycharmProjects/autoscore/im2latex/experiment/valid'
base_formula_lst = '/Users/mazeyu/PycharmProjects/autoscore/im2latex/data/im2latex_formulas.norm.lst'






# img_root = os.path.join(root, 'imgs')
txt_path = os.path.join(root, txt_name)

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

file_lst = open(os.path.join(save_dir, 'im2latex_valid_filter.lst'), 'w')
formula_lst = open(os.path.join(save_dir, 'im2latex_formulas.lst'), 'w')

if len(base_formula_lst) != 0:
    with open(base_formula_lst) as ff:
        formulas = [formula.strip('\n') for formula in ff.readlines()]
else:
    formulas = []

def getid(formula):
    return str(formulas.index(formula))



with open(txt_path, 'r') as file:
    for (id,dict_) in enumerate(file.readlines()):
        tmp = eval(str(dict_))
        if len(base_formula_lst) != 0:
            fid = getid(str(tmp['label']))
            file_lst.write(str(tmp['file']) + ' ' + fid + '\n')
        else:
            file_lst.write(str(tmp['file']) + ' ' + str(id) + '\n')
        formula_lst.write(str(tmp['label']) + '\n')
file_lst.close()
formula_lst.close()



