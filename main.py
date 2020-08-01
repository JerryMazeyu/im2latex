import os
import json
import shutil
import torch

post_part = True

with open('param1.json', 'r') as file:
    param = json.load(file)

misc_param = param['misc']
fine_tune_param = param['fine_tune']
jerry_preprocess_param = param['jerry_preprocess']
train_param = param['train']
infer_param = param['inference']
# misc.py 将原始数据做处理成训练集/测试集
if not post_part:
    misc_cmd = 'python misc.py '
    for (k, v) in misc_param.items():
        misc_cmd += '--%s %s ' % (k, v)
    print("+"*88)
    print("miscCmd is: ", misc_cmd)
    print("+"*88)
    os.system(misc_cmd)


    # 修改ID2FORMULA.json，加入微调数据
    try:
        shutil.rmtree(jerry_preprocess_param['ROOT'])
        print("+" * 88)
        print(jerry_preprocess_param['ROOT'], " has been deleted!")
        print("+" * 88)
    except:
        pass

    if not os.path.exists(jerry_preprocess_param['ROOT']):
        os.mkdir(jerry_preprocess_param['ROOT'])

    with open(fine_tune_param['ori_file'], 'r') as file:
        id2formula_list = json.load(file)
    id2formula_list.append({"(4,0)": fine_tune_param['ans']})
    id2formula_list.append({"(5,0)": "{ }"})
    with open(os.path.join(jerry_preprocess_param['ROOT'], 'ID2FORMULA.json'), 'w') as file:
        json.dump(id2formula_list, file)
    print("+"*88)
    print('Now Json Has Been Saved.')
    print("+"*88)



    # 生成待训练数据
    jerry_preprocess_cmd = "python jerry_preprocess.py "
    for (k, v) in jerry_preprocess_param.items():
        jerry_preprocess_cmd += '--%s %s ' % (k, "\"" + v + "\"")
    sup_ID2NAME = "\"" +str({os.path.join(misc_param['res_file'], 'true'): 4, os.path.join(misc_param['res_file'], 'false'): 5}) + "\""
    jerry_preprocess_cmd += '--SUP_NAME2ID %s' % sup_ID2NAME
    print("+"*88)
    print("jpCmd is: ", jerry_preprocess_cmd)
    print("+"*88)
    os.system(jerry_preprocess_cmd)

    # 训练（如果有GPU）
    if torch.cuda.is_available():
        train_cmd = "python train.py --data_path='./Jerry/Jerry2018T10' --save_dir='./ckpt' --dropout=0.4 --batch_size=16 --epoches=%s" % train_param['epoch']
    else:
        print("+" * 88)
        print("Cannot train right now.")
        print("+" * 88)
        raise RuntimeError("vocab.pkl should fit to checkpoint.")

# 推断


infer_cmd = 'python inference.py -i %s --expname %s --csvPath %s --ans %s --colName %s' % (infer_param['info_path'], infer_param['expname'], misc_param['score_csv'], "\"" + fine_tune_param['ans'] + "\"", misc_param['col_name'])
print("+"*88)
print("InferenceCmd is: ", infer_cmd)
print("+"*88)
res = os.system(infer_cmd)
print(res)












