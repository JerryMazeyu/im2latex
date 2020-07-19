


# Im2Latex

![License](https://img.shields.io/apm/l/vim-mode.svg)

Deep CNN Encoder + LSTM Decoder with Attention for Image to Latex, the pytorch implemention of the model architecture used by the [Seq2Seq for LaTeX generation](https://guillaumegenthial.github.io/image-to-latex.html)



## Sample results from this implemention



![sample_result](imgs/sample_result.png)





## Experimental results on the IM2LATEX-100K  test dataset

| BLUE-4 | Edit Distance | Exact Match |
| ------ | ------------- | ----------- |
| 40.80  | 44.23         | 0.27        |



## Getting Started



**Install dependency:**

```bash
pip install -r requirement.txt
```

**Download the dataset for training:**

```bash
cd data
wget http://lstm.seas.harvard.edu/latex/data/im2latex_validate_filter.lst
wget http://lstm.seas.harvard.edu/latex/data/im2latex_train_filter.lst
wget http://lstm.seas.harvard.edu/latex/data/im2latex_test_filter.lst
wget http://lstm.seas.harvard.edu/latex/data/formula_images_processed.tar.gz
wget http://lstm.seas.harvard.edu/latex/data/im2latex_formulas.norm.lst
tar -zxvf formula_images_processed.tar.gz
```

**Preprocess:**

```bash
python preprocess.py
```

**Build vocab**
```bash
python build_vocab.py
```

**Train:**

     python train.py \
          --data_path=[data dir] \
          --save_dir=[the dir for saving ckpts] \
          --dropout=0.2 --add_position_features \
          --epoches=25 --max_len=150
**Evaluate:**

```bash
python evaluate.py --split=test \
     --model_path=[the path to model] \
     --data_path=[data dir] \
     --batch_size=32 \
     --ref_path=[the file to store reference] \
     --result_path=[the file to store decoding result]
```



## Features

- [x] Schedule Sampling from [Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks](https://arxiv.org/pdf/1506.03099.pdf)
- [x] Positional Embedding from [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [x] Batch beam search
- [x] Training from checkpoint 
- [ ] Improve the code of data loading for cpu/cuda memery efficiency 
- [ ] **Finetune hyper parameters for better performance**
- [ ] A HTML Page allowing upload picture to decode

# JerryPart

Based On [luopeixiang](https://github.com/luopeixiang/im2latex)

针对本项目数据，做出了IO流程的修改以及过程上的调优（GPU自适应）

## jerry_preprocess.py 
支持 fine-tune/从0构建新数据集 前置条件是构造属于自己的NAME2ID和ID2FORMULA.json
```bash
python jerry_preprocess.py
```
**注意：finetune后需要修改train.py data_path**

## inference.py
利用训练好的模型测试新的数据集测试新的模型
```bash
python inference.py
```
## train.py
修改 from_check_point 后，还可以指定epoch训练几轮






































