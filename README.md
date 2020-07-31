# im2latex 
**(模型出自[luopeixiang](https://github.com/luopeixiang/im2latex))**
## 利用seq2seq模型从图像生成latex代码
Deep CNN Encoder + LSTM Decoder with Attention for Image to Latex
## Quick Start
```shell
git clone https://github.com/JerryMazeyu/im2latex.git
```
根据param1.json.example修改配置param1.json 和 param.json.   
如果有GPU的话，将`post_part`设置为`False`，直接运行：

```python
python main.py
```

如果没有GPU的话，将`post_part`设置为`False`，抛出`RunTimeError`，然后将`param1.json`中`jerry_preprocess`中`ROOT`文件夹放到服务器上，运行

```python
python train.py --data_path=<ROOT> --save_dir='./ckpt' --dropout=0.4 --batch_size=16 --epoches=<epoch>
```
训练好之后将`best_ckpt`放到`im2latex/ckpt/`，然后将`post_part`设置为`True`，运行`main.py`

训练好的结果将根据`param1.json`中的配置放在`results/`下。

如果只是做推断，直接修改param.json，运行

```python
python inference.py -i param.json
```


