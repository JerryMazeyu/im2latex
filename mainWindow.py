import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import os
import json
import shutil
import torch
from copy import deepcopy

class FineTune(QWidget):
    def __init__(self):
        self.title = "im2latex微调工具"
        super(FineTune, self).__init__()
        self.width = 1000
        self.height = 800
        self.initUI()
        self.file = None
        self.allFile = []

    def initUI(self):
        self.setWindowTitle(self.title)

        self.label1 = QLabel("微调文件")
        self.label2 = QLabel("答案")
        self.label3 = QLabel("实验名称")

        self.button1 = QPushButton("选择文件")
        self.button1.clicked.connect(lambda: self.chooseDirectory(self.button1))
        self.button2 = QPushButton("添加")
        self.button2.clicked.connect(self.addItem)
        self.button3 = QPushButton("微调")
        self.button3.clicked.connect(self.fineTune)

        self.input1 = QLineEdit()
        self.input1.setFixedWidth(100)
        self.input1.setPlaceholderText("输入")
        self.input2 = QLineEdit()
        self.input2.setFixedWidth(100)
        self.input2.setPlaceholderText("输入")

        self.spinbox = QSpinBox()
        self.spinbox.setRange(5, 1000)

        self.trainlayout = QGridLayout()
        self.trainlayout.addWidget(self.button3, 0, 0, 1, 6)
        self.trainlayout.addWidget(self.spinbox, 0, 6, 1, 1)
        self.trainlayout.addWidget(QLabel("轮"), 0, 7, 1, 1)




        self.frame = QTableWidget(100,4)
        self.frame.setHorizontalHeaderLabels(['文件', '答案', '实验名称', '操作'])
        self.frame.setShowGrid(False)
        self.frame.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.frame.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.frame.setSelectionBehavior(QAbstractItemView.SelectRows)


        self.status = QStatusBar()
        self.grid = QGridLayout()
        self.grid.setSpacing(10)

        self.grid.addWidget(self.label1, 0, 0)
        self.grid.addWidget(self.button1, 1, 0)
        self.grid.addWidget(self.label2, 0, 1)
        self.grid.addWidget(self.input1, 1, 1)
        self.grid.addWidget(self.label3, 0, 2)
        self.grid.addWidget(self.input2, 1, 2)
        self.grid.addWidget(self.button2, 2, 0, 1, 3)
        self.grid.addWidget(self.frame, 3, 0, 12, 3)
        # self.grid.addWidget(self.button3, 15, 0, 1, 2)
        self.grid.addLayout(self.trainlayout, 15, 0, 1, 3)
        self.grid.addWidget(self.status, 16, 0, 1, 3)
        self.setLayout(self.grid)

    def chooseDirectory(self, widget):
        filename = QFileDialog.getExistingDirectory(self, "选取文件夹", directory= os.getcwd())
        self.file = filename
        self.status.showMessage(filename)
        if filename != "":
            widget.setText(filename)

    def isEmpty(self, widget):
        return widget.text() == ""

    def getItemNum(self):
        return len(self.allFile)

    def addItem(self):
        if not self.isEmpty(widget=self.input1) and not self.isEmpty(widget=self.input2):
            num = len(self.allFile)
            pathItem = QTableWidgetItem(self.file)
            ansItem = QTableWidgetItem(self.input1.text())
            expItem = QTableWidgetItem(self.input2.text())

            self.frame.setItem(num, 0, pathItem)
            self.frame.setItem(num, 1, ansItem)
            self.frame.setItem(num, 2, expItem)
            tarId = deepcopy(num)
            if tarId != -1:
                button = QPushButton("删除")
                button.clicked.connect(lambda: self.delItem(tarId))
                self.frame.setCellWidget(num, 3, button)
                self.allFile.append({"id": tarId, "path": self.file, "ans": self.input1.text(), "exp": self.input2.text()})
                self.button1.setText("选择文件")
                self.input1.setText("")
                self.input2.setText("")

    def delItem(self, row=1):
        ans = self.__findDict(row)
        if ans:
            tarRow = self.__findIndex(ans)
            self.allFile.remove(ans)
            self.frame.removeRow(tarRow)



    def __findDict(self, value):
        for i in self.allFile:
            if i['id'] == value:
                return i
        return False

    def __findIndex(self, value):
        return self.allFile.index(value)

    def fineTune(self):
        import sys
        a = (os.path.abspath(sys.argv[0])[0: os.path.abspath(sys.argv[0]).find('im2latex')])
        b = 'im2latex'
        root = os.path.join(a, b)
        for info in self.allFile:
            tarFile, tarAns, expName = info["path"], info["ans"], info["exp"]
            trueFile = os.path.join(tarFile, 'right')
            falseFile = os.path.join(tarFile, 'wrong')
            dict = {trueFile: 4, falseFile: 5}
            finetuneResPath = os.path.join(root, 'finetune', expName)
            assetPath = os.path.join(root, 'asset', 'output')
            oriID2FORMULA = os.path.join(root, 'ID2FORMULA.json')
            pythonPath = os.path.join(root, 'jerry_preprocess.py')
            try:
                shutil.rmtree(finetuneResPath)
                print("+" * 88)
                print(finetuneResPath, " has been deleted!")
                self.status.showMessage(finetuneResPath + " has been deleted!")
                print("+" * 88)
            except:
                pass

            if not os.path.exists(finetuneResPath):
                os.mkdir(finetuneResPath)

            with open(oriID2FORMULA, 'r') as file:
                id2formula_list = json.load(file)
            id2formula_list.append({"(4,0)": tarAns})
            id2formula_list.append({"(5,0)": "{ }"})
            with open(os.path.join(finetuneResPath, 'ID2FORMULA.json'), 'w') as file:
                json.dump(id2formula_list, file)
            print("+" * 88)
            print('Now Json Has Been Saved.')
            self.status.showMessage('Now Json Has Been Saved.')
            print("+" * 88)
            cmd = "python %s --ROOT \"%s\" --ORIGIN_PATH \"%s\" --SUP_NAME2ID \"%s\"" % (pythonPath, finetuneResPath, assetPath, dict)
            self.status.showMessage("Waiting...")
            os.system(cmd)
            self.status.showMessage(expName + " preprocess OK!")
            saveDir = os.path.join(root, 'ckpt')
            epoches = self.spinbox.value()
            train_cmd = "python train.py --data_path=\"%s\" --save_dir=\"%s\" --dropout=0.4 --batch_size=16 --epoches=%s --exp=\"%s\"" % (
            finetuneResPath, saveDir, epoches, expName)
            print("+" * 88)
            print("train CMD: ", train_cmd)
            if torch.cuda.is_available():
                self.status.showMessage("Trianing " + expName)
                os.system(train_cmd)
            else:
                self.status.showMessage("NO CUDA!")




if __name__ == '__main__':
    app = QApplication(sys.argv)
    form = FineTune()
    form.show()
    sys.exit(app.exec_())