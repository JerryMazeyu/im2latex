import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import os


class Stream(QObject):
    """Redirects console output to text widget."""
    newText = pyqtSignal(str)

    def write(self, text):
        self.newText.emit(str(text))


class App(QWidget):

    def __init__(self):
        super().__init__()
        self.title = 'im2latex Finetune'
        self.left = 200
        self.top = 200
        self.width = 1000
        self.height = 800
        self.initUI()
        sys.stdout = Stream(newText=self.onUpdateText)


    def initUI(self):

        self.setWindowTitle(self.title)
        # self.setGeometry(self.left, self.top, self.width, self.height)
        # ===========================元素=============================
        # label1
        self.label1 = QLabel("param1.json Path")
        self.label1.setStyleSheet('border-width: 1px;border-style: solid;border-color: rgb(220, 220, 220);background-color: rgb(220, 220, 220);')
        # label2
        self.label2 = QLabel("param2.json Path")
        self.label2.setStyleSheet('border-width: 1px;border-style: solid;border-color: rgb(220, 220, 220);background-color: rgb(220, 220, 220);')

        # button1
        self.button1 = QPushButton('Open File')
        self.button1.clicked.connect(self.chooseFile1)

        # button2
        self.button2 = QPushButton('Open File')
        self.button2.clicked.connect(self.chooseFile2)

        # button
        self.button = QPushButton('Start')
        self.button.setToolTip('Start Fine-tune And Inference')
        self.button.clicked.connect(self.on_click)

        # progressBar
        self.progressBar = QProgressBar(self)

        # frame
        self.process = QTextEdit(self, readOnly=True)
        self.process.setFixedHeight(200)

        # statusBar
        self.statusBar = QStatusBar()

        # ===========================布局===========================
        vlayout = QVBoxLayout()  # 进度条和上面的部分分开

        spliter1 = QSplitter(Qt.Horizontal)
        spliter1.addWidget(self.label1)
        spliter1.addWidget(self.button1)


        spliter2 = QSplitter(Qt.Horizontal)
        spliter2.addWidget(self.label2)
        spliter2.addWidget(self.button2)


        spliter3 = QSplitter(Qt.Vertical)
        spliter3.addWidget(spliter1)
        spliter3.addWidget(spliter2)
        spliter3.setFixedSize(400, 200)



        vlayout.addWidget(spliter1)
        vlayout.addWidget(spliter2)
        vlayout.addWidget(self.process)
        vlayout.addWidget(self.progressBar)
        vlayout.addWidget(self.button)
        vlayout.addWidget(self.statusBar)
        self.setLayout(vlayout)

        self.show()

    def onUpdateText(self, text):
        """Write console output to text widget."""
        cursor = self.process.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        self.process.setTextCursor(cursor)
        self.process.ensureCursorVisible()

    def chooseFile1(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "选取文件", os.getcwd(), "All Files(*);;Text Files(*.txt)")
        self.file1 = fileName
        self.statusBar.showMessage(fileName)
        self.button1.setText(os.getcwd())

    def chooseFile2(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "选取文件", os.getcwd(), "All Files(*);;Text Files(*.txt)")
        self.file2 = fileName
        self.button2.setText(os.getcwd())




    def on_click(self):
        import os
        import json
        import shutil
        import torch

        post_part = False

        with open(self.file1, 'r') as file:
            param = json.load(file)

        misc_param = param['misc']
        fine_tune_param = param['fine_tune']
        jerry_preprocess_param = param['jerry_preprocess']
        train_param = param['train']
        infer_param = param['inference']
        # misc.py 将原始数据做处理成训练集/测试集
        if not post_part:
            misc_cmd = 'python /Users/mazeyu/PycharmProjects/autoscore/im2latex/misc.py '
            for (k, v) in misc_param.items():
                misc_cmd += '--%s %s ' % (k, v)
            print("+" * 88)
            print("miscCmd is: ", misc_cmd)
            print("+" * 88)
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
            print("+" * 88)
            print('Now Json Has Been Saved.')
            print("+" * 88)

            # 生成待训练数据
            jerry_preprocess_cmd = "python /Users/mazeyu/PycharmProjects/autoscore/im2latex/jerry_preprocess.py "
            for (k, v) in jerry_preprocess_param.items():
                jerry_preprocess_cmd += '--%s %s ' % (k, "\"" + v + "\"")
            sup_ID2NAME = "\"" + str({os.path.join(misc_param['res_file'], 'true'): 4,
                                      os.path.join(misc_param['res_file'], 'false'): 5}) + "\""
            jerry_preprocess_cmd += '--SUP_NAME2ID %s' % sup_ID2NAME
            print("+" * 88)
            print("jpCmd is: ", jerry_preprocess_cmd)
            print("+" * 88)
            os.system(jerry_preprocess_cmd)

            # 训练（如果有GPU）
            if torch.cuda.is_available():
                train_cmd = "python train.py --data_path='%s' --save_dir='./ckpt' --dropout=0.4 --batch_size=16 --epoches=%s" % (
                jerry_preprocess_param['ROOT'], train_param['epoch'])
                print("+" * 88)
                print("trainCmd is: ", train_cmd)
                os.system(train_cmd)
            else:
                print("+" * 88)
                print("Cannot train right now.")
                print("+" * 88)
                print("vocab.pkl should fit to checkpoint.")




if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())