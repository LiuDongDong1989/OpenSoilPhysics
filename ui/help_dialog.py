# ui/help_dialog.py
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton

class HelpDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        label = QLabel("这里是帮助信息的详细说明。")
        button = QPushButton("确定")
        button.clicked.connect(self.accept)

        layout.addWidget(label)
        layout.addWidget(button)

        self.setLayout(layout)
        self.setWindowTitle('帮助信息')