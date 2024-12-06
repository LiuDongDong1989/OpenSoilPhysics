# ui/about_dialog.py
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton

class AboutDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        label = QLabel("PySoilPhysics - 开源土壤物理学计算软件\n版权所有 © 2024 刘冬冬课题组")
        button = QPushButton("确定")
        button.clicked.connect(self.accept)

        layout.addWidget(label)
        layout.addWidget(button)

        self.setLayout(layout)
        self.setWindowTitle('版权声明')