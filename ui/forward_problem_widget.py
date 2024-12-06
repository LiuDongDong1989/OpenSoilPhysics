from PyQt5.QtWidgets import QWidget, QVBoxLayout, QSplitter, QLabel

class ForwardProblemWidget(QWidget):
    """
    正演问题界面组件，用于展示输入数据和水分特征曲线图形。
    包含两个主要部分：左侧用于输入数据，右侧用于展示图形。
    """

    def __init__(self):
        """
        初始化ForwardProblemWidget类的实例。
        
        :param self: 类实例本身。
        """
        super().__init__()  # 调用父类的构造函数
        self.initUI()  # 初始化用户界面

    def initUI(self):
        """
        初始化用户界面，设置布局和添加控件。
        
        :param self: 类实例本身。
        """
        # 创建一个QSplitter对象，用于分割窗口为左右两部分
        self.splitter = QSplitter(self)
        # 设置分割方向为水平
        self.splitter.setOrientation(1)  # 1代表水平分割，0代表垂直分割
        # 创建左侧和右侧的QWidget对象，用于放置各自的布局和控件
        self.left_widget = QWidget()
        self.right_widget = QWidget()
        # 将左侧和右侧的QWidget添加到QSplitter中
        self.splitter.addWidget(self.left_widget)
        self.splitter.addWidget(self.right_widget)
        # 设置QVBoxLayout作为主布局，并添加QSplitter
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.splitter)
        # 设置分割器的初始大小
        self.splitter.setSizes([400, 400])
        # 调用setup_left_widget和setup_right_widget方法来初始化左右部分的界面
        self.setup_left_widget()
        self.setup_right_widget()

    def setup_left_widget(self):
        """
        设置左侧QWidget的布局和控件，用于输入数据。
        
        :param self: 类实例本身。
        """
        # 创建QVBoxLayout布局，并设置为left_widget的布局
        self.left_layout = QVBoxLayout(self.left_widget)
        # 在左侧布局中添加一个QLabel，用于显示“输入数据”
        self.left_layout.addWidget(QLabel("输入数据"))
        # 在这里可以添加更多的输入控件，如QLineEdit, QComboBox等

    def setup_right_widget(self):
        """
        设置右侧QWidget的布局和控件，用于展示水分特征曲线图形。
        
        :param self: 类实例本身。
        """
        # 创建QVBoxLayout布局，并设置为right_widget的布局
        self.right_layout = QVBoxLayout(self.right_widget)
        # 在右侧布局中添加一个QLabel，用于显示“水分特征曲线”
        self.right_layout.addWidget(QLabel("水分特征曲线"))
        # 在这里可以添加一个自定义的绘图控件或者使用matplotlib等库来绘制图形