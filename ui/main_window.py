# ui/main_window.py
from PyQt5.QtWidgets import QMainWindow, QMenuBar, QMenu, QAction, QMessageBox
from ui.about_dialog import AboutDialog  # 从ui模块导入AboutDialog类
from ui.help_dialog import HelpDialog  # 从ui模块导入HelpDialog类

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()  # 调用QMainWindow的构造函数来初始化主窗口
        self.initUI()  # 初始化用户界面

    def initUI(self):
        # 设置窗口的标题和大小
        self.setWindowTitle('PySoilPhysics')
        self.setGeometry(100, 100, 800, 800)

        # 创建菜单栏
        menu_bar = self.menuBar()

        # 创建“问题选择”菜单，并添加到菜单栏
        problem_selection_menu = menu_bar.addMenu('问题选择')
        # 创建“帮助”菜单，并添加到菜单栏
        help_menu = menu_bar.addMenu('帮助')

        # 创建菜单项，并添加到“问题选择”菜单
        forward_problem_action = QAction('正演问题', self)
        inverse_problem_action = QAction('反演问题', self)
        transfer_function_action = QAction('传递函数', self)

        # 创建菜单项，并添加到“帮助”菜单
        about_action = QAction('版权声明', self)
        help_action = QAction('帮助', self)

        # 将菜单项添加到相应的菜单中
        problem_selection_menu.addAction(forward_problem_action)
        problem_selection_menu.addAction(inverse_problem_action)
        problem_selection_menu.addAction(transfer_function_action)
        help_menu.addAction(about_action)
        help_menu.addAction(help_action)

        # 连接菜单项的触发信号到相应的槽函数
        about_action.triggered.connect(self.show_about_dialog) 
        help_action.triggered.connect(self.show_help_dialog)

        forward_problem_action.triggered.connect(self.show_forward_problem)
        inverse_problem_action.triggered.connect(self.show_inverse_problem)
        transfer_function_action.triggered.connect(self.show_transfer_function)

    def show_about_dialog(self):
        # 实例化AboutDialog，并显示版权声明对话框
        about_dialog = AboutDialog()  
        about_dialog.exec_()  

    def show_help_dialog(self):
        # 实例化HelpDialog，并显示帮助信息对话框
        help_dialog = HelpDialog()
        help_dialog.exec_()

    def show_forward_problem(self):
        # 显示正演问题的详细说明
        QMessageBox.information(self, '正演问题', '正演问题的详细说明。')

    def show_inverse_problem(self):
        # 显示反演问题的详细说明
        QMessageBox.information(self, '反演问题', '反演问题的详细说明。')

    def show_transfer_function(self):
        # 显示传递函数计算的详细说明
        QMessageBox.information(self, '传递函数计算', '传递函数计算的详细说明。')