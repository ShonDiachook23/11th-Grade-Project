import sys
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QFileDialog, QLabel, QVBoxLayout, QWidget, QPushButton, QMainWindow, QGridLayout, QComboBox, QCheckBox, QRadioButton, QGroupBox, QSlider, QSpinBox, QDoubleSpinBox, QProgressBar, QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView, QSizePolicy, QFrame, QSplitter, QTabWidget, QTextEdit, QPlainTextEdit, QScrollArea, QSizePolicy, QSpacerItem, QLayout, QLayoutItem, QBoxLayout, QFormLayout, QApplication
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtCore import Qt

class MainWindow(QMainWindow):
    def __init__(self, model):
        super().__init__()

        self.model = model
        
        self.setWindowTitle("Shon Applications™")
        self.setGeometry(100, 100, 800, 600)




model = None
app = QApplication([])
window = MainWindow(model)
window.show()

app.exec_()
