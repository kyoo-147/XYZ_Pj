# import thư viện và các module cần thiết 
import os
# chia luồng xử lí 
import threading
import time
# một số phiên bản  cv2 không tương thích sẽ không truy xuất được 
# import cv2.xfeatures2d
import cv2
import numpy as np
from PyQt5 import uic, QtWidgets, QtGui, QtCore
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QSize, pyqtSlot
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QKeySequence, QPalette, QColor
from PyQt5.QtCore import Qt
import time
import sys
import PyQt5
from tensorflow.keras.models import load_model

classifier = load_model('ASLModel.h5')  # loading the model
currentMode = 'A'
recognizedResult = 'Z'
count = 0

def fileSearch():
    fileEntry = []
    for file in os.listdir("SampleGestures"):
        if file.endswith(".png"):
            fileEntry.append(file)
    return fileEntry

'''
predicator using Keras
Không nên có phụ thuộc PyQt.
'''

def predictor():
    import numpy as np
    from tensorflow.keras.preprocessing import image
    test_image = image.load_img('1.png', target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = classifier.predict(test_image)
    gesname = ''
    fileEntry = fileSearch()
    # đọc bảng``
    for i in range(len(fileEntry)):
        image_to_compare = cv2.imread("./SampleGestures/" + fileEntry[i])
        original = cv2.imread("1.png")
        sift = cv2.SIFT_create()
        kp_1, desc_1 = sift.detectAndCompute(original, None)
        kp_2, desc_2 = sift.detectAndCompute(image_to_compare, None)

        index_params = dict(algorithm=0, trees=5)
        search_params = dict()
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(desc_1, desc_2, k=2)

        good_points = []
        ratio = 0.6
        for m, n in matches:
            if m.distance < ratio * n.distance:
                good_points.append(m)
        if (abs(len(good_points) + len(
                matches)) > 20):
            gesname = fileEntry[i]
            gesname = gesname.replace('.png', '')
            if (gesname == 'sp'):
                gesname = ' '
            return gesname

    if result[0][0] == 1:
        return 'A'
    elif result[0][1] == 1:
        return 'B'
    elif result[0][2] == 1:
        return 'C'
    elif result[0][3] == 1:
        return 'D'
    elif result[0][4] == 1:
        return 'E'
    elif result[0][5] == 1:
        return 'F'
    elif result[0][6] == 1:
        return 'G'
    elif result[0][7] == 1:
        return 'H'
    elif result[0][8] == 1:
        return 'I'
    elif result[0][9] == 1:
        return 'J'
    elif result[0][10] == 1:
        return 'K'
    elif result[0][11] == 1:
        return 'L'
    elif result[0][12] == 1:
        return 'M'
    elif result[0][13] == 1:
        return 'N'
    elif result[0][14] == 1:
        return 'O'
    elif result[0][15] == 1:
        return 'P'
    elif result[0][16] == 1:
        return 'Q'
    elif result[0][17] == 1:
        return 'R'
    elif result[0][18] == 1:
        return 'S'
    elif result[0][19] == 1:
        return 'T'
    elif result[0][20] == 1:
        return 'U'
    elif result[0][21] == 1:
        return 'V'
    elif result[0][22] == 1:
        return 'W'
    elif result[0][23] == 1:
        return 'X'
    elif result[0][24] == 1:
        return 'Y'
    elif result[0][25] == 1:
        return 'Z'


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        uic.loadUi('pyqt_UI/last_ui_rabbit.ui', self)

        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(53, 53, 53))
        palette.setColor(QPalette.WindowText, Qt.white)
        palette.setColor(QPalette.Base, QColor(25, 25, 25))
        palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        palette.setColor(QPalette.ToolTipBase, Qt.white)
        palette.setColor(QPalette.ToolTipText, Qt.white)
        palette.setColor(QPalette.Text, Qt.white)
        palette.setColor(QPalette.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ButtonText, Qt.white)
        palette.setColor(QPalette.BrightText, Qt.red)
        palette.setColor(QPalette.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.HighlightedText, Qt.black)
        self.setPalette(palette)

        self.setWindowTitle('Rabbit AI-NaVin Tech')
        self.setWindowIcon(QIcon('icons/windowlogo.png'))
        # self.setWindowFlags(QtCore.Qt.FramelessWindowHint)

        """ Cho biết chế độ hiện tại (tất cả logic được xử lý bằng biến này, ví dụ: mode = A -> Hiển thị ảnh hướng dẫn 'A' """
        """ OnFirstRun """
        self.notifyModeChanged(currentMode)
        self.setTutorialButton()

        self.heightOfCamView = self.label_camView.height()
        self.widthOfCamView = self.label_camView.width()

        self.roiLeftTop = (550, 100)
        self.roiRightBottom = (750, 300)

        self.video_thread(MainWindow)

        self.playProgress()
        self.label_doneTutorial.hide()

    def playProgress(self):
        # Chơi ngay sau khi mở chế độ xem
        self.progressBarThread = ProgressThread()
        # Cài đặt sự kiện
        self.progressBarThread.countChanged.connect(self.onCountChanged)
        self.progressBarThread.start()

    # Được gọi bất cứ khi nào giá trị đếm thay đổi.
    def onCountChanged(self, value):
        self.progressBar.setValue(value)
        if value == 100:
            self.showDoneTutorial()

    # Hàm show kết quả thành công hướng dẫn 
    def showDoneTutorial(self):
        # tạo nhanx show label huớng dẫn show
        self.label_doneTutorial.show()
        # hàm khởi tại chiều cao của label
        height = self.label_doneTutorial.height()
        # hàm khởi tạo chiều ngang của label
        width = self.label_doneTutorial.width()
        # hiển thị hình ảnh label sau khi đã thực hiện detect thành công 
        pixmap = QPixmap("./resource/UI/aftertutorial.png")
        # hàm tạo cơ chế chiều cao, ngang, chuyển đổi smooth,
        pixmap = pixmap.scaled(width, height, QtCore.Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.label_doneTutorial.setPixmap(pixmap)

    def hideDoneTutorial(self):
        self.label_doneTutorial.hide()


    # TODO: Tôi không muốn các nút được liệt kê trông xấu nên tôi cần tối ưu hóa chúng (kế thừa và biến nó thành một lớp ??)
    @pyqtSlot()
    def setTutorialButton(self):
        self.button_A.setIcon(QtGui.QIcon("./resource/alphabet/button/A.png"))
        self.button_A.setIconSize(QSize(30, 30))
        self.button_A.clicked.connect(self.alphabetButtonClicked)

        self.button_B.setIcon(QtGui.QIcon("./resource/alphabet/button/B.png"))
        self.button_B.setIconSize(QSize(30, 30))
        self.button_B.clicked.connect(self.alphabetButtonClicked)

        self.button_C.setIcon(QtGui.QIcon("./resource/alphabet/button/C.png"))
        self.button_C.setIconSize(QSize(30, 30))
        self.button_C.clicked.connect(self.alphabetButtonClicked)

        self.button_D.setIcon(QtGui.QIcon("./resource/alphabet/button/D.png"))
        self.button_D.setIconSize(QSize(30, 30))
        self.button_D.clicked.connect(self.alphabetButtonClicked)

        self.button_E.setIcon(QtGui.QIcon("./resource/alphabet/button/E.png"))
        self.button_E.setIconSize(QSize(30, 30))
        self.button_E.clicked.connect(self.alphabetButtonClicked)

        self.button_F.setIcon(QtGui.QIcon("./resource/alphabet/button/F.png"))
        self.button_F.setIconSize(QSize(30, 30))
        self.button_F.clicked.connect(self.alphabetButtonClicked)

        self.button_G.setIcon(QtGui.QIcon("./resource/alphabet/button/G.png"))
        self.button_G.setIconSize(QSize(30, 30))
        self.button_G.clicked.connect(self.alphabetButtonClicked)

        self.button_H.setIcon(QtGui.QIcon("./resource/alphabet/button/H.png"))
        self.button_H.setIconSize(QSize(30, 30))
        self.button_H.clicked.connect(self.alphabetButtonClicked)

        self.button_I.setIcon(QtGui.QIcon("./resource/alphabet/button/I.png"))
        self.button_I.setIconSize(QSize(30, 30))
        self.button_I.clicked.connect(self.alphabetButtonClicked)

        self.button_J.setIcon(QtGui.QIcon("./resource/alphabet/button/J.png"))
        self.button_J.setIconSize(QSize(30, 30))
        self.button_J.clicked.connect(self.alphabetButtonClicked)

        self.button_K.setIcon(QtGui.QIcon("./resource/alphabet/button/K.png"))
        self.button_K.setIconSize(QSize(30, 30))
        self.button_K.clicked.connect(self.alphabetButtonClicked)

        self.button_L.setIcon(QtGui.QIcon("./resource/alphabet/button/L.png"))
        self.button_L.setIconSize(QSize(30, 30))
        self.button_L.clicked.connect(self.alphabetButtonClicked)

        self.button_M.setIcon(QtGui.QIcon("./resource/alphabet/button/M.png"))
        self.button_M.setIconSize(QSize(30, 30))
        self.button_M.clicked.connect(self.alphabetButtonClicked)

        self.button_N.setIcon(QtGui.QIcon("./resource/alphabet/button/N.png"))
        self.button_N.setIconSize(QSize(30, 30))
        self.button_N.clicked.connect(self.alphabetButtonClicked)

        self.button_O.setIcon(QtGui.QIcon("./resource/alphabet/button/O.png"))
        self.button_O.setIconSize(QSize(30, 30))
        self.button_O.clicked.connect(self.alphabetButtonClicked)

        self.button_P.setIcon(QtGui.QIcon("./resource/alphabet/button/P.png"))
        self.button_P.setIconSize(QSize(30, 30))
        self.button_P.clicked.connect(self.alphabetButtonClicked)

        self.button_Q.setIcon(QtGui.QIcon("./resource/alphabet/button/Q.png"))
        self.button_Q.setIconSize(QSize(30, 30))
        self.button_Q.clicked.connect(self.alphabetButtonClicked)

        self.button_R.setIcon(QtGui.QIcon("./resource/alphabet/button/R.png"))
        self.button_R.setIconSize(QSize(30, 30))
        self.button_R.clicked.connect(self.alphabetButtonClicked)

        self.button_S.setIcon(QtGui.QIcon("./resource/alphabet/button/S.png"))
        self.button_S.setIconSize(QSize(30, 30))
        self.button_S.clicked.connect(self.alphabetButtonClicked)

        self.button_T.setIcon(QtGui.QIcon("./resource/alphabet/button/T.png"))
        self.button_T.setIconSize(QSize(30, 30))
        self.button_T.clicked.connect(self.alphabetButtonClicked)

        self.button_U.setIcon(QtGui.QIcon("./resource/alphabet/button/U.png"))
        self.button_U.setIconSize(QSize(30, 30))
        self.button_U.clicked.connect(self.alphabetButtonClicked)

        self.button_V.setIcon(QtGui.QIcon("./resource/alphabet/button/V.png"))
        self.button_V.setIconSize(QSize(30, 30))
        self.button_V.clicked.connect(self.alphabetButtonClicked)

        self.button_W.setIcon(QtGui.QIcon("./resource/alphabet/button/W.png"))
        self.button_W.setIconSize(QSize(30, 30))
        self.button_W.clicked.connect(self.alphabetButtonClicked)

        self.button_X.setIcon(QtGui.QIcon("./resource/alphabet/button/X.png"))
        self.button_X.setIconSize(QSize(30, 30))
        self.button_X.clicked.connect(self.alphabetButtonClicked)

        self.button_Y.setIcon(QtGui.QIcon("./resource/alphabet/button/Y.png"))
        self.button_Y.setIconSize(QSize(30, 30))
        self.button_Y.clicked.connect(self.alphabetButtonClicked)

        self.button_Z.setIcon(QtGui.QIcon("./resource/alphabet/button/Z.png"))
        self.button_Z.setIconSize(QSize(30, 30))
        self.button_Z.clicked.connect(self.alphabetButtonClicked)

    def alphabetButtonClicked(self):
        button = self.sender()
        self.hideDoneTutorial()
        self.playProgress()
        objName = button.objectName()
        # Tôi biết đó là một vụ hack. lấy tên đối tượng 'button_A' và cắt thành 'A'
        self.notifyModeChanged(objName[-1])

    """ Tất cả các tiện ích trong chế độ xem được làm mới bằng cách gọi hàm này """
    # Hàm thông báo thay đổi chế độ
    def notifyModeChanged(self, modeName):
        global currentMode
        global count
        count = 0
        currentMode = modeName
        self.loadTutorialImageFromMode()
        self.statusBar().showMessage('Ký tự bạn hiện đang học là {}.'.format(modeName))
    """ label_tutorialView Chèn hình ảnh hướng dẫn phù hợp với chế độ hiện tại."""

    def loadTutorialImageFromMode(self):
         self.image = QPixmap("./resource/alphabet/overlay_image/{}.png".format(currentMode))
         self.image = self.image.scaled(180, 100, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
         self.label_tutorialView.setPixmap(self.image)

    def videoToFrame(self, MainWindow):
        global recognizedResult
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            resizedImage = cv2.resize(frame, (self.widthOfCamView, self.heightOfCamView))

            # Khung cho mô hình deep learning
            self.saveToPredictor(resizedImage)

            # Xử lý khung được hiển thị trong UI
            rgbImage = cv2.cvtColor(resizedImage, cv2.COLOR_BGR2RGB)
            # img1 = cv2.rectangle(rgbImage, (150, 50), (300, 200), (0, 255, 0), thickness=2, lineType=8, shift=0)
            img1 = cv2.rectangle(rgbImage, self.roiLeftTop, self.roiRightBottom, (0, 255, 0), thickness=1, lineType=8,
                                 shift=0)

            h, w, c = img1.shape
            qImg = QImage(img1.data, w, h, c * w, QImage.Format_RGB888)

            self.label_camView.setPixmap(QPixmap.fromImage(qImg))
            self.label_camView.update()

            # Cập nhật văn bản được nhận dạng
            self.updatePredictedResult()

        cap.release()
        cv2.destroyAllWindows()

    def saveToPredictor(self, frame):
        # Binarize thành inRange dựa trên giá trị HSV màu của bàn tay
        lower_blue = np.array([0, 58, 50])
        upper_blue = np.array([30, 255, 255])

        imcrop = frame[self.roiLeftTop[1]:self.roiRightBottom[1], self.roiLeftTop[0]:self.roiRightBottom[0]]
        hsv = cv2.cvtColor(imcrop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        save_img = cv2.resize(mask, (64, 64))
        cv2.imwrite('1.png', save_img)

    def updatePredictedResult(self):
        global recognizedResult
        recognizedResult = predictor()
        self.label_recognizedText.setText(recognizedResult)


    # video_to_frame - sử dụng như một chủ đề
    def video_thread(self, MainWindow):
        thread = threading.Thread(target=self.videoToFrame, args=(self,))
        thread.daemon = True  # Khi chương trình kết thúc, tiến trình cũng kết thúc (phát trong nền X)
        thread.start()


class ProgressThread(QThread):
    """
    Runs a counter thread.
    """
    # Cung cấp sự kiện
    countChanged = pyqtSignal(int)

    def run(self):
        TIME_LIMIT = 100
        global count
        count = 0
        while count < TIME_LIMIT:
            time.sleep(0.1)
            print(currentMode, recognizedResult)
            if currentMode == recognizedResult:
                count += 2
            self.countChanged.emit(count)

if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
