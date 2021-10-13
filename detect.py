import sys
import face_recognition
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import pandas as pd
import os
import cv2
import datetime
from ui.attend_ui import Ui_MainWindow


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.class_names = []
        self.encode_images = []

        now = QDate.currentDate()
        currentDate = now.toString('ddd dd MM yyyy')
        currentTime = datetime.datetime.now().strftime('%I:%M %p')
        self.ui.labelDate.setText(currentDate)
        self.ui.labelTime.setText(currentTime)

        self.timer = QTimer()
        self.start_video()
        self.timer.timeout.connect(self.next_frame)
        self.timer.start(20)

        self.get_information()
        self.ui.StartWork.clicked.connect(self.button_start_work)
        self.ui.EndWork.clicked.connect(self.button_end_work)

    def start_video(self):
        self.cap = cv2.VideoCapture(0)

    def next_frame(self):
        ret, image = self.cap.read()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)
        image = self.face_rec(image)
        height, width, channel = image.shape
        qImg = QImage(image.data, width, height, QImage.Format_RGB888)

        self.ui.imgLabel.setPixmap(QPixmap.fromImage(qImg))
        self.ui.imgLabel.setScaledContents(True)

    def face_rec(self, image):
        class_names = self.class_names
        known_faces_encoding = self.encode_images
        face_loc = face_recognition.face_locations(image)
        face_encode = face_recognition.face_encodings(image, face_loc)

        for encodeFace, faceLoc in zip(face_encode, face_loc):
            try:
                face_detect = face_recognition.compare_faces(known_faces_encoding, encodeFace, tolerance=0.5)
                detect_index = face_detect.index(True)
                name = class_names[detect_index]
            except:
                name = 'Unknowm'

            y1, x2, y2, x1 = faceLoc
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

        return image

    @pyqtSlot()
    def save_img(self):
        ret, image = self.cap.read()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)
        if ret:
            path = 'Image Storage'
            img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            try:
                cv2.imwrite(f'{path}/new_img.jpg', img)
                print('File Saved!')
            except Exception as e:
                print(e)

    def get_information(self):
        base_path = 'Image Storage'
        if not os.path.exists(base_path):
            os.mkdir(base_path)

        images = []
        attendance_list = os.listdir(base_path)
        df = pd.read_csv('database_person.csv')

        for img_file in attendance_list:
            image = cv2.imread(f'{base_path}/{img_file}')
            person = df.loc[df['Person File'] == img_file].to_numpy()
            name_class = person[0][0]
            images.append(image)
            self.class_names.append(name_class)

        for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(img)
            encode_img = face_recognition.face_encodings(img, boxes)[0]
            self.encode_images.append(encode_img)

    @pyqtSlot()
    def button_start_work(self):
        status = 'Start Work'
        self.detect_face(status)

    @pyqtSlot()
    def button_end_work(self):
        status = 'End Work'
        self.detect_face(status)

    def detect_face(self, status):
        encode_list_known = self.encode_images
        class_names = self.class_names

        ret, image = self.cap.read()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)

        encode_frame = face_recognition.face_encodings(image)[0]
        try:
            detect = face_recognition.compare_faces(encode_list_known, encode_frame)
            detect_index = detect.index(True)
            name = class_names[detect_index]
            self.display_information(name, status)
        except:
            msg = QMessageBox()
            msg.setWindowTitle("Sorry!")
            msg.setText("Your are not registered yet!")
            msg.exec()

    def display_information(self, name, status):
        with open('attendance.csv', 'a') as f:
            if name != 'unknown':
                if status == 'Start Work':
                    buttonReply = QMessageBox.question(self, 'Welcome ' + name, 'Are you going to start your work?',
                                                       QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                    if buttonReply == QMessageBox.Yes:
                        date_time_string = datetime.datetime.now().strftime("%y/%m/%d %H:%M:%S")
                        f.writelines(f'\n{name},{date_time_string}, Start Work')
                        df = pd.read_csv('database_person.csv')
                        person = df.loc[df['Name'] == name].to_numpy()
                        job = person[0][1]

                        self.ui.labelName.setText(name)
                        self.ui.labelJob.setText(job)

                elif status == 'End Work':
                    buttonReply = QMessageBox.question(self, 'Cheers ' + name, ', Are you going to end your work?',
                                                       QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                    if buttonReply == QMessageBox.Yes:
                        date_time_string = datetime.datetime.now().strftime("%y/%m/%d %H:%M:%S")
                        f.writelines(f'\n{name},{date_time_string}, End Work')
                        df = pd.read_csv('database_person.csv')
                        person = df.loc[df['Name'] == name].to_numpy()
                        job = person[0][1]
                        self.ui.labelName.setText(name)
                        self.ui.labelJob.setText(job)


if __name__ == "__main__":
    App = QApplication(sys.argv)
    Root = MainWindow()
    Root.show()
    sys.exit(App.exec())
