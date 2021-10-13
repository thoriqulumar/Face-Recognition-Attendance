from PyQt5.uic import loadUi
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import cv2
import face_recognition
import pandas as pd
import os
import sys
import cgitb

cgitb.enable(format='text')

class RegisterWindow(QMainWindow):
    def __init__(self):
        super(RegisterWindow, self).__init__()
        loadUi('ui/registwindow.ui', self)
        self.class_names = []
        self.encode_images = []

        self.name = ""
        self.get_information()
        self.timer = QTimer()
        self.start_video()
        self.timer.timeout.connect(self.next_frame)
        self.timer.start(20)
        self.btnSubmit.clicked.connect(self.save_data)

    def start_video(self):
        self.cap = cv2.VideoCapture(0)

    def next_frame(self):
        ret, image = self.cap.read()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)
        image = self.face_rec(image)
        height, width, channel = image.shape
        qImg = QImage(image.data, width, height, QImage.Format_RGB888)

        self.imgLabel.setPixmap(QPixmap.fromImage(qImg))
        self.imgLabel.setScaledContents(True)

    @pyqtSlot()
    def save_img(self, filename):
        ret, image = self.cap.read()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)
        if ret:
            path = 'Image Storage'
            img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            try:
                cv2.imwrite(f'{path}/{filename}.jpg', img)
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
                self.name = name
            except:
                name = 'Unknowm'
                self.name = name

            y1, x2, y2, x1 = faceLoc
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

        return image

    def save_data(self):
        full_name = self.editName.text()
        job = self.editJob.text()
        self.save_img(full_name)
        with open('database_person.csv', 'a') as f:
            buttonReply = QMessageBox.question(self, 'Confirmation', 'Your data is going to be save, are you sure?',
                                               QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if buttonReply == QMessageBox.Yes:
                if self.name == "Unknowm":
                    f.writelines(f'\n{full_name},{job},{full_name}.jpg')
                    QMessageBox.question(self, 'Notification', "Data Saved!", QMessageBox.Ok,
                                         QMessageBox.Ok)
                    print("Data is saved")

                else:
                    QMessageBox.question(self, 'Notification', "You have been registered!", QMessageBox.Ok,
                                         QMessageBox.Ok)


if __name__ == "__main__":
    App = QApplication(sys.argv)
    Root = RegisterWindow()
    Root.show()
    sys.exit(App.exec())
