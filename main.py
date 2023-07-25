import sys
import cv2
import time, random
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QLineEdit,
    QCheckBox,
)


class ImageProcessor:
    @staticmethod
    def crop(
        input_path: str, output_path: str, x: int, y: int, width: int, height: int
    ):
        cv2.imwrite(output_path, cv2.imread(input_path)[y : y + height, x : x + width])

    @staticmethod
    def quality(input_path: str, output_path: str, quality: int):
        cv2.imwrite(
            output_path, cv2.imread(input_path), [cv2.IMWRITE_JPEG_QUALITY, quality]
        )

    @staticmethod
    def resize(input_path: str, output_path: str, width: int, height: int):
        cv2.imwrite(output_path, cv2.resize(cv2.imread(input_path), (width, height)))

    @staticmethod
    def change_to_gray(input_path: str, output_path: str):
        cv2.imwrite(output_path, cv2.imread(input_path, cv2.IMREAD_GRAYSCALE))

    @staticmethod
    def find_face(input_path: str, output_path: str, is_many: bool):
        image = cv2.imread(input_path, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width, channels = image.shape
        cascade = cv2.CascadeClassifier("src/cascadeHaare.xml")
        rect = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

        def one():
            x, y, w, h = rect[0]
            multiply_w: int = int(200 * (width / 1920))
            multiply_h: int = int(240 * (height / 1920))

            y1 = y - multiply_h
            y2 = y + h + multiply_h

            x1 = x - multiply_w
            x2 = x + w + multiply_w

            cv2.imwrite(output_path, image[y1:y2, x1:x2])

        def many():
            index = 0
            for x, y, w, h in rect:
                if w < 50 or h < 50:
                    continue
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                index += 1

                multiply_w: int = int(200 * (width / 1920))
                multiply_h: int = int(240 * (height / 1920))

                y1 = y - multiply_h
                y2 = y + h + multiply_h

                x1 = x - multiply_w
                x2 = x + w + multiply_w

                cv2.imwrite(f"{output_path}{str(index)}_face.jpg", image[y1:y2, x1:x2])

        if is_many:
            many()
        else:
            one()

    @staticmethod
    def video_face(input_path: str, output_path: str):
        speed: float = 2.5
        cap: cv2.VideoCapture = cv2.VideoCapture(input_path)
        cascade = cv2.CascadeClassifier("src/cascadeHaare.xml")
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces_rect = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
            for x, y, w, h in faces_rect:
                if w < 20 or h < 20:
                    continue
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            time.sleep(1)
            cv2.imwrite(
                f"{output_path}/{str(random.randint(0, 1000000000))}_frame.jpg", frame
            )
            if cv2.waitKey(int(24 / speed)) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()


class Application:
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.window = QWidget()

        self.window.setWindowTitle("Image Decoder")
        self.window.setGeometry(100, 100, 600, 200)

        self.layout = QVBoxLayout()
        self.hbox1 = QHBoxLayout()
        self.hbox2 = QHBoxLayout()

        self.input_label = QLabel("Input")
        self.layout.addWidget(self.input_label)

        self.input_path = QLineEdit()
        self.layout.addWidget(self.input_path)

        self.output_label = QLabel("Output")
        self.layout.addWidget(self.output_label)

        self.output_path = QLineEdit()
        self.layout.addWidget(self.output_path)

        self.x_label = QLabel("X Position")
        self.hbox1.addWidget(self.x_label)

        self.y_label = QLabel("Y Position")
        self.hbox1.addWidget(self.y_label)

        self.width_label = QLabel("Width")
        self.hbox1.addWidget(self.width_label)

        self.height_label = QLabel("Height")
        self.hbox1.addWidget(self.height_label)

        self.quality_label = QLabel("Quality ( 0 to 100 )")
        self.hbox1.addWidget(self.quality_label)

        self.x_line = QLineEdit()
        self.hbox2.addWidget(self.x_line)

        self.y_line = QLineEdit()
        self.hbox2.addWidget(self.y_line)

        self.width_line = QLineEdit()
        self.hbox2.addWidget(self.width_line)

        self.height_line = QLineEdit()
        self.hbox2.addWidget(self.height_line)

        self.quality_line = QLineEdit()
        self.hbox2.addWidget(self.quality_line)

        self.layout.addLayout(self.hbox1)
        self.layout.addLayout(self.hbox2)

        self.crop_btn = QPushButton("Crop")
        self.crop_btn.clicked.connect(self.crop_image)
        self.layout.addWidget(self.crop_btn)

        self.quality_btn = QPushButton("Quality")
        self.quality_btn.clicked.connect(self.change_quality)
        self.layout.addWidget(self.quality_btn)

        self.rezise_btn = QPushButton("Rezise")
        self.rezise_btn.clicked.connect(self.resize_image)
        self.layout.addWidget(self.rezise_btn)

        self.change_to_gray_btn = QPushButton("Change to Gray")
        self.change_to_gray_btn.clicked.connect(self.change_to_gray)
        self.layout.addWidget(self.change_to_gray_btn)

        self.find_face_btn = QPushButton("Find Face")
        self.find_face_btn.clicked.connect(self.find_face)
        self.layout.addWidget(self.find_face_btn)

        self.video_face_btn = QPushButton("Video Face")
        self.video_face_btn.clicked.connect(self.video_face)
        self.layout.addWidget(self.video_face_btn)

        self.check_box_label = QLabel("Many Faces")
        self.layout.addWidget(self.check_box_label)

        self.check_box = QCheckBox()
        self.layout.addWidget(self.check_box)

        self.window.setLayout(self.layout)
        self.window.show()
        self.app.exec()

    def crop_image(self):
        input_path = self.input_path.text()
        output_path = self.output_path.text()
        x = int(self.x_line.text())
        y = int(self.y_line.text())
        width = int(self.width_line.text())
        height = int(self.height_line.text())
        ImageProcessor.crop(input_path, output_path, x, y, width, height)

    def change_quality(self):
        input_path = self.input_path.text()
        output_path = self.output_path.text()
        quality = int(self.quality_line.text())
        ImageProcessor.quality(input_path, output_path, quality)

    def resize_image(self):
        input_path = self.input_path.text()
        output_path = self.output_path.text()
        width = int(self.width_line.text())
        height = int(self.height_line.text())
        ImageProcessor.resize(input_path, output_path, width, height)

    def change_to_gray(self):
        input_path = self.input_path.text()
        output_path = self.output_path.text()
        ImageProcessor.change_to_gray(input_path, output_path)

    def find_face(self):
        input_path = self.input_path.text()
        output_path = self.output_path.text()
        is_many = self.check_box.isChecked()
        ImageProcessor.find_face(input_path, output_path, is_many)

    def video_face(self):
        input_path = self.input_path.text()
        output_path = self.output_path.text()
        ImageProcessor.video_face(input_path, output_path)


if __name__ == "__main__":
    app = Application()
