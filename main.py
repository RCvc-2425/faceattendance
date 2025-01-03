from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.image import Image
from kivy.core.window import Window
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.popup import Popup
import cv2
import os
import csv
from datetime import datetime
import numpy as np
import shutil

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
image_dir = 'datasets'
os.makedirs(image_dir, exist_ok=True)
csv_file = "Attendance_Log.csv"
if not os.path.exists(csv_file):
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "Date", "Time"])

name_mapping_file = "name_mapping.txt"
name_dict = {}
if os.path.exists(name_mapping_file):
    with open(name_mapping_file, "r") as file:
        for line in file:
            user_id, name = line.strip().split(",", 1) 
            name_dict[int(user_id)] = name

class BiometricApp(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.camera = None
        self.image_widget = Image()
        self.add_widget(self.image_widget)

        self.input_name = TextInput(hint_text='Enter Full Name', multiline=False, size_hint=(1, 0.1))
        self.add_widget(self.input_name)

        self.register_button = Button(text="Register Face", size_hint=(1, 0.1))
        self.register_button.bind(on_press=self.register_face)
        self.add_widget(self.register_button)

        self.train_button = Button(text="Train Model", size_hint=(1, 0.1))
        self.train_button.bind(on_press=self.train_model)
        self.add_widget(self.train_button)

        self.recognize_button = Button(text="Recognize Face", size_hint=(1, 0.1))
        self.recognize_button.bind(on_press=self.recognize_face)
        self.add_widget(self.recognize_button)

        self.attendance_button = Button(text="View Attendance", size_hint=(1, 0.1))
        self.attendance_button.bind(on_press=self.view_attendance)
        self.add_widget(self.attendance_button)

        self.save_button = Button(text="Save Attendance Log", size_hint=(1, 0.1))
        self.save_button.bind(on_press=self.save_attendance_log)
        self.add_widget(self.save_button)

        self.delete_button = Button(text="Delete All Saved Data", size_hint=(1, 0.1))
        self.delete_button.bind(on_press=self.delete_all_data)
        self.add_widget(self.delete_button)

        self.attendance_taken = set()
        self.is_recognition_active = False

    def register_face(self, instance):
        name = self.input_name.text.strip()
        if not name:
            self.show_popup("Error", "Please enter a valid name.")
            return

        self.camera = cv2.VideoCapture(0)
        count = 0

        def capture_faces(dt):
            nonlocal count
            ret, frame = self.camera.read()
            if not ret:
                return

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                count += 1
                face = gray[y:y + h, x:x + w]
                cv2.imwrite(f"{image_dir}/User.{name}.{count}.jpg", face)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            buf = cv2.flip(frame, 0).tobytes()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.image_widget.texture = texture

            if count >= 50:
                Clock.unschedule(capture_faces)
                self.camera.release()
                self.show_popup("Success", f"Faces for {name} registered successfully.")

        Clock.schedule_interval(capture_faces, 1.0 / 30.0)

    def train_model(self, instance):
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        faces = []
        ids = []

        for image_name in os.listdir(image_dir):
            if not image_name.endswith(".jpg"):
                continue
            path = os.path.join(image_dir, image_name)
            face = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            name = image_name.split(".")[1]
            faces.append(face)
            ids.append(len(name_dict) + 1)
            name_dict[len(name_dict) + 1] = name

        recognizer.train(faces, np.array(ids))
        recognizer.save("Trainer.yml")
        self.show_popup("Success", "Model trained successfully.")
        with open(name_mapping_file, "w") as file:
            for user_id, name in name_dict.items():
                file.write(f"{user_id},{name}\n")

    def recognize_face(self, instance):
        if self.is_recognition_active:
            self.show_popup("Error", "Face recognition is already active.")
            return

        self.is_recognition_active = True
        recognizer.read("Trainer.yml")
        self.camera = cv2.VideoCapture(0)
        recognition_start_time = datetime.now()

        def detect_faces(dt):
            nonlocal recognition_start_time
            ret, frame = self.camera.read()
            if not ret:
                return

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            for (x, y, w, h) in faces:
                face_id, conf = recognizer.predict(gray[y:y + h, x:x + w])
                name = name_dict.get(face_id, "Unknown")
                if (datetime.now() - recognition_start_time).seconds >= 10:
                    if name != "Unknown":
                        now = datetime.now()
                        date = now.strftime("%Y-%m-%d")
                        time = now.strftime("%H:%M:%S")
                        with open(csv_file, mode="r") as file:
                            reader = csv.reader(file)
                            for row in reader:
                                if row[0] == name and row[1] == date:
                                    self.show_popup("Duplicate Attendance", f"Attendance already taken for {name}.")
                                    Clock.unschedule(detect_faces)
                                    self.camera.release()
                                    self.is_recognition_active = False
                                    return
                        with open(csv_file, mode="a", newline="") as file:
                            writer = csv.writer(file)
                            writer.writerow([name, date, time])
                        self.show_popup("Attendance Taken", f"Attendance recorded for {name}.")
                        self.attendance_taken.add(name)
                        Clock.unschedule(detect_faces)
                        self.camera.release()
                        self.is_recognition_active = False
                        return
                    else:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                if name != "Unknown":
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            buf = cv2.flip(frame, 0).tobytes()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.image_widget.texture = texture

        Clock.schedule_interval(detect_faces, 1.0 / 30.0)

    def view_attendance(self, instance):
        with open(csv_file, mode="r") as file:
            reader = csv.reader(file)
            content = "\n".join([", ".join(row) for row in reader])

        self.show_popup("Attendance Log", content)

    def save_attendance_log(self, instance):
        now = datetime.now()
        new_file = f"Attendance_Log_{now.strftime('%Y-%m-%d_%H-%M-%S')}.csv"
        shutil.copy(csv_file, new_file)
        self.show_popup("Saved", f"Attendance log saved as {new_file}")

    def delete_all_data(self, instance):
        if os.path.exists(image_dir):
            shutil.rmtree(image_dir)
        if os.path.exists("Trainer.yml"):
            os.remove("Trainer.yml")
        if os.path.exists(name_mapping_file):
            os.remove(name_mapping_file)
        if os.path.exists(csv_file):
            os.remove(csv_file)

        self.show_popup("Deleted", "All saved data has been deleted.")

    def show_popup(self, title, message):
        content = BoxLayout(orientation='vertical')
        content.add_widget(Label(text=message))
        close_button = Button(text="Close", size_hint=(1, 0.2))
        close_button.bind(on_press=lambda *args: popup.dismiss())
        content.add_widget(close_button)

        popup = Popup(title=title, content=content, size_hint=(0.8, 0.6))
        popup.open()

    def check_and_create_new_excel(self):
        today = datetime.now()
        if today.weekday() == 0:
            new_csv_file = f"Attendance_Log_{today.strftime('%Y-%m-%d')}.csv"
            if os.path.exists(csv_file):
                shutil.copy(csv_file, new_csv_file)
            with open(csv_file, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Name", "Date", "Time"])

class BiometricAppMain(App):
    def build(self):
        Window.set_icon('face_recognition_icon.png')
        app = BiometricApp()
        app.check_and_create_new_excel()
        return app

if __name__ == '__main__':
    BiometricAppMain().run()
