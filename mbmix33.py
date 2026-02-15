import cv2
import face_recognition
import numpy as np
import os
import json
from datetime import datetime, timedelta
import threading
import speech_recognition as sr
from mbfinal3 import ImprovedChatbot
import warnings
import time
from cameramanager import CameraManager

warnings.filterwarnings("ignore", category=DeprecationWarning)

WAKE_WORDS = [
    'hey nova',
    'hi nova',
    'hello nova',
    'hoi nova',
    'nova',
    'innova',
    'inova',
    'hey innova',
    'hi innova',
    'hello innova',
    'hey inova',
    'hi inova',
    'hello inova',
    'no va',
    'noah',
    'hey noah',
    'hi noah',
    'hello noah',
    'noba',
    'hey noba',
    'hi noba',
    'hello noba',
    'nava',
    'hey nava',
    'hi nava',
    'hello nava',
    'novaa',
    'hey novaa',
    'hi novaa',
    'hello novaa'
]
GREETING_FILE = "last_greeting.json"
KNOWN_FACES_DIR = "known_faces"

class FaceChatSystem:
    def __init__(self, ui_callback=None):
        self.known_encodings, self.known_names = self.load_known_faces()
        self.camera_manager = None
        self.chatbot = ImprovedChatbot(log_callback=self.log)
        self.last_greeting = self.load_greeting_history()
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.wake_word_active = threading.Event()
        self.current_user = None
        self.listening = False
        self.chat_active = False
        self.last_frame_check = 0
        self.frame_check_interval = 5
        self.ui_callback = ui_callback

    def log(self, message):
        print(message)
        if self.ui_callback:
            self.ui_callback(message)

    def load_greeting_history(self):
        try:
            with open(GREETING_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}

    def save_greeting_history(self):
        with open(GREETING_FILE, 'w') as f:
            json.dump(self.last_greeting, f)

    def load_known_faces(self):
        known_encodings, known_names = [], []
        if not os.path.exists(KNOWN_FACES_DIR):
            self.log(f"❌ No folder named {KNOWN_FACES_DIR}")
            return known_encodings, known_names

        for file in os.listdir(KNOWN_FACES_DIR):
            if "_fixed.jpg" in file:
                path = os.path.join(KNOWN_FACES_DIR, file)
                image = face_recognition.load_image_file(path)
                encs = face_recognition.face_encodings(image)
                if encs:
                    known_encodings.append(encs[0])
                    known_names.append(file.replace("_fixed.jpg", ""))
        return known_encodings, known_names

    def should_greet(self, name):
        if name not in self.last_greeting:
            return True
        last = datetime.strptime(self.last_greeting[name], "%Y-%m-%d %H:%M:%S")
        return (datetime.now() - last) > timedelta(hours=24)

    def handle_known_user(self, name):
        self.chat_active = True
        self.wake_word_active.clear()
        if self.should_greet(name):
            msg = f"Welcome back {name}! How can I help you today?"
            self.log(f"🤖 {msg}")
            self.chatbot.speak(msg)
            self.last_greeting[name] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.save_greeting_history()
        else:
            self.log(f"🙋‍♂️ Recognized {name} (already greeted today)")

        self.chatbot.chat_interface(require_wake_word=False)
        self.chat_active = False
        self.wake_word_active.set()

    def wake_word_listener(self):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
        while True:
            if self.wake_word_active.is_set() and not self.chat_active:
                try:
                    self.log("🎤 Listening for wake word...")
                    with self.microphone as source:
                        audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=3)
                    text = self.recognizer.recognize_google(audio).lower()
                    if any(w in text for w in WAKE_WORDS):
                        self.log("👂 Wake word detected!")
                        self.chat_active = True
                        self.wake_word_active.clear()
                        self.chatbot.speak("Hello! How can I assist you today?")
                        self.chatbot.chat_interface(require_wake_word=False)
                        self.chat_active = False
                        self.wake_word_active.set()
                except:
                    pass
            time.sleep(0.5)

    def face_recognition_loop(self):
        self.log("🧠 Starting FaceChat System...")
        self.wake_word_active.set()
        threading.Thread(target=self.wake_word_listener, daemon=True).start()

        while True:
            frame = self.camera_manager.get_frame()
            if frame is None:
                if time.time() - self.last_frame_check > self.frame_check_interval:
                    #self.log("⚠️ No frame captured.")
                    self.last_frame_check = time.time()
                continue

            small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb = np.ascontiguousarray(small[:, :, ::-1])

            face_locations = face_recognition.face_locations(rgb)
            face_encodings = face_recognition.face_encodings(rgb, face_locations)

            for enc in face_encodings:
                matches = face_recognition.compare_faces(self.known_encodings, enc, tolerance=0.45)
                if True in matches:
                    idx = np.argmin(face_recognition.face_distance(self.known_encodings, enc))
                    name = self.known_names[idx]
                    if name != self.current_user and not self.chat_active:
                        self.log(f"🙋‍♂️ Recognized {name}")
                        self.current_user = name
                        self.handle_known_user(name)
                        break
            else:
                self.current_user = None
                if not self.chat_active:
                    self.wake_word_active.set()

if __name__ == "__main__":
    cam = CameraManager()
    system = FaceChatSystem()
    system.camera_manager = cam
    system.face_recognition_loop()
