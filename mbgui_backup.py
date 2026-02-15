import customtkinter as ctk
import pygame
import cv2
from abhi import FaceTracker
from mbmix33 import FaceChatSystem
from mbcameramanager import CameraManager
import threading
import queue
from PIL import Image, ImageDraw, ImageFont
import os
import platform
import time

ctk.set_appearance_mode("dark")

class MakerBotGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.geometry("1400x1000")
        self.title("🤖 Smart Face Tracker")
        self.configure(fg_color="#000000")
        self.chat_queue = queue.Queue()
        self.emoji_images = self.create_emoji_images()

        # HEADER with emoji and logos
        header_frame = ctk.CTkFrame(self, fg_color="transparent", border_color="#FF00FF",
                                    border_width=2, corner_radius=5)
        header_frame.pack(pady=(10, 5), padx=10, fill="x")

        logo_path_left = r"/home/pi/Desktop/makerbot/makerbhavan.jpeg"
        if os.path.exists(logo_path_left):
            img_left = Image.open(logo_path_left)
            logo_left = ctk.CTkImage(light_image=img_left,
                                     dark_image=img_left,
                                     size=(90, 70))
            ctk.CTkLabel(header_frame, image=logo_left, text="", fg_color="transparent")\
                .pack(side="left", padx=0, fill="y")

        logo_path_right = r"/home/pi/Desktop/makerbot/makerbhavan.jpeg"
        if os.path.exists(logo_path_right):
            img_right = Image.open(logo_path_right)
            logo_right = ctk.CTkImage(light_image=img_right,
                                      dark_image=img_right,
                                      size=(90, 70))
            ctk.CTkLabel(header_frame, image=logo_right, text="", fg_color="transparent")\
                .pack(side="right", padx=0, fill="y")

        if self.emoji_images.get("robot"):
            robot_label = ctk.CTkLabel(header_frame, text="", 
                                     image=self.emoji_images["robot"],
                                     fg_color="transparent")
            robot_label.pack(side="left", expand=True, fill="x", anchor="center", padx=(0, 10))

        self.title_label = ctk.CTkLabel(header_frame, text="MAKER BOT CONSOLE",
                                        font=("Consolas", 34, "bold"),  # Increased from 28
                                        text_color="#FF4EC2",
                                        fg_color="transparent")
        self.title_label.pack(side="left", expand=True, fill="x", anchor="center")

        if self.emoji_images.get("rocket"):
            rocket_label = ctk.CTkLabel(header_frame, text="", 
                                      image=self.emoji_images["rocket"],
                                      fg_color="transparent")
            rocket_label.pack(side="left", expand=True, fill="x", anchor="center", padx=(10, 0))

        # STATUS BAR
        self.status_label = ctk.CTkLabel(self, text="🔘 STATUS: Idle",
                                         font=("Consolas", 20, "bold"),  # Increased from 16
                                         text_color="#00BFFF",
                                         fg_color="transparent")
        self.status_label.pack(pady=5)

        # CHAT BOX
        self.chat_log = ctk.CTkTextbox(self, width=950, height=560,
                                       font=("Segoe UI", 20, "normal"),  # Increased from 15
                                       fg_color="#000000",
                                       text_color="#FFFFFF",
                                       corner_radius=10,
                                       border_color="#FF00FF",
                                       border_width=2)
        self.chat_log.pack(pady=(10, 5))
        self.chat_log.insert("end", "Console ready...\n")
        self.chat_log.configure(state="disabled")

        # SETUP
        self.camera_manager = CameraManager()
        self.chat_system = FaceChatSystem(ui_callback=self.log)
        self.chat_system.camera_manager = self.camera_manager
        self.chat_system.chat_queue = self.chat_queue

        self.tracker = FaceTracker(self.camera_manager)
        self.tracker.start()
        time.sleep(1)

        self.chat_thread = threading.Thread(target=self.chat_loop, daemon=True)
        self.chat_thread.start()

        threading.Thread(target=self.chat_system.face_recognition_loop, daemon=True).start()
        self.update_ui()

    def create_emoji_images(self):
        emoji_images = {}
        emojis = {
            "robot": "🤖",
            "rocket": "🚀",
            "green_circle": "🟢",
            "microphone": "🎤",
            "studio_microphone": "🎙️",
            "ear": "👂",
            "warning": "⚠️"
        }
        emoji_font_path = self.get_emoji_font_path()
        for name, emoji_char in emojis.items():
            try:
                emoji_images[name] = self.create_single_emoji_image(
                    emoji_char,
                    size=32 if name in ["robot", "rocket"] else 24,
                    font_path=emoji_font_path
                )
            except Exception as e:
                print(f"Failed to create emoji image for {name}: {e}")
                emoji_images[name] = None
        return emoji_images

    def get_emoji_font_path(self):
        system = platform.system()
        if system == "Windows":
            possible_paths = [
                "C:/Windows/Fonts/seguiemj.ttf",
                "C:/Windows/Fonts/NotoColorEmoji.ttf",
                "C:/Windows/Fonts/TwitterColorEmoji-SVGinOT.ttf"
            ]
        elif system == "Darwin":
            possible_paths = [
                "/System/Library/Fonts/Apple Color Emoji.ttc",
                "/Library/Fonts/Apple Color Emoji.ttc"
            ]
        else:
            possible_paths = [
                "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
                "/usr/share/fonts/TTF/NotoColorEmoji.ttf",
                "/usr/share/fonts/noto-color-emoji/NotoColorEmoji.ttf"
            ]
        for path in possible_paths:
            if os.path.exists(path):
                return path
        return None

    def create_single_emoji_image(self, emoji_text, size=32, font_path=None):
        try:
            if font_path and os.path.exists(font_path):
                font = ImageFont.truetype(font_path, size=int(size * 0.8))
            else:
                try:
                    font = ImageFont.truetype("arial.ttf", size=int(size * 0.8))
                except:
                    font = ImageFont.load_default()

            img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            bbox = draw.textbbox((0, 0), emoji_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            x = (size - text_width) // 2
            y = (size - text_height) // 2
            try:
                draw.text((x, y), emoji_text, font=font, embedded_color=True)
            except TypeError:
                draw.text((x, y), emoji_text, font=font, fill=(255, 255, 255, 255))
            return ctk.CTkImage(img, size=(size, size))
        except Exception as e:
            print(f"Error creating emoji image: {e}")
            return None

    def log(self, msg):
        self.chat_queue.put(msg)

    def chat_loop(self):
        while True:
            if not self.chat_queue.empty():
                msg = self.chat_queue.get()
                self.chat_log.configure(state="normal")
                self.chat_log.insert("end", msg + "\n")
                self.chat_log.see("end")
                self.chat_log.configure(state="disabled")

                if "Face Detected" in msg:
                    status_text = "STATUS: Face Detected | Listening..."
                    if self.emoji_images.get("green_circle") and self.emoji_images.get("microphone"):
                        self.update_status_with_emojis(status_text, ["green_circle", "microphone"], "#FFFFFF")
                    else:
                        self.status_label.configure(text="🟢 STATUS: Face Detected | 🎤 Listening...",
                                                    text_color="#0CE8F8")

                elif "Listening" in msg:
                    status_text = "STATUS: Listening..."
                    if self.emoji_images.get("studio_microphone"):
                        self.update_status_with_emojis(status_text, ["studio_microphone"], "#00BFFF")
                    else:
                        self.status_label.configure(text="🎙️ STATUS: Listening...",
                                                    text_color="#00BFFF")

                elif "Wake word" in msg:
                    status_text = "STATUS: Wake Word Detected"
                    if self.emoji_images.get("ear"):
                        self.update_status_with_emojis(status_text, ["ear"], "#FFD700")
                    else:
                        self.status_label.configure(text="👂 STATUS: Wake Word Detected",
                                                    text_color="#FFD700")

                elif "No frame" in msg:
                    status_text = "STATUS: Camera Error"
                    if self.emoji_images.get("warning"):
                        self.update_status_with_emojis(status_text, ["warning"], "#FF6347")
                    else:
                        self.status_label.configure(text="⚠️ STATUS: Camera Error",
                                                    text_color="#FF6347")

    def update_status_with_emojis(self, text, emoji_names, color):
        try:
            self.status_label.configure(text=f"{text}", text_color=color)
        except Exception as e:
            print(f"Error updating status with emojis: {e}")
            fallback_emojis = {
                "green_circle": "🟢",
                "microphone": "🎤",
                "studio_microphone": "🎙️",
                "ear": "👂",
                "warning": "⚠️"
            }
            emoji_text = " ".join([fallback_emojis.get(name, "") for name in emoji_names])
            self.status_label.configure(text=f"{emoji_text} {text}", text_color=color)

    def update_ui(self):
        self.after(500, self.update_ui)

if __name__ == "__main__":
    app = MakerBotGUI()
    app.mainloop()
