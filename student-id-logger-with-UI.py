import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np
import tensorflow.keras as tf
import os
import pytesseract
import re
import sqlite3
import datetime

# Define the CustomTkinterApp class
class CustomTkinterApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Student ID Logger")
        self.master.configure(bg="#1e1e1e")

        # Initialize webcam video object
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 640)
        self.cap.set(4, 480)

        # Initialize TensorFlow model
        self.model = tf.models.load_model("Model/keras_model.h5", compile=False)

        # Initialize pytesseract
        pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"

        # Initialize database connection
        self.conn = sqlite3.connect("logging.db")

        # Create left frame for video feed
        self.left_frame = tk.Frame(self.master, bg="#1e1e1e")
        self.left_frame.pack(side=tk.LEFT, padx=10, pady=10)

        # Create right frame for logging information
        self.right_frame = tk.Frame(self.master, bg="#1e1e1e")
        self.right_frame.pack(side=tk.RIGHT, padx=10, pady=10)

        # Create logging section
        self.log_label = tk.Label(self.right_frame, text="Logging Information", font=("Helvetica", 16), bg="#1e1e1e", fg="white")
        self.log_label.pack()

        self.log_text = tk.Text(self.right_frame, width=40, height=20, bg="#2e2e2e", fg="white")
        self.log_text.pack()

        # Show video feed
        self.show_video_feed()

    def show_video_feed(self):
        ret, frame = self.cap.read()
        if ret:
            # Convert frame to RGB format
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert frame to ImageTk format
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)

            # Display video feed on left frame
            self.video_label = tk.Label(self.left_frame, image=imgtk)
            self.video_label.image = imgtk
            self.video_label.pack()

            # Process frame for student ID detection
            self.process_frame(frame)

        # Repeat the process after 10 ms
        self.master.after(10, self.show_video_feed)

    def process_frame(self, frame):
        # Preprocess frame for student ID detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

        # Detect student ID using pytesseract
        text = pytesseract.image_to_string(thresh1)
        id_pattern = r"\d{4}-\d{4}"
        matches = re.findall(id_pattern, text)

        if matches:
            student_id = matches[0]
            self.log_student_entry(student_id)

    def log_student_entry(self, student_id):
        today = datetime.date.today()
        now = datetime.datetime.now()
        now_str = now.strftime("%H:%M:%S")
        cursor = self.conn.cursor()

        # Check last logged at time for student
        cursor.execute("SELECT last_logged_at FROM student_info WHERE student_id = ?", (student_id,))
        last_logged_at = cursor.fetchone()
        cooldown_period = 5  # Adjust cooldown period in seconds

        if last_logged_at:
            last_logged_at_seconds = last_logged_at[0]
            if last_logged_at_seconds:
                # Convert last_logged_at to actual seconds since epoch
                t1 = datetime.datetime.strptime(now_str, "%H:%M:%S")
                t2 = datetime.datetime.strptime(last_logged_at_seconds, "%H:%M:%S")
                time_diff = t1 - t2

                # Adjust cooldown period to seconds as well
                if time_diff.total_seconds() < cooldown_period:
                    print(f"Cooldown active for student {student_id}. Skipping entry.")
                    return  # Exit function if cooldown is active

        # Proceed with logging if cooldown is not active
        cursor.execute("INSERT INTO logs (student_id, date, time) VALUES (?, ?, ?)", (student_id, today, now_str))
        cursor.execute("UPDATE student_info SET last_logged_at = ? WHERE student_id = ?", (now_str, student_id))
        self.conn.commit()
        log_entry = f"{today} - {now_str}: Student {student_id} logged in successfully!\n"
        self.log_text.insert(tk.END, log_entry)
        self.log_text.see(tk.END)


def main():
    root = tk.Tk()
    app = CustomTkinterApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
