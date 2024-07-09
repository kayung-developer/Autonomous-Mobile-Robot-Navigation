import customtkinter as ctk
from tkinter import filedialog
import threading
import time
import requests

class RobotNavigationApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Autonomous Mobile Robot Navigation")
        self.geometry("800x600")

        self.start_button = ctk.CTkButton(self, text="Start Navigation", command=self.start_navigation)
        self.start_button.pack(pady=20)

        self.stop_button = ctk.CTkButton(self, text="Stop Navigation", command=self.stop_navigation)
        self.stop_button.pack(pady=20)

        self.upload_button = ctk.CTkButton(self, text="Upload Map", command=self.upload_map)
        self.upload_button.pack(pady=20)

        self.status_label = ctk.CTkLabel(self, text="Status: Idle")
        self.status_label.pack(pady=20)

        self.navigation_thread = None
        self.running = False

    def start_navigation(self):
        if not self.running:
            self.running = True
            self.navigation_thread = threading.Thread(target=self.run_navigation)
            self.navigation_thread.start()
            self.status_label.configure(text="Status: Running")

    def stop_navigation(self):
        if self.running:
            self.running = False
            if self.navigation_thread:
                self.navigation_thread.join()
            self.status_label.configure(text="Status: Stopped")

    def upload_map(self):
        map_file = filedialog.askopenfilename()
        if map_file:
            response = requests.post("http://localhost:8000/upload_map", files={"file": open(map_file, "rb")})
            if response.status_code == 200:
                self.status_label.configure(text="Status: Map Uploaded Successfully")
            else:
                self.status_label.configure(text="Status: Failed to Upload Map")

    def run_navigation(self):
        while self.running:
            response = requests.get("http://localhost:8000/navigate")
            if response.status_code == 200:
                navigation_data = response.json()
                # Process navigation data here (e.g., update UI, send commands to robot)
            time.sleep(1)

if __name__ == "__main__":
    app = RobotNavigationApp()
    app.mainloop()
