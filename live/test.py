

import platform
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np

class ThumbnailPanel:

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Vehicle Violations + Video")

        # -----------------------------
        # Frame for thumbnails + scrollbar
        # -----------------------------
        self.thumb_frame = ttk.Frame(self.root)
        self.thumb_frame.pack(side="right", fill="y", expand=False)

        # STATIC label at top (not scrollable)
        self.title_label = ttk.Label(self.thumb_frame, text="Detected Violations",
                                     font=("Arial", 14, "bold"))
        self.title_label.pack(pady=10)

        # Scrollable Canvas for thumbnails
        self.canvas = tk.Canvas(self.thumb_frame, width=320, height=700, bd=0, highlightthickness=0)
        self.scroll_y = tk.Scrollbar(self.thumb_frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scroll_y.set)

        # Frame inside canvas to hold thumbnails
        self.frame = ttk.Frame(self.canvas)
        self.frame_id = self.canvas.create_window((0, 0), window=self.frame, anchor="n")

        # Bind configure event to dynamically center the frame inside canvas
        self.frame.bind("<Configure>", self._center_frame)

        # Pack canvas and scrollbar inside thumb_frame
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scroll_y.pack(side="right", fill="y")

        # Keep references to avoid garbage collection
        self.thumbnails = []

        # Prevent duplicates
        self.seen = set()

        # Video display panel
        self.video_panel = ttk.Label(self.root)
        self.video_panel.pack(side="left", fill="both", expand=True)

    # -----------------------------
    # Center the frame inside the canvas
    # -----------------------------
    def _center_frame(self, event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        canvas_width = self.canvas.winfo_width()
        frame_width = self.frame.winfo_reqwidth()
        x = max((canvas_width - frame_width) // 2, 0)
        self.canvas.coords(self.frame_id, x, 0)

    # -----------------------------
    # Update video frame
    # -----------------------------
    def update_video(self, frame):
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = img.resize((800, 600))  # Resize as needed
            tk_img = ImageTk.PhotoImage(img)
            self.video_panel.imgtk = tk_img
            self.video_panel.configure(image=tk_img)
        except Exception as e:
            print(f"[VIDEO] failed to update frame: {e}")

    # -----------------------------
    # Add thumbnail once
    # -----------------------------
    def add_thumbnail_once(self, obj_id, vtype, img_array):
        key = (obj_id, vtype)
        if key in self.seen:
            return
        self.seen.add(key)
        self.add_thumbnail(img_array, vtype, obj_id)

    # -----------------------------
    # Add thumbnail
    # -----------------------------
    def add_thumbnail(self, img_array, vtype, obj_id):
        try:
            img = Image.fromarray(img_array)
            fixed_width = 250
            w_percent = (fixed_width / float(img.width))
            h_size = int((float(img.height) * w_percent))
            img = img.resize((fixed_width, h_size), Image.LANCZOS)
            tk_img = ImageTk.PhotoImage(img)
        except Exception as e:
            print(f"[THUMBNAIL] failed to create thumbnail: {e}")
            return

        # Create a block for each thumbnail
        block = ttk.Frame(self.frame)
        block.pack(pady=8)  # No anchor needed; frame is centered

        # Label for violation type + obj_id
        lbl_text = ttk.Label(block, text=f"{vtype}_{obj_id}", font=("Arial", 11, "bold"))
        lbl_text.pack()

        # Image label
        lbl_img = ttk.Label(block, image=tk_img)
        lbl_img.pack()

        # Keep reference to avoid garbage collection
        self.thumbnails.append(tk_img)

        # Center the frame after adding thumbnail
        self._center_frame()

    # -----------------------------
    # Run GUI
    # -----------------------------
    def run(self):
        self.root.mainloop()


# -----------------------------
# Dummy test
# -----------------------------
if __name__ == "__main__":
    panel = ThumbnailPanel()

    # Generate dummy video frames (gray gradient)
    def generate_dummy_video(frame_count=50):
        for i in range(frame_count):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, f"Video Frame {i+1}", (50, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            panel.update_video(frame)
            panel.root.update()  # Update GUI

    # Generate dummy thumbnail images (colored rectangles)
    colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255)]
    for idx, color in enumerate(colors):
        img = np.full((120, 200, 3), color, dtype=np.uint8)
        panel.add_thumbnail_once(idx+1, f"Violation{idx+1}", img)

    # Run dummy video updates in the background
    panel.root.after(100, generate_dummy_video)  # starts after 100ms
    panel.run()
