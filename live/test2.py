import platform
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import math

class ThumbnailPanel:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Violations")

        # Scrollable Canvas
        self.canvas = tk.Canvas(self.root, width=500, height=600)
        self.scroll_y = tk.Scrollbar(self.root, orient="vertical", command=self.canvas.yview)

        self.frame = ttk.Frame(self.canvas)
        self.frame_id = self.canvas.create_window((0, 0), window=self.frame, anchor="n")  # note anchor="n"

        # configure scrollregion when frame changes
        self.frame.bind(
            "<Configure>",
            self._on_frame_configure
        )
        self.canvas.configure(yscrollcommand=self.scroll_y.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scroll_y.pack(side="right", fill="y")

        # Store images to avoid garbage collection
        self.thumbnails = []

        # Track which (obj_id, violation_type) pairs were added
        self.seen = set()

        # Platform
        self._platform = platform.system()

        # Bind mouse enter/leave
        self.canvas.bind("<Enter>", self._bind_mousewheel_on_enter)
        self.canvas.bind("<Leave>", self._unbind_mousewheel_on_leave)

    def _on_frame_configure(self, event):
        # Update scrollregion
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        # Center the frame horizontally
        canvas_width = self.canvas.winfo_width()
        frame_width = self.frame.winfo_reqwidth()
        x = max((canvas_width - frame_width) // 2, 0)
        self.canvas.coords(self.frame_id, x, 0)

    def _bind_mousewheel_on_enter(self, event):
        try:
            self.canvas.focus_set()
        except Exception:
            pass
        if self._platform == "Windows" or self._platform == "Darwin":
            self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        else:
            self.canvas.bind("<Button-4>", self._on_mousewheel)
            self.canvas.bind("<Button-5>", self._on_mousewheel)

    def _unbind_mousewheel_on_leave(self, event):
        if self._platform == "Windows" or self._platform == "Darwin":
            self.canvas.unbind("<MouseWheel>")
        else:
            self.canvas.unbind("<Button-4>")
            self.canvas.unbind("<Button-5>")

    def _on_mousewheel(self, event):
        if self._platform == "Windows":
            steps = int(-1 * (event.delta / 120))
            if steps == 0:
                steps = -1 if event.delta < 0 else 1
            self.canvas.yview_scroll(steps, "units")
        elif self._platform == "Darwin":
            steps = int(-1 * math.copysign(1, event.delta))
            self.canvas.yview_scroll(steps, "units")
        else:
            if event.num == 4:
                self.canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                self.canvas.yview_scroll(1, "units")

    def add_thumbnail_once(self, obj_id, vtype, img_path):
        key = (obj_id, vtype)
        if key in self.seen:
            return
        self.seen.add(key)
        self.add_thumbnail(img_path, vtype)

    def add_thumbnail(self, img_path, vtype):
        try:
            img = Image.open(img_path)
        except Exception as e:
            print(f"Failed to open {img_path}: {e}")
            return

        img = img.resize((300, 170), Image.LANCZOS)
        tk_img = ImageTk.PhotoImage(img)

        # Container frame for each thumbnail block
        item_frame = ttk.Frame(self.frame)
        item_frame.pack(pady=6, anchor="n")  # pack vertically

        lbl_img = ttk.Label(item_frame, image=tk_img)
        lbl_img.pack(side="top")

        lbl_text = ttk.Label(item_frame, text=f"Violation: {vtype}", font=("Segoe UI", 10))
        lbl_text.pack(side="top", pady=(4,0))

        # prevent GC
        self.thumbnails.append(tk_img)

        # update canvas scrollregion and recenter
        self._on_frame_configure(None)

        # Update gui
        self.root.update_idletasks()

    def run(self):
        self.root.mainloop()


# --- Example usage ---
if __name__ == "__main__":
    panel = ThumbnailPanel()
    panel.add_thumbnail_once(1, "Speeding", "k1.jpg")
    panel.add_thumbnail_once(2, "Helmet", "k1.jpg")
    panel.add_thumbnail_once(3, "Red light", "k1.jpg")
    panel.run()
