import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from tkinterdnd2 import DND_FILES
import threading
import os
import pandas as pd

from data_io.generic_reader import read_file
from processing.resampling import resample_signal
from processing.laeq import compute_laeq
from processing.splmax import compute_spl_max
from config import FS_TARGET


class AcousticApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Acoustic Analyzer")
        self.root.state("zoomed")

        self.file_paths = []
        self.all_results = []

        self.setup_ui()

    # ================= UI =================
    def setup_ui(self):
        paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # LEFT PANEL
        left = tk.Frame(paned)
        paned.add(left, minsize=320)

        tk.Label(left, text="Selected Files", font=("Arial", 11, "bold")).pack(anchor="w", padx=5, pady=5)

        self.listbox = tk.Listbox(left, selectmode=tk.EXTENDED)
        self.listbox.pack(fill="both", expand=True, padx=5, pady=5)

        # Drag & Drop
        self.listbox.drop_target_register(DND_FILES)
        self.listbox.dnd_bind('<<Drop>>', self.drop_files)

        # -------- Buttons Row 1 --------
        row1 = tk.Frame(left)
        row1.pack(fill="x", padx=5, pady=2)

        tk.Button(row1, text="Add Files", command=self.add_files).pack(side="left", expand=True, fill="x")
        tk.Button(row1, text="Clear Selected", command=self.clear_selected).pack(side="left", expand=True, fill="x")
        tk.Button(row1, text="Clear All", command=self.clear_all).pack(side="left", expand=True, fill="x")

        # -------- Move Buttons --------
        row2 = tk.Frame(left)
        row2.pack(fill="x", padx=5, pady=5)

        tk.Button(row2, text="▲ Move Up", command=self.move_up, bg="#e6f2ff").pack(side="left", expand=True, fill="x")
        tk.Button(row2, text="▼ Move Down", command=self.move_down, bg="#e6f2ff").pack(side="left", expand=True, fill="x")

        # -------- Sort Buttons --------
        row3 = tk.Frame(left)
        row3.pack(fill="x", padx=5, pady=5)

        tk.Button(row3, text="Sort A→Z", command=self.sort_files, bg="#f0f8ff").pack(side="left", expand=True, fill="x")
        tk.Button(row3, text="Sort Z→A", command=self.sort_files_reverse, bg="#f0f8ff").pack(side="left", expand=True, fill="x")

        # RIGHT PANEL
        right = tk.Frame(paned)
        paned.add(right)

        tk.Button(right, text="Process LAeq", command=self.run_laeq_thread,
                  bg="#d9ead3", height=2).pack(fill="x", padx=10, pady=5)

        tk.Button(right, text="Process SPL Max", command=self.run_splmax_thread,
                  bg="#fff3e0", height=2).pack(fill="x", padx=10, pady=5)

        tk.Button(right, text="Export Excel", command=self.export_excel,
                  bg="#d0e0ff", height=2).pack(fill="x", padx=10, pady=5)

        # Progress
        self.progress_frame = tk.Frame(right)
        self.progress_frame.pack(fill="x", padx=10, pady=10)

        self.progress = ttk.Progressbar(self.progress_frame, orient="horizontal", mode="determinate")
        self.progress.pack(fill="x")

        self.progress_text = tk.Label(self.progress_frame, text="0%", bg="white")
        self.progress_text.place(relx=0.5, rely=0.5, anchor="center")

        # Log
        tk.Label(right, text="Console Log", font=("Arial", 10, "bold")).pack(anchor="w", padx=10)

        self.logbox = ScrolledText(right, bg="black", fg="lime", font=("Consolas", 9))
        self.logbox.pack(fill="both", expand=True, padx=10, pady=5)

    # ================= FILE HANDLING =================
    def refresh_listbox(self):
        self.listbox.delete(0, tk.END)
        for f in self.file_paths:
            self.listbox.insert(tk.END, os.path.basename(f))

    def add_files(self):
        files = filedialog.askopenfilenames()
        for f in files:
            if f not in self.file_paths:
                self.file_paths.append(f)
        self.refresh_listbox()

    def drop_files(self, event):
        files = self.root.tk.splitlist(event.data)
        for f in files:
            f = f.strip("{}")
            if f not in self.file_paths:
                self.file_paths.append(f)
        self.refresh_listbox()

    def clear_selected(self):
        sel = list(self.listbox.curselection())
        for i in reversed(sel):
            del self.file_paths[i]
        self.refresh_listbox()

    def clear_all(self):
        self.file_paths.clear()
        self.refresh_listbox()

    # ================= MOVE =================
    def move_up(self):
        sel = list(self.listbox.curselection())
        for i in sel:
            if i > 0:
                self.file_paths[i-1], self.file_paths[i] = self.file_paths[i], self.file_paths[i-1]
        self.refresh_listbox()
        for i in sel:
            if i > 0:
                self.listbox.selection_set(i-1)

    def move_down(self):
        sel = list(self.listbox.curselection())
        for i in reversed(sel):
            if i < len(self.file_paths) - 1:
                self.file_paths[i+1], self.file_paths[i] = self.file_paths[i], self.file_paths[i+1]
        self.refresh_listbox()
        for i in sel:
            if i < len(self.file_paths) - 1:
                self.listbox.selection_set(i+1)

    # ================= SORT =================
    def sort_files(self):
        self.file_paths.sort(key=lambda x: os.path.basename(x).lower())
        self.refresh_listbox()

    def sort_files_reverse(self):
        self.file_paths.sort(key=lambda x: os.path.basename(x).lower(), reverse=True)
        self.refresh_listbox()

    # ================= LOG =================
    def log(self, msg):
        self.logbox.insert("end", msg + "\n")
        self.logbox.see("end")

    # ================= PROGRESS =================
    def update_progress(self, current, total):
        percent = int((current / total) * 100)
        self.progress["value"] = percent
        self.progress_text.config(text=f"{percent}%")
        self.root.update_idletasks()

    # ================= LAEQ =================
    def run_laeq_thread(self):
        threading.Thread(target=self.process_laeq).start()

    def process_laeq(self):
        total = len(self.file_paths)

        for idx, f in enumerate(self.file_paths):
            self.log(f"\nProcessing {os.path.basename(f)}")

            df = read_file(f, self.log)
            t = df.iloc[:, 0].values

            for i in range(1, df.shape[1]):
                sig = df.iloc[:, i].values
                _, sig_r = resample_signal(t, sig, FS_TARGET)
                val = compute_laeq(sig_r, FS_TARGET)
                self.log(f"{df.columns[i]}: {val:.2f} dB(A)")

            self.update_progress(idx + 1, total)

    # ================= SPL =================
    def run_splmax_thread(self):
        threading.Thread(target=self.process_splmax).start()

    def process_splmax(self):
        total = len(self.file_paths)

        for idx, f in enumerate(self.file_paths):
            self.log(f"\nProcessing {os.path.basename(f)}")

            df = read_file(f, self.log)
            t = df.iloc[:, 0].values

            for i in range(1, df.shape[1]):
                sig = df.iloc[:, i].values
                _, sig_r = resample_signal(t, sig, FS_TARGET)
                val = compute_spl_max(sig_r, FS_TARGET)
                self.log(f"{df.columns[i]}: {val:.2f} dB(A)")

            self.update_progress(idx + 1, total)

    # ================= EXPORT =================
    def export_excel(self):
        if not self.all_results:
            messagebox.showwarning("No Data", "Run analysis first")
            return

        path = filedialog.asksaveasfilename(defaultextension=".xlsx")
        pd.DataFrame(self.all_results).to_excel(path, index=False)
        self.log(f"Exported: {path}")