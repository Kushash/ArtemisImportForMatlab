from tkinterdnd2 import TkinterDnD
from gui.app_ui import AcousticApp

if __name__ == "__main__":
    root = TkinterDnD.Tk()   # REQUIRED for drag & drop
    app = AcousticApp(root)
    root.mainloop()