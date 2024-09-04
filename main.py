# main.py

import tkinter as tk
from training_gui import TrainingApp

def main():
    root = tk.Tk()
    app = TrainingApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
