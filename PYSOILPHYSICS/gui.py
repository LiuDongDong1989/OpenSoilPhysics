import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np

# Assuming all models have similar structure for simplicity
import SWRC

class SWRC_GUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SWRC GUI")
        
        # Model Selection
        self.model_label = tk.Label(self, text="Select Model:")
        self.model_label.pack(pady=10)
        
        self.models = ['Model1', 'Model2', 'Model3', 'Model4', 'Model5']
        self.model_var = tk.StringVar()
        self.model_dropdown = ttk.Combobox(self, textvariable=self.model_var, values=self.models)
        self.model_dropdown.bind("<<ComboboxSelected>>", self.on_model_change)
        self.model_dropdown.pack(pady=10)
        
        # Parameter Input Frame
        self.param_frame = tk.Frame(self)
        self.param_frame.pack(pady=20)
        
        # Execute Button
        self.execute_btn = tk.Button(self, text="Execute", command=self.execute_model)
        self.execute_btn.pack(pady=20)
        
        # Placeholder for Matplotlib plot
        self.canvas = None
        
    def on_model_change(self, event=None):
        # Clear previous parameters
        for widget in self.param_frame.winfo_children():
            widget.destroy()
        
        # For simplicity, let's assume each model requires two parameters: alpha and beta
        self.alpha_var = tk.DoubleVar(value=0.5)
        self.beta_var = tk.DoubleVar(value=0.5)
        
        alpha_label = tk.Label(self.param_frame, text="Alpha:")
        alpha_label.grid(row=0, column=0)
        alpha_entry = tk.Entry(self.param_frame, textvariable=self.alpha_var)
        alpha_entry.grid(row=0, column=1)
        
        beta_label = tk.Label(self.param_frame, text="Beta:")
        beta_label.grid(row=1, column=0)
        beta_entry = tk.Entry(self.param_frame, textvariable=self.beta_var)
        beta_entry.grid(row=1, column=1)

    def execute_model(self):
        # For simplicity, let's assume each model just plots y = alpha * x + beta
        alpha = self.alpha_var.get()
        beta = self.beta_var.get()
        
        x = np.linspace(0, 10, 100)
        y = alpha * x + beta
        
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(x, y)
        ax.set_title(f"{self.model_var.get()} Result")
        
        # Embed the plot in the GUI
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        
        self.canvas = FigureCanvasTkAgg(fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(pady=20)

if __name__ == "__main__":
    app = SWRC_GUI()
    app.mainloop()
