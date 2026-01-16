import tkinter as tk
from tkinter import ttk
import multiprocessing
import soundcard as sc
from visualizer import run_visualizer
from generator import ToneGeneratorWindow

class ControlPanel:
    def __init__(self, root):
        self.root = root
        self.root.title("Visualizer Control")
        self.root.geometry("400x400")
        
        self.msg_queue = multiprocessing.Queue()
        self.status_queue = multiprocessing.Queue()
        self.viz_process = None
        
        try:
            self.devices = sc.all_microphones(include_loopback=True)
        except:
            self.devices = []
            
        lbl_title = tk.Label(root, text="Visualizer Settings", font=("Arial", 14, "bold"))
        lbl_title.pack(pady=15)
        
        lbl_device = tk.Label(root, text="Audio Source:")
        lbl_device.pack(pady=5)
        
        self.device_var = tk.StringVar()
        self.combo_devices = ttk.Combobox(root, textvariable=self.device_var, state="readonly", width=50)
        
        self.device_map = {}
        values = ["[Auto] Try to find Speaker Loopback"]
        self.device_map["[Auto] Try to find Speaker Loopback"] = "AUTO"
        
        for dev in self.devices:
            name = dev.name
            values.append(name)
            self.device_map[name] = dev.id
            
        self.combo_devices['values'] = values
        self.combo_devices.current(0)
        self.combo_devices.pack(pady=5)
        self.combo_devices.bind("<<ComboboxSelected>>", self.on_device_change)
        
        lbl_theme = tk.Label(root, text="Visual Theme:")
        lbl_theme.pack(pady=10)
        
        self.theme_var = tk.StringVar()
        self.combo_themes = ttk.Combobox(root, textvariable=self.theme_var, state="readonly", width=30)
        # [Update] Constellation 추가
        self.combo_themes['values'] = [
            "Default (Neon Cyberpunk)", 
            "Dynamic Shockwave", 
            "3D Waveform (Tunnel)",
            "3D Terrain (Stereo)",
            "Constellation (Star Network)"
        ]
        self.combo_themes.current(0)
        self.combo_themes.pack(pady=5)
        self.combo_themes.bind("<<ComboboxSelected>>", self.on_theme_change)
        
        self.lbl_status = tk.Label(root, text="Status: Running", fg="green")
        self.lbl_status.pack(pady=20)
        
        btn_gen = tk.Button(root, text="Open Frequency Generator", command=self.open_generator, bg="#ddffff", width=25)
        btn_gen.pack(pady=5)

        btn_quit = tk.Button(root, text="Quit All", command=self.close_app, bg="#ffdddd", width=20)
        btn_quit.pack(side=tk.BOTTOM, pady=20)

        self.start_visualizer()
        self.root.after(100, self.check_status)

    def start_visualizer(self):
        try:
            name = self.combo_devices.get()
            dev_id = self.device_map.get(name)
        except:
            dev_id = "AUTO"
            
        self.viz_process = multiprocessing.Process(
            target=run_visualizer, 
            args=(self.msg_queue, self.status_queue)
        )
        self.viz_process.start()
        
        if dev_id:
            self.msg_queue.put(("SET_DEVICE", dev_id))

    def on_device_change(self, event):
        try:
            name = self.combo_devices.get()
            dev_id = self.device_map.get(name)
            if dev_id:
                self.msg_queue.put(("SET_DEVICE", dev_id))
        except Exception as e:
            print(f"Error: {e}")

    def on_theme_change(self, event):
        try:
            selection = self.combo_themes.get()
            theme_code = "Default"
            if "Shockwave" in selection: theme_code = "Shockwave"
            elif "Waveform" in selection: theme_code = "3D Waveform"
            elif "Terrain" in selection: theme_code = "Terrain"
            elif "Constellation" in selection: theme_code = "Constellation"
            
            self.msg_queue.put(("SET_THEME", theme_code))
        except Exception as e:
            print(f"Error: {e}")

    def open_generator(self):
        ToneGeneratorWindow(self.root)

    def check_status(self):
        try:
            while not self.status_queue.empty():
                msg = self.status_queue.get_nowait()
                if msg == "CLOSED":
                    self.lbl_status.config(text="Status: Visualizer Closed", fg="red")
        except:
            pass
            
        if self.viz_process and not self.viz_process.is_alive():
             self.lbl_status.config(text="Status: Visualizer Stopped", fg="red")
             
        self.root.after(500, self.check_status)

    def close_app(self):
        if self.viz_process and self.viz_process.is_alive():
            self.msg_queue.put(("QUIT", None))
            self.viz_process.join(timeout=1)
            if self.viz_process.is_alive():
                self.viz_process.terminate()
        self.root.destroy()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    root = tk.Tk()
    app = ControlPanel(root)
    root.protocol("WM_DELETE_WINDOW", app.close_app)
    root.mainloop()
