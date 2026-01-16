import tkinter as tk
from tkinter import ttk, scrolledtext
import numpy as np
import sounddevice as sd
import ctypes
import os
import sys
import threading
import queue

class VoiceTunerWindow:
    def __init__(self, parent):
        self.window = tk.Toplevel(parent)
        self.window.title("C++ Powered Voice Tuner (Debug Mode)")
        self.window.geometry("500x650")
        self.window.resizable(False, False)

        # --- Variables ---
        self.is_active = False
        self.sample_rate = 44100
        self.block_size = 512
        self.stream = None
        self.log_queue = queue.Queue()
        
        # Audio Devices
        self.input_devices = []
        self.output_devices = []
        self.selected_input = tk.IntVar(value=-1)
        self.selected_output = tk.IntVar(value=-1)

        # C++ Engine Path
        if hasattr(sys, '_MEIPASS'):
            self.dll_path = os.path.join(sys._MEIPASS, "dsp_engine.dll")
        else:
            self.dll_path = os.path.abspath("dsp_engine.dll")

        self._setup_ui()
        self._load_devices()
        self._load_engine()
        
        # Start UI Update Loop
        self.window.after(100, self._update_ui_loop)
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)

    def log(self, msg):
        self.log_queue.put(msg)

    def _setup_ui(self):
        # 1. Device Selection Frame
        dev_frame = tk.LabelFrame(self.window, text="Audio Devices", padx=10, pady=5)
        dev_frame.pack(fill="x", padx=10, pady=5)
        
        tk.Label(dev_frame, text="Input (Mic):").pack(anchor="w")
        self.combo_in = ttk.Combobox(dev_frame, state="readonly")
        self.combo_in.pack(fill="x", pady=(0, 5))
        
        tk.Label(dev_frame, text="Output (Speaker):").pack(anchor="w")
        self.combo_out = ttk.Combobox(dev_frame, state="readonly")
        self.combo_out.pack(fill="x", pady=(0, 5))

        # 2. Main Control
        ctrl_frame = tk.Frame(self.window, padx=10, pady=5)
        ctrl_frame.pack(fill="x", padx=10)

        self.btn_toggle = tk.Button(ctrl_frame, text="▶ START MONITORING", command=self.toggle_stream, 
                                    bg="#ccffcc", font=("Arial", 11, "bold"), height=2)
        self.btn_toggle.pack(fill="x", pady=5)

        # Bypass Checkbox
        self.var_bypass = tk.BooleanVar(value=False)
        self.chk_bypass = tk.Checkbutton(ctrl_frame, text="Bypass C++ Engine (Test Output)", variable=self.var_bypass)
        self.chk_bypass.pack(anchor="w")

        # 3. Level Meters (Debug)
        meter_frame = tk.LabelFrame(self.window, text="Signal Levels", padx=10, pady=5)
        meter_frame.pack(fill="x", padx=10, pady=5)
        
        tk.Label(meter_frame, text="In:").grid(row=0, column=0)
        self.pb_in = ttk.Progressbar(meter_frame, orient="horizontal", length=300, mode="determinate")
        self.pb_in.grid(row=0, column=1, padx=5, sticky="ew")
        
        tk.Label(meter_frame, text="Out:").grid(row=1, column=0)
        self.pb_out = ttk.Progressbar(meter_frame, orient="horizontal", length=300, mode="determinate")
        self.pb_out.grid(row=1, column=1, padx=5, sticky="ew")

        # 4. Effects
        fx_frame = tk.LabelFrame(self.window, text="Effects", padx=10, pady=5)
        fx_frame.pack(fill="x", padx=10, pady=5)

        tk.Label(fx_frame, text="Ring Freq (Hz):").pack(anchor="w")
        self.scale_ring_f = tk.Scale(fx_frame, from_=20, to=2000, orient="horizontal")
        self.scale_ring_f.set(100)
        self.scale_ring_f.pack(fill="x")

        tk.Label(fx_frame, text="Ring Amount:").pack(anchor="w")
        self.scale_ring_a = tk.Scale(fx_frame, from_=0, to=100, orient="horizontal")
        self.scale_ring_a.set(50)
        self.scale_ring_a.pack(fill="x")
        
        tk.Label(fx_frame, text="Bit Depth:").pack(anchor="w")
        self.scale_bits = tk.Scale(fx_frame, from_=2, to=16, orient="horizontal")
        self.scale_bits.set(16)
        self.scale_bits.pack(fill="x")

        # 5. Log Console
        log_frame = tk.LabelFrame(self.window, text="Debug Log", padx=5, pady=5)
        log_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.txt_log = scrolledtext.ScrolledText(log_frame, height=8, state="disabled", font=("Consolas", 8))
        self.txt_log.pack(fill="both", expand=True)

    def _load_devices(self):
        try:
            devs = sd.query_devices()
            default_in = sd.query_devices(kind='input')
            default_out = sd.query_devices(kind='output')
            
            in_list = []
            out_list = []
            self.input_map = []
            self.output_map = []
            
            for i, d in enumerate(devs):
                if d['max_input_channels'] > 0:
                    name = f"{i}: {d['name']}"
                    if d['name'] == default_in['name']: name += " [Default]"
                    in_list.append(name)
                    self.input_map.append(i)
                
                if d['max_output_channels'] > 0:
                    name = f"{i}: {d['name']}"
                    if d['name'] == default_out['name']: name += " [Default]"
                    out_list.append(name)
                    self.output_map.append(i)
            
            self.combo_in['values'] = in_list
            self.combo_out['values'] = out_list
            
            if in_list: self.combo_in.current(0)
            if out_list: self.combo_out.current(0)
            
            self.log(f"Loaded {len(in_list)} input, {len(out_list)} output devices.")
        except Exception as e:
            self.log(f"Error loading devices: {e}")

    def _load_engine(self):
        self.log(f"Loading DLL from: {self.dll_path}")
        if not os.path.exists(self.dll_path):
            self.log("ERROR: DLL file not found!")
            return

        try:
            self.engine = ctypes.CDLL(self.dll_path)
            self.engine.init_engine.argtypes = [ctypes.c_float]
            self.engine.apply_ring_mod.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_float, ctypes.c_float]
            self.engine.apply_bit_crush.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
            self.engine.init_engine(44100.0)
            self.log("C++ Engine loaded successfully.")
        except Exception as e:
            self.log(f"Failed to load C++ engine: {e}")
            self.engine = None

    def audio_callback(self, indata, outdata, frames, time, status):
        if status:
            self.log(f"Audio Status: {status}")

        # Input Level Check
        vol_in = np.linalg.norm(indata) * 10
        self.window.after(0, lambda: self.pb_in.configure(value=min(100, vol_in)))

        if self.var_bypass.get():
            outdata[:] = indata
        elif self.engine:
            # [Fix] Do NOT use astype() here, it creates a copy! 
            # We enforce float32 in stream setup, so we pass pointers directly.
            in_ptr = indata.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            out_ptr = outdata.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            
            ring_f = float(self.scale_ring_f.get())
            ring_a = float(self.scale_ring_a.get()) / 100.0
            bits = int(self.scale_bits.get())
            
            self.engine.apply_bit_crush(in_ptr, out_ptr, frames, bits)
            self.engine.apply_ring_mod(out_ptr, out_ptr, frames, ring_f, ring_a)
        else:
            # Fallback if engine fails
            outdata[:] = indata

        # Output Level Check
        vol_out = np.linalg.norm(outdata) * 10
        self.window.after(0, lambda: self.pb_out.configure(value=min(100, vol_out)))

    def toggle_stream(self):
        if not self.is_active:
            try:
                # Get selected device indices
                idx_in = self.input_map[self.combo_in.current()]
                idx_out = self.output_map[self.combo_out.current()]
                
                self.log(f"Starting stream: In({idx_in}) -> Out({idx_out})")
                
                self.stream = sd.Stream(
                    device=(idx_in, idx_out),
                    samplerate=self.sample_rate,
                    blocksize=self.block_size,
                    channels=1,
                    dtype='float32',  # [Fix] Enforce float32 for direct C++ pointer access
                    callback=self.audio_callback
                )
                self.stream.start()
                self.is_active = True
                self.btn_toggle.config(text="■ STOP MONITORING", bg="#ffcccc")
            except Exception as e:
                self.log(f"Stream Error: {e}")
        else:
            if self.stream:
                self.stream.stop()
                self.stream.close()
            self.is_active = False
            self.btn_toggle.config(text="▶ START MONITORING", bg="#ccffcc")
            self.log("Stream stopped.")
            self.pb_in.configure(value=0)
            self.pb_out.configure(value=0)

    def _update_ui_loop(self):
        while not self.log_queue.empty():
            msg = self.log_queue.get_nowait()
            self.txt_log.configure(state="normal")
            self.txt_log.insert(tk.END, msg + "\n")
            self.txt_log.see(tk.END)
            self.txt_log.configure(state="disabled")
        self.window.after(100, self._update_ui_loop)

    def on_close(self):
        self.is_active = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = VoiceTunerWindow(root)
    root.mainloop()