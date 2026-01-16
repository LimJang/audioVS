import tkinter as tk
from tkinter import ttk
import numpy as np
import sounddevice as sd
import threading

class ToneGeneratorWindow:
    def __init__(self, parent):
        self.window = tk.Toplevel(parent)
        self.window.title("Advanced Tone Generator")
        self.window.geometry("600x600")
        self.window.resizable(False, False)
        self.window.configure(bg="#f0f0f0")

        # --- Audio Engine State ---
        self.sample_rate = 44100
        self.is_playing = False
        self.frequency = 440.0
        self.volume = 0.5
        self.wave_type = "Sine" # Sine, Square, Sawtooth, Triangle, Custom
        self.phase = 0.0
        self.stream = None
        
        # --- Custom Wave State ---
        # Normalized coordinates (x: 0~1, y: -1~1)
        # Default custom wave is a flat line
        self.custom_points = [{'x': 0.0, 'y': 0.0}, {'x': 1.0, 'y': 0.0}] 
        self.selected_point_idx = None
        self.wavetable = None
        self.update_wavetable() # Initial Build

        self._setup_ui()
        self._start_stream()
        
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)

    def _setup_ui(self):
        # 1. Canvas Area (Waveform Visualization & Editing)
        frame_canvas = tk.Frame(self.window, bg="black", bd=2, relief="sunken")
        frame_canvas.pack(pady=10, padx=10, fill="both", expand=True)
        
        self.canvas = tk.Canvas(frame_canvas, bg="#101020", height=250, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        
        # Canvas Events
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.canvas.bind("<Button-3>", self.on_canvas_right_click)

        # 2. Controls - Top Row
        control_frame = tk.Frame(self.window, bg="#f0f0f0")
        control_frame.pack(pady=5, padx=10, fill="x")

        # Play/Stop
        self.btn_play = tk.Button(control_frame, text="▶ START", command=self.toggle_play, 
                                  font=("Arial", 11, "bold"), bg="#ccffcc", width=10)
        self.btn_play.grid(row=0, column=0, rowspan=2, padx=5)

        # Waveform Selection
        tk.Label(control_frame, text="Waveform:", bg="#f0f0f0").grid(row=0, column=1, sticky="w")
        self.wave_var = tk.StringVar(value="Sine")
        waves = ["Sine", "Square", "Sawtooth", "Triangle", "Custom"]
        self.combo_wave = ttk.Combobox(control_frame, textvariable=self.wave_var, values=waves, state="readonly", width=12)
        self.combo_wave.grid(row=1, column=1, sticky="w")
        self.combo_wave.bind("<<ComboboxSelected>>", self.on_wave_change)

        # Volume
        tk.Label(control_frame, text="Volume:", bg="#f0f0f0").grid(row=0, column=2, sticky="w", padx=(15, 0))
        self.vol_scale = tk.Scale(control_frame, from_=0, to=100, orient="horizontal", 
                                  command=lambda v: self.update_params(), showvalue=False, length=150, bg="#f0f0f0")
        self.vol_scale.set(50)
        self.vol_scale.grid(row=1, column=2, sticky="w", padx=(15, 0))

        # 3. Frequency Controls (Min/Max Range + Slider)
        freq_frame = tk.LabelFrame(self.window, text="Frequency Control", padx=10, pady=10, bg="#f0f0f0")
        freq_frame.pack(pady=5, padx=10, fill="x")

        # Range Settings
        f_range_frame = tk.Frame(freq_frame, bg="#f0f0f0")
        f_range_frame.pack(fill="x", pady=(0, 5))
        
        tk.Label(f_range_frame, text="Range Min:", bg="#f0f0f0").pack(side="left")
        self.entry_min = tk.Entry(f_range_frame, width=6)
        self.entry_min.insert(0, "20")
        self.entry_min.pack(side="left", padx=5)
        self.entry_min.bind("<Return>", self.update_slider_range)

        tk.Label(f_range_frame, text="Max:", bg="#f0f0f0").pack(side="left")
        self.entry_max = tk.Entry(f_range_frame, width=6)
        self.entry_max.insert(0, "2000")
        self.entry_max.pack(side="left", padx=5)
        self.entry_max.bind("<Return>", self.update_slider_range)
        
        btn_apply = tk.Button(f_range_frame, text="Set", command=self.update_slider_range, height=1)
        btn_apply.pack(side="left", padx=5)

        # Current Freq
        self.freq_var = tk.StringVar(value="440")
        self.entry_freq = tk.Entry(f_range_frame, textvariable=self.freq_var, width=8, font=("Arial", 11, "bold"), justify="center")
        self.entry_freq.pack(side="right", padx=5)
        self.entry_freq.bind("<Return>", self.on_freq_entry)
        tk.Label(f_range_frame, text="Hz", bg="#f0f0f0").pack(side="right")

        # Slider
        self.slider_freq = tk.Scale(freq_frame, from_=20, to=2000, orient="horizontal", 
                                    command=self.on_freq_slide, showvalue=False, bg="#f0f0f0")
        self.slider_freq.set(440)
        self.slider_freq.pack(fill="x", expand=True)

        # 4. Custom Wave Tools (Only visible/active in Custom mode ideally, but kept simple)
        self.lbl_help = tk.Label(self.window, text="[Custom Mode] Left-Click: Add/Move Point | Right-Click: Delete", 
                                 fg="gray", bg="#f0f0f0", font=("Arial", 9))
        self.lbl_help.pack(side="bottom", pady=5)

        self.draw_waveform()

    # --- Canvas Interaction Logic ---
    def coord_to_norm(self, x, y):
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        nx = x / w
        # Screen Y is inverted. y=0 is top.
        # We want y=0 to be center. 
        # norm y: +1 (top) to -1 (bottom) is standard audio? 
        # Actually usually +1 is top amplitude visually, but let's map:
        # Screen y=0 -> +1.0
        # Screen y=h -> -1.0
        ny = 1.0 - 2.0 * (y / h)
        return nx, max(-1.0, min(1.0, ny))

    def norm_to_coord(self, nx, ny):
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        x = nx * w
        # ny=1 -> y=0, ny=-1 -> y=h
        y = (1.0 - ny) / 2.0 * h
        return x, y

    def on_canvas_click(self, event):
        if self.wave_type != "Custom": return
        
        nx, ny = self.coord_to_norm(event.x, event.y)
        
        # Find nearest point
        closest_idx = -1
        min_dist = 0.05 # Threshold
        
        for i, p in enumerate(self.custom_points):
            dx = p['x'] - nx
            # Visual y distance check needs aspect ratio consideration, but simple is fine
            if abs(dx) < min_dist: 
                closest_idx = i
                break
        
        if closest_idx != -1:
            self.selected_point_idx = closest_idx
        else:
            # Add new point
            self.custom_points.append({'x': nx, 'y': ny})
            self.custom_points.sort(key=lambda p: p['x'])
            self.selected_point_idx = self.custom_points.index({'x': nx, 'y': ny})
            self.update_wavetable()
        
        self.draw_waveform()

    def on_canvas_drag(self, event):
        if self.wave_type != "Custom" or self.selected_point_idx is None: return
        
        nx, ny = self.coord_to_norm(event.x, event.y)
        
        # Constraint x to be between neighbors to maintain order
        idx = self.selected_point_idx
        # If it's the first or last point, maybe lock x? 
        # For simplicity, let's keep x order valid
        min_x = 0.0
        max_x = 1.0
        
        if idx > 0: min_x = self.custom_points[idx-1]['x'] + 0.01
        if idx < len(self.custom_points) - 1: max_x = self.custom_points[idx+1]['x'] - 0.01
        
        # Allow moving start/end points freely within bounds?
        # Let's enforce 0 and 1 exist or just clamp.
        
        nx = max(min_x, min(nx, max_x))
        nx = max(0.0, min(1.0, nx))
        
        self.custom_points[idx]['x'] = nx
        self.custom_points[idx]['y'] = ny
        self.draw_waveform()

    def on_canvas_release(self, event):
        if self.selected_point_idx is not None:
            self.update_wavetable()
            self.selected_point_idx = None

    def on_canvas_right_click(self, event):
        if self.wave_type != "Custom": return
        nx, ny = self.coord_to_norm(event.x, event.y)
        
        for i, p in enumerate(self.custom_points):
            if abs(p['x'] - nx) < 0.05:
                # Don't delete if it's the only point?
                if len(self.custom_points) > 2:
                    del self.custom_points[i]
                    self.update_wavetable()
                    self.draw_waveform()
                break

    def update_wavetable(self):
        # Pre-calculate one cycle of the custom wave
        # Interpolate points onto a fixed size buffer
        table_size = 2048
        
        xs = [p['x'] for p in self.custom_points]
        ys = [p['y'] for p in self.custom_points]
        
        # Ensure x covers 0 to 1
        if xs[0] > 0:
            xs.insert(0, 0.0)
            ys.insert(0, ys[0]) # Repeat first val
        if xs[-1] < 1.0:
            xs.append(1.0)
            ys.append(ys[0]) # Wrap around to start for loop? or just repeat last.
                             # Let's wrap to first Y to make it loop smoothly if desired
                             # But users might want non-loop. Let's just extend last val.
                             
        target_x = np.linspace(0.0, 1.0, table_size)
        self.wavetable = np.interp(target_x, xs, ys)

    # --- Audio Logic ---
    def _audio_callback(self, outdata, frames, time, status):
        if status: print(status)
        
        if not self.is_playing:
            outdata.fill(0)
            return

        dt = 1.0 / self.sample_rate
        # Calculate phase increment
        phase_increment = (self.frequency * dt) 
        
        # Vectorized phase generation
        # phases = self.global_phase + np.arange(frames) * phase_increment
        # But this is linear.
        # We want wrapped phase 0..1
        
        indices = np.arange(frames)
        current_phases = (self.global_phase + indices * phase_increment) % 1.0
        
        # Update global phase for next block
        self.global_phase = (self.global_phase + frames * phase_increment) % 1.0

        if self.wave_type == "Sine":
            # 0..1 -> 0..2pi
            wave = np.sin(current_phases * 2 * np.pi)
        elif self.wave_type == "Square":
            wave = np.sign(np.sin(current_phases * 2 * np.pi))
        elif self.wave_type == "Sawtooth":
            # 0..1 -> -1..1
            wave = 2.0 * (current_phases - 0.5)
        elif self.wave_type == "Triangle":
            # 0..1 -> 0..2..0..-2..0 ?
            # Scipy triangle is abs((t % 1) - 0.5) * 4 - 1
            wave = 4.0 * np.abs(current_phases - 0.5) - 1.0
        elif self.wave_type == "Custom":
            # Lookup from wavetable
            if self.wavetable is None:
                wave = np.zeros(frames)
            else:
                # Map phase 0..1 to index 0..len-1
                idx = (current_phases * (len(self.wavetable) - 1)).astype(int)
                wave = self.wavetable[idx]

        outdata[:] = (wave * self.volume * 0.2).reshape(-1, 1)

    def _start_stream(self):
        self.global_phase = 0.0
        try:
            self.stream = sd.OutputStream(
                channels=1, 
                callback=self._audio_callback,
                samplerate=self.sample_rate,
                blocksize=1024
            )
            self.stream.start()
        except Exception as e:
            print(f"Audio Error: {e}")

    # --- Event Handlers ---
    def toggle_play(self):
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.btn_play.config(text="■ STOP", bg="#ffcccc")
        else:
            self.btn_play.config(text="▶ START", bg="#ccffcc")

    def on_wave_change(self, event):
        self.wave_type = self.combo_wave.get()
        self.draw_waveform()

    def update_params(self):
        self.volume = self.vol_scale.get() / 100.0

    def update_slider_range(self, event=None):
        try:
            min_v = float(self.entry_min.get())
            max_v = float(self.entry_max.get())
            if min_v >= max_v: max_v = min_v + 10
            
            self.slider_freq.config(from_=min_v, to=max_v)
            
            # Clamp current
            curr = float(self.freq_var.get())
            if curr < min_v: self.freq_var.set(min_v); self.frequency = min_v
            if curr > max_v: self.freq_var.set(max_v); self.frequency = max_v
            self.slider_freq.set(self.frequency)
            
        except ValueError: pass

    def on_freq_slide(self, val):
        self.frequency = float(val)
        self.freq_var.set(f"{self.frequency:.1f}")

    def on_freq_entry(self, event):
        try:
            val = float(self.freq_var.get())
            min_v = self.slider_freq.cget("from")
            max_v = self.slider_freq.cget("to")
            
            val = max(1, min(val, 22000)) # Absolute safety limits
            self.frequency = val
            
            # If out of slider range, just update slider visual if possible or ignore?
            # Standard behavior: slider moves to end if out of bounds
            if val < min_v: self.slider_freq.set(min_v)
            elif val > max_v: self.slider_freq.set(max_v)
            else: self.slider_freq.set(val)
            
        except ValueError: pass

    def draw_waveform(self):
        self.canvas.delete("all")
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w < 10: w = 400
        if h < 10: h = 250
        
        # 1. Draw Grid / Zero line
        cy = h / 2
        self.canvas.create_line(0, cy, w, cy, fill="#333333", dash=(2,2))
        
        points = []
        
        if self.wave_type == "Custom":
            # Draw lines between points
            # Sort for display
            sorted_pts = sorted(self.custom_points, key=lambda p: p['x'])
            
            prev_x, prev_y = None, None
            
            # Start from x=0 visual?
            if sorted_pts and sorted_pts[0]['x'] > 0:
                 # Draw line from 0 to first point?
                 sx, sy = self.norm_to_coord(0, sorted_pts[0]['y'])
                 # Actually just connect dots
            
            disp_points = []
            for i, p in enumerate(sorted_pts):
                sx, sy = self.norm_to_coord(p['x'], p['y'])
                disp_points.append(sx)
                disp_points.append(sy)
                
                # Draw Node
                r = 4
                color = "#00ffff" if i == self.selected_point_idx else "#ffffff"
                self.canvas.create_oval(sx-r, sy-r, sx+r, sy+r, fill=color, outline="")
            
            if len(disp_points) >= 4:
                self.canvas.create_line(disp_points, fill="#00ff00", width=2)
                
        else:
            # Draw standard preview
            num_points = 100
            display_points = []
            for i in range(num_points):
                x_norm = i / (num_points - 1)
                
                val = 0
                phase = x_norm # 0..1
                
                if self.wave_type == "Sine":
                    val = np.sin(phase * 2 * np.pi)
                elif self.wave_type == "Square":
                    val = np.sign(np.sin(phase * 2 * np.pi))
                elif self.wave_type == "Sawtooth":
                    val = 2.0 * (phase - 0.5)
                elif self.wave_type == "Triangle":
                    val = 4.0 * np.abs(phase - 0.5) - 1.0
                
                sx, sy = self.norm_to_coord(x_norm, val)
                display_points.append(sx)
                display_points.append(sy)
                
            self.canvas.create_line(display_points, fill="#00ff00", width=2)

    def on_close(self):
        self.is_playing = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = ToneGeneratorWindow(root)
    root.mainloop()