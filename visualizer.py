import pygame
import pygame.gfxdraw
import numpy as np
import soundcard as sc
import math
import queue
import time
import random
import threading
from collections import deque

# --- 설정 상수 ---
WIDTH, HEIGHT = 1000, 800
FPS = 60
CHUNK = 1024 
NOISE_GATE = 0.002 

# ==========================================
# 1. 렌더러 베이스
# ==========================================
class Renderer:
    def __init__(self, width, height):
        self.width = width; self.height = height; self.cx = width // 2; self.cy = height // 2
    def draw(self, screen, bars_height, radius_info, kick_trigger, bass_energy, stereo_data=None, fft_mag=None): pass

# (DefaultRenderer, ShockwaveRenderer, Waveform3DRenderer, Terrain3DRenderer는 기존 유지)
# ... 전체 코드 작성을 위해 아래에 모두 포함합니다 ...

class DefaultRenderer(Renderer):
    def __init__(self, width, height):
        super().__init__(width, height)
        self.particles = []
    def draw(self, screen, bars_height, radius_info, kick_trigger, bass_energy, stereo_data=None, fft_mag=None):
        current_radius, radius_velocity = radius_info
        bg_color = (20, 20, 35) if kick_trigger else (5, 5, 10)
        screen.fill(bg_color)
        if kick_trigger:
            for _ in range(4):
                hue = random.randint(200, 280)
                p_color = pygame.Color(0); p_color.hsla = (hue, 100, 70, 100)
                self.particles.append(SimpleParticle(self.cx, self.cy, (p_color.r, p_color.g, p_color.b)))
        self.particles = [p for p in self.particles if p.update()]
        for p in self.particles: p.draw(screen)
        
        draw_radius = current_radius
        if current_radius > 130 and bass_energy > 5:
            draw_radius += random.uniform(-0.8, 0.8) * ((current_radius - 130) * 0.2)
        final_r = int(draw_radius)
        
        pygame.gfxdraw.filled_circle(screen, self.cx, self.cy, final_r + 10, (20, 30, 60))
        pygame.gfxdraw.filled_circle(screen, self.cx, self.cy, final_r, (10, 10, 20))
        pygame.gfxdraw.aacircle(screen, self.cx, self.cy, final_r, (100, 200, 255))
        pygame.gfxdraw.aacircle(screen, self.cx, self.cy, final_r-1, (100, 200, 255))

        BARS = len(bars_height)
        angle_step = 360 / BARS
        gap = 0.1
        for i in range(BARS):
            h = bars_height[i]
            if h < 2: continue
            angle = i * angle_step + 90 
            rad_start = math.radians(angle + gap)
            rad_end = math.radians(angle + angle_step - gap)
            p1 = (self.cx + math.cos(rad_start) * draw_radius, self.cy + math.sin(rad_start) * draw_radius)
            p2 = (self.cx + math.cos(rad_end) * draw_radius, self.cy + math.sin(rad_end) * draw_radius)
            p3 = (self.cx + math.cos(rad_end) * (draw_radius + h), self.cy + math.sin(rad_end) * (draw_radius + h))
            p4 = (self.cx + math.cos(rad_start) * (draw_radius + h), self.cy + math.sin(rad_start) * (draw_radius + h))
            hue = 190 + (i * 90 / BARS)
            c = pygame.Color(0); c.hsla = (hue % 360, 100, 80, 100)
            pygame.gfxdraw.filled_polygon(screen, [p1, p2, p3, p4], (c.r//2, c.g//2, c.b//2))
            pygame.gfxdraw.aapolygon(screen, [p1, p2, p3, p4], c)
        
        if bass_energy > 10:
             pygame.gfxdraw.filled_circle(screen, int(self.cx), int(self.cy), int(final_r * 0.3), (255, 255, 255, 50))

class ShockwaveRenderer(Renderer):
    def __init__(self, width, height):
        super().__init__(width, height)
        self.shockwaves = []
        self.particles = []
        self.bg_pulse = 0
    def draw(self, screen, bars_height, radius_info, kick_trigger, bass_energy, stereo_data=None, fft_mag=None):
        current_radius, radius_velocity = radius_info
        if kick_trigger: self.bg_pulse = 20
        self.bg_pulse *= 0.9
        bg_val = int(5 + self.bg_pulse)
        screen.fill((bg_val, bg_val, bg_val + 5))
        if kick_trigger:
            self.shockwaves.append({'r': current_radius, 'a': 255})
            for _ in range(10):
                hue = random.choice([0, 30, 330])
                self.particles.append(GravityParticle(self.cx, self.cy, hue))
        for wave in self.shockwaves[:]:
            wave['r'] += 10; wave['a'] -= 10
            if wave['a'] <= 0: self.shockwaves.remove(wave)
            else:
                c = max(0, min(255, wave['a']))
                pygame.gfxdraw.aacircle(screen, self.cx, self.cy, int(wave['r']), (c, c, 255))
        self.particles = [p for p in self.particles if p.update()]
        for p in self.particles: p.draw(screen)
        
        draw_radius = current_radius + self.bg_pulse
        final_r = int(draw_radius)
        pygame.gfxdraw.filled_circle(screen, self.cx, self.cy, final_r, (40, 10, 10))
        pygame.gfxdraw.aacircle(screen, self.cx, self.cy, final_r, (255, 100, 100))
        
        BARS = len(bars_height)
        angle_step = 360 / BARS
        gap = 0.1
        for i in range(BARS):
            h = bars_height[i]
            if h < 2: continue
            angle = i * angle_step + 90 
            rad_start = math.radians(angle + gap); rad_end = math.radians(angle + angle_step - gap)
            p1 = (self.cx + math.cos(rad_start) * draw_radius, self.cy + math.sin(rad_start) * draw_radius)
            p2 = (self.cx + math.cos(rad_end) * draw_radius, self.cy + math.sin(rad_end) * draw_radius)
            p3 = (self.cx + math.cos(rad_end) * (draw_radius + h), self.cy + math.sin(rad_end) * (draw_radius + h))
            p4 = (self.cx + math.cos(rad_start) * (draw_radius + h), self.cy + math.sin(rad_start) * (draw_radius + h))
            hue = (60 - (i * 60 / BARS)) % 360
            c = pygame.Color(0); c.hsla = (hue, 100, 50, 100)
            pygame.gfxdraw.filled_polygon(screen, [p1, p2, p3, p4], c)
            pygame.gfxdraw.aapolygon(screen, [p1, p2, p3, p4], (255, 200, 200))

class Waveform3DRenderer(Renderer):
    def __init__(self, width, height):
        super().__init__(width, height)
        self.history_limit = 16
        self.bar_limit = 32
        self.history = deque(maxlen=self.history_limit)
        self.tilt = 0.5
        self.ring_spacing = 25
        angle_step = 2 * np.pi / self.bar_limit
        angles = np.linspace(0, 2 * np.pi - angle_step, self.bar_limit)
        self.cos_vals = np.cos(angles)
        self.sin_vals = np.sin(angles)

    def project_3d(self, x, y, z):
        if z < -500: z = -500
        scale = 1.0 / (1.0 + z * 0.0025)
        sx = self.cx + x * scale
        sy = self.cy + (z * self.tilt * scale) - (y * scale) + 150
        return sx, sy

    def draw(self, screen, bars_height, radius_info, kick_trigger, bass_energy, stereo_data=None, fft_mag=None):
        screen.fill((5, 5, 8))
        step = len(bars_height) // self.bar_limit
        if step < 1: step = 1
        current_ring = bars_height[::step][:self.bar_limit]
        if len(current_ring) < self.bar_limit:
            current_ring = np.pad(current_ring, (0, self.bar_limit - len(current_ring)))
        self.history.appendleft(current_ring)
        
        hist_array = np.array(self.history)
        count = len(hist_array)
        if count == 0: return

        r_indices = np.arange(count).reshape(-1, 1)
        base_radii = 60 + r_indices * self.ring_spacing
        
        wz = base_radii * self.sin_vals
        wx = base_radii * self.cos_vals
        wy = hist_array
        wz = np.maximum(wz, -500)
        
        scale_factors = 1.0 - (r_indices * 0.03)
        scale_factors = np.maximum(scale_factors, 0.1)
        
        sx = self.cx + wx * scale_factors
        sy = self.cy + (wz * self.tilt * scale_factors) - (wy * scale_factors) + 150
        
        for i in range(count - 1, -1, -1):
            points = np.column_stack((sx[i], sy[i])).astype(int).tolist()
            hue = (200 + i * 8) % 360
            brightness = max(20, 80 - i * 4)
            color = pygame.Color(0)
            color.hsla = (hue, 90, brightness, 100)
            thickness = 2 if i < 5 else 1
            pygame.draw.lines(screen, color, True, points, thickness)
            if i == 0: pygame.draw.polygon(screen, (10, 10, 20), points)

class Terrain3DRenderer(Renderer):
    def __init__(self, width, height):
        super().__init__(width, height)
        self.grid_w = 32; self.grid_h = 24
        self.hist_L = deque(maxlen=self.grid_h)
        self.hist_R = deque(maxlen=self.grid_h)
        self.cell_w = 12; self.cell_h = 15; self.fov = 300.0

    def project(self, x, y, z, is_left):
        world_x = (x + 0.5) * self.cell_w
        world_z = z * self.cell_h
        depth = world_z + 100
        scale = self.fov / (depth + 0.1)
        sx = self.cx + world_x * scale
        floor_y = (self.cy - 100) + 300 * scale
        sy = floor_y - (y * scale)
        return sx, sy

    def draw(self, screen, bars_height, radius_info, kick_trigger, bass_energy, stereo_data=None, fft_mag=None):
        screen.fill((5, 0, 10))
        if stereo_data is None: left_data = right_data = bars_height
        else: left_data, right_data = stereo_data
        step = len(left_data) // self.grid_w
        if step < 1: step = 1
        l_row = left_data[::step][:self.grid_w]; r_row = right_data[::step][:self.grid_w]
        if len(l_row) < self.grid_w: l_row = np.pad(l_row, (0, self.grid_w - len(l_row)))
        if len(r_row) < self.grid_w: r_row = np.pad(r_row, (0, self.grid_w - len(r_row)))
        self.hist_L.appendleft(l_row[::-1]); self.hist_R.appendleft(r_row)
        
        def draw_grid(hist, is_left):
            for z in range(len(hist) - 1, -1, -1):
                row = hist[z]
                points = []
                for i in range(self.grid_w):
                    h = row[i] * 0.5
                    gx = -self.grid_w + i if is_left else i
                    sx, sy = self.project(gx, h, z, is_left)
                    points.append((sx, sy))
                if len(points) > 1:
                    hue = (260 + z * 3) % 360
                    lightness = max(10, 80 - z * 3)
                    if z == 0: lightness = 80
                    c = pygame.Color(0); c.hsla = (hue, 80, lightness, 100)
                    pygame.draw.lines(screen, c, False, points, 2)
        draw_grid(self.hist_L, True); draw_grid(self.hist_R, False)

# =========================================================
# [NEW] Constellation Renderer (Star Network)
# =========================================================
class ConstellationRenderer(Renderer):
    """[Profile 5] Constellation: 별자리 노드 연결 시각화"""
    def __init__(self, width, height):
        super().__init__(width, height)
        self.num_nodes = 150
        self.nodes = []
        
        # 노드 생성 (위치 랜덤, 주파수 할당)
        for i in range(self.num_nodes):
            x = random.randint(50, width - 50)
            y = random.randint(50, height - 50)
            
            # Y축에 따라 주파수 할당 (아래=저음, 위=고음)
            # 0(저음) ~ 1(고음) 정규화된 값 저장
            freq_factor = 1.0 - (y / height)
            
            # 주파수 인덱스 미리 계산 (로그 스케일 매핑)
            # 512 bin 중 어디를 볼 것인가 (0 ~ 256 유효)
            # freq_factor 0 -> index 2 (Low Bass)
            # freq_factor 1 -> index 200 (High Treble)
            fft_index = int(2 + freq_factor * 200)
            
            self.nodes.append({
                'x': x, 'y': y,
                'base_x': x, 'base_y': y, # 원래 위치 (돌아오기 위함)
                'freq_idx': fft_index,
                'size': 2,
                'val': 0, # 현재 반응값
                'color_hue': int(freq_factor * 240) # 0(Red) ~ 240(Blue)
            })

    def draw(self, screen, bars_height, radius_info, kick_trigger, bass_energy, stereo_data=None, fft_mag=None):
        screen.fill((5, 5, 10)) # 우주 배경
        
        if fft_mag is None: return
        
        # 1. 노드 업데이트
        active_nodes = []
        
        for node in self.nodes:
            # 해당 주파수의 에너지 가져오기
            idx = node['freq_idx']
            if idx < len(fft_mag):
                val = fft_mag[idx] * 2.0 # 증폭
            else:
                val = 0
            
            # 부드러운 감쇠
            if val > node['val']: node['val'] += (val - node['val']) * 0.5
            else: node['val'] += (val - node['val']) * 0.1
            
            current_val = node['val']
            
            # 시각적 크기
            node['size'] = 2 + current_val * 0.5
            
            # 위치 미세 진동 (살아있는 느낌)
            node['x'] = node['base_x'] + random.uniform(-1, 1) * (current_val * 0.1)
            node['y'] = node['base_y'] + random.uniform(-1, 1) * (current_val * 0.1)
            
            # 일정 강도 이상인 노드만 선 연결 후보로 등록
            if current_val > 10:
                active_nodes.append(node)
            
            # 노드 그리기 (빛나는 별)
            alpha = min(255, int(current_val * 10))
            if alpha > 20:
                color = pygame.Color(0)
                color.hsla = (node['color_hue'], 80, 60, 100)
                
                # Glow
                # gfxdraw는 alpha 블렌딩이 약해서 원을 겹쳐 그림
                # r, g, b = color.r, color.g, color.b
                # pygame.gfxdraw.filled_circle(screen, int(node['x']), int(node['y']), int(node['size'])+2, (r, g, b, 50))
                pygame.draw.circle(screen, color, (int(node['x']), int(node['y'])), int(node['size']))

        # 2. 선 연결 (Network)
        # 활성 노드끼리 거리가 가까우면 선을 그림
        # O(N^2) 이지만 활성 노드가 적으면 괜찮음
        
        # 너무 많으면 느려지니 최대 50개 정도만 체크
        check_nodes = active_nodes[:60] 
        
        for i in range(len(check_nodes)):
            n1 = check_nodes[i]
            for j in range(i + 1, len(check_nodes)):
                n2 = check_nodes[j]
                
                dist = math.hypot(n1['x'] - n2['x'], n1['y'] - n2['y'])
                
                if dist < 150: # 연결 거리 제한
                    # 가까울수록 진하게
                    alpha = int(255 * (1 - dist / 150))
                    width = 1
                    if dist < 50: width = 2
                    
                    # 선 색상은 두 노드의 중간색
                    c1 = pygame.Color(0); c1.hsla = (n1['color_hue'], 80, 50, 100)
                    # Alpha 적용을 위해 Surface 사용 안하고 선 색을 어둡게 처리
                    # (검은 배경이라 어두운 색 = 투명해 보임)
                    line_color = (
                        int(c1.r * (alpha/255)), 
                        int(c1.g * (alpha/255)), 
                        int(c1.b * (alpha/255))
                    )
                    
                    pygame.draw.line(screen, line_color, (n1['x'], n1['y']), (n2['x'], n2['y']), width)

# --- 파티클 & 오디오 워커 (기존 유지) ---
class SimpleParticle:
    def __init__(self, x, y, color):
        self.x = x; self.y = y; self.color = color
        angle = random.uniform(0, 2 * math.pi); speed = random.uniform(2, 6)
        self.vx = math.cos(angle) * speed; self.vy = math.sin(angle) * speed
        self.life = random.randint(20, 40); self.size = random.randint(2, 4)
    def update(self):
        self.x += self.vx; self.y += self.vy; self.life -= 1
        return self.life > 0
    def draw(self, surface):
        if self.life > 0: pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), int(self.size))

class GravityParticle:
    def __init__(self, x, y, hue):
        self.x = x; self.y = y; self.hue = hue
        angle = random.uniform(0, 2 * math.pi); speed = random.uniform(5, 15)
        self.vx = math.cos(angle) * speed; self.vy = math.sin(angle) * speed
        self.gravity = 0.5; self.life = 60; self.size = random.randint(2, 5)
    def update(self):
        self.x += self.vx; self.y += self.vy; self.vy += self.gravity; self.life -= 1
        return self.life > 0
    def draw(self, surface):
        if self.life > 0:
            c = pygame.Color(0); c.hsla = (self.hue, 100, 60, 100)
            pygame.draw.line(surface, c, (int(self.x), int(self.y)), (int(self.x-self.vx), int(self.y-self.vy)), 2)
            pygame.draw.circle(surface, (255, 255, 255), (int(self.x), int(self.y)), int(self.size))

class AudioWorker(threading.Thread):
    def __init__(self, device_id):
        super().__init__()
        self.device_id = device_id
        self.running = True
        self.data_queue = queue.Queue(maxsize=3)
        self.daemon = True
    def stop(self): self.running = False
    def run(self):
        while self.running:
            try:
                mic = None
                if self.device_id == "AUTO":
                    mics = sc.all_microphones(include_loopback=True)
                    for m in mics:
                        if "Speaker" in m.name or "스피커" in m.name: mic = m; break
                    if mic is None and mics: mic = mics[0]
                else: mic = sc.get_microphone(self.device_id, include_loopback=True)
                if mic:
                    with mic.recorder(samplerate=48000, blocksize=CHUNK) as recorder:
                        while self.running:
                            raw_data = recorder.record(numframes=CHUNK)
                            if raw_data.shape[1] > 1:
                                mono = np.mean(raw_data, axis=1)
                                left = raw_data[:, 0]
                                right = raw_data[:, 1]
                                package = (mono, left, right)
                            else:
                                flat = raw_data.flatten()
                                package = (flat, flat, flat)
                            if self.data_queue.full():
                                try: self.data_queue.get_nowait()
                                except: pass
                            self.data_queue.put(package, timeout=0.1)
                else: time.sleep(1)
            except Exception: time.sleep(0.5)

def run_visualizer(msg_queue, status_queue):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Visualizer")
    clock = pygame.time.Clock()

    BARS = 120 
    bars_height = np.zeros(BARS, dtype=np.float64)
    bars_L = np.zeros(BARS, dtype=np.float64)
    bars_R = np.zeros(BARS, dtype=np.float64)
    
    base_radius = 100
    current_radius = base_radius
    radius_velocity = 0
    avg_volume = 0.01 
    gain = 50.0 
    current_device_id = "AUTO"
    renderer = DefaultRenderer(WIDTH, HEIGHT)
    audio_thread = AudioWorker(current_device_id)
    audio_thread.start()
    
    print("--- Visualizer GUI Started ---")

    running = True
    while running:
        try:
            while not msg_queue.empty():
                cmd, value = msg_queue.get_nowait()
                if cmd == "SET_DEVICE":
                    audio_thread.stop()
                    audio_thread.join(timeout=1.0)
                    if audio_thread.is_alive(): print("Force killing thread")
                    current_device_id = value
                    audio_thread = AudioWorker(current_device_id)
                    audio_thread.start()
                elif cmd == "SET_THEME":
                    if value == "Default": renderer = DefaultRenderer(WIDTH, HEIGHT)
                    elif value == "Shockwave": renderer = ShockwaveRenderer(WIDTH, HEIGHT)
                    elif "3D Waveform" in value: renderer = Waveform3DRenderer(WIDTH, HEIGHT)
                    elif "Terrain" in value: renderer = Terrain3DRenderer(WIDTH, HEIGHT)
                    elif "Constellation" in value: renderer = ConstellationRenderer(WIDTH, HEIGHT) # NEW
                elif cmd == "QUIT":
                    running = False
        except queue.Empty: pass

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                status_queue.put("CLOSED")

        package = None
        try:
            if not audio_thread.data_queue.empty():
                package = audio_thread.data_queue.get_nowait()
        except: pass

        # FFT 결과 변수 (렌더러 전달용)
        fft_mag = None

        if package is not None:
            mono_data, left_data, right_data = package
            raw_vol = np.mean(np.abs(mono_data))
            if raw_vol < NOISE_GATE:
                mono_data[:] = 0.0; left_data[:] = 0.0; right_data[:] = 0.0
                frame_vol = 0.0
            else: frame_vol = raw_vol

            if frame_vol > 0: avg_volume = avg_volume * 0.98 + frame_vol * 0.02
            if avg_volume > 0.00001:
                target_gain = 0.08 / avg_volume 
                target_gain = min(target_gain, 250.0); target_gain = max(target_gain, 1.0)
            else: target_gain = 250.0
            
            gain = gain * 0.9 + target_gain * 0.1
            
            mono_data = mono_data * gain * np.hanning(len(mono_data))
            fft_mono = np.abs(np.fft.rfft(mono_data))
            fft_mag = fft_mono # 저장
            
            left_data = left_data * gain * np.hanning(len(left_data))
            fft_L = np.abs(np.fft.rfft(left_data))
            right_data = right_data * gain * np.hanning(len(right_data))
            fft_R = np.abs(np.fft.rfft(right_data))

            samplerate = 48000
            bin_width = samplerate / CHUNK
            bass_energy = np.mean(fft_mono[1:4])
            kick_trigger = (fft_mono[1] + fft_mono[2]) > 25 

            def map_bars(fft_data):
                res = []
                start_bin = int(300 / bin_width)
                end_bin = int(10000 / bin_width)
                boundaries = np.logspace(np.log10(start_bin), np.log10(end_bin), num=BARS+1)
                for i in range(BARS):
                    s = int(boundaries[i]); e = int(boundaries[i+1])
                    if e <= s: e = s + 1
                    val = np.max(fft_data[s:e]) if e < len(fft_data) else 0
                    comp = 1.0 + (i / BARS) * 2.0
                    h = (val ** 0.5) * 20 * comp
                    res.append(max(0, min(h, 300)))
                return np.array(res, dtype=np.float64)

            target_h_mono = map_bars(fft_mono)
            target_h_L = map_bars(fft_L)
            target_h_R = map_bars(fft_R)
            
            diff = target_h_L - target_h_R
            boost = 4.0
            target_h_L += np.maximum(diff, 0) * boost
            target_h_R += np.maximum(-diff, 0) * boost
            
            for i in range(BARS):
                diff_mono = (target_h_mono[i] - bars_height[i])
                if diff_mono > 0: bars_height[i] += diff_mono * 0.7
                else: bars_height[i] += diff_mono * 0.25
                
                diff_L = (target_h_L[i] - bars_L[i])
                if diff_L > 0: bars_L[i] += diff_L * 0.7
                else: bars_L[i] += diff_L * 0.25
                
                diff_R = (target_h_R[i] - bars_R[i])
                if diff_R > 0: bars_R[i] += diff_R * 0.7
                else: bars_R[i] += diff_R * 0.25

            target_r = base_radius
            if kick_trigger: radius_velocity += 10
            if bass_energy > 5: target_r += min(bass_energy * 0.5, 15)
            force = (target_r - current_radius) * 0.2
            radius_velocity += force
            radius_velocity *= 0.65
            current_radius += radius_velocity
            current_radius = max(80, min(current_radius, 200))
        else:
            kick_trigger = False; bass_energy = 0; radius_velocity *= 0.9
            current_radius += (base_radius - current_radius) * 0.1
            bars_height *= 0.9; bars_L *= 0.9; bars_R *= 0.9

        stereo_package = (bars_L, bars_R)
        # [수정] fft_mag 전달
        renderer.draw(screen, bars_height, (current_radius, radius_velocity), kick_trigger, bass_energy, stereo_package, fft_mag)

        pygame.display.flip()
        clock.tick(FPS)

    audio_thread.stop()
    audio_thread.join()
    pygame.quit()
