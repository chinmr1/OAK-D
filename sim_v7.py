"""
sim.py — 4-DOF Robot Arm 3D Stick-Figure Simulator
===================================================
Faithfully translates the IK, FK, coordinated S-curve motion profile,
and PID control logic from the BBCServo16 C++ firmware (v10.0).

Controls:
  Type coordinates : "150 0 50"  or  "150, 0, 50"  or  "[150 0 50]"
  Speed command    : "v 50"      (degrees per second)
  Home command     : "g"  or  "home"
  Cube place       : "c 100 50"  (place a cube at x=100 y=50 on the floor)
  Cube delete tip  : "c d"       (delete cube currently touching/attached to tip)
  Cube reset       : "c r"       (delete all cubes)
  Mouse drag       : Rotate 3D view
  Scroll wheel     : Zoom in/out

Physical constants, servo limits, calibration offsets, and compensation
coefficients are hardcoded from the C++ source to ensure 1:1 accuracy.
"""

import math
import time
import threading
import queue
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# =====================================================================
# CONFIGURABLE
# =====================================================================
TARGET_FPS = 30          # Animation frame rate (change as needed)
LINE_THICKNESS = 1.2     # Multiplier for robot arm line widths (1.0 = default)
ROBOT_ALPHA = 1.0        # Transparency for the 3D robot arm (0.0 to 1.0)
CUBE_SIDE = 25.0         # Cube size in mm
CUBE_GRAB_DIST = 10.0    # Distance (mm) for tip to grab a cube
INVERT_Y = False         # True: flip Y axis direction (positive Y = left instead of right)

# =====================================================================
# WORKING AREA LIMITS (mm)
# =====================================================================
WORK_X_MIN = -50
WORK_X_MAX = 300
WORK_Y_MIN = -300
WORK_Y_MAX = 300
WORK_Z_MIN = -30
WORK_Z_MAX = 200

# =====================================================================
# PHYSICAL CONSTANTS  (from BBCServo16_WORKING.ino)
# =====================================================================
H1 = 92.8                          # Base height to shoulder pivot (mm)
L1 = 103.5                         # Upper arm length (mm)
L2 = 145.5                         # Forearm length (mm)
L3 = 83.0                          # Wrist-to-tip / end-effector (mm)
SHOULDER_FORWARD_OFFSET = 10.0     # Shoulder pivot forward offset (mm)

# =====================================================================
# SERVO ANGLE LIMITS  [waist, shoulder, elbow, wrist]
# =====================================================================
SERVO_MIN = [0.0,   32.0,  40.0,  85.0]
SERVO_MAX = [160.0, 160.0, 251.0, 233.0]

# =====================================================================
# CALIBRATION OFFSETS
# =====================================================================
WAIST_OFFSET    =  0.0
SHOULDER_OFFSET =  0.0
ELBOW_OFFSET    =  0.0
WRIST_OFFSET    =  0.0

# =====================================================================
# IK GLOBAL OFFSETS  (hand-tuned in firmware)
# =====================================================================
IK_OFFSET_X =  0.0
IK_OFFSET_Y =  0.0
IK_OFFSET_Z =  0.0

# =====================================================================
# COMPENSATION COEFFICIENTS  (53-point calibration, quadratic model)
# error = c0 + c1*r + c2*z + c3*r² + c4*r*z + c5*z²
# =====================================================================
COMP_R = [ 48.179516, -0.12332323, -0.51895783,
            0.0000481339,  0.0008664521,  0.0017162520]
COMP_Z = [-10.900375,  0.10396278,  0.43701631,
            0.0002494156, -0.0016885057, -0.0011007065]

# =====================================================================
# MOTION DEFAULTS
# =====================================================================
DEFAULT_SPEED   = 30.0    # deg/s  (userBaseSpeed in firmware)
MIN_JOINT_SPEED = 18.0    # deg/s  (floor for coordinated motion)

# =====================================================================
# PID GAINS  (from firmware setup)
# =====================================================================
KP = 1.00
KI = 0.03
KD = 0.05

# =====================================================================
# EMA FILTER
# =====================================================================
EMA_ALPHA = 0.15    # 15 % new, 85 % old  (firmware default)


# #####################################################################
#  SIMULATED SERVO
# #####################################################################
class SimServo:
    def __init__(self, name, min_ang, max_ang, initial=90.0):
        self.name     = name
        self.min_ang  = min_ang
        self.max_ang  = max_ang

        self._start    = initial
        self._target   = initial
        self._duration = 0.0
        self._elapsed  = 0.0
        self._moving   = False
        self._speed    = DEFAULT_SPEED

        self._setpoint = initial
        self._smoothed = initial

        self._integral = 0.0
        self._last_err = 0.0
        self._pid_out  = 0.0

    def set_target(self, deg):
        if abs(deg - self._target) < 0.01:
            return
        self._start    = self._setpoint          
        self._target   = deg
        self._integral = 0.0                     
        dist = abs(self._target - self._start)
        self._duration = max(0.1, dist / self._speed) if self._speed > 0 else 1.0
        self._elapsed  = 0.0
        self._moving   = True

    def set_speed(self, dps):
        self._speed = max(1.0, min(200.0, dps))

    def update(self, dt):
        if dt <= 0:
            return

        if self._moving:
            self._elapsed += dt
            if self._elapsed >= self._duration:
                self._setpoint = self._target
                self._moving   = False
            else:
                t = self._elapsed / self._duration
                curve = (1.0 - math.cos(math.pi * t)) / 2.0
                self._setpoint = (self._start
                                  + (self._target - self._start) * curve)
        else:
            self._setpoint = self._target

        self._smoothed += EMA_ALPHA * (self._setpoint - self._smoothed)
        if abs(self._setpoint - self._smoothed) < 0.01:
            self._smoothed = self._setpoint

        raw_err = self._setpoint - self._smoothed
        if abs(raw_err) < 1.0:             
            error = 0.0
            self._integral *= 0.95         
        else:
            error = raw_err
        self._integral = max(-100.0, min(100.0,
                                         self._integral + error * dt))
        d_term = ((error - self._last_err) / max(dt, 1e-6)) * KD
        self._pid_out  = error * KP + self._integral * KI + d_term
        self._last_err = error

    @property
    def angle(self):
        return self._smoothed

    def is_moving(self):
        return self._moving

    def is_settled(self, tol=0.05):
        return (not self._moving
                and abs(self._target - self._smoothed) < tol)


# #####################################################################
#  INVERSE KINEMATICS  
# #####################################################################
def solve_ik(tx, ty, tz):
    x = tx
    y = ty
    z = tz

    waist = 90.0 + math.degrees(math.atan2(y, x))
    waist = max(SERVO_MIN[0], min(SERVO_MAX[0], waist))

    r_target = math.sqrt(x * x + y * y)

    z_wrist_rel = (z + L3) - H1
    r_arm       = r_target - SHOULDER_FORWARD_OFFSET

    if r_arm < 0.0:
        return None, "Behind shoulder"

    D = math.sqrt(r_arm * r_arm + z_wrist_rel * z_wrist_rel)
    MAX_REACH = L1 + L2
    MIN_REACH = abs(L1 - L2)

    if D > MAX_REACH:
        return None, f"Too far  D={D:.1f}  max={MAX_REACH:.1f}"
    if D < MIN_REACH:
        return None, f"Too close  D={D:.1f}  min={MIN_REACH:.1f}"

    cos_elbow = (L1*L1 + L2*L2 - D*D) / (2.0 * L1 * L2)
    cos_elbow = max(-1.0, min(1.0, cos_elbow))
    elbow_inner_deg = math.degrees(math.acos(cos_elbow))

    alpha_1 = math.atan2(z_wrist_rel, r_arm)
    cos_a2  = max(-1.0, min(1.0, (L1*L1 + D*D - L2*L2) / (2.0*L1*D)))
    alpha_2 = math.acos(cos_a2)
    shoulder_deg = math.degrees(alpha_1 + alpha_2)

    ss = (180.0 - shoulder_deg)
    se = 90.0 + (180.0 - elbow_inner_deg)
    wu = (450.0 - ss - se)

    ss = max(SERVO_MIN[1], min(SERVO_MAX[1], ss))
    se = max(SERVO_MIN[2], min(SERVO_MAX[2], se))
    sw = max(SERVO_MIN[3], min(SERVO_MAX[3], wu))

    return (waist, ss, se, sw), None

# #####################################################################
#  FORWARD KINEMATICS 
# #####################################################################
def compute_fk(s_wa, s_sh, s_el, _s_wr):
    wr = math.radians(s_wa - 90.0)                     
    sr = math.radians(180.0 - s_sh)                     
    ei = 180.0 - (s_el - 90.0)                          
    er = sr - math.radians(180.0 - ei)                   

    cw, sw = math.cos(wr), math.sin(wr)
    pts = np.zeros((6, 3))

    pts[0] = [0.0, 0.0, 0.0]
    pts[1] = [0.0, 0.0, H1]
    pts[2] = [SHOULDER_FORWARD_OFFSET * cw, SHOULDER_FORWARD_OFFSET * sw, H1]

    r_e = SHOULDER_FORWARD_OFFSET + L1 * math.cos(sr)
    z_e = H1 + L1 * math.sin(sr)
    pts[3] = [r_e * cw, r_e * sw, z_e]

    r_w = r_e + L2 * math.cos(er)
    z_w = z_e + L2 * math.sin(er)
    pts[4] = [r_w * cw, r_w * sw, z_w]

    pts[5] = [r_w * cw, r_w * sw, z_w - L3]
    return pts


# #####################################################################
#  CUBE MANAGER & GEOMETRY
# #####################################################################
class CubeManager:
    def __init__(self, grab_dist=CUBE_GRAB_DIST):
        self.grab_dist = grab_dist
        self._cubes = []       

    def place_cube(self, x, y):
        cube = {
            'pos': np.array([float(x), float(y), CUBE_SIDE / 2.0]),
            'attached': False,
        }
        self._cubes.append(cube)

    def delete_attached(self):
        before = len(self._cubes)
        self._cubes = [c for c in self._cubes if not c['attached']]
        return before - len(self._cubes)

    def delete_all(self):
        n = len(self._cubes)
        self._cubes.clear()
        return n

    def update(self, tip_pos):
        for cube in self._cubes:
            if cube['attached']:
                cube['pos'] = tip_pos.copy()
                cube['pos'][2] -= CUBE_SIDE / 2.0  
            else:
                half = CUBE_SIDE / 2.0
                nearest = np.clip(tip_pos, cube['pos'] - half, cube['pos'] + half)
                dist = np.linalg.norm(tip_pos - nearest)
                if dist <= self.grab_dist:
                    cube['attached'] = True

    def get_all_cubes(self):
        return self._cubes


def _cube_verts(center, side=CUBE_SIDE):
    s = side / 2.0
    cx, cy, cz = center
    corners = np.array([
        [cx - s, cy - s, cz - s], [cx + s, cy - s, cz - s],
        [cx + s, cy + s, cz - s], [cx - s, cy + s, cz - s],
        [cx - s, cy - s, cz + s], [cx + s, cy - s, cz + s],
        [cx + s, cy + s, cz + s], [cx - s, cy + s, cz + s],
    ])
    faces = [
        [corners[0], corners[1], corners[2], corners[3]],  
        [corners[4], corners[5], corners[6], corners[7]],  
        [corners[0], corners[1], corners[5], corners[4]],  
        [corners[2], corners[3], corners[7], corners[6]],  
        [corners[0], corners[3], corners[7], corners[4]],  
        [corners[1], corners[2], corners[6], corners[5]],  
    ]
    return faces


def _prism_between(p0, p1, width, depth):
    """Generates 6 faces of a rectangular prism between two 3D coordinates."""
    v = np.array(p1) - np.array(p0)
    L = np.linalg.norm(v)
    if L < 1e-6: return []
    v_dir = v / L
    
    if abs(v_dir[2]) > 0.99:
        up = np.array([1.0, 0.0, 0.0])
    else:
        up = np.array([0.0, 0.0, 1.0])
        
    orth1 = np.cross(v_dir, up)
    orth1 /= np.linalg.norm(orth1)
    orth2 = np.cross(v_dir, orth1)

    w2 = width / 2.0
    d2 = depth / 2.0
    
    c0 = p0 - orth1*w2 - orth2*d2
    c1 = p0 + orth1*w2 - orth2*d2
    c2 = p0 + orth1*w2 + orth2*d2
    c3 = p0 - orth1*w2 + orth2*d2

    c4 = c0 + v
    c5 = c1 + v
    c6 = c2 + v
    c7 = c3 + v

    return [
        [c0, c1, c2, c3], [c4, c5, c6, c7], [c0, c1, c5, c4],
        [c1, c2, c6, c5], [c2, c3, c7, c6], [c3, c0, c4, c7]
    ]


def _cylinder_faces(p0, p1, radius, n_sides=16):
    """Generates faces for a cylinder between two 3D coordinates."""
    v = np.array(p1) - np.array(p0)
    L = np.linalg.norm(v)
    if L < 1e-6: return []
    v_dir = v / L
    
    if abs(v_dir[2]) > 0.99:
        up = np.array([1.0, 0.0, 0.0])
    else:
        up = np.array([0.0, 0.0, 1.0])
        
    orth1 = np.cross(v_dir, up)
    orth1 /= np.linalg.norm(orth1)
    orth2 = np.cross(v_dir, orth1)

    faces = []
    angles = np.linspace(0, 2*np.pi, n_sides+1)
    cap0 = []
    cap1 = []
    
    for a in angles[:-1]:
        offset = radius * (np.cos(a)*orth1 + np.sin(a)*orth2)
        cap0.append(p0 + offset)
        cap1.append(p1 + offset)

    for i in range(n_sides):
        next_i = (i+1)%n_sides
        faces.append([cap0[i], cap0[next_i], cap1[next_i], cap1[i]])

    faces.append(cap0[::-1])
    faces.append(cap1)
    return faces


# #####################################################################
#  ARM SIMULATOR
# #####################################################################
class ArmSimulator:
    def __init__(self):
        result, _ = solve_ik(75, 0, 65)
        if result:
            init = list(result)
        else:
            init = [90.0, 90.0, 180.0, 180.0]

        self.servos = [
            SimServo("WAIST",    SERVO_MIN[0], SERVO_MAX[0], init[0]),
            SimServo("SHOULDER", SERVO_MIN[1], SERVO_MAX[1], init[1]),
            SimServo("ELBOW",    SERVO_MIN[2], SERVO_MAX[2], init[2]),
            SimServo("WRIST",    SERVO_MIN[3], SERVO_MAX[3], init[3]),
        ]
        self.base_speed  = DEFAULT_SPEED
        self.status      = "Ready  |  Enter coordinates: x y z"
        self.target      = None
        self._last_time  = time.perf_counter()
        self.cubes = CubeManager()
        self._cmd_queue = queue.Queue()

    def send_command(self, text):
        self._cmd_queue.put(text)

    def send_xyz(self, x, y, z):
        self._cmd_queue.put(f"{x} {y} {z}")

    def command(self, text):
        text = text.strip().replace('[', '').replace(']', '')
        if not text:
            return

        if text[0].lower() == 'v':
            try:
                spd = float(text[1:].strip())
                if spd > 0:
                    self.base_speed = spd
                    self.status = f"Speed set to {spd:.0f} deg/s"
                else:
                    self.status = "Speed must be > 0"
            except ValueError:
                self.status = "Usage: v <speed>"
            return

        if text.lower() in ('g', 'home'):
            self._move_xyz(75, 0, 65)
            return

        if text[0].lower() == 'c':
            rest = text[1:].strip()
            if rest.lower() == 'd':
                n = self.cubes.delete_attached()
                self.status = f"Deleted {n} attached cube(s)" if n else "No cube attached to tip"
                return
            if rest.lower() == 'r':
                n = self.cubes.delete_all()
                self.status = f"Removed all {n} cube(s)" if n else "No cubes to remove"
                return
            parts = rest.replace(',', ' ').split()
            if len(parts) == 2:
                try:
                    cx, cy = float(parts[0]), float(parts[1])
                    self.cubes.place_cube(cx, cy)
                    self.status = f"Placed cube at ({cx:.0f}, {cy:.0f})"
                except ValueError:
                    self.status = "Usage: c <x> <y>  |  c d  |  c r"
            else:
                self.status = "Usage: c <x> <y>  |  c d  |  c r"
            return

        parts = text.replace(',', ' ').split()
        if len(parts) == 3:
            try:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                self._move_xyz(x, y, z)
            except ValueError:
                self.status = "Bad input.  Use: x y z  (e.g. 150 0 50)"
        else:
            self.status = "Bad input.  Use: x y z  |  v <speed>  |  g (home)  |  c <x> <y>"

    def _move_xyz(self, x, y, z):
        result, err = solve_ik(x, y, z)
        if result is None:
            self.status = f"IK Error: {err}"
            return

        targets = list(result)
        fk_pts = compute_fk(*targets)
        self.target = (fk_pts[5][0], fk_pts[5][1], fk_pts[5][2])

        dists = [abs(targets[i] - self.servos[i].angle) for i in range(4)]
        max_d = max(max(dists), 0.1)
        move_time = max_d / self.base_speed

        for i in range(4):
            spd = dists[i] / move_time
            if spd < MIN_JOINT_SPEED:
                spd = MIN_JOINT_SPEED
            self.servos[i].set_speed(spd)
            self.servos[i].set_target(targets[i])

        self.status = (f"Moving to ({x:.0f}, {y:.0f}, {z:.0f})  |  "
                       f"W={targets[0]:.1f}  S={targets[1]:.1f}  "
                       f"E={targets[2]:.1f}  Wr={targets[3]:.1f}")

    def tick(self):
        while not self._cmd_queue.empty():
            try:
                cmd = self._cmd_queue.get_nowait()
                self.command(cmd)
            except queue.Empty:
                break

        now = time.perf_counter()
        dt  = min(now - self._last_time, 0.1)   
        self._last_time = now

        for s in self.servos:
            s.update(dt)

        pts = self.get_joints()
        tip = pts[-1]
        self.cubes.update(tip)

        if self.target and all(s.is_settled() for s in self.servos):
            self.status = (f"Settled at ({self.target[0]:.0f}, "
                           f"{self.target[1]:.0f}, {self.target[2]:.0f})  |  "
                           f"Tip FK: ({tip[0]:.1f}, {tip[1]:.1f}, "
                           f"{tip[2]:.1f})")

    def get_joints(self):
        return compute_fk(*[s.angle for s in self.servos])


# #####################################################################
#  MAIN  — MATPLOTLIB GUI & ANIMATION
# #####################################################################
def main():
    sim = ArmSimulator()

    plt.rcParams['toolbar'] = 'None'          
    fig = plt.figure(figsize=(16, 9), facecolor='#f5f5f5')
    fig.canvas.manager.set_window_title("4-DOF Robot Arm Simulator")

    ax = fig.add_axes([-0.08, -0.15, 1.16, 1.28], projection='3d', facecolor='#fafafa')
    ax.computed_zorder = False   

    # Joint and Target markers
    joints_plot, = ax.plot([], [], [], 'o', color='#212121', ms=7, zorder=5)
    tip_plot,    = ax.plot([], [], [], 's', color='#E64A19', ms=9, zorder=5)
    tgt_plot,    = ax.plot([], [], [], '*', color='#D32F2F', ms=14, markeredgewidth=0.5, markeredgecolor='black', zorder=6)
    shadow,      = ax.plot([], [], [], '--', color='#bdbdbd', lw=1, alpha=0.5, zorder=1)

    theta = np.linspace(0, 2 * np.pi, 120)
    reach = L1 + L2 + SHOULDER_FORWARD_OFFSET
    ax.plot(reach * np.cos(theta), reach * np.sin(theta), np.zeros_like(theta), ':', color='#e0e0e0', lw=0.8, zorder=1)

    gx_lo, gx_hi = WORK_X_MIN, WORK_X_MAX
    gy_lo, gy_hi = WORK_Y_MIN, WORK_Y_MAX
    grid_step = 50
    floor_cells = []
    xs = list(range(gx_lo, gx_hi, grid_step)) + [gx_hi]
    ys = list(range(gy_lo, gy_hi, grid_step)) + [gy_hi]
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            floor_cells.append([
                (xs[i],   ys[j],   0), (xs[i+1], ys[j],   0),
                (xs[i+1], ys[j+1], 0), (xs[i],   ys[j+1], 0),
            ])
    floor_poly = Poly3DCollection(floor_cells, alpha=0.8, zorder=0, facecolors='#1a1a1a', edgecolors='#ffffff', linewidths=0.6)
    ax.add_collection3d(floor_poly)

    ax.plot(25 * np.cos(theta), 25 * np.sin(theta), np.zeros_like(theta), '-', color='#9e9e9e', lw=1.5, zorder=1)

    ax.set_xlim(WORK_X_MIN, WORK_X_MAX)
    if INVERT_Y:
        ax.set_ylim(WORK_Y_MAX, WORK_Y_MIN)   
    else:
        ax.set_ylim(WORK_Y_MIN, WORK_Y_MAX)
    ax.set_zlim(WORK_Z_MIN, WORK_Z_MAX)
    ax.set_xlabel('X', fontsize=7, labelpad=2)
    ax.set_ylabel('Y', fontsize=7, labelpad=2)
    ax.set_zlabel('Z', fontsize=7, labelpad=2)
    ax.tick_params(labelsize=6, pad=1)
    ax.view_init(elev=22, azim=-55)
    
    x_range = WORK_X_MAX - WORK_X_MIN   
    y_range = WORK_Y_MAX - WORK_Y_MIN   
    z_range = WORK_Z_MAX - WORK_Z_MIN   
    max_range = max(x_range, y_range, z_range)
    ax.set_box_aspect([x_range / max_range, y_range / max_range, z_range / max_range])

    cube_collection = None
    arm_collection = None

    angle_txt  = fig.text(0.005, 0.045, '', fontsize=6.5, va='bottom', ha='left', family='monospace', color='#555555')
    status_txt = fig.text(0.995, 0.045, sim.status, ha='right', fontsize=7, family='monospace', color='#333333')

    ax_input = fig.add_axes([0.04, 0.015, 0.92, 0.025])
    text_box = TextBox(ax_input, '> ', initial='', textalignment='left', label_pad=0.02)
    text_box.label.set_fontsize(8)
    text_box.label.set_fontweight('bold')

    def on_submit(text):
        sim.command(text)
        text_box.set_val('')

    text_box.on_submit(on_submit)

    for keymap in ['keymap.back', 'keymap.forward', 'keymap.fullscreen',
                   'keymap.home', 'keymap.save', 'keymap.quit',
                   'keymap.quit_all', 'keymap.grid', 'keymap.grid_minor',
                   'keymap.yscale', 'keymap.xscale', 'keymap.pan',
                   'keymap.zoom', 'keymap.help']:
        try:
            plt.rcParams[keymap] = []
        except (KeyError, ValueError):
            pass

    def animate(_frame):
        nonlocal cube_collection, arm_collection

        sim.tick()
        pts = sim.get_joints()  # (6, 3)

        # -------------------------------------------------------------
        #  Draw 3D Robot Geometries
        # -------------------------------------------------------------
        if arm_collection is not None:
            arm_collection.remove()
            arm_collection = None

        arm_faces = []
        # Base column
        arm_faces.extend(_prism_between(pts[0], pts[1], 45, 45))
        # Shoulder offset
        arm_faces.extend(_prism_between(pts[1], pts[2], 10, 10))
        # Upper arm (L1) - double width from 35 to 70
        arm_faces.extend(_prism_between(pts[2], pts[3], 70, 25))
        
        # Forearm (L2) - double width from 30 to 60
        arm_faces.extend(_prism_between(pts[3], pts[4], 60, 20))
        
        # Tool Block (L3 structure) + Cylinder Tip
        dir_vec = pts[4] - pts[5]
        tool_len = np.linalg.norm(dir_vec)
        if tool_len > 1e-6:
            dir_norm = dir_vec / tool_len
            p_cyl_top = pts[5] + dir_norm * 20.0  
            # L3 block - double width from 25 to 50
            arm_faces.extend(_prism_between(pts[4], p_cyl_top, 50, 25))
            arm_faces.extend(_cylinder_faces(p_cyl_top, pts[5], radius=12.5, n_sides=16))

        arm_collection = Poly3DCollection(
            arm_faces, alpha=ROBOT_ALPHA, zorder=2,
            facecolors='#7A7A7A', edgecolors='#202020', linewidths=0.6
        )
        ax.add_collection3d(arm_collection)

        # -------------------------------------------------------------
        #  Draw markers & shadows
        # -------------------------------------------------------------
        jx = [pts[2][0], pts[3][0], pts[4][0]]
        jy = [pts[2][1], pts[3][1], pts[4][1]]
        jz = [pts[2][2], pts[3][2], pts[4][2]]
        joints_plot.set_data_3d(jx, jy, jz)
        tip_plot.set_data_3d([pts[5][0]], [pts[5][1]], [pts[5][2]])

        if sim.target:
            tgt_plot.set_data_3d([sim.target[0]], [sim.target[1]], [sim.target[2]])

        sx, sy = pts[:, 0], pts[:, 1]
        sz = np.zeros(len(pts))
        shadow.set_data_3d(sx, sy, sz)

        # -------------------------------------------------------------
        #  Draw cubes
        # -------------------------------------------------------------
        if cube_collection is not None:
            cube_collection.remove()
            cube_collection = None

        all_cubes = sim.cubes.get_all_cubes()
        if all_cubes:
            all_faces = []
            face_colors = []
            for cube in all_cubes:
                faces = _cube_verts(cube['pos'])
                all_faces.extend(faces)
                if cube['attached']:
                    face_colors.extend(['#D4AF37'] * 6)
                else:
                    face_colors.extend(['#E0E0E0'] * 6)

            cube_collection = Poly3DCollection(
                all_faces, alpha=1.0, zorder=4,
                facecolors=face_colors, edgecolors='#404040', linewidths=0.8
            )
            ax.add_collection3d(cube_collection)

        status_txt.set_text(sim.status)
        parts = []
        for s in sim.servos:
            flag = '*' if s.is_moving() else ' '
            parts.append(f"{s.name}:{s.angle:6.1f}{flag}")
        angle_txt.set_text('  |  '.join(parts))

        return (joints_plot, tip_plot, tgt_plot, shadow, status_txt, angle_txt)

    _ani = FuncAnimation(fig, animate, interval=1000 // TARGET_FPS, blit=False, cache_frame_data=False)
    plt.show()

# #####################################################################
if __name__ == '__main__':
    main()