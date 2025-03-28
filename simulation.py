import cv2
import tkinter as tk
from tkinter import filedialog
import numpy as np
import threading
import time
import random
import pygame
import sys
import os

# ---------------- GLOBAL VARIABLES & STATE ----------------
class AppState:
    def __init__(self):
        self.points = []
        self.polygon_selected = False
        self.polygons = []  # Stores polygon coordinates for each video/image
        self.temp_image = None
        self.frame = None

# Global variable for detection counts for each of the 4 lanes.
detection_counts = [3, 5, 2, 4]

# Global exit event to gracefully shut down threads.
exit_event = threading.Event()

# -------------- UTILITY FUNCTIONS --------------
def sort_points_clockwise(pts):
    centroid = np.mean(pts, axis=0)
    return sorted(pts, key=lambda x: np.arctan2(x[1]-centroid[1], x[0]-centroid[0]))

def draw_polygon(event, x, y, flags, param):
    state = param
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(state.points) < 4:
            state.points.append((x, y))
            print(f"Point {len(state.points)}: ({x}, {y})")
            state.temp_image = state.frame.copy()
            for pt in state.points:
                cv2.circle(state.temp_image, pt, 5, (0,255,0), -1)
            if len(state.points) > 1:
                for i in range(1, len(state.points)):
                    cv2.line(state.temp_image, state.points[i-1], state.points[i], (0,255,0), 2)
            cv2.imshow("Image", state.temp_image)
            if len(state.points) == 4:
                state.points = sort_points_clockwise(state.points)
                print(f"Sorted Points: {state.points}")
                cv2.polylines(state.temp_image, [np.array(state.points)], isClosed=True, color=(0,255,0), thickness=2)
                cv2.imshow("Image", state.temp_image)
                print(f"Polygon selected with points: {state.points}")
                state.polygons.append(state.points[:])
                state.points = []
                state.polygon_selected = True
        else:
            print("Only 4 points can be selected. Polygon is complete.")

def select_files():
    print("Simulated file selection - no actual files needed")
    return ["simulated_file_1.mp4", "simulated_file_2.mp4", "simulated_file_3.mp4", "simulated_file_4.mp4"]

def process_file(file_path, state):
    default_polygons = [
        [(100,100), (200,100), (200,200), (100,200)],  # Lane 0
        [(300,100), (400,100), (400,200), (300,200)],  # Lane 1
        [(300,300), (400,300), (400,400), (300,400)],  # Lane 2
        [(100,300), (200,300), (200,400), (100,400)]   # Lane 3
    ]
    file_index = int(file_path.split("_")[-1].split(".")[0]) - 1
    state.polygon_selected = True
    state.polygons.append(default_polygons[file_index])
    print(f"Default polygon created for {file_path}: {default_polygons[file_index]}")

def is_bbox_in_polygon(bbox, polygon):
    x1,y1,x2,y2 = bbox
    polygon_np = np.array(polygon, dtype=np.int32)
    for corner in [(x1,y1),(x2,y1),(x2,y2),(x1,y2)]:
        if cv2.pointPolygonTest(polygon_np, corner, False) >= 0:
            return True
    return False

# ---------------- CONSTANTS FOR SIMULATION ----------------
SIM_WIDTH, SIM_HEIGHT = 800, 600
FPS = 60

# Timing constants:
DYNAMIC_MIN_GREEN = 10.0       # Base green time (sec)
EXTRA_GREEN_PER_VEHICLE = 1.0  # Extra sec per vehicle detected
YELLOW_DURATION = 3.0          # Yellow duration (sec)
MAX_GREEN = 50.0               # Maximum allowed green time (sec)

# Vehicle & lane constants:
VEHICLE_SIZE = 28
SPACING = 50
STOP_LINE_OFFSET = 70
SPAWN_OFFSET = 50
UP_LANE_X_OFFSET    = -45
DOWN_LANE_X_OFFSET  = -2
RIGHT_LANE_Y_OFFSET = -52
LEFT_LANE_Y_OFFSET  = -15
SPEED = 30.0

# ---------------- PYGAME ASSET LOADING ----------------
def load_signal_images():
    signals = {}
    try:
        signals["red"] = pygame.image.load("red.png").convert_alpha()
        signals["green"] = pygame.image.load("green.png").convert_alpha()
        signals["yellow"] = pygame.image.load("yellow.png").convert_alpha()
    except pygame.error:
        print("Signal images not found, creating placeholder signals")
        signals["red"] = pygame.Surface((40,40)); signals["red"].fill((255,0,0))
        signals["green"] = pygame.Surface((40,40)); signals["green"].fill((0,255,0))
        signals["yellow"] = pygame.Surface((40,40)); signals["yellow"].fill((255,255,0))
    for key in signals:
        signals[key] = pygame.transform.scale(signals[key], (40,40))
    return signals

def load_and_scale_image(filename, size=(VEHICLE_SIZE,VEHICLE_SIZE), color=(255,0,0)):
    try:
        img = pygame.image.load(filename).convert_alpha()
        return pygame.transform.scale(img, size)
    except pygame.error:
        print(f"Image {filename} not found, creating placeholder")
        surf = pygame.Surface(size, pygame.SRCALPHA)
        pygame.draw.rect(surf, color, (0,0,size[0],size[1]))
        return surf

def load_vehicle_images():
    car_color = (200,0,0)
    bike_color = (0,200,0)
    bus_color = (0,0,200)
    truck_color = (200,200,0)
    rickshaw_color = (200,0,200)
    vehicle_imgs = {
        0: [load_and_scale_image("upcar.png", color=car_color),
            load_and_scale_image("upbike.png", color=bike_color),
            load_and_scale_image("upbus.png", color=bus_color),
            load_and_scale_image("uptruck.png", color=truck_color),
            load_and_scale_image("uprickshaw.png", color=rickshaw_color)],
        1: [load_and_scale_image("rightcar.png", color=car_color),
            load_and_scale_image("rightbike.png", color=bike_color),
            load_and_scale_image("rightbus.png", color=bus_color),
            load_and_scale_image("rightrickshaw.png", color=rickshaw_color)],
        2: [load_and_scale_image("downcar.png", color=car_color),
            load_and_scale_image("downbike.png", color=bike_color),
            load_and_scale_image("downbus.png", color=bus_color),
            load_and_scale_image("downtruck.png", color=truck_color),
            load_and_scale_image("downrickshaw.png", color=rickshaw_color)],
        3: [load_and_scale_image("leftcar.png", color=car_color),
            load_and_scale_image("leftbike.png", color=bike_color),
            load_and_scale_image("leftbus.png", color=bus_color),
            load_and_scale_image("lefttruck.png", color=truck_color),
            load_and_scale_image("leftrickshaw.png", color=rickshaw_color)]
    }
    return vehicle_imgs

def already_passed_stop_line(stop_lines, lane, vx, vy):
    if lane == 0:  # Up
        return vy < stop_lines[0]
    elif lane == 1:  # Right
        return vx > stop_lines[1]
    elif lane == 2:  # Down
        return vy > stop_lines[2]
    elif lane == 3:  # Left
        return vx < stop_lines[3]
    return False

# ---------------- DYNAMIC TRAFFIC SIMULATION CLASS ----------------
class TrafficSimDynamicUpdated:
    def __init__(self, surface, vehicle_imgs, initial_counts, signal_imgs):
        self.surface = surface
        self.vehicle_imgs = vehicle_imgs
        self.width = SIM_WIDTH
        self.height = SIM_HEIGHT
        self.signal_imgs = signal_imgs

        self.start_time = time.time()
        self.cleared_time = None
        self.cleared_count = 0

        # Signal image positions.
        self.signal_positions = {
            0: (self.width//2 - 20, 18),
            1: (self.width - 60, self.height//2 - 20),
            2: (self.width//2 - 20, self.height - 60),
            3: (20, self.height//2 - 20)
        }
        self.stop_lines = [
            self.height/2 + STOP_LINE_OFFSET,
            self.width/2 - STOP_LINE_OFFSET,
            self.height/2 - STOP_LINE_OFFSET,
            self.width/2 + STOP_LINE_OFFSET
        ]
        # Initialize lane vehicles.
        self.lanes = [[], [], [], []]
        self.init_lane_vehicles(initial_counts)

        # --- Compute fixed cycle ordering & green time allocations ---
        # Each lane’s green time is computed as:
        #     green = min(DYNAMIC_MIN_GREEN + (# vehicles)*EXTRA_GREEN_PER_VEHICLE, MAX_GREEN)
        self.lane_order = sorted(range(4), key=lambda i: len(self.lanes[i]), reverse=True)
        self.order_index = 0  # current index in lane_order
        self.cycle_allocations = [min(DYNAMIC_MIN_GREEN + len(self.lanes[i]) * EXTRA_GREEN_PER_VEHICLE, MAX_GREEN) for i in range(4)]
        self.current_lane = self.lane_order[self.order_index]
        self.current_green_duration = self.cycle_allocations[self.current_lane]
        self.light_state = "green"  # "green" or "yellow"
        self.phase_start_time = time.time()

        self.last_update_time = time.time()
        self.update_display_count = 0

    def init_lane_vehicles(self, counts):
        # Lane 0: Up
        baseY = self.stop_lines[0] + SPAWN_OFFSET
        up_lane_x = (self.width/2) + UP_LANE_X_OFFSET
        for i in range(counts[0]):
            img = random.choice(self.vehicle_imgs[0])
            self.lanes[0].append({"img": img, "x": up_lane_x, "y": baseY + i * SPACING})
        # Lane 1: Right
        baseX = self.stop_lines[1] - SPAWN_OFFSET
        right_lane_y = (self.height/2) + RIGHT_LANE_Y_OFFSET
        for i in range(counts[1]):
            img = random.choice(self.vehicle_imgs[1])
            self.lanes[1].append({"img": img, "x": baseX - i * SPACING, "y": right_lane_y})
        # Lane 2: Down
        baseY = self.stop_lines[2] - SPAWN_OFFSET
        down_lane_x = (self.width/2) + DOWN_LANE_X_OFFSET
        for i in range(counts[2]):
            img = random.choice(self.vehicle_imgs[2])
            self.lanes[2].append({"img": img, "x": down_lane_x, "y": baseY - i * SPACING})
        # Lane 3: Left
        baseX = self.stop_lines[3] + SPAWN_OFFSET
        left_lane_y = (self.height/2) + LEFT_LANE_Y_OFFSET
        for i in range(counts[3]):
            img = random.choice(self.vehicle_imgs[3])
            self.lanes[3].append({"img": img, "x": baseX + i * SPACING, "y": left_lane_y})
        print(f"Initialized lanes with: {[len(lane) for lane in self.lanes]} vehicles")

    def update_lane_vehicles(self, new_counts):
        # Append new vehicles (fixed cycle allocations do not update mid-cycle)
        current_time = time.time()
        if current_time - self.last_update_time < 2.0:
            return
        self.last_update_time = current_time
        self.update_display_count += 1
        print(f"Update #{self.update_display_count}: Updating lanes with counts: {new_counts}")
        print(f"Current lane counts: {[len(lane) for lane in self.lanes]}")
        for lane in range(4):
            current_count = len(self.lanes[lane])
            if new_counts[lane] > current_count:
                diff = new_counts[lane] - current_count
                print(f"Adding {diff} vehicles to lane {lane}")
                if lane == 0:
                    up_lane_x = (self.width/2) + UP_LANE_X_OFFSET
                    last_y = self.lanes[0][-1]["y"] if self.lanes[0] else self.stop_lines[0] + SPAWN_OFFSET
                    for i in range(diff):
                        img = random.choice(self.vehicle_imgs[0])
                        new_y = last_y + SPACING
                        self.lanes[0].append({"img": img, "x": up_lane_x, "y": new_y})
                        last_y = new_y
                elif lane == 1:
                    right_lane_y = (self.height/2) + RIGHT_LANE_Y_OFFSET
                    last_x = self.lanes[1][-1]["x"] if self.lanes[1] else self.stop_lines[1] - SPAWN_OFFSET
                    for i in range(diff):
                        img = random.choice(self.vehicle_imgs[1])
                        new_x = last_x - SPACING
                        self.lanes[1].append({"img": img, "x": new_x, "y": right_lane_y})
                        last_x = new_x
                elif lane == 2:
                    down_lane_x = (self.width/2) + DOWN_LANE_X_OFFSET
                    last_y = self.lanes[2][-1]["y"] if self.lanes[2] else self.stop_lines[2] - SPAWN_OFFSET
                    for i in range(diff):
                        img = random.choice(self.vehicle_imgs[2])
                        new_y = last_y - SPACING
                        self.lanes[2].append({"img": img, "x": down_lane_x, "y": new_y})
                        last_y = new_y
                elif lane == 3:
                    left_lane_y = (self.height/2) + LEFT_LANE_Y_OFFSET
                    last_x = self.lanes[3][-1]["x"] if self.lanes[3] else self.stop_lines[3] + SPAWN_OFFSET
                    for i in range(diff):
                        img = random.choice(self.vehicle_imgs[3])
                        new_x = last_x + SPACING
                        self.lanes[3].append({"img": img, "x": new_x, "y": left_lane_y})
                        last_x = new_x
        print(f"After update: {[len(lane) for lane in self.lanes]} vehicles")

    def select_next_lane(self):
        # Advance in the fixed cycle order.
        self.order_index += 1
        if self.order_index >= 4:
            print("Cycle complete. Recalculating cycle order and fixed green times based on current counts.")
            # Recalculate order and fixed green times (capped at MAX_GREEN)
            self.lane_order = sorted(range(4), key=lambda i: len(self.lanes[i]), reverse=True)
            self.cycle_allocations = [min(DYNAMIC_MIN_GREEN + len(self.lanes[i]) * EXTRA_GREEN_PER_VEHICLE, MAX_GREEN) for i in range(4)]
            self.order_index = 0
        return self.lane_order[self.order_index]

    def compute_red_timer(self, target_lane):
        """
        Compute the red timer for a waiting lane (target_lane) as the time until its green phase starts.
        For the immediate next lane:
          - If the active lane is in green, red = (remaining green time) + YELLOW_DURATION.
          - If in yellow, red = remaining yellow time.
        For lanes further down, add the full (green+yellow) durations of intermediate lanes.
        """
        if target_lane == self.current_lane:
            return 0
        current_time = time.time()
        if self.light_state == "green":
            rem_current = max(0, self.current_green_duration - (current_time - self.phase_start_time)) + YELLOW_DURATION
        else:
            rem_current = max(0, YELLOW_DURATION - (current_time - self.phase_start_time))
        pos_current = self.order_index
        pos_target = self.lane_order.index(target_lane)
        total = 0
        if pos_target > pos_current:
            total = rem_current + sum(self.cycle_allocations[j] + YELLOW_DURATION for j in range(pos_current+1, pos_target))
        else:
            total = rem_current + sum(self.cycle_allocations[j] + YELLOW_DURATION for j in range(pos_current+1, len(self.lane_order)))
            total += sum(self.cycle_allocations[j] + YELLOW_DURATION for j in range(0, pos_target))
        return total

    def update(self, dt):
        current_time = time.time()
        elapsed = current_time - self.phase_start_time

        # Signal phase management.
        if self.light_state == "green":
            if elapsed >= self.current_green_duration:
                self.light_state = "yellow"
                self.phase_start_time = current_time
                print(f"Lane {self.current_lane} changing to YELLOW")
        elif self.light_state == "yellow":
            if elapsed >= YELLOW_DURATION:
                self.current_lane = self.select_next_lane()
                self.current_green_duration = self.cycle_allocations[self.current_lane]
                self.light_state = "green"
                self.phase_start_time = current_time
                print(f"Lane {self.current_lane} changing to GREEN for {self.current_green_duration:.2f}s")

        # Move vehicles.
        for lane in range(4):
            for v in self.lanes[lane]:
                can_move = False
                if lane == self.current_lane:
                    if self.light_state == "green":
                        can_move = True
                    elif self.light_state == "yellow":
                        if lane == 0 and v["y"] <= self.stop_lines[0]:
                            can_move = True
                        elif lane == 1 and v["x"] >= self.stop_lines[1]:
                            can_move = True
                        elif lane == 2 and v["y"] >= self.stop_lines[2]:
                            can_move = True
                        elif lane == 3 and v["x"] <= self.stop_lines[3]:
                            can_move = True
                else:
                    if already_passed_stop_line(self.stop_lines, lane, v["x"], v["y"]):
                        can_move = True
                if can_move:
                    if lane == 0:
                        v["y"] -= SPEED * dt
                    elif lane == 1:
                        v["x"] += SPEED * dt
                    elif lane == 2:
                        v["y"] += SPEED * dt
                    elif lane == 3:
                        v["x"] -= SPEED * dt

        self.remove_offscreen()
        if self.cleared_time is None and sum(len(lane) for lane in self.lanes) == 0:
            self.cleared_time = current_time - self.start_time

    def remove_offscreen(self):
        new_lanes = [[], [], [], []]
        for lane in range(4):
            current_lane_vehicles = len(self.lanes[lane])
            kept = []
            for v in self.lanes[lane]:
                if lane == 0 and v["y"] + VEHICLE_SIZE > 0:
                    kept.append(v)
                elif lane == 1 and v["x"] < self.width:
                    kept.append(v)
                elif lane == 2 and v["y"] < self.height:
                    kept.append(v)
                elif lane == 3 and v["x"] + VEHICLE_SIZE > 0:
                    kept.append(v)
            new_lanes[lane] = kept
            vehicles_cleared = current_lane_vehicles - len(kept)
            if vehicles_cleared > 0:
                print(f"{vehicles_cleared} vehicles cleared from lane {lane}")
                self.cleared_count += vehicles_cleared
        self.lanes = new_lanes

    def draw(self, bg_image):
        self.surface.blit(bg_image, (0, 0))
        font = pygame.font.SysFont(None, 24)
        y_pos = 60
        for lane in range(4):
            lane_count_text = f"Lane {lane}: {len(self.lanes[lane])} vehicles"
            lane_surface = font.render(lane_count_text, True, (255,255,255))
            self.surface.blit(lane_surface, (10, y_pos))
            y_pos += 25
        for lane in range(4):
            for v in self.lanes[lane]:
                self.surface.blit(v["img"], (v["x"], v["y"]))
        for lane in range(4):
            if lane == self.current_lane:
                if self.light_state == "green":
                    sig = self.signal_imgs["green"]
                    remaining_time = max(0, self.current_green_duration - (time.time()-self.phase_start_time))
                    text_color = (0,255,0)
                else:
                    sig = self.signal_imgs["yellow"]
                    remaining_time = max(0, YELLOW_DURATION - (time.time()-self.phase_start_time))
                    text_color = (255,255,0)
                remaining_text = f"{int(remaining_time)}s"
            else:
                sig = self.signal_imgs["red"]
                red_timer = self.compute_red_timer(lane)
                remaining_text = f"{int(red_timer)}s" if red_timer > 0 else "GO"
                text_color = (255,0,0)
            self.surface.blit(sig, self.signal_positions[lane])
            font_big = pygame.font.SysFont(None, 32)
            text_surface = font_big.render(remaining_text, True, text_color)
            text_x = self.signal_positions[lane][0] + 10
            text_y = self.signal_positions[lane][1] + 45
            padding = 4
            text_width, text_height = text_surface.get_size()
            rect = pygame.Rect(text_x - padding, text_y - padding, text_width + 2*padding, text_height + 2*padding)
            pygame.draw.rect(self.surface, (0,0,0), rect)
            self.surface.blit(text_surface, (text_x, text_y))
        font_small = pygame.font.SysFont(None, 24)
        info_x = self.width - 250
        info_y = 10
        cleared_message = f"Cleared: {self.cleared_count}"
        cleared_txt = font_small.render(cleared_message, True, (255,255,0))
        self.surface.blit(cleared_txt, (info_x, info_y))
        active_message = f"Active Lane: {self.current_lane} ({self.light_state.upper()})"
        active_txt = font_small.render(active_message, True, (255,255,255))
        self.surface.blit(active_txt, (info_x, info_y+25))

# ---------------- SIMULATED DETECTION THREAD ----------------
def simulate_detection_thread():
    """Simulates vehicle detection by randomly updating counts."""
    global detection_counts
    print("Starting simulated detection thread")
    while not exit_event.is_set():
        for i in range(4):
            detection_counts[i] += random.randint(0,3)
        print(f"Updated simulated detection counts: {detection_counts}")
        time.sleep(3)

# ---------------- MAIN LOOP ----------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((SIM_WIDTH, SIM_HEIGHT))
    pygame.display.set_caption("Dynamic Traffic Simulation")
    clock = pygame.time.Clock()
    try:
        bg_image = pygame.image.load("mod_int.png").convert()
        bg_image = pygame.transform.scale(bg_image, (SIM_WIDTH, SIM_HEIGHT))
    except pygame.error:
        print("Background image not found, creating a simple one")
        bg_image = pygame.Surface((SIM_WIDTH, SIM_HEIGHT))
        bg_image.fill((50,50,50))
        road_color = (80,80,80)
        line_color = (255,255,255)
        pygame.draw.rect(bg_image, road_color, (SIM_WIDTH//2 - 50, 0, 100, SIM_HEIGHT))
        pygame.draw.rect(bg_image, road_color, (0, SIM_HEIGHT//2 - 50, SIM_WIDTH, 100))
        for i in range(0, SIM_WIDTH, 40):
            pygame.draw.rect(bg_image, line_color, (i, SIM_HEIGHT//2, 20, 2))
        for i in range(0, SIM_HEIGHT, 40):
            pygame.draw.rect(bg_image, line_color, (SIM_WIDTH//2, i, 2, 20))
    vehicle_imgs = load_vehicle_images()
    signal_imgs = load_signal_images()
    print("Simulation mode - using default files and polygons")
    files = select_files()
    state = AppState()
    for file in files:
        process_file(file, state)
    print("Default polygons created for all files")
    simulation_thread = threading.Thread(target=simulate_detection_thread, daemon=True)
    simulation_thread.start()
    print("Started simulation thread")
    initial_counts = detection_counts[:]
    print(f"Initial counts: {initial_counts}")
    sim_dynamic = TrafficSimDynamicUpdated(screen, vehicle_imgs, initial_counts, signal_imgs)
    running = True
    print("Starting main simulation loop")
    while running:
        dt = clock.tick(FPS)/1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                exit_event.set()
        sim_dynamic.update_lane_vehicles(detection_counts)
        sim_dynamic.update(dt)
        sim_dynamic.draw(bg_image)
        pygame.display.flip()
    simulation_thread.join(timeout=1.0)
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        pygame.quit()
        sys.exit(1)
