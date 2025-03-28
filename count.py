import cv2
from ultralytics import YOLO
from collections import defaultdict
import tkinter as tk
from tkinter import filedialog
import numpy as np

# Global Variables
class AppState:
    def __init__(self):
        self.points = []
        self.polygon_selected = False
        self.polygons = []  # To store polygon coordinates for each video/image
        self.temp_image = None
        self.frame = None

# YOLO model setup
model = YOLO("C:\Vehicle-detection-and-tracking-classwise-using-YOLO11-main\yolo11x.pt")
  # Use YOLOv8 large (update to your model)
class_list = model.names

# Function to sort points clockwise
def sort_points_clockwise(pts):
    centroid = np.mean(pts, axis=0)
    sorted_pts = sorted(pts, key=lambda x: np.arctan2(x[1] - centroid[1], x[0] - centroid[0]))
    return sorted_pts

# Function to draw a polygon
def draw_polygon(event, x, y, flags, param):
    state = param
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(state.points) < 4:  # Allow only 4 points
            state.points.append((x, y))
            print(f"Point {len(state.points)}: ({x}, {y})")  # Debug: Print the point
            state.temp_image = state.frame.copy()
            for pt in state.points:
                cv2.circle(state.temp_image, pt, 5, (0, 255, 0), -1)
            if len(state.points) > 1:
                for i in range(1, len(state.points)):
                    cv2.line(state.temp_image, state.points[i-1], state.points[i], (0, 255, 0), 2)
            cv2.imshow("Image", state.temp_image)

            if len(state.points) == 4:  # Polygon is complete
                state.points = sort_points_clockwise(state.points)
                print(f"Sorted Points: {state.points}")  # Debug: Print sorted points
                cv2.polylines(state.temp_image, [np.array(state.points)], isClosed=True, color=(0, 255, 0), thickness=2)
                cv2.imshow("Image", state.temp_image)
                print(f"Polygon selected with points: {state.points}")
                state.polygons.append(state.points[:])
                state.points = []
                state.polygon_selected = True
        else:
            print("Only 4 points can be selected. Polygon is complete.")

# Function to select files
def select_files():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_paths = filedialog.askopenfilenames(
        title="Select 4 video/image files",
        filetypes=[("Video and Image Files", "*.mp4;*.avi;*.mov;*.jpg;*.png;*.jpeg")],
    )
    return list(file_paths)  # Convert tuple to list

# Function to process a file for polygon selection
def process_file(file_path, state):
    state.polygon_selected = False
    state.points = []

    cap = cv2.VideoCapture(file_path) if file_path.endswith(('.mp4', '.avi', '.mov')) else None

    if cap:  # Video
        while True:
            ret, state.frame = cap.read()
            if not ret or state.frame is None or state.frame.size == 0:
                print(f"Skipping corrupted frame in {file_path}")
                continue

            state.temp_image = state.frame.copy()
            cv2.imshow("Image", state.frame)
            cv2.setMouseCallback("Image", draw_polygon, state)
            print(f"Select four points on the image to draw a polygon for {file_path}. Press 'n' to skip frame, 'r' to reset points, any other key to confirm.")

            key = cv2.waitKey(0)
            if key == ord('n'):  # Skip frame
                continue
            elif key == ord('r'):  # Reset points
                state.points = []
                state.temp_image = state.frame.copy()
                cv2.imshow("Image", state.temp_image)
            else:
                break

        cap.release()
    else:  # Image
        state.frame = cv2.imread(file_path)
        if state.frame is None:
            print(f"Error: Unable to load {file_path}.")
            return

        state.temp_image = state.frame.copy()
        cv2.imshow("Image", state.frame)
        cv2.setMouseCallback("Image", draw_polygon, state)
        print(f"Select four points on the image to draw a polygon for {file_path}. Press 'r' to reset points, any other key after selection.")
        key = cv2.waitKey(0)
        if key == ord('r'):  # Reset points
            state.points = []
            state.temp_image = state.frame.copy()
            cv2.imshow("Image", state.temp_image)

    cv2.destroyAllWindows()

    if not state.polygon_selected:
        print(f"No polygon was selected for {file_path}. Skipping.")
        return

    print(f"Polygon saved for {file_path}.")

# Function to check if a bounding box overlaps with a polygon
def is_bbox_in_polygon(bbox, polygon):
    x1, y1, x2, y2 = bbox
    polygon_np = np.array(polygon, dtype=np.int32)

    # Check if any corner of the bounding box is inside the polygon
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    for corner in corners:
        if cv2.pointPolygonTest(polygon_np, corner, False) >= 0:
            return True
    return False

# Function to count vehicles in a video
def count_vehicles(file_path, polygon, num_frames=10):
    cap = cv2.VideoCapture(file_path) if file_path.endswith(('.mp4', '.avi', '.mov')) else None

    if not cap or not cap.isOpened():
        print(f"Error: Unable to open video file {file_path}.")
        return 0  # Return 0 if the file cannot be opened

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate the starting frame to process the last num_frames
    start_frame = max(0, total_frames - num_frames)
    
    # Set the video capture to the starting frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    crossed_ids = set()  # Track IDs of objects inside the polygon
    class_counts = defaultdict(int)  # Count vehicles by class

    frame_count = 0

    while frame_count < num_frames:
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            print(f"Skipping corrupted or unreadable frame in {file_path}")
            continue  # Skip invalid frames

        # Get original dimensions before resizing
        orig_height, orig_width = frame.shape[:2]

        # Resize frame to a consistent size (e.g., 640x480)
        new_width, new_height = 640, 480
        frame = cv2.resize(frame, (new_width, new_height))

        # Scale polygon coordinates to match resized frame
        scale_x = new_width / orig_width
        scale_y = new_height / orig_height
        scaled_polygon = [(int(x * scale_x), int(y * scale_y)) for x, y in polygon]

        frame_count += 1

        # Perform YOLO detection with tracking
        try:
            results = model.track(frame, persist=True, classes=[2, 3, 5, 7])  # Vehicle classes
        except Exception as e:
            print(f"Error during YOLO tracking: {e}")
            continue  # Skip this frame if tracking fails

        if results[0].boxes.data is None or len(results[0].boxes) == 0:
            print("No objects detected in this frame.")
            continue  # Skip frame if no objects are detected

        # Get detected boxes, track IDs, and classes
        boxes = results[0].boxes.xyxy.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else []
        class_indices = results[0].boxes.cls.int().cpu().tolist()

        # Draw the counting polygon
        cv2.polylines(frame, [np.array(scaled_polygon)], isClosed=True, color=(0, 255, 0), thickness=2)

        for box, track_id, class_idx in zip(boxes, track_ids, class_indices):
            x1, y1, x2, y2 = map(int, box)
            class_name = class_list[class_idx]

            # Check if the bounding box overlaps with the polygon
            if is_bbox_in_polygon((x1, y1, x2, y2), scaled_polygon):
                if track_id not in crossed_ids:
                    crossed_ids.add(track_id)
                    class_counts[class_name] += 1

                # Draw bounding box and ID
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{class_name}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Display counts on the frame
        y_offset = 30
        for class_name, count in class_counts.items():
            cv2.putText(frame, f"{class_name}: {count}", (50, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            y_offset += 30

        cv2.imshow(f"Counting: {file_path}", frame)

        # Break the loop if 'q' is pressed (quit)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Quit the entire program
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()  # Ensure all windows are closed after the loop

    # Return the total count of vehicles
    return sum(class_counts.values())

# Main Execution
if __name__ == "__main__":
    print("Please select 4 video files.")
    files = select_files()

    if not files or len(files) != 4:
        print("Error: You must select exactly 4 files.")
        exit()

    print("Selected files:", files)  # Debugging: Print selected files

    state = AppState()
    for file in files:
        process_file(file, state)

    print("Polygons selected for all files:")
    for i, polygon in enumerate(state.polygons):
        print(f"File {i + 1}: {polygon}")

    # Save polygon coordinates
    with open("polygons.txt", "w") as f:
        for i, polygon in enumerate(state.polygons):
            f.write(f"File {i + 1}: {polygon}\n")

    print("Polygon coordinates saved to polygons.txt.")

    # Perform vehicle counting for each video
    
    # Store vehicle counts for each video
    vehicle_counts = []  # Global list to store vehicle counts for each camera

    
    for i, file in enumerate(files):
        print(f"Processing file {i + 1}: {file}")
        count = count_vehicles(file, state.polygons[i], num_frames=10)  # Get vehicle count
        vehicle_counts.append(count)  # Store count in list
        print(f"Total vehicles counted in {file}: {count}")
    
    print("Final vehicle counts:", vehicle_counts)  # Debugging


    # Determine which video has the most vehicles
    max_count = max(vehicle_counts)
    max_index = vehicle_counts.index(max_count)
    print(f"Video {max_index + 1} has the most vehicles with a count of {max_count}.")
    
    # Print all counts
    for i, count in enumerate(vehicle_counts):
        print(f"Video {i + 1}: {count} vehicles")
