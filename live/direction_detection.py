import cv2
import os

vehicle_sides = {}       # last side per vehicle
vehicle_sequence = {}    # true crossing sequence
violated_vehicles = set()  # track vehicles already saved

# Make sure the folder exists
os.makedirs("violations/direction", exist_ok=True)

def check_vehicle_direction(obj_id, cy, lines, allowed_direction, frame, box, original_frame=None):
    """
    Track vehicle line-crossing sequence.
    Draw green box for normal, red box if sequence >1 and violates allowed direction.
    If violation occurs, save cropped vehicle image (only once per vehicle).
    """
    x1, y1, x2, y2 = map(int, box)
    cx = (x1 + x2) // 2

    # Initialize if first time
    if obj_id not in vehicle_sides:
        initial_sides = []
        for p1, p2 in lines:
            A = p2[1] - p1[1]
            B = p1[0] - p2[0]
            C = p2[0]*p1[1] - p1[0]*p2[1]
            side = 1 if (A*cx + B*cy + C) >= 0 else -1
            initial_sides.append(side)

        vehicle_sides[obj_id] = initial_sides
        vehicle_sequence[obj_id] = []
        return

    # Compute current sides
    new_sides = []
    for p1, p2 in lines:
        A = p2[1] - p1[1]
        B = p1[0] - p2[0]
        C = p2[0]*p1[1] - p1[0]*p2[1]
        side = 1 if (A*cx + B*cy + C) >= 0 else -1
        new_sides.append(side)

    # Detect crossings (side change)
    for idx, (old, new) in enumerate(zip(vehicle_sides[obj_id], new_sides)):
        if old != new:
            vehicle_sequence[obj_id].append(idx + 1)

    # Update last sides
    vehicle_sides[obj_id] = new_sides

    # Full sequence
    seq = vehicle_sequence[obj_id]

    # Determine violation
    draw_red = False
    if len(seq) >= 2:
        if allowed_direction is not None:
            if seq[-2:] != allowed_direction:
                draw_red = True
        else:
            draw_red = True

    # Draw box
    if draw_red:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 3)
        cv2.putText(frame, "Direction Violation", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        # --- Save cropped original frame (only once per vehicle) ---
        if obj_id not in violated_vehicles and original_frame is not None:
            crop = original_frame[y1:y2, x1:x2]
            filename = f"violations/direction/vehicle_{obj_id}.jpg"
            cv2.imwrite(filename, crop)
            violated_vehicles.add(obj_id)
