import cv2


def check_direction_violation(obj_id, cx, cy, last_positions, directions, frame, x1, y1):
    """
    Checks for direction violation and annotates frame.
    """
    if obj_id in last_positions:
        last_cx, last_cy = last_positions[obj_id]
        dy = cy - last_cy
        direction = "Forward" if dy > 0 else "Reverse"
        if obj_id in directions and directions[obj_id] != direction:
            cv2.putText(frame, "DIRECTION VIOLATION", (x1, y1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        directions[obj_id] = direction
    return directions
