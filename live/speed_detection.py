import math

def calculate_speed(obj_id, cx, cy, last_positions, PIXELS_PER_METER, frame_time, speeds):
    """
    Returns updated speeds dictionary after calculating speed.
    """
    if obj_id in last_positions:
        last_cx, last_cy = last_positions[obj_id]
        dx, dy = cx - last_cx, cy - last_cy
        pixel_dist = math.sqrt(dx**2 + dy**2)
        dist_m = pixel_dist / PIXELS_PER_METER
        speed = (dist_m / frame_time) * 3.6  # km/h
        speeds[obj_id] = 0.8 * speeds.get(obj_id, speed) + 0.2 * speed
    return speeds
