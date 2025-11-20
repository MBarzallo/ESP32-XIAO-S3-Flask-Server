import cv2
import numpy as np
from collections import deque

N_FRAMES = 40

bg_buffer = deque(maxlen=N_FRAMES)
BG_READY = False
BG_MEDIAN = None

def update_background(gray_frame):
    global BG_READY, BG_MEDIAN

    bg_buffer.append(gray_frame.copy())

    if len(bg_buffer) < N_FRAMES:
        return

    BG_MEDIAN = np.median(np.array(bg_buffer), axis=0).astype(np.uint8)
    BG_READY = True
    
def remove_background(gray_frame):
    global BG_READY, BG_MEDIAN

    update_background(gray_frame)

    if not BG_READY:
        empty = np.zeros_like(gray_frame)
        return empty, empty

    gray_blur = cv2.GaussianBlur(gray_frame, (9,9), 0)
    bg_blur   = cv2.GaussianBlur(BG_MEDIAN,   (9,9), 0)

    diff = cv2.absdiff(gray_blur, bg_blur)

    t = np.mean(diff) * 1.2
    _, mask = cv2.threshold(diff, t, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=1)

    fg = cv2.bitwise_and(gray_frame, gray_frame, mask=mask)

    return mask, fg
