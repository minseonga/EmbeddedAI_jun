import numpy as np
from collections import deque
import time

class GestureDetector:
    """
    Detects hand gestures:
    1. Fist: Capture trigger
    2. Swipe (Left/Right): Gallery navigation
    """
    def __init__(self, history_len=10, swipe_thresh=50, cooldown=1.0):
        self.history_len = history_len
        self.swipe_thresh = swipe_thresh
        self.cooldown = cooldown
        
        # History for Swipe
        self.x_history = deque(maxlen=history_len)
        
        # State
        self.last_capture_time = 0
        self.last_swipe_time = 0
        self.is_fist_held = False
        
        # Keypoint indices
        self.TIPS = [4, 8, 12, 16, 20]
        self.PIPS = [2, 6, 10, 14, 18] # PIP or MCP (depending on joint def)
        # MobileHand/MANO joints: 
        # 0:Wrist
        # Thumb: 1, 2, 3, 4
        # Index: 5, 6, 7, 8
        # Middle: 9, 10, 11, 12
        # Ring: 13, 14, 15, 16
        # Pinky: 17, 18, 19, 20
        # PIPs are typically indices: 2, 6, 10, 14, 18? 
        # Actually:
        # Index: 5(MCP), 6(PIP), 7(DIP), 8(TIP)
        # So we compare TIP(8) with PIP(6) or MCP(5). Using PIP(6) is safer.
        self.FINGER_TIPS = [8, 12, 16, 20]
        self.FINGER_PIPS = [6, 10, 14, 18] 

    def update(self, landmarks):
        """
        Update state with new landmarks.
        Returns: gesture_name (str or None)
        """
        if landmarks is None or len(landmarks) == 0:
            self.x_history.clear()
            return None
            
        # Assuming single hand for now (landmarks[0])
        # kpts: (21, 3)
        kpts = landmarks[0] # Take first hand
        
        now = time.time()
        
        # 1. Detect Fist
        # Condition: 4 fingers (Index~Pinky) tips are below PIPs (y coordinate higher because y increases downwards)
        # Or simply check if Tip is "lower" in screen (higher Y value) than PIP?
        # Wait, usually "curled" means Tip is closer to Wrist/Palm.
        # Simple heuristic: Tip.y > PIP.y (assuming hand is upright).
        # Better heuristic: Distance from Wrist(0) to Tip < Threshold? 
        # Let's use: Tip.y > PIP.y check (Folded)
        # Note: In screen coords, Y increases downwards.
        # Upright hand: Wrist at bottom (High Y), Fingers at top (Low Y).
        # Open hand: Tip.y < PIP.y
        # Fist: Tip.y > PIP.y (folded down)
        
        isfist = True
        for tip, pip in zip(self.FINGER_TIPS, self.FINGER_PIPS):
            if kpts[tip][1] < kpts[pip][1]: # If Tip is above PIP (Open)
                isfist = False
                break
        
        gesture = None
        
        # Fist Trigger
        if isfist:
            if not self.is_fist_held and (now - self.last_capture_time > self.cooldown):
                gesture = "FIST"
                self.last_capture_time = now
                self.is_fist_held = True
        else:
            self.is_fist_held = False
            
        # 2. Detect Swipe (only if not fist)
        if not isfist:
            wrist_x = kpts[0][0]
            self.x_history.append(wrist_x)
            
            if len(self.x_history) == self.history_len and (now - self.last_swipe_time > self.cooldown):
                # Calculate simple velocity
                dx = self.x_history[-1] - self.x_history[0]
                
                if dx > self.swipe_thresh:
                    gesture = "SWIPE_RIGHT"
                    self.last_swipe_time = now
                    self.x_history.clear()
                elif dx < -self.swipe_thresh:
                    gesture = "SWIPE_LEFT"
                    self.last_swipe_time = now
                    self.x_history.clear()
                    
        return gesture
