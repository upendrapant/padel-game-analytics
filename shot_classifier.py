class ShotClassifier:
    def __init__(self):
        # Dictionary to maintain state for each tracked player
        # track_id -> { "prev_lw": (x,y), "prev_rw": (x,y), "cooldown": int, "last_shot": str, "last_shot_frames": int }
        self.player_states = {}
        # We'll use a velocity threshold to detect a swing (event detection trigger)
        self.VEL_THRESHOLD = 20.0 
        # Ball proximity threshold to ensure player is actually hitting the ball
        self.BALL_PROXIMITY_THRESHOLD = 200.0

    def classify_shots(self, detections, ball_coords, frame_num, fps=30.0):
        shot_events = []
        for det in detections:
            track_id = det.get("track_id")
            if track_id is None:
                continue

            if track_id not in self.player_states:
                self.player_states[track_id] = {
                    "prev_lw": None,
                    "prev_rw": None,
                    "cooldown": 0,
                    "last_shot": None,
                    "last_shot_frames": 0
                }

            state = self.player_states[track_id]

            if state["cooldown"] > 0:
                state["cooldown"] -= 1
            if state["last_shot_frames"] > 0:
                state["last_shot_frames"] -= 1
            else:
                state["last_shot"] = None

            kps = det.get("keypoints")
            if not kps:
                det["shot_type"] = state["last_shot"]
                continue

            def get_kp(idx):
                if idx < len(kps) and kps[idx]["visibility"] > 0.5:
                    return kps[idx]
                return None

            ls = get_kp(11) # Left Shoulder
            rs = get_kp(12) # Right Shoulder
            lw = get_kp(15) # Left Wrist
            rw = get_kp(16) # Right Wrist
            lh = get_kp(23) # Left Hip
            rh = get_kp(24) # Right Hip

            if not (ls and rs and lw and rw and lh and rh):
                self._update_prev_wrists(state, lw, rw)
                det["shot_type"] = state["last_shot"]
                continue

            midline_x = (lh["x"] + rh["x"]) / 2.0

            # Calculate wrist velocities
            lw_vel = 0
            rw_vel = 0
            if state["prev_lw"]:
                lw_vel = ((lw["x"] - state["prev_lw"]["x"])**2 + (lw["y"] - state["prev_lw"]["y"])**2)**0.5
            if state["prev_rw"]:
                rw_vel = ((rw["x"] - state["prev_rw"]["x"])**2 + (rw["y"] - state["prev_rw"]["y"])**2)**0.5

            self._update_prev_wrists(state, lw, rw)

            active_wrist = None
            active_vel = 0
            active_shoulder = None

            if lw_vel > rw_vel and lw_vel > self.VEL_THRESHOLD:
                active_wrist = lw
                active_vel = lw_vel
                active_shoulder = ls
            elif rw_vel > lw_vel and rw_vel > self.VEL_THRESHOLD:
                active_wrist = rw
                active_vel = rw_vel
                active_shoulder = rs

            # Event trigger: velocity spike + ball proximity
            is_valid_shot = False
            if state["cooldown"] == 0 and active_wrist is not None:
                if ball_coords is not None:
                    dist_to_ball = ((active_wrist["x"] - ball_coords[0])**2 + (active_wrist["y"] - ball_coords[1])**2)**0.5
                    if dist_to_ball < self.BALL_PROXIMITY_THRESHOLD:
                        is_valid_shot = True

            if is_valid_shot:
                shot_type = "Forehand" # Default fallback

                # 1. Smash: Wrist Y significantly less than shoulder Y (smaller Y = higher)
                # pyrefly: ignore [unsupported-operation]
                if active_wrist["y"] < active_shoulder["y"] - 30:
                    shot_type = "Smash"
                else:
                    # 2. Backhand: Active wrist crosses to the non-dominant side of body midline
                    # Assuming Right-Handed player: right wrist crosses to the left side (smaller X)
                    # pyrefly: ignore [unsupported-operation]
                    if active_wrist["x"] < midline_x:
                        shot_type = "Backhand"
                    else:
                        # 3. Forehand: Active wrist on dominant side
                        shot_type = "Forehand"

                state["last_shot"] = shot_type
                state["cooldown"] = 20 # Cooldown to avoid multiple classifications for the same swing
                state["last_shot_frames"] = 30 # Show the text for 30 frames
                print(f"Player {track_id} hit a {shot_type}!")
                
                shot_events.append({
                    "frame": frame_num,
                    "timestamp": round(frame_num / fps, 2),
                    "track_id": track_id,
                    "shot_type": shot_type
                })

            det["shot_type"] = state["last_shot"]

        return shot_events

    def _update_prev_wrists(self, state, lw, rw):
        if lw:
            state["prev_lw"] = {"x": lw["x"], "y": lw["y"]}
        if rw:
            state["prev_rw"] = {"x": rw["x"], "y": rw["y"]}
