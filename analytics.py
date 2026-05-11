import json
import os
import pandas as pd
from collections import defaultdict

class ShotLogger:
    def __init__(self):
        self.shots = []   # list of all shot event dicts

    def log(self, shot_events):
        """Call this every frame with the list returned by classifier.process()"""
        if shot_events:
            self.shots.extend(shot_events)

    def save(self, output_dir="output/"):
        os.makedirs(output_dir, exist_ok=True)
        
        # save self.shots as JSON
        json_path = os.path.join(output_dir, "shots.json")
        with open(json_path, "w") as f:
            json.dump(self.shots, f, indent=2)
            
        # convert self.shots to DataFrame and save as CSV
        csv_path = os.path.join(output_dir, "shots.csv")
        if self.shots:
            df = pd.DataFrame(self.shots)
            df.to_csv(csv_path, index=False)
        else:
            pd.DataFrame(columns=["frame", "timestamp", "track_id", "shot_type"]).to_csv(csv_path, index=False)

    def get_analytics(self):
        total_shots = len(self.shots)
        by_type = defaultdict(int)
        by_player = defaultdict(int)
        by_player_and_type = defaultdict(lambda: defaultdict(int))
        
        for shot in self.shots:
            by_type[shot["shot_type"]] += 1
            by_player[shot["track_id"]] += 1
            by_player_and_type[shot["track_id"]][shot["shot_type"]] += 1
            
        return {
            "total_shots": total_shots,
            "by_type": dict(by_type),
            "by_player": dict(by_player),
            "by_player_and_type": {player: dict(types) for player, types in by_player_and_type.items()}
        }


class Analytics:
    def __init__(self, shots):
        self.df = pd.DataFrame(shots) if shots else pd.DataFrame(columns=["frame", "timestamp", "track_id", "shot_type"])

    def shot_counts_by_type(self):
        if self.df.empty:
            return pd.Series(dtype=int)
        return self.df['shot_type'].value_counts()

    def shot_counts_by_player(self):
        if self.df.empty:
            return pd.Series(dtype=int)
        return self.df['track_id'].value_counts()

    def shots_per_minute(self):
        if self.df.empty or self.df['timestamp'].max() == 0:
            return 0.0
        max_timestamp = self.df['timestamp'].max()
        return len(self.df) / (max_timestamp / 60.0)

    def print_summary(self):
        print("\n=== SHOT ANALYTICS ===")
        print(f"Total shots detected: {len(self.df)}")
        print("\nBy type:")
        print(self.shot_counts_by_type().to_string() if not self.df.empty else "No shots")
        print("\nBy player:")
        print(self.shot_counts_by_player().to_string() if not self.df.empty else "No shots")
        print(f"\nShots per minute: {self.shots_per_minute():.1f}")
