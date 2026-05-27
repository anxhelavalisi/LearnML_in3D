import numpy as np
import argparse
from game_client import RoomBot
from drive2win import nn as nn_mod
from drive2win.normalize import normalize_states

ap = argparse.ArgumentParser()
ap.add_argument("--room", required=True)
ap.add_argument("--name", default="Anxhi")
args = ap.parse_args()

w = nn_mod.load("nav_v25-balanced.npz")
tick = [0]

def controller(obs):
    nav = obs["navigation"]
    raw = np.array([
        obs["speed"],
        nav["heading_error"],
        nav["distance"],
        *obs["rays"],
        obs["ground_friction"],
    ], dtype=np.float32)
    features = normalize_states(raw[None, :])[0]
    throttle, steering = nn_mod.forward(features, w)
    tick[0] += 1
    if tick[0] % 20 == 0:
        print(f"tick={tick[0]} spd={obs['speed']:.1f} t={throttle:.2f} s={steering:.2f} phase={obs['race_phase']}")
    return float(np.clip(throttle, -1, 1)), float(np.clip(steering, -1, 1))

bot = RoomBot("https://ml.ferit.tech", room=args.room, name=args.name)
standings = bot.run(controller, hz=20.0)
print(standings)
