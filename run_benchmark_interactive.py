"""
Wrapper per benchmark qe pret browserin para se te filloje.
"""
import time
import json
from pathlib import Path
from game_client import GameClient
from drive2win.normalize import sensors_to_input, clip_action
from drive2win import nn as nn_mod
from drive2win.eval import run_policy, score_runs
from drive2win import viz
import argparse
import numpy as np

TARGET_CHECKPOINTS = 8

def make_policy(weights_path):
    w = nn_mod.load(weights_path)
    def policy(state):
        x = sensors_to_input(state["sensors"])
        return clip_action(nn_mod.forward(x, w))
    return policy

def run_one(weights, seed, runs=5, duration=60.0):
    policy = make_policy(weights)
    client = GameClient("https://ml.ferit.tech")
    runs_out = []

    for i in range(runs):
        session = client.create_session(
            mode="time_trial",
            player_name=f"benchmark_run{i+1}",
            config={"seed": seed, "wind_enabled": False},
        )
        print(f"\n  run {i+1}/{runs} — HAP BROWSERIN:")
        print(f"  {session['browser_url']}")
        input("  Pasi shohesh botin shtyp ENTER...")

        client.connect_ws()
        time.sleep(2.0)

        result = run_policy(client, policy, duration=duration, hz=20.0)
        print(f"    checkpoints={result['checkpoints_passed']}/{TARGET_CHECKPOINTS}  crashes={result['crashes']}")
        runs_out.append(result)
        client.disconnect_ws()
        try: client.delete_session()
        except: pass

    return runs_out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--duration", type=float, default=60.0)
    args = ap.parse_args()

    weights = f"nav_{args.tag}.npz"
    out_dir = Path("benchmarks"); out_dir.mkdir(exist_ok=True)
    all_results = []

    for seed in args.seeds:
        print(f"\n=== seed {seed} ===")
        runs_out = run_one(weights, seed, args.runs, args.duration)
        summary = score_runs(runs_out, TARGET_CHECKPOINTS)
        all_results.append({"seed": seed, "summary": summary, "runs": runs_out})

    print("\n" + "=" * 56)
    print(f"  iteration: {args.tag}    weights: {weights}")
    for r in all_results:
        s = r["summary"]
        print(f"  seed {r['seed']:>4}  complete={int(s['completion_rate']*s['n_runs'])}/{s['n_runs']}  "
              f"median_lap={s['median_lap_time']:.1f}s  crashes={s['mean_crashes']:.1f}  "
              f"max_cp={s['max_checkpoints']}")
    print("=" * 56)

    log = {"tag": args.tag, "weights": weights, "runs_per_seed": args.runs,
           "duration_s": args.duration,
           "seeds": [{"seed": r["seed"], "summary": r["summary"], "runs": r["runs"]} for r in all_results]}
    log_path = out_dir / f"{args.tag}.json"
    log_path.write_text(json.dumps(log, indent=2, default=float))
    print(f"\nwrote {log_path}")

    flat_runs = [run for r in all_results for run in r["runs"]]
    viz.plot_multi_run_paths(flat_runs, out=str(out_dir / f"{args.tag}_paths.png"), title=f"All paths — {args.tag}")
    viz.plot_checkpoint_progress(flat_runs, out=str(out_dir / f"{args.tag}_progress.png"))

if __name__ == "__main__":
    main()
