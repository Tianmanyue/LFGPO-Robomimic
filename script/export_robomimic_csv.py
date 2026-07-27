#!/usr/bin/env python3
"""Export Robomimic text logs to one plotting-friendly CSV file."""

import argparse
import csv
import re
from pathlib import Path


STAMP = re.compile(r"^\[(?P<time>[^]]+)].*?\[INFO] - (?P<msg>.*)$")
EVAL = re.compile(
    r"eval: success rate\s+(?P<success>[-+\d.e]+) \| avg episode reward\s+"
    r"(?P<reward>[-+\d.e]+) \| avg best reward\s+(?P<best>[-+\d.e]+)"
)
TRAIN = re.compile(
    r"(?P<itr>\d+): step\s+(?P<step>\d+).*?reward\s+(?P<reward>[-+\d.e]+)"
)
REINFLOW_ITR = re.compile(r"itr\s+(?P<itr>\d+) \| Total Step\s+(?P<step>[\d.]+)\s*(?P<unit>[MK]?)")
NUMBER = r"([-+\d.e]+)"


def scaled_step(value, unit):
    scale = {"": 1, "K": 1_000, "M": 1_000_000}[unit]
    return int(float(value) * scale)


def rows_from_log(path):
    rows, pending = [], None
    eval_index = 0
    for raw in path.read_text(errors="replace").splitlines():
        match = STAMP.match(raw)
        timestamp, msg = (match.group("time"), match.group("msg")) if match else ("", raw)
        if m := EVAL.search(msg):
            rows.append((timestamp, "eval", eval_index, "", m["success"], m["reward"], m["best"]))
            eval_index += 1
        elif m := TRAIN.search(msg):
            rows.append((timestamp, "train", m["itr"], m["step"], "", m["reward"], ""))
        elif m := REINFLOW_ITR.search(msg):
            pending = [timestamp, "train", m["itr"], scaled_step(m["step"], m["unit"]), "", "", ""]
        elif pending and (m := re.search(r"Episode Reward:\s*" + NUMBER, msg)):
            pending[5] = m.group(1)
        elif pending and (m := re.search(r"Success Rate:\s*" + NUMBER + r"%", msg)):
            pending[4] = float(m.group(1)) / 100.0
        elif pending and (m := re.search(r"(?:Avg )?Best Reward(?: \(per action\))?:\s*" + NUMBER, msg)):
            pending[6] = m.group(1)
            rows.append(tuple(pending))
            pending = None
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("runtime/log/robomimic/finetune"))
    parser.add_argument("--output", type=Path, default=Path("experiment_metrics.csv"))
    args = parser.parse_args()
    fields = ["run", "timestamp", "phase", "iteration", "env_steps", "success_rate",
              "effective_success_rate", "avg_episode_reward", "avg_best_reward"]
    with args.output.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        for log in sorted(args.root.glob("**/run.log")):
            run = str(log.parent.relative_to(args.root))
            for row in rows_from_log(log):
                timestamp, phase, iteration, steps, success, reward, best = row
                # Existing jobs used the broken divided best-reward metric. For
                # these sparse 0/1 Robomimic tasks, mean episode reward is the
                # exact empirical success rate and preserves historical runs.
                effective_success = reward if phase == "eval" else ""
                writer.writerow((run, timestamp, phase, iteration, steps, success,
                                 effective_success, reward, best))


if __name__ == "__main__":
    main()
