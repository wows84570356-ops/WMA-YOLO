# -*- coding: utf-8 -*-
"""x86 CPU latency of the four models of the paper (Table 11), one thread, FP32, 640x640, batch 1.

Protocol: 10 warm-up passes, then 5 rounds of 30 images; the reported value is the mean over rounds
(the paper reports the mean and standard deviation over two such sessions). Only the network forward
pass is timed.

    python scripts/cpu_latency.py
"""
import json
import os
import statistics as st
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402

torch.set_num_threads(1)
MODELS = {"YOLOv11n baseline": "yolov11n_baseline_run0.pt", "YOLOv11n + WMA": "yolov11n_wma_run0.pt",
          "YOLO26n baseline": "yolo26n_baseline_run0.pt", "YOLO26n + WMA": "yolo26n_wma_run0.pt"}
WARMUP, ROUNDS, PER_ROUND, SZ = 10, 5, 30, 640


def time_model(path):
    net = YOLO(path).model.eval().float()
    x = torch.rand(1, 3, SZ, SZ)
    with torch.inference_mode():
        for _ in range(WARMUP):
            net(x)
        rounds = []
        for _ in range(ROUNDS):
            t = []
            for _ in range(PER_ROUND):
                t0 = time.perf_counter()
                net(x)
                t.append((time.perf_counter() - t0) * 1000)
            rounds.append(st.mean(t))
    return st.mean(rounds), st.stdev(rounds)


if __name__ == "__main__":
    res = {}
    for name, f in MODELS.items():
        mean, sd = time_model(os.path.join(ROOT, "weights", f))
        res[name] = dict(mean_ms=mean, sd_ms=sd)
        print("%-18s %7.1f ms  (sd over rounds %.1f)" % (name, mean, sd))
    for det in ("YOLOv11n", "YOLO26n"):
        b, w = res[det + " baseline"]["mean_ms"], res[det + " + WMA"]["mean_ms"]
        print("%-9s WMA adds %.1f ms (%+.0f%%)" % (det, w - b, 100 * (w / b - 1)))
    with open(os.path.join(ROOT, "cpu_latency_results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
