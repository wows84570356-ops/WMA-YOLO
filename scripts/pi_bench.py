# -*- coding: utf-8 -*-
"""Raspberry Pi 5 latency for the four models of the paper. Runs ON THE PI.

    pip install onnxruntime numpy      # if missing
    python3 pi_bench.py                # in the bundle folder; ~25 min
    PI_ROUNDS=1 python3 pi_bench.py    # quick smoke test

Protocol: FP32, CPU (onnxruntime
CPUExecutionProvider), 640x640, batch 1, 10 warm-up runs, 5 rounds x 100 images; latency is the
session.run() call only (normalisation happens before the clock starts).

Extra controls because the table will mix old and new measurements:
  * The first arm is timed again at the end (drift / thermal check).
  * Before each arm the script waits until the SoC is below COOL_C, and records temperature,
    clock and the firmware throttle flags before and after.
Writes pi_latency_<hostname>_<timestamp>.json next to this file.
"""
import json
import os
import platform
import socket
import statistics as st
import subprocess
import sys
import time

import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = ["v11n", "v11n_WMA", "A0", "A4"]
# Same weights as the WMA arms, IDWT written without ScatterND; tells whether the ScatterND lowering
# is a material part of the WMA overhead on this CPU. Not a table row.
EXTRA = ["v11n_WMA_stackidwt", "A4_stackidwt"]
WARMUP = 10
ROUNDS = int(os.environ.get("PI_ROUNDS", "5"))
THREADS = int(os.environ.get("PI_THREADS", "0"))  # 0 = onnxruntime default (all cores)
COOL_C, COOL_MAX_S = 55.0, 300


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception as e:
        return "n/a (%s)" % e


def temp_c():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return int(f.read()) / 1000.0
    except Exception:
        return float("nan")


def state():
    return dict(temp_c=temp_c(), throttled=sh("vcgencmd get_throttled"),
                arm_clock=sh("vcgencmd measure_clock arm"),
                governor=sh("cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"))


def cool_down():
    t0 = time.time()
    while temp_c() > COOL_C and time.time() - t0 < COOL_MAX_S:
        time.sleep(5)
    return round(time.time() - t0, 1)


def bench(name, imgs):
    so = ort.SessionOptions()
    if THREADS:
        so.intra_op_num_threads = THREADS
    sess = ort.InferenceSession(os.path.join(HERE, name + ".onnx"), so, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    waited = cool_down()
    before = state()
    for i in range(WARMUP):
        sess.run(None, {inp: imgs[i % len(imgs)]})
    rounds, all_ms = [], []
    for r in range(ROUNDS):
        ts = []
        for x in imgs:
            t0 = time.perf_counter()
            sess.run(None, {inp: x})
            ts.append((time.perf_counter() - t0) * 1000.0)
        rounds.append(st.mean(ts))
        all_ms += ts
        print("  %-10s round %d  %.2f ms  (%.1f C)" % (name, r + 1, rounds[-1], temp_c()), flush=True)
    after = state()
    m = st.mean(rounds)
    res = dict(mean_ms=m, sd_rounds_ms=st.pstdev(rounds), fps=1000.0 / m, median_ms=st.median(all_ms),
               p95_ms=float(np.percentile(all_ms, 95)), rounds=rounds, cooldown_s=waited,
               before=before, after=after)
    print("%-10s %.2f +- %.2f ms  %.2f FPS  throttled %s" % (name, m, res["sd_rounds_ms"], res["fps"],
                                                           after["throttled"]), flush=True)
    return res


def main():
    raw = np.load(os.path.join(HERE, "images_u8.npy"))
    imgs = [(raw[i:i + 1].astype(np.float32) / 255.0) for i in range(len(raw))]
    missing = [a for a in ARMS + EXTRA if not os.path.exists(os.path.join(HERE, a + ".onnx"))]
    assert not missing, "missing onnx: %s" % missing
    assert len(imgs) == 100, len(imgs)

    out = dict(host=socket.gethostname(), model=sh("cat /proc/device-tree/model | tr -d '\\0'"),
               os=sh("cat /etc/os-release | grep PRETTY"), kernel=platform.release(),
               python=sys.version.split()[0], onnxruntime=ort.__version__, numpy=np.__version__,
               threads=THREADS or "default", protocol=dict(warmup=WARMUP, rounds=ROUNDS, per_round=len(imgs),
                                                          precision="FP32", size=640),
               start=time.strftime("%Y-%m-%d %H:%M:%S"), start_state=state(), arms={})
    print(out["model"], "| onnxruntime", ort.__version__, "| threads", out["threads"], flush=True)
    for a in ARMS:
        out["arms"][a] = bench(a, imgs)
    for a in EXTRA:
        out["arms"][a] = bench(a, imgs)
    out["arms"][ARMS[0] + "_repeat"] = bench(ARMS[0], imgs)
    out["end"] = time.strftime("%Y-%m-%d %H:%M:%S")

    print("\n== summary ==")
    for a, r in out["arms"].items():
        print("%-14s %8.2f ms  %5.2f FPS" % (a, r["mean_ms"], r["fps"]))
    for b, w in (("v11n", "v11n_WMA"), ("A0", "A4")):
        B, W = (out["arms"][k]["mean_ms"] for k in (b, w))
        print("%-5s  WMA vs baseline %+.1f%%" % (b, 100 * (W / B - 1)))
    for w, e in (("v11n_WMA", "v11n_WMA_stackidwt"), ("A4", "A4_stackidwt")):
        print("%-9s ScatterND IDWT %.2f ms vs stack IDWT %.2f ms" % (w, out["arms"][w]["mean_ms"], out["arms"][e]["mean_ms"]))
    drift = out["arms"][ARMS[0] + "_repeat"]["mean_ms"] / out["arms"][ARMS[0]]["mean_ms"] - 1
    print("drift (first arm, start vs end): %+.1f%%" % (100 * drift))

    fn = os.path.join(HERE, "pi_latency_%s_%s.json" % (out["host"], time.strftime("%Y%m%d_%H%M%S")))
    with open(fn, "w") as f:
        json.dump(out, f, indent=1)
    print("->", fn)


if __name__ == "__main__":
    main()
