# -*- coding: utf-8 -*-
r"""Build the Raspberry Pi 5 benchmark bundle for the four models of the paper (first training run each).

Run on a workstation, from the repository root:
    python scripts/export_onnx.py /path/to/bridge20/test/images

Produces in bundle/
    <arm>.onnx      FP32, batch 1, 640x640, static shape, opset 17 (from the trained checkpoint)
    images_u8.npy   100 bridge20 test images, letterboxed to 640, uint8 NCHW (content does not change
                    the timing of a static graph, but real images keep the protocol identical to the
                    original "five rounds of 100 images")
    manifest.json   sha256 of every file, parameter count, torch-vs-onnxruntime max abs diff
"""
import hashlib
import json
import os
import shutil
import sys

FORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the repository root (this fork)
sys.path.insert(0, FORK)

import numpy as np
import torch
import ultralytics
from ultralytics import YOLO

assert os.path.normcase(os.path.dirname(os.path.dirname(ultralytics.__file__))) == os.path.normcase(os.path.normpath(FORK)), \
    "imported site-packages ultralytics instead of the fork: %s" % ultralytics.__file__

WEIGHTS = os.path.join(FORK, "weights")
ARMS = ["v11n", "v11n_WMA", "A0", "A4"]
OUT = os.path.join(FORK, "bundle")
TEST = sys.argv[1] if len(sys.argv) > 1 else "bridge20/test/images"
PT = {"A0": "yolo26n_baseline_run0.pt", "A4": "yolo26n_wma_run0.pt",
      "v11n": "yolov11n_baseline_run0.pt", "v11n_WMA": "yolov11n_wma_run0.pt"}
N_IMG, SZ = 100, 640


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def letterbox(path):
    import cv2
    im = cv2.imread(path)
    h, w = im.shape[:2]
    r = SZ / max(h, w)
    nh, nw = round(h * r), round(w * r)
    im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((SZ, SZ, 3), 114, np.uint8)
    top, left = (SZ - nh) // 2, (SZ - nw) // 2
    canvas[top:top + nh, left:left + nw] = im
    return canvas[:, :, ::-1].transpose(2, 0, 1)  # BGR->RGB, HWC->CHW


def _haar_idwt_stack(x):
    """Same arithmetic as conv._haar_idwt, interleaving with stack+flatten instead of strided writes
    into torch.empty() (which export as ScatterND). Used only for the optional *_stackidwt graphs."""
    C = x.shape[1] // 4
    LL, LH, HL, HH = x[:, 0:C], x[:, C:2 * C], x[:, 2 * C:3 * C], x[:, 3 * C:4 * C]
    L = torch.stack(((LL + LH) * 0.5, (LL - LH) * 0.5), dim=-1).flatten(-2)
    Hi = torch.stack(((HL + HH) * 0.5, (HL - HH) * 0.5), dim=-1).flatten(-2)
    return torch.stack((L + Hi, L - Hi), dim=-2).flatten(-3, -2)


def export_and_check(net, n_wma, imgs, dst):
    """torch.onnx.export (TorchScript exporter) + op-count guard + onnxruntime-vs-torch check.

    Not YOLO.export(): on this fork it silently drops every WaveletAttention (the WMA arms came out
    with the baseline's 2.586 M initialisers and no LayerNorm ReduceMean), so the file would time
    the baseline under the WMA name."""
    import onnx
    import onnxruntime as ort
    x = torch.from_numpy(imgs[:1].astype(np.float32) / 255.0)
    torch.onnx.export(net, x, dst, opset_version=17, input_names=["images"], output_names=["output0"],
                      dynamo=False, do_constant_folding=True)
    g = onnx.load(dst)
    n_rm = sum(nd.op_type == "ReduceMean" for nd in g.graph.node)
    assert n_rm == 2 * n_wma, (dst, "ReduceMean", n_rm, "expected", 2 * n_wma)
    sess = ort.InferenceSession(dst, providers=["CPUExecutionProvider"])
    diff = 0.0
    for i in range(0, len(imgs), 10):
        xi = imgs[i:i + 1].astype(np.float32) / 255.0
        with torch.no_grad():
            ref = net(torch.from_numpy(xi))
        ref = (ref[0] if isinstance(ref, (list, tuple)) else ref).numpy()[0]
        got = sess.run(None, {"images": xi})[0][0]
        assert got.shape == ref.shape, (dst, got.shape, ref.shape)
        if got.shape[-1] == 6:  # end-to-end (300, 6): top-k order can swap near-ties, compare sets
            ref, got = ref[ref[:, 4] > 0.25], got[got[:, 4] > 0.25]
            assert len(ref) == len(got), (dst, i, "detections", len(ref), len(got))
            ref, got = ref[np.lexsort(ref.T[::-1])], got[np.lexsort(got.T[::-1])]
        if len(ref):
            diff = max(diff, float(np.abs(got - ref).max()))
    assert diff < 1e-2, (dst, "onnx differs from torch", diff)
    return dict(onnx=os.path.basename(dst), sha256=sha(dst), output_shape=list(got.shape),
                reducemean=n_rm, scatternd=sum(nd.op_type == "ScatterND" for nd in g.graph.node),
                max_abs_diff_vs_torch=diff)


def main():
    sys.stdout.reconfigure(encoding="utf8", errors="replace")
    import copy
    import onnxruntime as ort
    import ultralytics.nn.modules.conv as conv
    os.makedirs(OUT, exist_ok=True)

    files = sorted(f for f in os.listdir(TEST) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    assert len(files) >= N_IMG, len(files)
    imgs = np.stack([letterbox(os.path.join(TEST, f)) for f in files[:N_IMG]])
    assert imgs.shape == (N_IMG, 3, SZ, SZ) and imgs.dtype == np.uint8
    np.save(OUT + "/images_u8.npy", imgs)

    man = dict(images=dict(n=N_IMG, source=TEST, first=files[0], last=files[N_IMG - 1],
                           sha256=sha(OUT + "/images_u8.npy")),
               torch=torch.__version__, onnxruntime_export_check=ort.__version__,
               exporter="torch.onnx.export(dynamo=False, opset 17), fused BN, Detect.export=True", arms={})
    t = torch.randn(2, 16, 10, 12)
    assert torch.equal(conv._haar_idwt(t), _haar_idwt_stack(t)), "stack IDWT is not identical"
    orig_idwt = conv._haar_idwt

    for a in ARMS:
        pt = os.path.join(WEIGHTS, PT[a])
        m = YOLO(pt)
        n_par = sum(p.numel() for p in m.model.parameters())
        net = copy.deepcopy(m.model).fuse().eval().float().cpu()
        for mm in net.modules():  # what the Ultralytics exporter sets on the head
            if hasattr(mm, "export"):
                mm.export = True
            if hasattr(mm, "format"):
                mm.format = "onnx"
        wma = [mm for mm in net.modules() if type(mm).__name__ == "WaveletAttention"]
        assert all(type(mm).__module__ == conv.__name__ for mm in wma), "WaveletAttention not from conv.py"
        assert (len(wma) > 0) == ("WMA" in a or a == "A4"), (a, len(wma))

        conv._haar_idwt = orig_idwt
        info = export_and_check(net, len(wma), imgs, "%s/%s.onnx" % (OUT, a))
        info.update(checkpoint=pt, params_M=round(n_par / 1e6, 4), n_wma=len(wma))
        man["arms"][a] = info
        print("%-10s %.2f M  WMA x%d  out %s  ScatterND %d  max|onnx-torch| %.2e"
              % (a, n_par / 1e6, len(wma), info["output_shape"], info["scatternd"], info["max_abs_diff_vs_torch"]))
        if wma:  # optional second graph: same weights, IDWT without ScatterND
            conv._haar_idwt = _haar_idwt_stack
            s = export_and_check(net, len(wma), imgs, "%s/%s_stackidwt.onnx" % (OUT, a))
            man["arms"][a + "_stackidwt"] = s
            print("%-10s stack IDWT  ScatterND %d  max|onnx-torch| %.2e" % (a, s["scatternd"], s["max_abs_diff_vs_torch"]))
            conv._haar_idwt = orig_idwt

    assert len([k for k in man["arms"] if not k.endswith("_stackidwt")]) == len(ARMS)
    shutil.copy(os.path.join(FORK, "scripts", "pi_bench.py"), OUT + "/pi_bench.py")
    with open(OUT + "/manifest.json", "w", encoding="utf8") as f:
        json.dump(man, f, indent=1)
    print("bundle ->", OUT)


if __name__ == "__main__":
    main()
