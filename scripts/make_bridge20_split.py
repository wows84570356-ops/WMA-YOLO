# -*- coding: utf-8 -*-
"""Bridge-level stratified 20-class train/val/test split (70/10/20).

Population: the source train + val folders merged (7,037 images, 27,669 boxes = paper Table 1).
Group unit: bridge_id = first token of the file name; every photo of a bridge (all years) goes to ONE split,
so no bridge appears in two splits.  Stratification: greedy iterative
stratification over bridges, rarest class first, so every class has >= 1 bridge in each split and instance
shares are close to 70/10/20.  md5-identical images are inside the same bridge, hence never cross splits.
Output (hard links, source untouched): damage_detection_img_yolo/bridge20/{train,val,test}/{images,labels},
data.yaml, split_report.txt, {train,val,test}_images.txt (release lists).  Deterministic (SEED).
"""
import os, io, sys, random, shutil, hashlib, collections
D = sys.argv[1] if len(sys.argv) > 1 else "."  # folder that holds source/ (train, val) and receives bridge20/
SRC = os.path.join(D, "source"); OUT = os.path.join(D, "bridge20")
SEED = 0; RATIO = {"train": 0.70, "val": 0.10, "test": 0.20}
NAMES = ["crack","corrosion","paint degradation","lime","delamination","rebar corrosion","floating","pavement crack",
         "poor drainage","bearing mortar damage","joint damage","pavement damage","water accumulation/leakage","deformation",
         "appurtenant structures damage","concrete deformation/defect","bearing damage","sediment accumulation","bolt falling off","movement of abutment"]
SPL = list(RATIO)

imgs = {}
for sp in ("train", "val"):
    for f in os.listdir(os.path.join(SRC, sp, "images")):
        st = os.path.splitext(f)[0]; lp = os.path.join(SRC, sp, "labels", st + ".txt")
        cls = [int(float(l.split()[0])) for l in io.open(lp, encoding="utf8", errors="ignore") if l.strip()] if os.path.exists(lp) else []
        imgs[st] = (os.path.join(SRC, sp, "images", f), lp if os.path.exists(lp) else None, cls)
tot = collections.Counter(c for _, _, cls in imgs.values() for c in cls); assert sum(tot.values()) == 27669
bridges = collections.defaultdict(list)
for st in imgs: bridges[st.split("_")[0]].append(st)
bcnt = {b: collections.Counter(c for st in sts for c in imgs[st][2]) for b, sts in bridges.items()}
# md5 duplicates must share a bridge
md = collections.defaultdict(set)
for st, (ip, _, _) in imgs.items(): md[hashlib.md5(open(ip, "rb").read()).hexdigest()].add(st.split("_")[0])
assert all(len(v) == 1 for v in md.values()), "duplicate image across bridges"

rng = random.Random(SEED)
assign = {}; cur = {s: collections.Counter() for s in SPL}; nimg = collections.Counter()
order = sorted(range(20), key=lambda c: tot[c])                     # rarest class first
for c in order:
    bs = [b for b in bridges if b not in assign and bcnt[b][c] > 0]; rng.shuffle(bs)
    bs.sort(key=lambda b: -bcnt[b][c])                              # big contributors first
    for b in bs:
        # split with the largest remaining deficit for class c; tie -> largest overall image deficit
        def deficit(s): return RATIO[s] * tot[c] - cur[s][c]
        def imgdef(s): return RATIO[s] * len(imgs) - nimg[s]
        s = max(SPL, key=lambda s: (round(deficit(s) / max(tot[c], 1), 4), imgdef(s)))
        assign[b] = s; cur[s].update(bcnt[b]); nimg[s] += len(bridges[b])
for b in bridges:                                                    # background-only bridges
    if b not in assign:
        s = max(SPL, key=lambda s: RATIO[s] * len(imgs) - nimg[s]); assign[b] = s; nimg[s] += len(bridges[b])

# ---- refinement: hill-climb bridge moves to bring every class's share close to 70/10/20 (coverage kept) ----
N = len(imgs)
def cost():
    c1 = sum(abs(cur[s][c] / tot[c] - RATIO[s]) for c in range(20) for s in SPL)
    c2 = sum(abs(nimg[s] / N - RATIO[s]) for s in SPL)
    return c1 + 4 * c2
def covered():
    return all(any(bcnt[b][c] > 0 and assign[b] == s for b in bridges) for c in range(20) for s in SPL)
best = cost(); blist = list(bridges); rng2 = random.Random(SEED + 1)
for it in range(60000):
    b = rng2.choice(blist); s0 = assign[b]; s1 = rng2.choice([s for s in SPL if s != s0])
    assign[b] = s1; cur[s0].subtract(bcnt[b]); cur[s1].update(bcnt[b]); nimg[s0] -= len(bridges[b]); nimg[s1] += len(bridges[b])
    c = cost()
    if c < best and covered(): best = c
    else:
        assign[b] = s0; cur[s1].subtract(bcnt[b]); cur[s0].update(bcnt[b]); nimg[s1] -= len(bridges[b]); nimg[s0] += len(bridges[b])
print("refined cost", round(best, 4))

if os.path.exists(OUT): shutil.rmtree(OUT)
lists = {s: [] for s in SPL}; nbg = collections.Counter()
for s in SPL: os.makedirs(os.path.join(OUT, s, "images")); os.makedirs(os.path.join(OUT, s, "labels"))
for st, (ip, lp, cls) in imgs.items():
    s = assign[st.split("_")[0]]; lists[s].append(st)
    os.link(ip, os.path.join(OUT, s, "images", os.path.basename(ip)))
    if lp: os.link(lp, os.path.join(OUT, s, "labels", st + ".txt"))
    else: nbg[s] += 1
for s in SPL: io.open(os.path.join(OUT, f"{s}_images.txt"), "w", encoding="utf8").write("\n".join(sorted(lists[s])) + "\n")
io.open(os.path.join(OUT, "data.yaml"), "w", encoding="utf8").write(
    "# 20-class bridge-level stratified split 70/10/20 (seed 0). One bridge = one split, all years together.\n"
    "# Built by make_bridge20_split.py from the source train+val (7,037 images, 27,669 boxes). 2026-09-03.\n"
    f"path: {OUT.replace(os.sep, '/')}\ntrain: train/images\nval: val/images\ntest: test/images\nnc: 20\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES)))
nb = {s: sum(1 for b in bridges if assign[b] == s) for s in SPL}
lines = [f"bridge20 split  seed={SEED}  ratios={RATIO}", f"bridges: " + ", ".join(f"{s}={nb[s]}" for s in SPL) + f"  (total {len(bridges)})",
         "images:  " + ", ".join(f"{s}={len(lists[s])} (bg {nbg[s]})" for s in SPL), "",
         f"{'id':>2} {'class':30s} {'total':>6} | {'train':>6} {'val':>5} {'test':>5} | {'val%':>5} {'test%':>5} | bridges tr/va/te"]
bad = []
for c, n in enumerate(NAMES):
    bb = [sum(1 for b in bridges if assign[b] == s and bcnt[b][c] > 0) for s in SPL]
    if min(bb) == 0: bad.append(n)
    lines.append(f"{c:>2} {n:30s} {tot[c]:>6} | {cur['train'][c]:>6} {cur['val'][c]:>5} {cur['test'][c]:>5} | "
                 f"{100*cur['val'][c]/tot[c]:>4.1f}% {100*cur['test'][c]/tot[c]:>4.1f}% | {bb[0]}/{bb[1]}/{bb[2]}")
lines.append(""); lines.append("classes missing from some split: " + (", ".join(bad) if bad else "none"))
rep = "\n".join(lines); io.open(os.path.join(OUT, "split_report.txt"), "w", encoding="utf8").write(rep + "\n"); print(rep)
