"""
Make the model file about half the size by storing the weights in float16.
GitHub does not accept files over 100 MB (ConvNeXt-Tiny is 111 MB -> about 56 MB after this).

The app still runs in float32: when the app loads the file, the weights are converted back.
The script checks that the predictions stay the same on the sample images.

Usage (from the project folder):
    python scripts/shrink_model.py path/to/downloaded/best_model.pth
It writes model/best_model.pth
"""
import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.predictor import TumorPredictor  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("src", help="the full-size .pth downloaded from Kaggle")
ap.add_argument("--out", default=str(ROOT / "model" / "best_model.pth"))
args = ap.parse_args()

src, out = Path(args.src), Path(args.out)
ckpt = torch.load(src, map_location="cpu", weights_only=False)
# Halve only the big weight tensors (conv / linear, 2+ dimensions). Small 1-D tensors such as
# batch-norm statistics stay float32, because they lose too much precision in float16.
ckpt["state_dict"] = {k: (v.half() if v.is_floating_point() and v.dim() >= 2 else v)
                      for k, v in ckpt["state_dict"].items()}
ckpt["stored_as"] = "float16 weights, float32 norm layers"
tmp = out.with_suffix(".tmp.pth")
torch.save(ckpt, tmp)

# Check: same predictions with the full and the small file
samples = sorted((ROOT / "data" / "samples").glob("*/*.jpg"))[:: max(1, len(sorted((ROOT / "data" / "samples").glob('*/*.jpg'))) // 40)]
full, small = TumorPredictor(src), TumorPredictor(tmp)
same, same_refer, max_diff = 0, 0, 0.0
for p in samples:
    a, b = full.predict(p, with_heatmap=False), small.predict(p, with_heatmap=False)
    same += a.label == b.label
    same_refer += a.refer == b.refer
    max_diff = max(max_diff, max(abs(a.probs[c] - b.probs[c]) for c in a.probs))
n = len(samples)
print(f"Model: {ckpt['model_name']} | checked {n} sample images")
print(f"Same prediction: {same}/{n} | same 'refer' decision: {same_refer}/{n} | "
      f"biggest probability difference: {max_diff:.4f}")
# What matters for the app: identical predictions and identical safety-rule decisions.
if same != n or same_refer != n or max_diff > 0.03:
    tmp.unlink()
    sys.exit("Predictions changed too much — keeping the original file. Ask Claude.")

tmp.replace(out)
print(f"Saved {out}  ({src.stat().st_size / 1e6:.0f} MB -> {out.stat().st_size / 1e6:.0f} MB)")
