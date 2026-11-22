"""Evaluate the deployed model (model/best_model.pth) on a test folder.

Uses the app's own TumorPredictor, so preprocessing, temperature scaling and the
review rule are exactly the same as on the website.

Usage (from the project root):
    python scripts/eval_final_model.py --data /kaggle/input/brain-tumor-mri-dataset/Testing --out eval_out

The data folder must contain one sub-folder per class: glioma, meningioma, notumor, pituitary.
Outputs in --out:
    confusion_matrix.png      counts + row-normalised (recall) matrices
    reliability_diagram.png   calibration plot (confidence vs accuracy)
    per_class_metrics.csv     precision, recall, F1, support per class
    summary.json              accuracy, macro-F1, ECE, missed / silent missed tumors, false alarms
    predictions.csv           one row per image
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.predictor import TumorPredictor  # noqa: E402

NAMES = {"glioma": "Glioma", "meningioma": "Meningioma", "notumor": "No tumor", "pituitary": "Pituitary"}


def ece_score(conf, correct, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    ece, rows = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            gap = abs(correct[m].mean() - conf[m].mean())
            ece += m.mean() * gap
            rows.append((lo, hi, conf[m].mean(), correct[m].mean(), int(m.sum())))
    return ece, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default=str(ROOT / "model" / "best_model.pth"))
    ap.add_argument("--out", default="eval_out")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    pr = TumorPredictor(a.model)
    classes = list(pr.class_names)
    rows = []
    files = [(c, f) for c in classes for f in sorted((Path(a.data) / c).glob("*"))
             if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    print(f"{len(files)} images, classes = {classes}")
    for i, (c, f) in enumerate(files, 1):
        r = pr.predict(Image.open(f).convert("RGB"), with_heatmap=False)
        rows.append(dict(file=f.name, true=c, pred=r.label, conf=r.confidence, p_tumor=r.p_tumor,
                         priority=r.priority, refer=bool(r.refer), **{f"p_{k}": v for k, v in r.probs.items()}))
        if i % 200 == 0:
            print(f"  {i}/{len(files)}")
    df = pd.DataFrame(rows); df.to_csv(out / "predictions.csv", index=False)

    y, p = df.true.values, df.pred.values
    cm = confusion_matrix(y, p, labels=classes)
    prec, rec, f1, sup = precision_recall_fscore_support(y, p, labels=classes, zero_division=0)
    pd.DataFrame({"class": [NAMES[c] for c in classes], "precision": prec.round(4), "recall": rec.round(4),
                  "f1": f1.round(4), "support": sup}).to_csv(out / "per_class_metrics.csv", index=False)

    tumor = df.true != "notumor"
    missed = tumor & (df.pred == "notumor")
    silent = missed & (~df.refer)
    correct = (df.true == df.pred).values.astype(float)
    ece, bins = ece_score(df.conf.values, correct)
    summary = dict(images=len(df), accuracy=round(accuracy_score(y, p), 4),
                   macro_f1=round(f1_score(y, p, labels=classes, average="macro"), 4),
                   ece_15_bins=round(ece, 4), tumor_images=int(tumor.sum()),
                   missed_tumors=int(missed.sum()), silent_missed_tumors=int(silent.sum()),
                   false_alarms=int(((~tumor) & (df.pred != "notumor")).sum()),
                   flagged=int(df.refer.sum()), errors_flagged=int((df.refer & (correct == 0)).sum()),
                   errors=int((correct == 0).sum()))
    json.dump(summary, open(out / "summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))
    print(pd.read_csv(out / "per_class_metrics.csv").to_string(index=False))

    # confusion matrices
    labels = [NAMES[c] for c in classes]
    cmn = cm / cm.sum(1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=200)
    for ax, mat, title, fmt in [(axes[0], cm, "Counts", "d"), (axes[1], cmn, "Row-normalised (recall)", ".1%")]:
        ax.imshow(mat, cmap="Blues", vmin=0, vmax=mat.max())
        for i in range(len(classes)):
            for j in range(len(classes)):
                v = mat[i, j]
                txt = format(v, fmt) if fmt != "d" else str(v)
                ax.text(j, i, txt, ha="center", va="center", fontsize=10,
                        color="white" if v > mat.max() * 0.6 else "#1f2937")
        ax.set_xticks(range(len(classes))); ax.set_xticklabels(labels, rotation=20)
        ax.set_yticks(range(len(classes))); ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted class"); ax.set_ylabel("True class"); ax.set_title(title, loc="left", fontsize=11)
    fig.suptitle(f"Final model (Xception) on {len(df):,} test images – accuracy {summary['accuracy']:.2%}, macro-F1 {summary['macro_f1']:.2%}",
                 fontsize=11, x=0.02, ha="left")
    fig.tight_layout(); fig.savefig(out / "confusion_matrix.png", facecolor="white"); plt.close(fig)

    # reliability diagram
    fig, ax = plt.subplots(figsize=(5.2, 4.6), dpi=200)
    ax.plot([0, 1], [0, 1], ls="--", color="#9ca3af", lw=1, label="Perfect calibration")
    xs = [b[2] for b in bins]; ys = [b[3] for b in bins]; ns = [b[4] for b in bins]
    ax.plot(xs, ys, marker="o", color="#2a5ea8", lw=2, label="Final model")
    for x_, y_, n_ in zip(xs, ys, ns):
        ax.annotate(str(n_), (x_, y_), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=7, color="#4b5563")
    ax.set_xlim(0.3, 1.0); ax.set_ylim(0.3, 1.02)
    ax.set_xlabel("Confidence (after temperature scaling)"); ax.set_ylabel("Accuracy")
    ax.set_title(f"Reliability diagram – ECE {ece:.3f}", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="lower right"); ax.grid(color="#e5e7eb")
    fig.tight_layout(); fig.savefig(out / "reliability_diagram.png", facecolor="white"); plt.close(fig)
    print(f"Saved to {out.resolve()}")


if __name__ == "__main__":
    main()
