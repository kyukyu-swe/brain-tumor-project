# Paste this as a NEW CELL at the end of the Kaggle training notebook and run it.
# It uses variables that already exist there: test_logits, test_y, TEMPERATURE, CLASS_NAMES.
# It measures how well the "refer to a doctor" safety rule works on all 1,600 test images.

import torch.nn.functional as F

MIN_CONFIDENCE, MIN_MARGIN, NOTUMOR_TRUST = 0.80, 0.20, 0.95   # same as core/predictor.py
NT = CLASS_NAMES.index("notumor")

probs = F.softmax(test_logits / TEMPERATURE, 1)
conf, pred = probs.max(1)
top2 = probs.topk(2, 1).values
refer = (conf < MIN_CONFIDENCE) | ((top2[:, 0] - top2[:, 1]) < MIN_MARGIN) | ((pred == NT) & (probs[:, NT] < NOTUMOR_TRUST))

wrong = pred != test_y
missed_tumor = (pred == NT) & (test_y != NT)

print(f"Test images            : {len(test_y)}")
print(f"Wrong predictions      : {wrong.sum().item()}")
print(f"  -> flagged by rule   : {(wrong & refer).sum().item()}  ({(wrong & refer).sum().item() / max(wrong.sum().item(), 1):.0%})")
print(f"Missed tumors          : {missed_tumor.sum().item()}  (tumor predicted as 'no tumor')")
print(f"  -> flagged by rule   : {(missed_tumor & refer).sum().item()}")
print(f"Correct but flagged    : {(~wrong & refer).sum().item()}  (extra work for doctors)")
print(f"Total sent to doctor   : {refer.sum().item()}  ({refer.float().mean().item():.1%} of all scans)")
print(f"Accuracy on scans NOT flagged: {(~wrong[~refer]).float().mean().item():.2%}")
