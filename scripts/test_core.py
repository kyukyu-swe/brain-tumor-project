"""
Quick test for Part B. Run from the project folder:

    python scripts/test_core.py path/to/mri.jpg
    python scripts/test_core.py path/to/mri.jpg --llm --language Thai

It prints the prediction and saves the heatmap as heatmap_test.png.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.predictor import TumorPredictor  # noqa: E402
from core.llm import explain  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("image")
parser.add_argument("--model", default="model/best_model.pth")
parser.add_argument("--llm", action="store_true", help="also call the LLM")
parser.add_argument("--language", default="English")
parser.add_argument("--mode", default="patient", choices=["patient", "student", "doctor_note"])
args = parser.parse_args()

predictor = TumorPredictor(args.model)
r = predictor.predict(args.image)

print(f"Prediction : {r.display_name} ({r.confidence:.1%})")
print("Probs      :", {k: f"{v:.1%}" for k, v in r.probs.items()})
print(f"P(tumor)   : {r.p_tumor:.1%}  -> priority {r.priority}")
print(f"Refer      : {r.refer}  {r.reasons}")
r.heatmap.save("heatmap_test.png")
print("Heatmap saved to heatmap_test.png")

if args.llm:
    print("\n--- LLM explanation ---")
    print(explain(r, language=args.language, mode=args.mode, true_label=None))
