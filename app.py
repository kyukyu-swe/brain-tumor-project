"""
Brain Tumor MRI Assistant — Group 1 final app (Idea 3 Triage + Idea 1 Patient explainer).

Run locally (from the project folder):
    python app.py
Then open http://127.0.0.1:7860

Optional settings (in .env locally, or "Variables and secrets" on Hugging Face):
    GEMINI_API_KEY          Gemini key (without it, the app shows the fact sheet instead)
    GEMINI_MODEL            default: gemini-flash-latest
    APP_PASSWORD            if set, the app asks for a login (username: APP_USERNAME, default "group1")
    MAX_LLM_CALLS_PER_HOUR  default: 100 (protects your Gemini bill)
"""
# ---- Hugging Face ZeroGPU support ----
# Free Hugging Face accounts can host Gradio apps only on "ZeroGPU" hardware.
# There, model code must run inside a function decorated with @spaces.GPU.
# On your laptop the `spaces` package is not installed, so `gpu` does nothing.
try:
    import spaces
    gpu = spaces.GPU(duration=30)
except ImportError:
    def gpu(fn):
        return fn
# (`spaces` must be imported before torch, so this block is at the very top.)

import os
import random
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

from core.llm import LANGUAGES, explain
from core.predictor import (DISPLAY_NAMES, MIN_CONFIDENCE, MIN_MARGIN, NOTUMOR_TRUST,
                            PRIORITY_HIGH, PRIORITY_MEDIUM, TumorPredictor)

load_dotenv()

ROOT = Path(__file__).parent
SAMPLES_DIR = ROOT / "data" / "samples"
MAX_TRIAGE_FILES = 20

predictor = TumorPredictor(ROOT / "model" / "best_model.pth")
CLASSES = predictor.class_names

DISCLAIMER = (
    "⚠️ **Student project prototype — not a medical device.** "
    f"The AI ({predictor.model_name}) was trained on a public dataset of 2D MRI slices "
    f"(test accuracy {predictor.test_accuracy:.2%}). It can be wrong. Only a doctor can diagnose."
)


def list_samples():
    items = []
    for cls in CLASSES:
        for p in sorted((SAMPLES_DIR / cls).glob("*")):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                items.append((str(p), cls))
    return items


SAMPLES = list_samples()
EXAMPLES = [[next(p for p, l in SAMPLES if l == c)] for c in CLASSES if any(l == c for _, l in SAMPLES)]


@gpu
def predict_one(image):
    return predictor.predict(image)


@gpu
def predict_many(paths):
    return [(p, predictor.predict(p)) for p in paths]


def result_markdown(r):
    lines = [f"### AI result: {r.display_name} ({r.confidence:.0%})"]
    if r.refer:
        lines.append("⚠️ **The AI is not sure. A specialist should check this scan.**")
        lines += [f"- {reason}" for reason in r.reasons]
    else:
        lines.append("✅ The AI is confident (but it can still be wrong).")
    return "\n".join(lines)


# ------------------------------------------------------------ Tab 1: Triage
PRIORITY_LABEL = {"High": "🔴 High", "Medium": "🟠 Medium", "Low": "🟢 Low"}


def _paths(files):
    return [f if isinstance(f, str) else f.name for f in (files or [])]


def triage_load_samples():
    return [p for p, _ in random.sample(SAMPLES, min(12, len(SAMPLES)))]


def triage_run(files):
    paths = _paths(files)
    empty = ([], [], [], "Upload scans or click **Load 12 sample scans**.", gr.update(choices=[], value=None))
    if not paths:
        return empty
    note = ""
    if len(paths) > MAX_TRIAGE_FILES:
        note = f" Only the first {MAX_TRIAGE_FILES} scans were processed (demo limit)."
        paths = paths[:MAX_TRIAGE_FILES]

    results = predict_many(paths)
    # Flagged scans go up within the same priority, so a doctor checks them early.
    results.sort(key=lambda x: (x[1].p_tumor >= PRIORITY_HIGH, x[1].refer, x[1].p_tumor), reverse=True)

    table = [[i + 1, Path(p).name, PRIORITY_LABEL[r.priority], r.display_name, f"{r.confidence:.0%}",
              f"{r.p_tumor:.0%}", "⚠️ Review" if r.refer else ""] for i, (p, r) in enumerate(results)]
    gallery = [(r.heatmap, f"#{i + 1} {r.display_name} ({r.priority})") for i, (_, r) in enumerate(results)]
    n_high = sum(r.priority == "High" for _, r in results)
    n_flag = sum(r.refer for _, r in results)
    summary = (f"**{len(results)} scans sorted.** {n_high} high priority, {n_flag} flagged for specialist review."
               f"{note} Pick a scan below to see details.")
    choices = [f"#{i + 1} {Path(p).name}" for i, (p, _) in enumerate(results)]
    return table, gallery, results, summary, gr.update(choices=choices, value=None)


def _selected(choice, results):
    if not results or not choice:
        return None
    return results[int(choice.split()[0].lstrip("#")) - 1][1]


def triage_select(choice, results):
    r = _selected(choice, results)
    if r is None:
        return None, "", ""
    return r.heatmap, result_markdown(r), explain(r, mode="doctor_note")


def triage_patient_text(choice, results, language):
    r = _selected(choice, results)
    if r is None:
        return "Pick a scan first."
    return explain(r, language=language, mode="patient")


# ------------------------------------------------------------ Tab 2: Patient
def patient_explain(image, language):
    if image is None:
        return "Please upload an MRI image.", None, None, ""
    r = predict_one(image)
    return result_markdown(r), r.heatmap, r.probs, explain(r, language=language, mode="patient")


# ------------------------------------------------------------ About text
ABOUT = f"""
## How this app works
1. **AI model** (`{predictor.model_name}`) reads one MRI slice and gives the probability of 4 classes:
   glioma, meningioma, pituitary tumor, no tumor.
2. **Heatmap (Grad-CAM)**: red = the area that most influenced the AI. Check if the red area is on the tumor.
3. **Safety rule**: the scan is flagged for a specialist if
   - confidence < {MIN_CONFIDENCE:.0%}, or
   - the top two answers are within {MIN_MARGIN:.0%}, or
   - the answer is "no tumor" but P(no tumor) < {NOTUMOR_TRUST:.0%} (a missed tumor is the worst error).
4. **Triage priority** uses P(tumor) = 1 − P(no tumor): High ≥ {PRIORITY_HIGH:.0%}, Medium ≥ {PRIORITY_MEDIUM:.0%}, else Low.
5. **Explanations** come from Gemini, which never sees the image. It only gets the AI result and a fact sheet
   checked by our group, and it must use only those facts.

## Limitations
- Trained on a public dataset of 2D slices, not real hospital data.
- The AI can be confidently wrong; the safety rule cannot catch those cases.
- Not a medical device. It does not diagnose.

Dataset: [Brain Tumor MRI Dataset (Kaggle, CC0)](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
"""

# ------------------------------------------------------------ UI
with gr.Blocks(title="Brain Tumor MRI Assistant") as demo:
    gr.Markdown("# 🧠 Brain Tumor MRI Assistant\nHelp hospitals read urgent scans first, and help patients understand their results.")
    gr.Markdown(DISCLAIMER)

    with gr.Tab("1. Triage list (hospital staff)"):
        gr.Markdown(f"Upload up to {MAX_TRIAGE_FILES} scans. Likely tumors and uncertain scans go to the top.")
        t_results = gr.State([])
        with gr.Row():
            t_files = gr.File(file_count="multiple", file_types=["image"], label="MRI scans")
            with gr.Column():
                t_sample = gr.Button("Load 12 sample scans")
                t_btn = gr.Button("Sort by priority", variant="primary")
        t_summary = gr.Markdown()
        t_table = gr.Dataframe(headers=["#", "File", "Priority", "AI prediction", "Confidence", "P(tumor)", "Flag"],
                               interactive=False)
        t_pick = gr.Dropdown([], label="Show details for scan", interactive=True)
        with gr.Row():
            t_heat = gr.Image(label="Where the AI looked")
            with gr.Column():
                t_result = gr.Markdown()
                t_note = gr.Markdown()
        with gr.Accordion("Explain this scan to the patient", open=False):
            with gr.Row():
                t_lang = gr.Dropdown(LANGUAGES, value="English", label="Patient's language")
                t_explain = gr.Button("Make patient explanation")
            t_patient = gr.Markdown()
        t_gallery = gr.Gallery(label="All heatmaps (same order)", columns=6, height="auto")

        t_sample.click(triage_load_samples, None, t_files)
        t_btn.click(triage_run, [t_files], [t_table, t_gallery, t_results, t_summary, t_pick])
        t_pick.change(triage_select, [t_pick, t_results], [t_heat, t_result, t_note])
        t_explain.click(triage_patient_text, [t_pick, t_results, t_lang], t_patient)

    with gr.Tab("2. Understand my scan (patients)"):
        gr.Markdown("Upload an MRI slice. The AI explains the result in simple words, in your language, "
                    "with questions to ask your doctor.")
        with gr.Row():
            with gr.Column():
                p_img = gr.Image(type="pil", label="MRI image")
                p_lang = gr.Dropdown(LANGUAGES, value="English", label="Language")
                p_btn = gr.Button("Explain", variant="primary")
                if EXAMPLES:
                    gr.Examples(EXAMPLES, inputs=[p_img], label="Try a sample")
            with gr.Column():
                p_res = gr.Markdown()
                p_heat = gr.Image(label="Where the AI looked (red = important)")
                p_probs = gr.Label(label="Probabilities", num_top_classes=4)
        p_text = gr.Markdown()
        p_btn.click(patient_explain, [p_img, p_lang], [p_res, p_heat, p_probs, p_text])

    with gr.Tab("About"):
        gr.Markdown(ABOUT)

demo.queue(default_concurrency_limit=2)

if __name__ == "__main__":
    password = os.getenv("APP_PASSWORD")
    auth = (os.getenv("APP_USERNAME", "group1"), password) if password else None
    demo.launch(auth=auth)
