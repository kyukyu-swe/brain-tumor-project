# Brain Tumor MRI — Group 1 Project

📖 **New to the project? Read [PROJECT_GUIDE.md](PROJECT_GUIDE.md) first.**

One trained classifier (EfficientNet-B0, test accuracy 94.25%) shared by four demo ideas.

```
brain_tumor_project/
├── training/
│   ├── group1-braintumortraining.ipynb   ← Part A (run on Kaggle)
│   ├── 02_safety_rule_check.py           ← extra Kaggle cell: tests the "refer" rule
│   ├── 03_make_sample_set.py             ← extra Kaggle cell: 100-image sample set
│   ├── 04_model_comparison.ipynb         ← train 7 models (from our reference papers), compare, pick the best
│   └── 05_seed_check.ipynb               ← re-train the top 2 with 2 more seeds (is the winner really better?)
├── model/
│   └── best_model.pth                    ← trained model (Part A output)
├── core/                                 ← Part B: shared by all apps
│   ├── predictor.py                      ← prediction + Grad-CAM + safety rule
│   ├── llm.py                            ← Gemini explanations (facts only)
│   └── tumor_facts.json                  ← checked facts the LLM is allowed to use
├── scripts/
│   ├── test_core.py                      ← quick test of Part B
│   ├── deploy_cloudrun.sh                ← one-command deploy of the backend to Google Cloud Run
│   ├── deploy_to_hf.py                   ← (alternative) deploy to Hugging Face
│   └── shrink_model.py                   ← halve the model file size (GitHub limit is 100 MB)
├── data/samples/                         ← Kaggle Testing images (see data/README.md)
├── app.py                                ← AI backend (Gradio): model API + simple built-in UI
├── web/                                  ← the website (HTML/CSS/JS) — calls the backend API
├── Dockerfile, .gcloudignore             ← container for Google Cloud Run (the backend)
├── deploy/cloudrun/requirements.txt      ← Python packages inside the container
├── deploy/                               ← (alternative) Hugging Face Space settings
├── DEPLOY_GUIDE.md                       ← how to host for free: backend on Cloud Run, website on Vercel
├── requirements.txt
└── .env.example                          ← copy to .env and add your Gemini key
```

## Setup (once)

```bash
cd brain_tumor_project
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # then open .env and paste your Gemini key
```

## Test Part B

```bash
python scripts/test_core.py data/samples/glioma/<any-image>.jpg
python scripts/test_core.py data/samples/glioma/<any-image>.jpg --llm --language Thai
```

## Run the website on your laptop

Two terminals:
```bash
# terminal 1 — AI backend
source .venv/bin/activate
python app.py

# terminal 2 — website
cd web
python -m http.server 5173
```
Open http://localhost:5173 . (The backend's own simple UI is still at http://127.0.0.1:7860.)

## Run the app

```bash
python app.py
```
Open http://127.0.0.1:7860. Tabs: **1. Triage list** (hospital staff) and **2. Understand my scan** (patients).
Without a Gemini key the app still works and shows the fact sheet instead of the LLM text.

To host it for free (Cloud Run + Vercel), see [DEPLOY_GUIDE.md](DEPLOY_GUIDE.md).

## How the safety rule works (core/predictor.py)

A scan is flagged **"refer to a specialist"** if any of these is true:
- top-class confidence < 80%
- the top two classes are within 20% of each other
- the answer is "no tumor" but P(no tumor) < 95% (a missed tumor is the worst error)

## Known limitations (say these in the presentation)
- Trained on 2D slices from a public dataset, not real hospital data. Prototype only, not a medical device.
- On our 100 sample images: 92 correct; the rule flagged 4 of the 8 errors (2 of 3 missed tumors).
- Weakest class: glioma (recall 80.7%). Some gliomas are predicted as meningioma or "no tumor".
- Confidently wrong predictions cannot be caught by the confidence rule.
- `tumor_facts.json` must be checked by the group against the listed sources.
