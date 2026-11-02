# Brain Tumor MRI Assistant — Project Guide (Group 1)

> **Update (29 Oct):** the group chose **Idea 3 (Triage) + Idea 1 (Patient explainer)**.
> The app now has these two tabs, linked: staff sort the scans, then make a patient explanation for a chosen scan.
> Next: compare 6 models (`training/04_model_comparison.ipynb`) and host the app for free (`DEPLOY_GUIDE.md`).

This guide explains the whole project: the problem, the four ideas, how the system works, what we found, and how to run it.

---

## 1. The big picture

We have one dataset (brain MRI images) and we must build something with **social impact**.

Our approach:

1. Train **one AI model** that looks at a brain MRI slice and says which type of tumor it shows (or no tumor).
2. Add tools that make the AI **safer and easier to understand**:
  - a **heatmap** that shows where the AI looked,
  - a **safety rule** that says "send this to a doctor" when the AI is not sure,
  - an **LLM (Gemini)** that explains the result in simple words, in several languages.
3. Build **four demo apps** on top of the same model, one for each project idea. The group chooses one to develop fully.

```
MRI image ──► AI model (EfficientNet-B0) ──► tumor type + probabilities
                     │
                     ├──► Grad-CAM heatmap (where the AI looked)
                     ├──► Safety rule (refer to a doctor?)
                     └──► Gemini LLM (simple explanation, many languages)
                                │
                     4 apps: Patients · Students · Hospitals · Doctors
```

---



## 2. The dataset

- **Source:** Kaggle, "Brain Tumor MRI Dataset" by Masoud Nickparvar
([https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset))
- **Size:** 5,600 training images + 1,600 testing images (current Kaggle version)
- **4 classes, balanced:** 1,400 training and 400 testing images each


| Class      | What it is (simple)                                                      |
| ---------- | ------------------------------------------------------------------------ |
| Glioma     | Tumor that grows **inside** the brain tissue, from glial (support) cells |
| Meningioma | Tumor that grows from the **covering** of the brain, under the skull     |
| Pituitary  | Tumor in the small hormone gland at the **base** of the brain            |
| No tumor   | Normal brain scan                                                        |


**Important limits of the data:**

- Images are single **2D slices**, not full 3D scans like in a hospital.
- There is **no patient information**, and images come from several public sources mixed together.
- So our model is a **prototype for learning and demonstration**, not a medical device.

---



## 3. The four project ideas

All four use the same model. They differ in **who the user is** and **what problem we solve**.

### Idea 1 — "Understand my scan" (patients and families)

- **Problem:** After a diagnosis, patients get reports full of medical words, often in a language they don't read well (for example, migrant workers in Thailand or Japan). They feel scared and don't know what to ask.
- **Solution:** Upload the MRI → the app shows the tumor type, a heatmap, and a simple explanation in Thai, Burmese, Japanese or English, plus **questions to ask the doctor**.
- **Strength:** Strongest social story (language barrier, patient understanding).
- **Risk:** Patients might trust the AI more than the doctor. We must frame it as "help you understand *after* the doctor's diagnosis", not "diagnose yourself".



### Idea 2 — "Learn" (medical students)

- **Problem:** Medical students in low-resource countries see too few brain MRI cases to learn the differences between tumor types.
- **Solution:** A quiz. The app shows an MRI, the student guesses the type, then sees the correct answer, the heatmap, and an explanation of the **visual clues**. It keeps a score.
- **Strength:** **Safest** idea. No medical claim, and the correct answer comes from the dataset label, so nobody is harmed if the AI is wrong.
- **Risk:** Social impact is smaller / less direct than idea 1.



### Idea 3 — "Triage list" (hospitals with few radiologists)

- **Problem:** Rural hospitals may have MRI machines but few radiologists, so urgent scans wait for days.
- **Solution:** Upload many scans → the app sorts them by **probability of tumor**, so likely tumors are read first. It also makes a short **draft note** for the doctor.
- **Strength:** Sounds impressive, clear real-world use.
- **Risk:** Hard to defend: our data is not real hospital data, so we cannot prove it works there.



### Idea 4 — "Second opinion" (general doctors)

- **Problem:** A general doctor has no brain specialist nearby to ask.
- **Solution:** Upload a scan → the app gives its answer, confidence, heatmap, and clearly says **"refer to a specialist"** when it is unsure.
- **Strength:** Simple, shows the safety rule clearly.
- **Risk:** Same as idea 3 (real clinical use needs much more testing).



### Quick comparison


|                         | 1. Patients | 2. Students | 3. Triage   | 4. Second opinion |
| ----------------------- | ----------- | ----------- | ----------- | ----------------- |
| Social impact story     | ⭐⭐⭐         | ⭐⭐          | ⭐⭐⭐         | ⭐⭐                |
| Safety / easy to defend | ⭐⭐          | ⭐⭐⭐         | ⭐           | ⭐⭐                |
| Uses LLM                | Yes         | Yes         | Yes (notes) | No                |
| Demo is easy to show    | ⭐⭐⭐         | ⭐⭐⭐         | ⭐⭐          | ⭐⭐                |


**Possible combination:** one app with two modes — "Learn" for students and "Understand" for patients — both multilingual.

---



## 4. How the system works (step by step)

The project has three parts.

### Part A — Train the model (Kaggle notebook)

File: `training/group1-braintumortraining.ipynb` (run on Kaggle with a free GPU)


| Step                  | What we do                                                                                                                      | Why                                                                                                            |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| 1. Load data          | Read the Training/ and Testing/ folders; the folder name is the label                                                           | Simple and reliable                                                                                            |
| 2. Look at data       | Count images per class, check sizes, show samples                                                                               | Check balance and image sizes                                                                                  |
| 3. Split & preprocess | Training → 85% train / 15% validation. Resize to 224×224, normalize. Small rotation, flip, brightness changes for training only | Validation picks the best model; the test set stays "unseen". Augmentation reduces overfitting                 |
| 4. Model              | **EfficientNet-B0**, pretrained on ImageNet, last layer changed to 4 classes (**transfer learning**)                            | 5,600 images is small; a pretrained model already knows edges and shapes. It is small and fast (4M parameters) |
| 5. Train              | 12 epochs, AdamW, learning rate 1e-4, keep the epoch with the best validation accuracy                                          | Avoid keeping an overfitted model                                                                              |
| 6. Evaluate           | Accuracy, precision/recall per class, confusion matrix on the **test set (used once)**                                          | Honest final score                                                                                             |
| 7. Calibrate          | **Temperature scaling**: learn one number T to make confidence values honest                                                    | The safety rule depends on confidence                                                                          |
| 8. Grad-CAM           | Make heatmaps and check them by eye                                                                                             | See if the AI looks at the tumor                                                                               |
| 9. Save               | Save the model + settings to `model/best_model.pth`                                                                             | The apps load this file                                                                                        |




### Part B — Shared core (used by all apps)

Folder: `core/`

- `predictor.py` — loads the model once. One function `predict(image)` returns:
  - tumor type and probabilities for all 4 classes
  - P(tumor) = 1 − P(no tumor), and a priority (High / Medium / Low)
  - the Grad-CAM heatmap
  - **refer** = True/False, with the reason
- `llm.py` — sends the prediction to **Gemini** and gets an explanation in three styles: patient, student, or doctor note.
- `tumor_facts.json` — short, checked facts about each tumor type. **The LLM may use only these facts.**



### Part C — The app

File: `app.py` — one **Gradio** web app with four tabs (one per idea). Runs in the browser.

---



## 5. Results



### Overall


| Measure                                 | Result                                   |
| --------------------------------------- | ---------------------------------------- |
| Validation accuracy (best epoch)        | 98.3%                                    |
| **Test accuracy** (1,600 unseen images) | **94.25%** ← the honest number to report |
| Our 100 sample images                   | 92 / 100 correct                         |


Why is test lower than validation? Validation images come from the same pool as training images (maybe similar slices or the same patients), so they are easier. The Testing folder is less similar.

### Per class (test set)


| Class      | Precision | Recall    | Comment                                      |
| ---------- | --------- | --------- | -------------------------------------------- |
| Glioma     | 99.1%     | **80.7%** | Weakest class. 77 of 400 gliomas were missed |
| Meningioma | 89.8%     | 96.5%     |                                              |
| No tumor   | 91.5%     | 100%      | Never misses a normal scan                   |
| Pituitary  | 98.0%     | 99.8%     | Almost perfect                               |


- **Precision** = when the AI says "X", how often is it right?
- **Recall** = of all real X cases, how many did the AI find? For tumors, recall matters most.



### The most important errors (confusion matrix)

- 43 gliomas → predicted as meningioma
- **31 gliomas → predicted as "no tumor"** ← most dangerous: a tumor is missed completely
- 3 gliomas → predicted as pituitary



### Confidence calibration

- Learned temperature T = 1.29 (the model was a little overconfident)
- Calibration error (ECE): 0.041 → 0.034 after scaling (lower is better)

---



## 6. Safety features



### 6.1 The "refer to a doctor" rule

A scan is flagged for a specialist if **any** of these is true:

1. Confidence of the top answer is **below 80%**
2. The top two answers are **within 20%** of each other (the AI can't decide)
3. The answer is "no tumor" but P(no tumor) is **below 95%** (stricter, because missing a tumor is the worst error)

**Result on our 100 sample images:** 8 errors, and the rule flagged 4 of them, including 2 of the 3 missed tumors.

**Honest limitation:** when the AI is **confidently wrong** (for example, one glioma was called "no tumor" with 99% confidence), no confidence rule can catch it.
To measure the rule on all 1,600 test images, run `training/02_safety_rule_check.py` as a cell in the Kaggle notebook.

### 6.2 Grad-CAM heatmap — "where did the AI look?"

- **Red/yellow** = areas that most influenced the answer. **Blue** = ignored.
- The heatmap is blurry by design (it comes from a 7×7 grid), so it won't trace the exact tumor shape.

**How to check a heatmap:**

1. First find the tumor yourself on the original image: a bright blob or ring, a dark area with a bright edge, left–right asymmetry, or pushed structures.
2. Then look at the red area.
3. Judge it: ✅ Good (red on tumor), 🟡 Partial (red near or on part of the tumor), ❌ Wrong (red on the skull, edges, background, or the other side).


| Tumor type | Usual location                               | Usual look                                        |
| ---------- | -------------------------------------------- | ------------------------------------------------- |
| Glioma     | Inside brain tissue                          | Irregular, often a bright ring with a dark center |
| Meningioma | Edge of the brain, touching the skull lining | Round, smooth, evenly bright                      |
| Pituitary  | Base of the brain, in the middle             | Small; best seen in side or front views           |


**What we found:** when the AI is correct, it usually looks at the tumor. When it is wrong, it often looks at the wrong place. Sometimes it is **right for the wrong reason** (correct answer, but red on the wrong area). This is why the heatmap helps people decide whether to trust a result.
Note: heatmaps on "no tumor" images have little meaning (there is nothing to point at).

### 6.3 LLM safety design

- Gemini **never sees the image**. It only gets the prediction + the facts in `tumor_facts.json`.
- The system prompt tells it: use only these facts, never say the person has a disease, never give treatment decisions, use simple words, and warn clearly when the AI is unsure.
- This is a simple form of **RAG** (retrieval-augmented generation): it reduces made-up answers ("hallucinations").
- If there is no API key or the API fails, the app still works and shows the fact sheet directly.

**Group task:** someone must check `tumor_facts.json` against reliable sources (National Cancer Institute, Mayo Clinic, Cleveland Clinic) and add the exact links before the presentation.

---



## 7. Project files

```
brain_tumor_project/
├── PROJECT_GUIDE.md                      ← this file
├── README.md                             ← short technical setup
├── app.py                                ← the web app (4 tabs)
├── requirements.txt                      ← Python packages
├── .env.example                          ← copy to .env and add the Gemini key
├── core/
│   ├── predictor.py                      ← model + heatmap + safety rule
│   ├── llm.py                            ← Gemini explanations
│   └── tumor_facts.json                  ← facts the LLM may use
├── model/
│   └── best_model.pth                    ← trained model (16 MB)
├── data/samples/                         ← 100 test images (25 per class)
├── scripts/test_core.py                  ← quick test from the terminal
└── training/
    ├── group1-braintumortraining.ipynb   ← training notebook (Kaggle)
    ├── 02_safety_rule_check.py           ← extra Kaggle cell: test the safety rule
    └── 03_make_sample_set.py             ← extra Kaggle cell: make the 100-image sample set
```

---



## 8. How to run it on your laptop

```bash
cd brain_tumor_project
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # then paste the Gemini API key into .env
python app.py
```

Open [http://127.0.0.1:7860](http://127.0.0.1:7860) in your browser.

---



## 9. Limitations (we must say these in the presentation)

1. **Not real hospital data:** 2D slices from a public dataset, no patient info. Results may not hold on real scans.
2. **Glioma is weak:** recall 80.7%; some gliomas are called meningioma or "no tumor".
3. **Confident mistakes** cannot be caught by the safety rule.
4. **Heatmaps are rough** and were checked by non-experts (us), not radiologists.
5. **The LLM text** depends on our fact sheet; it must be checked by the group.
6. **Not a medical device.** The app supports understanding and learning; it does not diagnose.

---



## 10. Next steps

1. **Choice** choose the appropriate project or combination of the projects for the app
2. **Deploy** the chosen app on Hugging Face Spaces so anyone can try it from a link.
3. **Prepare slides:** problem → solution → demo → results → limitations → impact.

