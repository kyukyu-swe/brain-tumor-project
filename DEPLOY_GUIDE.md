# Deployment Guide — free hosting (Google Cloud Run + Vercel)

Our app has two parts, hosted in two free places:

| Part | What it is | Where | Cost |
|---|---|---|---|
| **AI backend** | `app.py` + `core/` + model (Python, PyTorch) | **Google Cloud Run** | Free tier (see below) |
| **Website** | `web/` folder (HTML/CSS/JS) | **Vercel** | Free (Hobby plan) |

The website runs in the visitor's browser and calls the backend's API.

**Cloud Run free tier (per month):** 180,000 vCPU-seconds, 360,000 GiB-seconds, 2 million requests.
One scan uses about 1 second, so a class project stays far below this.
The backend **sleeps when nobody uses it** (no cost) and wakes up on the first visit (about 20–40 seconds).
It uses the same Google billing account as your Gemini key. Image storage may cost a few cents per month.

---

## Part A — Backend on Google Cloud Run

### A1. One-time setup on your Mac
```bash
brew install --cask google-cloud-sdk      # installs the gcloud command
gcloud auth login                          # opens the browser — log in with your Google account
```
Choose the Google Cloud project (the one with billing; it can be the same project as your Gemini key):
```bash
gcloud projects list                       # find your PROJECT_ID
gcloud config set project YOUR_PROJECT_ID
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
```

### A2. Set a budget alert (important, 2 minutes)
Google Cloud Console → **Billing → Budgets & alerts → Create budget** → amount **$1** → alerts at 50%, 90%, 100%.
You will get an email if anything ever costs money.

### A3. Deploy (one command)
From the project folder:
```bash
bash scripts/deploy_cloudrun.sh
```
- The first time, it may ask to create an "Artifact Registry repository" → answer **Y**.
- Google builds the container from `Dockerfile` (about 5–10 minutes the first time).
- At the end it prints the backend URL, like `https://brain-tumor-api-xxxxx.asia-northeast1.run.app`.
- Your Gemini key is read from `.env` and given to the backend as a setting. `.env` itself is never uploaded (see `.gcloudignore`).

Settings used (in the script): 2 CPU, 2 GB memory, **max 1 instance** (this also limits cost), sleeps when idle, region Tokyo.

### A4. Check it
Open the backend URL in your browser. You should see the simple built-in Gradio app. Try one sample.
If something fails: Google Cloud Console → **Cloud Run → brain-tumor-api → Logs**, copy the red error, send it to Claude.

### Update the backend later
Change code / model → run `bash scripts/deploy_cloudrun.sh` again.

---

## Part B — Website on Vercel

### B1. Point the website to the backend
In `web/js/config.js`:
```js
export const BACKEND = "https://brain-tumor-api-xxxxx.asia-northeast1.run.app/";
```
Commit and push to GitHub.

### B2. Create the Vercel project (one time)
1. https://vercel.com → **Sign up with GitHub**
2. **Add New → Project** → import `brain-tumor-project`
3. **Root Directory: `web`** · Framework Preset: **Other** · no build command → **Deploy**
4. You get a link like `https://brain-tumor-project.vercel.app` — share this with your group.

Every `git push` updates the website automatically.

---

## Protecting your Gemini key and budget
- You already set a spending limit for Gemini. ✅
- The backend limits Gemini to `MAX_LLM_CALLS_PER_HOUR` (default 100, all users together). After that it shows the fact sheet.
- Cloud Run uses **max 1 instance**, so it cannot grow big.
- Do **not** set `APP_PASSWORD` when you use the website — the login would block the website's requests.

## Testing everything on your laptop first
```bash
# terminal 1 — backend
source .venv/bin/activate
python app.py
# terminal 2 — website (web/js/config.js must point to http://127.0.0.1:7860/)
cd web && python3 -m http.server 5173
```
Open http://localhost:5173

## Backup plan for presentation day
If the cloud has a problem, run the backend on your laptop with a temporary public link:
change the last line of `app.py` to `demo.launch(share=True)`, run it, and put the printed
`https://xxxx.gradio.live` link into `web/js/config.js`. It works while your laptop is on.

## Other free option (not used)
Hugging Face Spaces (ZeroGPU) also works: `scripts/deploy_to_hf.py` and the `deploy/` folder are still in the project.
