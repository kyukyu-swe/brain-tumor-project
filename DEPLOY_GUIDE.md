# Deployment Guide — free hosting on Hugging Face Spaces (ZeroGPU)

## Why ZeroGPU?
Hugging Face changed its rules: **normal (CPU) Gradio Spaces now need a paid plan**.
Free personal accounts can still host **up to 2 Gradio Spaces on ZeroGPU** if the account is
**older than 30 days with a verified email**. Our app is prepared for this:
the model runs inside a function marked `@spaces.GPU` (see the top of `app.py`).

ZeroGPU limits (free): each visitor gets a small **daily GPU quota** (about 2 minutes without login,
5 minutes with a free Hugging Face login). One prediction uses about 1 second, so this is enough for a demo.
The Space sleeps when nobody uses it and wakes up when someone opens the link (first load is slower).

---

## Step 1 — Put the final model in place
After running `training/04_model_comparison.ipynb` on Kaggle, download `best_model.pth`
and replace `model/best_model.pth`. Test locally first:
```bash
source .venv/bin/activate
python app.py
```

## Step 2 — Protect your paid Gemini key (important!)
A public link means **anyone** can make the app call Gemini with your key.
1. **Set a spending limit** for the Gemini API in Google AI Studio / Google Cloud billing.
2. The app already limits Gemini to `MAX_LLM_CALLS_PER_HOUR` (default 100). After that, it shows the fact sheet.
3. Optional: add a login. Put these in your local `.env`:
   ```
   APP_PASSWORD=choose-a-password
   APP_USERNAME=group1
   ```
   Then only people with the password can use the app (good for a class demo).

## Step 3 — Log in to Hugging Face (one time)
1. Go to https://huggingface.co/settings/tokens → **Create new token** → type **Write** → copy it.
2. In the terminal:
   ```bash
   pip install -U huggingface_hub
   hf auth login        # paste the token
   ```

## Step 4 — Deploy with one command
```bash
python scripts/deploy_to_hf.py --space YOUR_USERNAME/brain-tumor-assistant --set-secrets
```
This:
- creates the Space with **Gradio + ZeroGPU** (if it doesn't exist yet),
- uploads the app, model, sample images, and the Space settings in `deploy/`,
- copies `GEMINI_API_KEY` and `APP_PASSWORD` from `.env` into the Space **secrets** (they are not visible in the code).

If the script cannot create the Space, create it in the browser:
https://huggingface.co/new-space → name `brain-tumor-assistant` → SDK **Gradio** → Hardware **ZeroGPU** → Create.
Then run the same command again.

## Step 5 — Wait for the build and test
Open `https://huggingface.co/spaces/YOUR_USERNAME/brain-tumor-assistant`.
The first build takes about 5–10 minutes. Check:
- [ ] The app opens (and asks for the password, if you set one)
- [ ] Triage tab: "Load 12 sample scans" → "Sort by priority" → table and heatmaps appear
- [ ] Pick a scan → draft note appears → "Make patient explanation" in Thai works
- [ ] Patient tab: upload an image → result + explanation

**If something fails:** open the **Logs** tab of the Space, copy the red error message, and send it to Claude.

## Step 6 — Update the app later
Change the code, test locally, then run the Step 4 command again. It uploads the new version.

## Where the settings live

| Setting | Where | Secret? |
|---|---|---|
| `GEMINI_API_KEY` | Space → Settings → Variables and secrets | Yes |
| `APP_PASSWORD` | same | Yes |
| `GEMINI_MODEL`, `APP_USERNAME`, `MAX_LLM_CALLS_PER_HOUR` | same | No (variables) |

## Backup plan (if ZeroGPU is not possible)
Run the app on your laptop with a temporary public link:
```python
demo.launch(share=True)     # change the last line of app.py
```
Gradio prints a `https://xxxx.gradio.live` link that works while your laptop is on (link expires after about a week).
Good enough for a live presentation.
