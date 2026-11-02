"""
Deploy the app to a free Hugging Face Space (ZeroGPU hardware).

One-time setup:
    pip install -U huggingface_hub
    hf auth login            # paste a token with WRITE access (huggingface.co/settings/tokens)

Deploy (run from the project folder):
    python scripts/deploy_to_hf.py --space YOUR_HF_USERNAME/brain-tumor-assistant --set-secrets

    --set-secrets   copies GEMINI_API_KEY / APP_PASSWORD from your local .env into the Space secrets
    --private       make the Space private (only you + invited people can open it)
    --dry-run       only list the files that would be uploaded

Run the same command again after you change the code; it uploads the new version.
"""
import argparse
import os
import sys
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]

# Files copied from the project (same path in the Space)
PROJECT_FILES = ["app.py", "core/__init__.py", "core/predictor.py", "core/llm.py",
                 "core/tumor_facts.json", "model/best_model.pth"]
SAMPLE_GLOB = "data/samples/*/*"
# Space-only files: deploy/<name> -> <name> in the Space
DEPLOY_FILES = ["README.md", "requirements.txt", "packages.txt"]

SECRET_KEYS = ["GEMINI_API_KEY", "APP_PASSWORD"]
VARIABLE_KEYS = ["GEMINI_MODEL", "APP_USERNAME", "MAX_LLM_CALLS_PER_HOUR"]


def collect_files():
    files = []
    for f in PROJECT_FILES:
        p = ROOT / f
        if not p.is_file():
            sys.exit(f"Missing file: {f}")
        files.append((p, f))
    samples = sorted(p for p in ROOT.glob(SAMPLE_GLOB) if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not samples:
        print("Warning: no images in data/samples/ — the 'Load sample scans' button will not work.")
    files += [(p, p.relative_to(ROOT).as_posix()) for p in samples]
    files += [(ROOT / "deploy" / f, f) for f in DEPLOY_FILES]
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", required=True, help="e.g. la-min/brain-tumor-assistant")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--set-secrets", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    files = collect_files()
    size = sum(p.stat().st_size for p, _ in files) / 1e6
    print(f"{len(files)} files, {size:.1f} MB")
    if args.dry_run:
        for p, dest in files:
            print("  ", dest)
        return

    from huggingface_hub import CommitOperationAdd, HfApi
    api = HfApi()
    print("Logged in as:", api.whoami()["name"])

    try:
        api.create_repo(args.space, repo_type="space", space_sdk="gradio", space_hardware="zero-a10g",
                        private=args.private, exist_ok=True)
    except Exception as e:
        sys.exit(f"Could not create the Space automatically: {e}\n"
                 "Create it in the browser instead (huggingface.co/new-space → SDK: Gradio → "
                 "Hardware: ZeroGPU), then run this command again.")

    ops = [CommitOperationAdd(path_in_repo=dest, path_or_fileobj=str(p)) for p, dest in files]
    api.create_commit(args.space, repo_type="space", operations=ops, commit_message="Deploy app")
    print("Uploaded.")

    if args.set_secrets:
        env = dotenv_values(ROOT / ".env")
        for k in SECRET_KEYS:
            if env.get(k):
                api.add_space_secret(args.space, k, env[k]); print("secret set:", k)
        for k in VARIABLE_KEYS:
            if env.get(k):
                api.add_space_variable(args.space, k, env[k]); print("variable set:", k)

    print(f"\nDone. The Space is building (first build takes ~5-10 min):\n  https://huggingface.co/spaces/{args.space}")


if __name__ == "__main__":
    main()
