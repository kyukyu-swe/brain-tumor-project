#!/usr/bin/env bash
# Deploy the AI backend to Google Cloud Run (free tier).
# Run from the project folder:   bash scripts/deploy_cloudrun.sh
# First-time setup: see DEPLOY_GUIDE.md (install gcloud, log in, choose project).
set -euo pipefail
cd "$(dirname "$0")/.."

SERVICE="${SERVICE:-brain-tumor-api}"
REGION="${REGION:-asia-northeast1}"   # Tokyo

if [ ! -f .env ]; then echo "No .env file found (needs GEMINI_API_KEY)."; exit 1; fi
set -a; source .env; set +a
if [ -z "${GEMINI_API_KEY:-}" ]; then echo "GEMINI_API_KEY is empty in .env"; exit 1; fi
if [ ! -f model/best_model.pth ]; then echo "model/best_model.pth not found"; exit 1; fi

echo "Deploying '$SERVICE' to region $REGION (project: $(gcloud config get-value project 2>/dev/null))"
gcloud run deploy "$SERVICE" \
  --source . \
  --region "$REGION" \
  --allow-unauthenticated \
  --cpu 2 --memory 2Gi \
  --min-instances 0 --max-instances 1 \
  --concurrency 10 --timeout 300 \
  --session-affinity \
  --set-env-vars "^@^GEMINI_API_KEY=${GEMINI_API_KEY}@GEMINI_MODEL=${GEMINI_MODEL:-gemini-flash-latest}@MAX_LLM_CALLS_PER_HOUR=${MAX_LLM_CALLS_PER_HOUR:-100}"

URL=$(gcloud run services describe "$SERVICE" --region "$REGION" --format 'value(status.url)')
echo ""
echo "Done! Backend URL: $URL"
echo "Next: put this in web/js/config.js ->  export const BACKEND = \"$URL/\";"
