// Where the AI backend (our Python Gradio app) runs.
//
// On your laptop:   "http://127.0.0.1:7860/"   (run `python app.py` first)
// After deploying:  your Hugging Face Space id, for example "la-min/brain-tumor-assistant"
//export const BACKEND = "http://127.0.0.1:7860/"; // for local testing

export const BACKEND = "https://brain-tumor-api-ji3g5bhe7a-an.a.run.app/";

// Must match LANGUAGES in core/llm.py
export const LANGUAGES = [
  { value: "English", label: "English" },
  { value: "Thai", label: "ไทย Thai" },
  { value: "Burmese (Myanmar)", label: "မြန်မာ Burmese" },
  { value: "Japanese", label: "日本語 Japanese" },
];

export const MAX_TRIAGE_FILES = 20;
