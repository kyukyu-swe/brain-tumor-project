// Where the AI backend (our Python Gradio app) runs.
//
// On your laptop:   "http://127.0.0.1:7860/"   (run `python app.py` first)
// After deploying:  your Hugging Face Space id, for example "la-min/brain-tumor-assistant"
//export const BACKEND = "http://127.0.0.1:7860/"; // for local testing

export const BACKEND = "https://brain-tumor-api-ji3g5bhe7a-an.a.run.app/";

// The full language list comes from the backend (core/languages.py).
// This short list is only used if the backend cannot be reached.
export const FALLBACK_LANGUAGES = [
  { name: "English", native: "English", code: "en", official: true, rtl: false },
  { name: "Thai", native: "ไทย", code: "th", official: true, rtl: false },
  { name: "Burmese (Myanmar)", native: "မြန်မာ", code: "my", official: false, rtl: false },
  { name: "Japanese", native: "日本語", code: "ja", official: true, rtl: false },
];
export const SUGGESTED_LANGUAGES = ["English", "Thai", "Burmese (Myanmar)", "Japanese"];

export const MAX_TRIAGE_FILES = 20;
