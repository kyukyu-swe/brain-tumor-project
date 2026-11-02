"""
Languages for the explanations.

"official": listed by Google as supported by the Gemini API (ai.google.dev, "Available languages").
"official": False = beta: Gemini can usually write it, but Google does not officially support it,
so a native speaker should check the text. We keep these because many patients in Thailand and
Japan speak them (for example migrant workers).
"""

LANGUAGE_INFO = [
    # name (sent to Gemini)   native name            code   official
    ("English",               "English",              "en",  True),
    ("Arabic",                "العربية",              "ar",  True),
    ("Bengali",               "বাংলা",                "bn",  True),
    ("Bulgarian",             "Български",            "bg",  True),
    ("Chinese (Simplified)",  "简体中文",              "zh",  True),
    ("Chinese (Traditional)", "繁體中文",              "zh",  True),
    ("Croatian",              "Hrvatski",             "hr",  True),
    ("Czech",                 "Čeština",              "cs",  True),
    ("Danish",                "Dansk",                "da",  True),
    ("Dutch",                 "Nederlands",           "nl",  True),
    ("Estonian",              "Eesti",                "et",  True),
    ("Finnish",               "Suomi",                "fi",  True),
    ("French",                "Français",             "fr",  True),
    ("German",                "Deutsch",              "de",  True),
    ("Greek",                 "Ελληνικά",             "el",  True),
    ("Hebrew",                "עברית",                "he",  True),
    ("Hindi",                 "हिन्दी",                "hi",  True),
    ("Hungarian",             "Magyar",               "hu",  True),
    ("Indonesian",            "Bahasa Indonesia",     "id",  True),
    ("Italian",               "Italiano",             "it",  True),
    ("Japanese",              "日本語",                "ja",  True),
    ("Korean",                "한국어",                "ko",  True),
    ("Latvian",               "Latviešu",             "lv",  True),
    ("Lithuanian",            "Lietuvių",             "lt",  True),
    ("Norwegian",             "Norsk",                "no",  True),
    ("Polish",                "Polski",               "pl",  True),
    ("Portuguese",            "Português",            "pt",  True),
    ("Romanian",              "Română",               "ro",  True),
    ("Russian",               "Русский",              "ru",  True),
    ("Serbian",               "Српски",               "sr",  True),
    ("Slovak",                "Slovenčina",           "sk",  True),
    ("Slovenian",             "Slovenščina",          "sl",  True),
    ("Spanish",               "Español",              "es",  True),
    ("Swahili",               "Kiswahili",            "sw",  True),
    ("Swedish",               "Svenska",              "sv",  True),
    ("Thai",                  "ไทย",                  "th",  True),
    ("Turkish",               "Türkçe",               "tr",  True),
    ("Ukrainian",             "Українська",           "uk",  True),
    ("Vietnamese",            "Tiếng Việt",           "vi",  True),
    # ---- beta (not on Google's official list) ----
    ("Burmese (Myanmar)",     "မြန်မာ",               "my",  False),
    ("Khmer",                 "ខ្មែរ",                 "km",  False),
    ("Lao",                   "ລາວ",                  "lo",  False),
    ("Malay",                 "Bahasa Melayu",        "ms",  False),
    ("Filipino",              "Filipino",             "fil", False),
    ("Nepali",                "नेपाली",               "ne",  False),
]

RTL_CODES = {"ar", "he"}

LANGUAGES = [name for name, *_ in LANGUAGE_INFO]
SUGGESTED = ["English", "Thai", "Burmese (Myanmar)", "Japanese"]


def languages_json():
    """For the website: everything it needs to build the language picker."""
    return [
        {"name": n, "native": native, "code": code, "official": official, "rtl": code in RTL_CODES}
        for n, native, code, official in LANGUAGE_INFO
    ]


def is_beta(name):
    return any(n == name and not official for n, _, _, official in LANGUAGE_INFO)
