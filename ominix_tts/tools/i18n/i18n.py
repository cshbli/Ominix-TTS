import json
import locale
import os
import importlib.resources
import pkg_resources

# I18N_JSON_DIR : os.PathLike = os.path.join(os.path.dirname(os.path.relpath(__file__)), 'locale')

# def load_language_list(language):
#     with open(os.path.join(I18N_JSON_DIR, f"{language}.json"), "r", encoding="utf-8") as f:
#         language_list = json.load(f)
#     return language_list

# Use importlib.resources to find the locale directory
try:
    # Python 3.9+
    with importlib.resources.files("ominix_tts.tools.i18n") as p:
        I18N_JSON_DIR = p / "locale"
except AttributeError:
    # Fallback for earlier Python versions
    I18N_JSON_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "locale")
    # If that fails, try to use pkg_resources
    if not os.path.exists(I18N_JSON_DIR):
        I18N_JSON_DIR = os.path.join(pkg_resources.resource_filename("ominix_tts.tools.i18n", ""), "locale")

def load_language_list(language):
    try:
        with open(os.path.join(I18N_JSON_DIR, f"{language}.json"), "r", encoding="utf-8") as f:
            language_list = json.load(f)
        return language_list
    except FileNotFoundError as e:
        print(f"Error: Language file not found: {os.path.join(I18N_JSON_DIR, f'{language}.json')}")
        print(f"Available path: {I18N_JSON_DIR}")
        print(f"Files in directory: {os.listdir(I18N_JSON_DIR) if os.path.exists(I18N_JSON_DIR) else 'Directory not found'}")
        # Fall back to English if available
        if language != "en_US":
            try:
                return load_language_list("en_US")
            except:
                pass
        # Return an empty dict as last resort
        return {}

def scan_language_list():
    language_list = []
    for name in os.listdir(I18N_JSON_DIR):
        if name.endswith(".json"):language_list.append(name.split('.')[0])
    return language_list

class I18nAuto:
    def __init__(self, language=None):
        if language in ["Auto", None]:
            language = locale.getdefaultlocale()[0]  
            # getlocale can't identify the system's language ((None, None))
        if not os.path.exists(os.path.join(I18N_JSON_DIR, f"{language}.json")):
            language = "en_US"
        self.language = language
        self.language_map = load_language_list(language)

    def __call__(self, key):
        return self.language_map.get(key, key)

    def __repr__(self):
        return "Use Language: " + self.language

if __name__ == "__main__":
    i18n = I18nAuto(language='en_US')
    print(i18n)