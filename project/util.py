import os
from pathlib import Path

ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = ROOT_DIR / 'data'

MUSIC_FANDOM_URL = "https://music.fandom.com/api.php"
WIKIPEDIA_URL = "https://en.wikipedia.org/w/api.php"
