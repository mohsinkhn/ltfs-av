from pathlib import Path

ROOT = "data"
UTILITY = "utility"
SUBMISSIONS = "submissions"
LOGS = "logs"

Path(UTILITY).mkdir(exist_ok=True)
Path(SUBMISSIONS).mkdir(exist_ok=True)
Path(LOGS).mkdir(exist_ok=True)

