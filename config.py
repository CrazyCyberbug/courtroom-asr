from pathlib import Path
import os


# dataset csv from assignment
DATASET_URL = "https://docs.google.com/spreadsheets/d/1JCIEEKTpBh1LG5jzSeEhz8wNzh7jcxKV146-9ClEaD8/export?format=csv&gid=0"

HF_USERNAME = "CrazyCyberBug2"

BASE_DIR = Path("data")
os.makedirs(BASE_DIR, exist_ok= True)


