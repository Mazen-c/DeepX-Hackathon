from pathlib import Path
import time

Path("pythonw_alive.txt").write_text("alive", encoding="utf-8")
time.sleep(60)
