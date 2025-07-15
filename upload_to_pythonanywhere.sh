import os
from datetime import datetime

# Your credentials
username = "sameenshahid"
remote_path = f"/home/{username}/nids_dashboard/nids_alerts.csv"
local_path = r"D:/multilabels/processed_data/nids_alerts.csv"

# SCP command (needs scp from Git Bash or WSL)
scp_command = f"scp {local_path} {username}@ssh.pythonanywhere.com:{remote_path}"

print(f"[{datetime.now()}] Uploading file to PythonAnywhere...")
os.system(scp_command)
print("Done.")
