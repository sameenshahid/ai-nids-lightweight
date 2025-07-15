🚨 AI-Powered Lightweight Network Intrusion Detection System (NIDS)

A real-time, AI-powered Network Intrusion Detection System (NIDS) designed for **lightweight and edge devices** like Raspberry Pi, Android phones, and iPads. This system uses **TinyBERT** for semantic inspection and **LightGBM** for fast multiclass classification of network traffic. It features live packet sniffing, hybrid AI classification, and a fully interactive Dash dashboard for real-time threat visualization — with **automated email alerts** on intrusion detection.


🌟 Key Features

- 🧠 Hybrid AI model (TinyBERT + LightGBM)
- 🔍 Real-time packet sniffing and parsing
- 📊 Dash dashboard with live threat updates
- 📧 **Automated email alerts** for suspicious activity
- 🪶 Lightweight design optimized for edge devices
- 🧩 Modular codebase with ONNX + quantization support
- ☁️ Deployment-ready (PythonAnywhere / Raspberry Pi)



Project Structure

AI-POWERED-NIDS/
├── assets/
│ └── custom.css
├── results/
│ ├── full_model/
│ ├── classification_report.txt
│ ├── confusion_matrix.png
│ ├── tinybert_model.pth
│ └── tinybert_onnx/
├── utils/
│ └── packet_utils.py
├── lightgbm1.py
├── main.py
├── nids_alerts.csv
├── processed data.zip
├── quantization.py
├── test.py
├── tinybert_label.py
├── tinybert_lgbm_pipeline.pkl
├── upload_to_pythonanywhere.sh
├── .gitignore
├── README.md

yaml
Copy
Edit

---

 Getting Started

1. Clone the Repository


git clone https://github.com/sameenshahid/ai-nids-lightweight.git
cd ai-nids-lightweight

2. (Optional) Create a Virtual Environment

python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

3. Install Required Packages

pip install -r requirements.txt

🚀 Running the System
To start the full NIDS pipeline:
python main.py
It will:

Start capturing network packets

Run TinyBERT + LightGBM detection

Send email alerts for malicious traffic

Launch a Dash dashboard in your browser

📧 Email Alerting System
This system includes automatic alerting via email whenever a potential threat or attack is detected. You can configure the sender/receiver settings in your alert module (utils/ or inside main.py).

Make sure to:

Allow access for less secure apps (if using Gmail)

Provide an app password if 2FA is enabled

📊 Visualization & Outputs
confusion_matrix.png: Visual display of model performance

classification_report.txt: Precision, recall, and F1 scores

nids_alerts.csv: Logged alerts with timestamps and predicted labels

🧠 Models Used
tinybert_model.pth: TinyBERT model for semantic feature extraction

tinybert_lgbm_pipeline.pkl: Pre-trained LightGBM classifier

tinybert_onnx/: Exported ONNX version for edge devices

quantization.py: Script for optimizing model size and inference

Optimized For

Linux-based lightweight virtual servers




Author
Sameen Shahid
GitHub: @sameenshahid
📧 Email alert system integrated and tested.
