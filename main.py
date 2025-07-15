# === Imports ===
import torch
import numpy as np
import joblib
import time
import csv
import threading
import os
from datetime import datetime
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from utils.packet_utils import get_capture, packet_to_text
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import asyncio
import psutil  # << Added for CPU and memory monitoring

# Dash imports
import pandas as pd
from dash import Dash, dcc, html, dash_table, no_update
from dash.dependencies import Input, Output, State
import plotly.express as px
import dash_bootstrap_components as dbc

# === Device & Paths ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = r"D:\faizan\results\full_model"
log_file = r"D:\faizan\nids_alerts.csv"
pipeline_path = "tinybert_lgbm_pipeline.pkl"

# === Load models ===
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
bert_classifier = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True).to(device).eval()
bert_model = AutoModel.from_pretrained(model_path, local_files_only=True).to(device).eval()
pipeline = joblib.load(pipeline_path)

# === Label map ===
label_map = {
    0: "BENIGN", 1: "DoS Hulk", 2: "DoS GoldenEye", 3: "DoS slowloris", 4: "DoS Slowhttptest",
    5: "DDoS", 6: "PortScan", 7: "Brute Force -XSS", 8: "Brute Force -Web",
    9: "Brute Force -Sql Injection", 10: "Bot", 11: "Infiltration",
    12: "Heartbleed", 13: "FTP-Patator", 14: "SSH-Patator"
}

# === Log header ===
if not os.path.exists(log_file) or os.stat(log_file).st_size == 0:
    with open(log_file, "w", newline="") as f:
        csv.writer(f).writerow(["Timestamp", "Label", "Confidence", "Packet_Summary"])

# === Utility Functions ===
def send_email_alert(subject, body, to_email):
    from_email = "####"
    from_password = "#####"
    msg = MIMEMultipart()
    msg["From"], msg["To"], msg["Subject"] = from_email, to_email, subject
    msg.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(from_email, from_password)
        server.sendmail(from_email, to_email, msg.as_string())
        server.quit()
    except Exception as e:
        print(f"[ERROR] Email failed: {e}")

def get_bert_probs(texts, batch_size=32):
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tokenizer(texts[i:i + batch_size], padding=True, truncation=True, return_tensors="pt", max_length=128).to(device)
            probs = torch.softmax(bert_classifier(**enc).logits, dim=1).cpu().numpy()
            all_probs.append(probs)
    return np.vstack(all_probs)

def get_cls_embeddings(texts, batch_size=32):
    all_embeds = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tokenizer(texts[i:i + batch_size], padding=True, truncation=True, return_tensors="pt", max_length=128).to(device)
            cls_tok = bert_model(**enc).last_hidden_state[:, 0, :]
            all_embeds.append(cls_tok.cpu().numpy())
    return np.vstack(all_embeds)

def run_detection_loop(mode, source):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    capture = get_capture(mode, source)
    buffer = []
    batch_size = 16

    while True:
        try:
            for packet in capture.sniff_continuously():
                pkt_text = packet_to_text(packet)
                buffer.append(pkt_text)

                if len(buffer) >= batch_size:
                    bert_probs = get_bert_probs(buffer)
                    cls_embeddings = get_cls_embeddings(buffer)
                    lgbm_probs = pipeline.predict_proba(cls_embeddings)

                    num_classes = lgbm_probs.shape[1]
                    severity = np.array([0.1, 0.8, 0.8, 0.9, 0.9, 0.85, 0.75,
                                         0.6, 0.65, 0.7, 0.85, 0.9, 0.95, 0.7, 0.7])[:num_classes]

                    final_probs = 0.6 * bert_probs[:, :num_classes] + 0.4 * lgbm_probs[:, :num_classes]
                    max_conf = np.max(final_probs, axis=1, keepdims=True)
                    low_conf_mask = (max_conf < 0.75).astype(float)
                    adjusted = final_probs * severity * (1 + 0.4 * low_conf_mask)
                    preds = np.argmax(adjusted, axis=1)

                    trimmed_label_map = {i: label_map[i] for i in range(num_classes) if i in label_map}
                    for pkt, idx, conf in zip(buffer, preds, max_conf):
                        label = trimmed_label_map.get(idx, 'Unknown')
                        if label.lower() != 'benign' and conf >= 0.8:
                            send_email_alert(
                                f"NIDS Alert: {label}",
                                f"Detected:\n\n{pkt[:120]}...\nConfidence: {conf[0]:.2f}",
                                "example@email.com"
                            )

                        with open(log_file, "a", newline="") as f:
                            csv.writer(f).writerow([
                                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                label, f"{conf[0]:.2f}", pkt[:120]
                            ])
                    buffer.clear()

                if mode == "live":
                    time.sleep(0.9)

        except Exception as e:
            print(f"[ERROR] Detection loop failed: {e}")

# === Start Detection in Background ===
detection_thread = threading.Thread(target=run_detection_loop, args=("live", "Wi-Fi"), daemon=True)
detection_thread.start()

# === DASH APP WITH LOGIN ===
USER_CREDENTIALS = {"admin": "password123"}

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "NIDS Threat Dashboard"

app.layout = html.Div([
    dcc.Location(id="url"),
    dcc.Store(id='auth-status', storage_type='session'),
    html.Div([
        dbc.Navbar(
            dbc.Container([
                html.A([
                    dbc.Row([
                        dbc.Col(html.Img(src="/assets/logo.png", height="40px")),
                        dbc.Col(dbc.NavbarBrand("AI-Powered NIDS", className="ms-2", style={
                            'fontSize': '32px', 'fontWeight': 'bold', 'color': 'rgb(44, 62, 80)'})
                        )
                    ], align="center", className="g-0")
                ], href="#", style={"textDecoration": "none"})
            ]),
            color="light",
            dark=False,
            style={'boxShadow': '0 2px 6px rgba(0,0,0,0.1)'}
        ),
        html.Div(id="page-content")
    ])
], style={
    'fontFamily': 'Segoe UI, sans-serif',
    'backgroundColor': '#f0f4f8',
    'minHeight': '100vh',
    'padding': '0'
})

login_layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Img(src="/assets/logo.png", style={
                    'width': '120px',
                    'marginBottom': '20px',
                    'filter': 'drop-shadow(0 4px 4px rgba(66, 133, 244, 0.3))'}),
                html.H2("Secure Access", className="mb-3", style={'color': '#4285F4', 'fontWeight': 'bold'}),
                html.H5("Login to NIDS Dashboard", className="mb-4"),
                dbc.Input(id="username", placeholder="Username", type="text", className="mb-3"),
                dbc.Input(id="password", placeholder="Password", type="password", className="mb-3"),
                dbc.Button("Login", id="login-button", color="primary", className="w-100", style={
                    'backgroundColor': '#34A853',
                    'border': 'none',
                    'fontWeight': 'bold'
                }),
                html.Div(id="login-alert", className="mt-3")
            ], style={
                'textAlign': 'center',
                'padding': '50px',
                'borderRadius': '16px',
                'backgroundColor': '#ffffff',
                'boxShadow': '0 8px 24px rgba(0,0,0,0.1)',
                'marginTop': '80px'
            })
        ], width=5)
    ], justify="center")
])

# dashboard_layout remains unchanged from previous version


dashboard_layout = html.Div([
    html.Div([
        dbc.Button("Logout", id="logout-button", color="primary", className="mb-3", style={
            'borderRadius': '20px',
            'float': 'right',
            'marginRight': '20px'
        })
    ]),

    html.H1("NIDS Threat Dashboard", style={
        'textAlign': 'center',
        'fontWeight': 'bold',
        'color': '#2c3e50',
        'marginTop': '20px',
        'marginBottom': '30px',
        'fontSize': '36px'
    }),

    html.Hr(style={'borderTop': '1px solid #ccc'}),

    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H5("CPU Usage", className="card-title"),
                html.P(id='cpu-usage', className="card-text", style={'fontWeight': 'bold', 'fontSize': '20px', 'color': '#007bff'})
            ])
        ], color="light", inverse=False), width=6),

        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H5("Memory Usage", className="card-title"),
                html.P(id='memory-usage', className="card-text", style={'fontWeight': 'bold', 'fontSize': '20px', 'color': '#28a745'})
            ])
        ], color="light", inverse=False), width=6)
    ], className="mb-4"),

    dcc.Interval(id='interval-component', interval=5000, n_intervals=0),

    html.Div([
        html.Label("Minimum Confidence Threshold:", style={'fontWeight': 'bold'}),
        dcc.Slider(id='confidence-slider', min=0.0, max=1.0, step=0.01,
                   marks={0.0: '0.0', 0.5: '0.5', 0.75: '0.75', 1.0: '1.0'}, value=0.0),
        html.Br(),
        html.Label("Show Only High Severity (Confidence ≥ 0.8):", style={'fontWeight': 'bold'}),
        dcc.Checklist(id='severity-filter', options=[{'label': 'Enable', 'value': 'high'}], value=[])
    ], style={
        'backgroundColor': '#ffffff',
        'padding': '20px',
        'borderRadius': '10px',
        'boxShadow': '0 2px 6px rgba(0,0,0,0.1)',
        'marginBottom': '20px',
        'width': '60%',
        'margin': 'auto'
    }),

    dbc.Card([
        dbc.CardHeader("Threat Type Counts", style={'fontWeight': 'bold'}),
        dbc.CardBody([
            dcc.Graph(id='attack-bar-chart')
        ])
    ], className="mb-4", style={'backgroundColor': 'white', 'boxShadow': '0 2px 6px rgba(0,0,0,0.1)', 'borderRadius': '10px'}),

    dbc.Card([
        dbc.CardHeader("Threat Trends Over Time", style={'fontWeight': 'bold'}),
        dbc.CardBody([
            dcc.Graph(id='attack-time-series')
        ])
    ], className="mb-4", style={'backgroundColor': 'white', 'boxShadow': '0 2px 6px rgba(0,0,0,0.1)', 'borderRadius': '10px'}),

    dbc.Card([
        dbc.CardHeader("Recent Detections", style={'fontWeight': 'bold'}),
        dbc.CardBody([
            dash_table.DataTable(
                id='recent-attacks-table',
                columns=[{"name": i, "id": i} for i in ["Timestamp", "Label", "Confidence", "Packet_Summary"]],
                style_table={'overflowX': 'auto'},
                style_cell={
                    'textAlign': 'center',
                    'padding': '10px',
                    'fontFamily': 'Segoe UI',
                    'fontSize': '14px',
                    'backgroundColor': '#fefefe'
                },
                style_header={
                    'backgroundColor': '#2c3e50',
                    'color': 'white',
                    'fontWeight': 'bold'
                },
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': '#f7f7f7'
                    }
                ],
                page_size=10
            )
        ])
    ], style={'backgroundColor': 'white', 'boxShadow': '0 2px 6px rgba(0,0,0,0.1)', 'borderRadius': '10px'})
])


# === Callbacks ===
@app.callback(
    Output("auth-status", "data"),
    Output("login-alert", "children"),
    Input("login-button", "n_clicks"),
    State("username", "value"),
    State("password", "value"),
    prevent_initial_call=True
)
def login(n_clicks, username, password):
    if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
        return True, ""
    return False, dbc.Alert("Invalid credentials", color="danger")

@app.callback(
    Output("page-content", "children"),
    Input("auth-status", "data")
)
def display_page(auth_status):
    if auth_status:
        return dashboard_layout
    return login_layout

@app.callback(
    Output('auth-status', 'clear_data'),
    Input('logout-button', 'n_clicks'),
    prevent_initial_call=True
)
def logout(n_clicks):
    if n_clicks:
        return True
    return dash.no_update


@app.callback(
    Output('attack-bar-chart', 'figure'),
    Output('attack-time-series', 'figure'),
    Output('recent-attacks-table', 'data'),
    Output('cpu-usage', 'children'),
    Output('memory-usage', 'children'),
    Input('interval-component', 'n_intervals'),
    State('confidence-slider', 'value'),
    State('severity-filter', 'value')
)
def update_dashboard(n, confidence_threshold, severity_filter):
    # CPU and Memory Usage
    cpu_percent = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory()
    mem_percent = mem.percent

    if not os.path.exists(log_file):
        return {}, {}, [], f"{cpu_percent} %", f"{mem_percent} %"

    df = pd.read_csv(log_file)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Confidence'] = df['Confidence'].astype(float)
    df = df[df['Confidence'] >= confidence_threshold]
    if 'high' in severity_filter:
        df = df[df['Confidence'] >= 0.8]
    if df.empty:
        return {}, {}, [], f"{cpu_percent} %", f"{mem_percent} %"

    label_counts = df['Label'].value_counts().reset_index()
    label_counts.columns = ['Label', 'count']
    bar_fig = px.bar(label_counts, x='Label', y='count', title='Detected Attacks by Type')

    trend_df = df.groupby([pd.Grouper(key='Timestamp', freq='1min'), 'Label']).size().reset_index(name='count')
    time_fig = px.line(trend_df, x='Timestamp', y='count', color='Label', title='Attack Frequency Over Time')

    recent_data = df.sort_values(by='Timestamp', ascending=False).head(10).to_dict('records')

    return bar_fig, time_fig, recent_data, f"{cpu_percent} %", f"{mem_percent} %"

if __name__ == '__main__':
    app.run(debug=False)
