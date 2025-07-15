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
    from_email = "example@gmail.com"
    from_password = "######"
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
                                "example@gmail.com"
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

USER_CREDENTIALS = {"admin": "password123"}

external_stylesheets = [
    dbc.themes.FLATLY,
    "https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap"
]

app = Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "NIDS Threat Dashboard"

# Global Layout
app.layout = html.Div([
    dcc.Location(id="url"),
    dcc.Store(id='auth-status', storage_type='session'),
    html.Div([
        dbc.Navbar(
            dbc.Container([
                html.A([
                    dbc.Row([
                        dbc.Col(html.Img(src="/assets/logo.png", height="45px", style={'marginRight': '15px'})),
                        dbc.Col(dbc.NavbarBrand("AI-Powered NIDS", className="ms-2", style={
                            'fontSize': '24px', 'fontWeight': '700', 'color': '#ffffff',
                            'fontFamily': 'Poppins'}))
                    ], align="center", className="g-0")
                ], href="#", style={"textDecoration": "none"}),

                dbc.Button("Logout", id="logout-button", style={
                    'borderRadius': '8px',
                    'fontSize': '14px',
                    'fontWeight': '600',
                    'padding': '6px 14px',
                    'backgroundColor': '#484848',
                    'color': 'white',
                    'border': 'none'
                }, className='ms-auto')
            ]),
            color="dark", dark=True, style={'backgroundColor': '#20283E', 'padding': '8px 0'}
        ),
        html.Div(id="page-content")
    ])
], style={
    'fontFamily': 'Poppins, Segoe UI, sans-serif',
    'backgroundColor': '#DADADA',
    'minHeight': '100vh',
    'padding': '0'
})

# Login Layout
login_layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Img(src="/assets/logo.png", style={
                    'width': '120px', 'marginBottom': '20px',
                    'filter': 'drop-shadow(0 4px 8px rgba(0,0,0,0.15))'}),
                html.H2("Secure Access", className="mb-3", style={'color': '#000000', 'fontWeight': '700'}),
                html.H5("Login to NIDS Dashboard", className="mb-4", style={'color': '#488A99'}),
                dbc.Input(id="username", placeholder="Username", type="text", className="mb-3"),
                dbc.Input(id="password", placeholder="Password", type="password", className="mb-3"),
                dbc.Button("Login", id="login-button", className="w-100", style={
                    'backgroundColor': 'rgb(72, 72, 72)', 'border': 'none',
                    'fontWeight': '600', 'fontSize': '16px'
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

# Dashboard Layout
dashboard_layout = html.Div([

    html.H1("Threat Analysis Dashboard", style={
        'textAlign': 'center',
        'fontWeight': '700',
        'color': '#000000',
        'marginTop': '20px',
        'marginBottom': '15px',
        'fontSize': '30px'
    }),

    html.Hr(style={'borderTop': '1px solid #999'}),

    dbc.Row([
        dbc.Col(
            dbc.Card([
                dbc.CardBody([
                    html.H5("CPU Usage", style={'fontSize': '16px', 'textAlign': 'center', 'color': '#484848'}),
                    html.P(id='cpu-usage', style={
                        'fontWeight': '700', 'fontSize': '26px', 'color': '#DBAE58', 'textAlign': 'center'})
                ])
            ], style={'backgroundColor': '#ffffff', 'borderRadius': '14px',
                      'boxShadow': '0 3px 8px rgba(0,0,0,0.1)', 'padding': '10px'}),
            width=6
        ),
        dbc.Col(
            dbc.Card([
                dbc.CardBody([
                    html.H5("Memory Usage", style={'fontSize': '16px', 'textAlign': 'center', 'color': '#484848'}),
                    html.P(id='memory-usage', style={
                        'fontWeight': '700', 'fontSize': '26px', 'color': '#488A99', 'textAlign': 'center'})
                ])
            ], style={'backgroundColor': '#ffffff', 'borderRadius': '14px',
                      'boxShadow': '0 3px 8px rgba(0,0,0,0.1)', 'padding': '10px'}),
            width=6
        ),
    ], className="mb-4", style={'margin': '0 5%'}),

    dcc.Interval(id='interval-component', interval=5000, n_intervals=0),

    html.Div([
        html.Label("Confidence Threshold", style={
            'fontWeight': '600', 'color': '#000000', 'fontSize': '15px', 'marginBottom': '8px'
        }),
        dcc.Slider(id='confidence-slider', min=0.0, max=1.0, step=0.01, marks={
            0.0: '0', 0.5: '0.5', 0.75: '0.75', 1.0: '1.0'
        }, value=0.0, tooltip={"placement": "bottom", "always_visible": False}, updatemode='drag'),
        html.Br(),
        dbc.Checklist(
            options=[{'label': ' High Severity Only (≥ 0.8)', 'value': 'high'}],
            value=[], id='severity-filter', switch=True,
            style={'fontWeight': '600', 'color': '#000000', 'fontSize': '14px'}
        )
    ], style={
        'backgroundColor': '#ffffff', 'padding': '18px',
        'borderRadius': '14px', 'boxShadow': '0 3px 10px rgba(0,0,0,0.15)',
        'marginBottom': '30px', 'width': '50%', 'margin': '20px auto'
    }),

    dbc.Row([
        dbc.Col(
            dbc.Card([
                dbc.CardHeader("Threat Type Counts", style={
                    'fontWeight': '700', 'textAlign': 'center', 'backgroundColor': '#20283E', 'color': 'white'}),
                dbc.CardBody([dcc.Graph(id='attack-bar-chart')])
            ], style={'backgroundColor': '#ffffff', 'borderRadius': '14px',
                      'boxShadow': '0 3px 8px rgba(0,0,0,0.1)'}),
            width=6
        ),
        dbc.Col(
            dbc.Card([
                dbc.CardHeader("Threat Trends Over Time", style={
                    'fontWeight': '700', 'textAlign': 'center', 'backgroundColor': '#20283E', 'color': 'white'}),
                dbc.CardBody([dcc.Graph(id='attack-time-series')])
            ], style={'backgroundColor': '#ffffff', 'borderRadius': '14px',
                      'boxShadow': '0 3px 8px rgba(0,0,0,0.1)'}),
            width=6
        ),
    ], className="mb-4", style={'margin': '0 5%'}),

    html.Div([
        dbc.Card([
            dbc.CardHeader("Recent Detections", style={
                'fontWeight': '700', 'textAlign': 'center',
                'backgroundColor': '#20283E', 'color': 'white'
            }),
            dbc.CardBody([
                dash_table.DataTable(
                    id='recent-attacks-table',
                    columns=[{"name": i, "id": i} for i in ["Timestamp", "Label", "Confidence", "Packet_Summary"]],
                    style_table={'overflowX': 'auto'},
                    style_cell={
                        'textAlign': 'center',
                        'padding': '10px',
                        'fontFamily': 'Poppins',
                        'fontSize': '13px',
                        'backgroundColor': '#fafafa'
                    },
                    style_header={
                        'backgroundColor': '#000000',
                        'color': 'white',
                        'fontWeight': 'bold'
                    },
                    style_data_conditional=[
                        {'if': {'row_index': 'odd'}, 'backgroundColor': '#DADADA'}
                    ],
                    page_size=10
                )
            ])
        ], style={'backgroundColor': '#ffffff', 'borderRadius': '14px',
                  'boxShadow': '0 3px 10px rgba(0,0,0,0.1)'})
    ], style={'width': '90%', 'margin': 'auto', 'marginBottom': '40px'})
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

    # Updated color palette (new one)
    colors = ['#7E909A', '#1C4E80', '#A5D8DD', '#EA6A47', '#0091D5']

    # Bar Chart
    bar_fig = px.bar(
        label_counts,
        x='Label',
        y='count',
        title='Detected Attacks by Type',
        color='Label',
        color_discrete_sequence=colors
    )

    # Time Series Chart
    trend_df = df.groupby([pd.Grouper(key='Timestamp', freq='1min'), 'Label']).size().reset_index(name='count')
    time_fig = px.line(
        trend_df,
        x='Timestamp',
        y='count',
        color='Label',
        title='Attack Frequency Over Time',
        color_discrete_sequence=colors
    )

    recent_data = df.sort_values(by='Timestamp', ascending=False).head(10).to_dict('records')

    return bar_fig, time_fig, recent_data, f"{cpu_percent} %", f"{mem_percent} %"

if __name__ == '__main__':
    app.run(debug=False)
