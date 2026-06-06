"""
Valencia Emocional — comparador de 3 modelos (NN1 MLP / NN2 LSTM / NN3 BiGRU+Attn).
Sube un video, extrae Action Units con MediaPipe, predice valencia y compara los
tres modelos en el tiempo.

Features: upload, gráfica valencia-vs-tiempo (3 modelos), toggle de puntos de
tracking, reproducción, reset, muestreo ajustable, comparación viva (stats + acuerdo).
"""
import json, tempfile, time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import torch
import torch.nn as nn
import joblib
import cv2
import imageio
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ── Paleta (data-viz: un color por modelo) ────────────────────────
COL = {
    "NN1 MLP":        "#6FB1E0",
    "NN2 LSTM":       "#4FD0A8",
    "NN3 BiGRU+Attn": "#F07167",
}
ROL = {
    "NN1 MLP":        "Baseline",
    "NN2 LSTM":       "Recurrente",
    "NN3 BiGRU+Attn": "Atención",
}
ACCENT = "#E0A33C"; MUTED = "#A8A296"; PANEL = "#1F1D18"
BORDER = "#322E26"; TEXT = "#ECE8E0"; BG = "#161512"

AQUI = Path(__file__).parent
CFG  = json.loads((AQUI / "model_config.json").read_text())
TIMESTEPS = CFG["TIMESTEPS"]; GRU_UNITS = CFG["GRU_UNITS"]
NUM_HEADS = CFG["NUM_HEADS"]; DROPOUT = CFG["DROPOUT"]
AU_COLS = CFG["AU_COLS"]; ESCALA = CFG["ESCALA_FACS"]
TASK = str(AQUI / "face_landmarker.task")

AU_MAP = {
    "AU01_r": ["browInnerUp"], "AU02_r": ["browOuterUpLeft", "browOuterUpRight"],
    "AU04_r": ["browDownLeft", "browDownRight"], "AU05_r": ["eyeWideLeft", "eyeWideRight"],
    "AU06_r": ["cheekSquintLeft", "cheekSquintRight"], "AU07_r": ["eyeSquintLeft", "eyeSquintRight"],
    "AU12_r": ["mouthSmileLeft", "mouthSmileRight"], "AU14_r": ["mouthDimpleLeft", "mouthDimpleRight"],
    "AU15_r": ["mouthFrownLeft", "mouthFrownRight"],
    "AU17_r": ["mouthShrugUpper", "mouthPressLeft", "mouthPressRight"],
    "AU20_r": ["mouthStretchLeft", "mouthStretchRight"], "AU25_r": ["jawOpen"],
}

class LSTMModel(nn.Module):
    def __init__(self, n_features=12, hidden=64, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout); self.fc = nn.Linear(hidden, 1); self.tanh = nn.Tanh()
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.tanh(self.fc(self.drop(out[:, -1, :])))

class BiGRUAttention(nn.Module):
    def __init__(self, n_features=12, gru_units=32, num_heads=4, dropout=0.3):
        super().__init__()
        hidden = gru_units * 2
        self.bigru = nn.GRU(n_features, gru_units, batch_first=True, bidirectional=True)
        self.ln1 = nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(hidden, num_heads, dropout=dropout, batch_first=True)
        self.fc1 = nn.Linear(hidden, 32); self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU(); self.tanh = nn.Tanh(); self.drop = nn.Dropout(dropout)
    def forward(self, x):
        out, _ = self.bigru(x); out = self.ln1(out)
        attn, _ = self.attn(out, out, out)
        out = (out + attn).mean(dim=1)
        out = self.drop(self.relu(self.fc1(out)))
        return self.tanh(self.fc2(out))

@st.cache_resource(show_spinner=False)
def cargar_modelos():
    scaler = joblib.load(AQUI / "scaler.pkl")
    mlp = joblib.load(AQUI / "modelo_mlp.pkl")
    lstm = LSTMModel(); lstm.load_state_dict(torch.load(AQUI / "modelo_lstm.pt", map_location="cpu")); lstm.eval()
    bigru = BiGRUAttention(12, GRU_UNITS, NUM_HEADS, DROPOUT)
    bigru.load_state_dict(torch.load(AQUI / "modelo_bigru.pt", map_location="cpu")); bigru.eval()
    return scaler, mlp, lstm, bigru

@st.cache_resource(show_spinner=False)
def cargar_detector():
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=TASK),
        output_face_blendshapes=True, num_faces=1,
        running_mode=mp_vision.RunningMode.IMAGE)
    return mp_vision.FaceLandmarker.create_from_options(opts)

def bs_to_aus(bs):
    d = {b.category_name: b.score for b in bs}
    return [float(np.clip(np.mean([d.get(n, 0.0) for n in AU_MAP[au]]) * ESCALA, 0, 5)) for au in AU_COLS]

def ccc(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 2: return float("nan")
    cov = ((a - a.mean()) * (b - b.mean())).mean()
    return float(2 * cov / (a.var() + b.var() + (a.mean() - b.mean())**2 + 1e-8))

def procesar_video(ruta, stride, dibujar, max_seg, detector, prog):
    cap = cv2.VideoCapture(str(ruta))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    max_frames = int(max_seg * fps) if max_seg else total
    tiempos, aus, anot = [], [], []
    fn = 0
    while True:
        ok, frame = cap.read()
        if not ok or fn >= max_frames: break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        muestrear = (fn % stride == 0)
        if muestrear or dibujar:
            res = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            tiene = bool(res.face_blendshapes)
            if muestrear and tiene:
                tiempos.append(fn / fps); aus.append(bs_to_aus(res.face_blendshapes[0]))
            if dibujar:
                if tiene:
                    h, w = rgb.shape[:2]
                    for lm in res.face_landmarks[0]:
                        cv2.circle(rgb, (int(lm.x * w), int(lm.y * h)), 1, (79, 208, 168), -1)
                anot.append(rgb)
        fn += 1
        if total:
            prog.progress(min(fn / max(1, min(total, max_frames)), 1.0))
    cap.release()
    return fps, np.array(tiempos), np.array(aus, dtype=np.float32), anot

def predecir(aus, tiempos, scaler, mlp, lstm, bigru):
    Xs = scaler.transform(aus).astype(np.float32)
    out = {"NN1 MLP": (tiempos, np.clip(mlp.predict(Xs), -1, 1))}
    if len(Xs) > TIMESTEPS:
        seqs = np.stack([Xs[i:i+TIMESTEPS] for i in range(len(Xs)-TIMESTEPS)])
        t_seq = tiempos[TIMESTEPS:]
        with torch.no_grad():
            t = torch.tensor(seqs)
            out["NN2 LSTM"] = (t_seq, lstm(t).numpy().flatten())
            out["NN3 BiGRU+Attn"] = (t_seq, bigru(t).numpy().flatten())
    return out

# ══════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Valencia Emocional", page_icon="◐", layout="centered")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
html, body, [class*="css"], .stMarkdown, button, input, textarea {{ font-family:'Inter',system-ui,sans-serif; }}
.block-container {{ max-width:780px !important; padding-top:3rem; padding-bottom:6rem; }}
#MainMenu, footer, [data-testid="stDecoration"] {{ visibility:hidden; }}
hr {{ border:0; border-top:1px solid {BORDER}; margin:1.6rem 0; }}

.vh-eyebrow {{ font-size:.7rem; letter-spacing:.22em; text-transform:uppercase;
  color:{ACCENT}; font-weight:600; margin-bottom:.4rem; }}
.vh-title {{ font-size:2.25rem; font-weight:700; line-height:1.05; letter-spacing:-.02em;
  margin:0 0 .55rem 0; color:{TEXT}; }}
.vh-sub {{ color:{MUTED}; font-size:.98rem; line-height:1.55; max-width:60ch; margin:0; }}
.vh-legend {{ display:flex; gap:1.3rem; flex-wrap:wrap; margin:1.1rem 0 0; }}
.vh-chip {{ display:flex; align-items:center; gap:.5rem; font-size:.8rem; color:{MUTED}; }}
.vh-dot {{ width:.6rem; height:.6rem; border-radius:50%; flex:0 0 auto; }}
.vh-chip b {{ color:{TEXT}; font-weight:600; }}
.vh-sec {{ font-size:.72rem; letter-spacing:.14em; text-transform:uppercase;
  color:{MUTED}; font-weight:600; margin:.2rem 0 .7rem; }}
.vh-how {{ display:flex; gap:1.7rem; flex-wrap:wrap; }}
.vh-how > div {{ font-size:.9rem; color:{MUTED}; line-height:1.4; }}
.vh-how b {{ color:{ACCENT}; font-family:'IBM Plex Mono',monospace; font-weight:500; margin-right:.5rem; }}

/* Upload zone: calm, single focus */
[data-testid="stFileUploaderDropzone"] {{ background:{PANEL}; border:1px dashed {BORDER};
  border-radius:12px; padding:1.6rem; }}
[data-testid="stFileUploaderDropzone"]:hover {{ border-color:{ACCENT}; }}

/* Settings popover trigger + secondary buttons: ghost style */
button[kind="secondary"], [data-testid="stPopover"] button {{
  border:1px solid {BORDER} !important; background:transparent !important;
  color:{MUTED} !important; font-weight:500 !important; border-radius:9px !important; }}
button[kind="secondary"]:hover, [data-testid="stPopover"] button:hover {{
  border-color:{ACCENT} !important; color:{ACCENT} !important; }}

[data-testid="stDataFrame"] {{ font-feature-settings:"tnum"; }}
[data-testid="stExpander"] {{ border:1px solid {BORDER}; border-radius:10px; background:transparent; }}
</style>
""", unsafe_allow_html=True)

legend = "".join(
    f'<div class="vh-chip"><span class="vh-dot" style="background:{COL[m]}"></span>'
    f'<b>{m}</b> · {ROL[m]}</div>' for m in COL)
st.markdown(f"""
<div class="vh-eyebrow">Computación afectiva · TEC</div>
<h1 class="vh-title">Valencia emocional desde el rostro</h1>
<p class="vh-sub">MediaPipe extrae las Action Units faciales y tres modelos predicen
la valencia (de -1 a +1) cuadro a cuadro. Sube un video para comparar cómo se
comportan en el tiempo.</p>
<div class="vh-legend">{legend}</div>
""", unsafe_allow_html=True)
st.markdown("<hr/>", unsafe_allow_html=True)

if "k" not in st.session_state:
    st.session_state.k = 0

# ── Cómo usar ─────────────────────────────────────────────────────
st.markdown('<div class="vh-sec">Cómo usar</div>', unsafe_allow_html=True)
st.markdown("""
<div class="vh-how">
  <div><b>1</b>Sube un video del rostro</div>
  <div><b>2</b>Ajusta el muestreo y el rango</div>
  <div><b>3</b>Compara los tres modelos en el tiempo</div>
</div>
""", unsafe_allow_html=True)

# ── Ajustes (visibles, inline) ────────────────────────────────────
st.markdown('<div class="vh-sec" style="margin-top:1.7rem">Ajustes</div>', unsafe_allow_html=True)
s1, s2, s3, s4 = st.columns([3, 3, 3, 2], gap="medium", vertical_alignment="bottom")
with s1:
    stride = st.number_input("1 de cada N cuadros", 1, 60, 10, step=1,
                             help="Menor N: curvas más suaves, más cómputo.")
with s2:
    max_seg = st.number_input("Segundos máximos", 5, 120, 30, step=5,
                              help="Acota el cómputo en videos largos.")
with s3:
    dots = st.toggle("Puntos faciales", value=False,
                     help="Dibuja la malla facial sobre el video. Reprocesa al cambiar.")
with s4:
    reset = st.button("Reiniciar", use_container_width=True, type="secondary")

if reset:
    for key in list(st.session_state.keys()):
        if key != "k": del st.session_state[key]
    st.session_state.k += 1
    st.rerun()

# Controles vivos: si cambia un parámetro de proceso, invalida el resultado
sig = (stride, max_seg, dots)
if st.session_state.get("sig") != sig:
    for key in ("preds", "aus", "tiempos", "fps", "in_path", "anot_path", "t_proc"):
        st.session_state.pop(key, None)
    st.session_state.sig = sig

st.markdown('<div class="vh-sec" style="margin-top:1.7rem">Video</div>', unsafe_allow_html=True)
video = st.file_uploader("Video", type=["mp4", "mov", "avi", "mkv"],
                         key=f"up_{st.session_state.k}", label_visibility="collapsed")

# ── Empty state: mínimo y calmo ───────────────────────────────────
if video is None:
    st.caption("Un primer plano del rostro funciona mejor.")
    ref = AQUI / "comparacion_modelos.csv"
    if ref.exists():
        with st.expander("Precisión de referencia de los modelos"):
            st.dataframe(pd.read_csv(ref), use_container_width=True, hide_index=True)
            st.caption("CCC / MSE / R² sobre las etiquetas reales del dataset (validación K-Fold). "
                       "No es recalculable sobre un video sin etiquetar.")
    st.stop()

# ── Procesa ───────────────────────────────────────────────────────
if "preds" not in st.session_state:
    tmp = Path(tempfile.gettempdir()) / f"in_{st.session_state.k}_{video.name}"
    tmp.write_bytes(video.getbuffer())
    scaler, mlp, lstm, bigru = cargar_modelos()
    detector = cargar_detector()
    prog = st.progress(0.0, text="Extrayendo Action Units…")
    t0 = time.time()
    fps, tiempos, aus, anot = procesar_video(tmp, stride, dots, max_seg, detector, prog)
    prog.empty()
    if len(aus) <= TIMESTEPS:
        st.error(f"Solo se detectó rostro en {len(aus)} muestras (se necesitan más de "
                 f"{TIMESTEPS}). Baja el N de muestreo o usa un video más largo o nítido.")
        st.stop()
    preds = predecir(aus, tiempos, scaler, mlp, lstm, bigru)
    anot_path = None
    if anot:
        anot_path = str(Path(tempfile.gettempdir()) / f"anot_{st.session_state.k}.mp4")
        imageio.mimwrite(anot_path, anot, fps=fps, codec="libx264", quality=7, macro_block_size=None)
    st.session_state.update(preds=preds, aus=aus, tiempos=tiempos, fps=fps,
                            in_path=str(tmp), anot_path=anot_path, t_proc=time.time()-t0)

preds = st.session_state.preds
n_muestras = len(st.session_state.tiempos)
dur = float(st.session_state.tiempos[-1]) if n_muestras else 0.0

# ── Video ─────────────────────────────────────────────────────────
if dots and st.session_state.get("anot_path"):
    st.video(st.session_state.anot_path)
else:
    st.video(st.session_state.in_path)
st.caption(f"{n_muestras} muestras · {dur:.1f}s analizados · {st.session_state.t_proc:.1f}s de proceso"
           + (" · puntos activos" if dots and st.session_state.get("anot_path") else ""))

# ── Gráfica ───────────────────────────────────────────────────────
st.markdown("<hr/>", unsafe_allow_html=True)
gc1, gc2 = st.columns([1, 1], gap="small", vertical_alignment="center")
with gc1:
    st.markdown('<div class="vh-sec" style="margin-bottom:0">Valencia vs tiempo</div>', unsafe_allow_html=True)
with gc2:
    modelos_sel = st.segmented_control(
        "Modelos", list(COL.keys()), selection_mode="multi",
        default=list(COL.keys()), label_visibility="collapsed")

fig = go.Figure()
for m in (modelos_sel or []):
    if m in preds:
        t, v = preds[m]
        fig.add_trace(go.Scatter(x=t, y=v, name=m.split()[0], mode="lines",
                      line=dict(color=COL[m], width=2.4),
                      hovertemplate=f"<b>{m}</b><br>%{{x:.1f}}s · %{{y:.3f}}<extra></extra>"))
fig.add_hline(y=0, line=dict(color=BORDER, width=1, dash="dot"))
fig.update_layout(
    height=320, margin=dict(l=4, r=4, t=24, b=4),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=MUTED, family="Inter", size=12),
    yaxis=dict(range=[-1.05, 1.05], title="Valencia", gridcolor=BORDER, zeroline=False),
    xaxis=dict(title="Tiempo (s)", gridcolor=BORDER, zeroline=False),
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"))
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
if not modelos_sel:
    st.caption("Selecciona al menos un modelo.")

# ── Comparación ───────────────────────────────────────────────────
st.markdown("<hr/>", unsafe_allow_html=True)
st.markdown('<div class="vh-sec">Resumen por modelo · este video</div>', unsafe_allow_html=True)
filas = []
for m in (modelos_sel or list(COL.keys())):
    if m in preds:
        _, v = preds[m]
        filas.append({"Modelo": m, "Media": round(float(v.mean()), 3),
                      "Desv.": round(float(v.std()), 3),
                      "Mín": round(float(v.min()), 3), "Máx": round(float(v.max()), 3),
                      "% positiva": round(float((v > 0).mean()) * 100, 1)})
if filas:
    st.dataframe(pd.DataFrame(filas), use_container_width=True, hide_index=True,
                 column_config={"% positiva": st.column_config.NumberColumn(format="%.1f%%")})

st.markdown('<div class="vh-sec" style="margin-top:1.6rem">Acuerdo entre modelos</div>', unsafe_allow_html=True)
nombres = [n for n in (modelos_sel or list(COL.keys())) if n in preds]
ag = []
for i in range(len(nombres)):
    for j in range(i + 1, len(nombres)):
        _, va = preds[nombres[i]]; _, vb = preds[nombres[j]]
        n = min(len(va), len(vb))
        if n >= 2:
            va2, vb2 = va[-n:], vb[-n:]
            ag.append({"Par": f"{nombres[i].split()[0]} ↔ {nombres[j].split()[0]}",
                       "CCC": round(ccc(va2, vb2), 3),
                       "Pearson": round(float(np.corrcoef(va2, vb2)[0, 1]), 3)})
if ag:
    st.dataframe(pd.DataFrame(ag), use_container_width=True, hide_index=True)
st.caption("Concordancia entre las predicciones de los modelos, no su exactitud: "
           "el video no tiene valencia real etiquetada.")
