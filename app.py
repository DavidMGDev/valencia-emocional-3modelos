"""
Predicción de Valencia Emocional — comparador de 3 modelos.
Sube un video → extrae Action Units (MediaPipe) → predice valencia con NN1/NN2/NN3
→ grafica valencia vs tiempo y compara los 3 modelos.

Features:
  1. Upload de video + gráficas valencia-vs-tiempo (los 3 modelos).
  2. Toggle de puntos de tracking facial (video anotado).
  3. Reproducción del video.
  4. Botón de reset.

Sin datasets: solo viajan los modelos + el .task de MediaPipe.
"""
import json, tempfile, time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

import torch
import torch.nn as nn
import joblib
import cv2
import imageio
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

AQUI = Path(__file__).parent
CFG  = json.loads((AQUI / "model_config.json").read_text())
TIMESTEPS = CFG["TIMESTEPS"]
GRU_UNITS = CFG["GRU_UNITS"]
NUM_HEADS = CFG["NUM_HEADS"]
DROPOUT   = CFG["DROPOUT"]
AU_COLS   = CFG["AU_COLS"]
ESCALA    = CFG["ESCALA_FACS"]
TASK      = str(AQUI / "face_landmarker.task")

# blendshape -> AU (igual que extraer_aus.py)
AU_MAP = {
    "AU01_r": ["browInnerUp"],
    "AU02_r": ["browOuterUpLeft", "browOuterUpRight"],
    "AU04_r": ["browDownLeft", "browDownRight"],
    "AU05_r": ["eyeWideLeft", "eyeWideRight"],
    "AU06_r": ["cheekSquintLeft", "cheekSquintRight"],
    "AU07_r": ["eyeSquintLeft", "eyeSquintRight"],
    "AU12_r": ["mouthSmileLeft", "mouthSmileRight"],
    "AU14_r": ["mouthDimpleLeft", "mouthDimpleRight"],
    "AU15_r": ["mouthFrownLeft", "mouthFrownRight"],
    "AU17_r": ["mouthShrugUpper", "mouthPressLeft", "mouthPressRight"],
    "AU20_r": ["mouthStretchLeft", "mouthStretchRight"],
    "AU25_r": ["jawOpen"],
}

# ── Arquitecturas (idénticas a entrenar_modelos.py) ───────────────
class LSTMModel(nn.Module):
    def __init__(self, n_features=12, hidden=64, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden, 1)
        self.tanh = nn.Tanh()
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.tanh(self.fc(self.drop(out[:, -1, :])))

class BiGRUAttention(nn.Module):
    def __init__(self, n_features=12, gru_units=32, num_heads=4, dropout=0.3):
        super().__init__()
        hidden = gru_units * 2
        self.bigru = nn.GRU(n_features, gru_units, batch_first=True, bidirectional=True)
        self.ln1   = nn.LayerNorm(hidden)
        self.attn  = nn.MultiheadAttention(hidden, num_heads, dropout=dropout, batch_first=True)
        self.fc1   = nn.Linear(hidden, 32)
        self.fc2   = nn.Linear(32, 1)
        self.relu  = nn.ReLU()
        self.tanh  = nn.Tanh()
        self.drop  = nn.Dropout(dropout)
    def forward(self, x):
        out, _  = self.bigru(x)
        out     = self.ln1(out)
        attn, _ = self.attn(out, out, out)
        out     = (out + attn).mean(dim=1)
        out     = self.drop(self.relu(self.fc1(out)))
        return self.tanh(self.fc2(out))

# ── Carga cacheada de modelos ─────────────────────────────────────
@st.cache_resource
def cargar_modelos():
    scaler = joblib.load(AQUI / "scaler.pkl")
    mlp    = joblib.load(AQUI / "modelo_mlp.pkl")
    lstm   = LSTMModel()
    lstm.load_state_dict(torch.load(AQUI / "modelo_lstm.pt", map_location="cpu"))
    lstm.eval()
    bigru  = BiGRUAttention(12, GRU_UNITS, NUM_HEADS, DROPOUT)
    bigru.load_state_dict(torch.load(AQUI / "modelo_bigru.pt", map_location="cpu"))
    bigru.eval()
    return scaler, mlp, lstm, bigru

@st.cache_resource
def cargar_detector():
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=TASK),
        output_face_blendshapes=True, num_faces=1,
        running_mode=mp_vision.RunningMode.IMAGE)
    return mp_vision.FaceLandmarker.create_from_options(opts)

def bs_to_aus(blendshapes):
    d = {b.category_name: b.score for b in blendshapes}
    return [float(np.clip(np.mean([d.get(n, 0.0) for n in AU_MAP[au]]) * ESCALA, 0, 5))
            for au in AU_COLS]

def ccc(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 2: return float("nan")
    cov = ((a - a.mean()) * (b - b.mean())).mean()
    return float(2 * cov / (a.var() + b.var() + (a.mean() - b.mean())**2 + 1e-8))

# ── Procesa el video: AUs muestreadas + video anotado opcional ────
def procesar_video(ruta, stride, dibujar_dots, max_seg, detector, progreso):
    cap = cv2.VideoCapture(str(ruta))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    max_frames = int(max_seg * fps) if max_seg else total

    tiempos, aus, frames_anot = [], [], []
    fn = 0
    while True:
        ok, frame = cap.read()
        if not ok or fn >= max_frames:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        muestrear = (fn % stride == 0)

        if muestrear or dibujar_dots:
            res = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            tiene = bool(res.face_blendshapes)
            if muestrear and tiene:
                tiempos.append(fn / fps)
                aus.append(bs_to_aus(res.face_blendshapes[0]))
            if dibujar_dots:
                if tiene:
                    h, w = rgb.shape[:2]
                    for lm in res.face_landmarks[0]:
                        cv2.circle(rgb, (int(lm.x * w), int(lm.y * h)), 1,
                                   (0, 255, 0), -1)
                frames_anot.append(rgb)
        fn += 1
        if total:
            progreso.progress(min(fn / max(1, min(total, max_frames)), 1.0))
    cap.release()
    return fps, np.array(tiempos), np.array(aus, dtype=np.float32), frames_anot

def predecir(aus, tiempos, scaler, mlp, lstm, bigru):
    Xs = scaler.transform(aus).astype(np.float32)               # (N,12)
    out = {"NN1 MLP": (tiempos, mlp.predict(Xs))}
    if len(Xs) > TIMESTEPS:
        seqs = np.stack([Xs[i:i+TIMESTEPS] for i in range(len(Xs)-TIMESTEPS)])
        t_seq = tiempos[TIMESTEPS:]
        with torch.no_grad():
            t = torch.tensor(seqs)
            out["NN2 LSTM"]        = (t_seq, lstm(t).numpy().flatten())
            out["NN3 BiGRU+Attn"]  = (t_seq, bigru(t).numpy().flatten())
    return out

# ══════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Valencia Emocional — 3 modelos", layout="wide")
st.title("Predicción de Valencia Emocional — comparación de 3 modelos")
st.caption("NN1 MLP · NN2 LSTM · NN3 BiGRU+Attention — desde Action Units faciales (MediaPipe)")

if "k" not in st.session_state:
    st.session_state.k = 0   # key del uploader, se incrementa al resetear

with st.sidebar:
    st.header("Controles")
    stride = st.slider("Muestreo: 1 frame cada N", 1, 60, 10,
                       help="Bajo = curvas suaves (más lento). Alto = más rápido.")
    dots   = st.toggle("Puntos de tracking facial", value=False)
    max_seg = st.slider("Máx. segundos a procesar", 5, 120, 30,
                        help="Acota el costo de cómputo en videos largos.")
    modelos_sel = st.multiselect("Modelos a mostrar",
                        ["NN1 MLP", "NN2 LSTM", "NN3 BiGRU+Attn"],
                        default=["NN1 MLP", "NN2 LSTM", "NN3 BiGRU+Attn"])
    if st.button("🔄 Reset", use_container_width=True):
        for key in list(st.session_state.keys()):
            if key != "k":
                del st.session_state[key]
        st.session_state.k += 1
        st.rerun()

video = st.file_uploader("Sube un video (.mp4 / .mov / .avi)",
                         type=["mp4", "mov", "avi", "mkv"],
                         key=f"up_{st.session_state.k}")

# Tabla de referencia (precisión sobre el dataset etiquetado, estática)
ref = AQUI / "comparacion_modelos.csv"
if ref.exists():
    with st.expander("📊 Precisión de referencia (dataset etiquetado, K-Fold)"):
        st.dataframe(pd.read_csv(ref), use_container_width=True)
        st.caption("CCC/MSE/R² vs etiquetas reales. No recalculable sobre un "
                   "video sin etiquetas; sirve de referencia de exactitud.")

if video is not None:
    if "preds" not in st.session_state:
        tmp = Path(tempfile.gettempdir()) / f"in_{st.session_state.k}_{video.name}"
        tmp.write_bytes(video.getbuffer())
        scaler, mlp, lstm, bigru = cargar_modelos()
        detector = cargar_detector()
        prog = st.progress(0.0, text="Procesando video…")
        t0 = time.time()
        fps, tiempos, aus, frames_anot = procesar_video(
            tmp, stride, dots, max_seg, detector, prog)
        prog.empty()
        if len(aus) <= TIMESTEPS:
            st.error(f"Solo {len(aus)} muestras con cara detectada. "
                     f"Baja el muestreo (N menor) o sube un video más largo "
                     f"(se necesitan > {TIMESTEPS}).")
            st.stop()
        preds = predecir(aus, tiempos, scaler, mlp, lstm, bigru)
        # Video anotado -> mp4 H.264 reproducible en navegador
        anot_path = None
        if frames_anot:
            anot_path = str(Path(tempfile.gettempdir()) / f"anot_{st.session_state.k}.mp4")
            imageio.mimwrite(anot_path, frames_anot, fps=fps, codec="libx264",
                             quality=7, macro_block_size=None)
        st.session_state.preds   = preds
        st.session_state.aus     = aus
        st.session_state.tiempos = tiempos
        st.session_state.fps     = fps
        st.session_state.in_path = str(tmp)
        st.session_state.anot_path = anot_path
        st.session_state.t_proc  = time.time() - t0

    preds   = st.session_state.preds
    col_v, col_g = st.columns([1, 2])

    with col_v:
        st.subheader("Reproducción")
        if dots and st.session_state.get("anot_path"):
            st.video(st.session_state.anot_path)
            st.caption("Con puntos de tracking.")
        else:
            st.video(st.session_state.in_path)
            if dots:
                st.info("Activa el toggle ANTES de subir para generar el video anotado.")
        st.metric("Tiempo de procesamiento", f"{st.session_state.t_proc:.1f} s")

    with col_g:
        st.subheader("Valencia vs tiempo")
        colores = {"NN1 MLP": "#5B9BD5", "NN2 LSTM": "#ED7D31",
                   "NN3 BiGRU+Attn": "#70AD47"}
        fig, ax = plt.subplots(figsize=(9, 4))
        for nombre in modelos_sel:
            if nombre in preds:
                t, v = preds[nombre]
                ax.plot(t, v, label=nombre, color=colores[nombre], linewidth=2)
        ax.axhline(0, color="gray", ls=":", lw=0.8)
        ax.set_xlabel("Tiempo (s)"); ax.set_ylabel("Valencia [-1, 1]")
        ax.set_ylim(-1.1, 1.1); ax.grid(alpha=0.3); ax.legend()
        st.pyplot(fig)

    # Comparación viva de los 3 modelos
    st.subheader("Comparación de los 3 modelos (sobre este video)")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Estadísticas por modelo**")
        filas = []
        for nombre in modelos_sel:
            if nombre in preds:
                _, v = preds[nombre]
                filas.append({"Modelo": nombre, "Valencia media": round(float(v.mean()), 3),
                              "Desv.": round(float(v.std()), 3),
                              "Mín": round(float(v.min()), 3),
                              "Máx": round(float(v.max()), 3),
                              "% positiva": round(float((v > 0).mean()) * 100, 1)})
        st.dataframe(pd.DataFrame(filas), use_container_width=True, hide_index=True)
    with c2:
        st.markdown("**Acuerdo entre modelos** (CCC / Pearson)")
        nombres = [n for n in modelos_sel if n in preds]
        ag = []
        for i in range(len(nombres)):
            for j in range(i+1, len(nombres)):
                ta, va = preds[nombres[i]]; tb, vb = preds[nombres[j]]
                n = min(len(va), len(vb))
                if n >= 2:
                    va2, vb2 = va[-n:], vb[-n:]
                    ag.append({"Par": f"{nombres[i]} ↔ {nombres[j]}",
                               "CCC": round(ccc(va2, vb2), 3),
                               "Pearson": round(float(np.corrcoef(va2, vb2)[0, 1]), 3)})
        if ag:
            st.dataframe(pd.DataFrame(ag), use_container_width=True, hide_index=True)
        st.caption("Acuerdo entre las predicciones de los modelos (no exactitud: el "
                   "video no tiene valencia real etiquetada).")
else:
    st.info("⬆️ Sube un video para comenzar. Ajusta el muestreo y los toggles en la barra lateral.")
