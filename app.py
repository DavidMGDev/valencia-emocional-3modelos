"""
Valencia Emocional — comparador de 3 modelos (NN1 MLP / NN2 LSTM / NN3 BiGRU+Attn).
Sube un video, extrae Action Units con MediaPipe, predice valencia y compara los
tres modelos en el tiempo.

Features: upload, gráfica valencia-vs-tiempo (3 modelos), toggle de puntos de
tracking, reproducción, reset, muestreo ajustable, comparación viva (stats + acuerdo).
"""
import gc, json, tempfile, time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import joblib
import onnxruntime as ort
import cv2
import imageio
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ── Paleta (data-viz: un color por modelo) ────────────────────────
COL = {
    "MLP":        "#6FB1E0",
    "LSTM":       "#4FD0A8",
    "BiGRU+Attn": "#F07167",
}
ROL = {
    "MLP":        "Baseline",
    "LSTM":       "Recurrente",
    "BiGRU+Attn": "Atención",
}
ACCENT = "#E0A33C"; MUTED = "#A8A296"; PANEL = "#1F1D18"
BORDER = "#322E26"; TEXT = "#ECE8E0"; BG = "#161512"

AQUI = Path(__file__).parent
CFG  = json.loads((AQUI / "model_config.json").read_text())
TIMESTEPS = CFG["TIMESTEPS"]; GRU_UNITS = CFG["GRU_UNITS"]
NUM_HEADS = CFG["NUM_HEADS"]; DROPOUT = CFG["DROPOUT"]
AU_COLS = CFG["AU_COLS"]; ESCALA = CFG["ESCALA_FACS"]
TASK = str(AQUI / "face_landmarker.task")

# Valencia real (ground truth) por nombre de archivo. Solo etiquetas, sin rostros.
GROUND_TRUTH = json.loads((AQUI / "ground_truth.json").read_text())
GT_NAME = "Real (dataset)"
GT_COL  = "#ECE8E0"
DATASET_SKIP = 141   # cadencia de muestreo del dataset (~4.7s a 30fps)

AU_MAP = {
    "AU01_r": ["browInnerUp"], "AU02_r": ["browOuterUpLeft", "browOuterUpRight"],
    "AU04_r": ["browDownLeft", "browDownRight"], "AU05_r": ["eyeWideLeft", "eyeWideRight"],
    "AU06_r": ["cheekSquintLeft", "cheekSquintRight"], "AU07_r": ["eyeSquintLeft", "eyeSquintRight"],
    "AU12_r": ["mouthSmileLeft", "mouthSmileRight"], "AU14_r": ["mouthDimpleLeft", "mouthDimpleRight"],
    "AU15_r": ["mouthFrownLeft", "mouthFrownRight"],
    "AU17_r": ["mouthShrugUpper", "mouthPressLeft", "mouthPressRight"],
    "AU20_r": ["mouthStretchLeft", "mouthStretchRight"], "AU25_r": ["jawOpen"],
}

@st.cache_resource(show_spinner=False)
def cargar_modelos():
    scaler = joblib.load(AQUI / "scaler.pkl")
    mlp = joblib.load(AQUI / "modelo_mlp.pkl")
    so = ort.SessionOptions(); so.intra_op_num_threads = 1
    lstm = ort.InferenceSession(str(AQUI / "modelo_lstm.onnx"), so, providers=["CPUExecutionProvider"])
    bigru = ort.InferenceSession(str(AQUI / "modelo_bigru.onnx"), so, providers=["CPUExecutionProvider"])
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

MAX_W = 640   # downscale para acotar RAM/CPU en el plan gratuito

def procesar_video(ruta, fps_obj, dibujar, max_seg, detector, prog, anot_path):
    cap = cv2.VideoCapture(str(ruta))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, round(fps / max(0.1, fps_obj)))   # cuadros/seg -> 1 de cada N
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    max_frames = int(max_seg * fps) if max_seg else total
    tiempos, aus = [], []
    writer = None   # escritura en streaming -> RAM constante
    fn = 0
    while True:
        ok, frame = cap.read()
        if not ok or fn >= max_frames: break
        h, w = frame.shape[:2]
        if w > MAX_W:
            frame = cv2.resize(frame, (MAX_W, int(h * MAX_W / w)))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        muestrear = (fn % stride == 0)
        if muestrear or dibujar:
            res = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            tiene = bool(res.face_blendshapes)
            if muestrear and tiene:
                tiempos.append(fn / fps); aus.append(bs_to_aus(res.face_blendshapes[0]))
            if dibujar:
                if tiene:
                    hh, ww = rgb.shape[:2]
                    for lm in res.face_landmarks[0]:
                        cv2.circle(rgb, (int(lm.x * ww), int(lm.y * hh)), 1, (79, 208, 168), -1)
                if writer is None:
                    writer = imageio.get_writer(anot_path, fps=fps, codec="libx264",
                                                quality=7, macro_block_size=None)
                writer.append_data(rgb)
        fn += 1
        if total:
            prog.progress(min(fn / max(1, min(total, max_frames)), 1.0))
    cap.release()
    if writer is not None:
        writer.close()
    return fps, np.array(tiempos), np.array(aus, dtype=np.float32), (writer is not None)

def predecir(aus, tiempos, scaler, mlp, lstm, bigru):
    Xs = scaler.transform(aus).astype(np.float32)
    out = {"MLP": (tiempos, np.clip(mlp.predict(Xs), -1, 1))}
    if len(Xs) > TIMESTEPS:
        seqs = np.stack([Xs[i:i+TIMESTEPS] for i in range(len(Xs)-TIMESTEPS)]).astype(np.float32)
        t_seq = tiempos[TIMESTEPS:]
        out["LSTM"] = (t_seq, lstm.run(["output"], {"input": seqs})[0].flatten())
        out["BiGRU+Attn"] = (t_seq, bigru.run(["output"], {"input": seqs})[0].flatten())
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
.vh-note {{ border:1px solid {BORDER}; background:{PANEL}; border-radius:10px;
  padding:.75rem .95rem; font-size:.82rem; color:{MUTED}; line-height:1.55; margin-top:.4rem; }}
.vh-note b {{ color:{TEXT}; font-weight:600; }}

/* Upload zone: calm, single focus */
[data-testid="stFileUploaderDropzone"] {{ background:{PANEL}; border:1px dashed {BORDER};
  border-radius:12px; padding:1.6rem; }}
[data-testid="stFileUploaderDropzone"]:hover {{ border-color:{ACCENT}; }}

/* Secondary buttons: ghost outline */
button[kind="secondary"] {{
  border:1px solid {BORDER} !important; background:transparent !important;
  color:{MUTED} !important; font-weight:500 !important; border-radius:9px !important; }}
button[kind="secondary"]:hover {{ border-color:{ACCENT} !important; color:{ACCENT} !important; }}

/* Popover trigger styled as an inline clickable word */
[data-testid="stPopover"] {{ margin-top:.7rem; }}
[data-testid="stPopover"] > button {{
  border:none !important; background:transparent !important; padding:0 !important;
  min-height:auto !important; color:{ACCENT} !important; font-size:.82rem !important;
  font-weight:500 !important; text-decoration:underline; text-underline-offset:3px; }}
[data-testid="stPopover"] > button:hover {{ color:{TEXT} !important; }}

[data-testid="stDataFrame"] {{ font-feature-settings:"tnum"; }}
[data-testid="stExpander"] {{ border:1px solid {BORDER}; border-radius:10px; background:transparent; }}
</style>
""", unsafe_allow_html=True)

legend = "".join(
    f'<div class="vh-chip"><span class="vh-dot" style="background:{COL[m]}"></span>'
    f'<b>{m}</b> · {ROL[m]}</div>' for m in COL)
st.markdown(f"""
<div class="vh-eyebrow">Valencia Emocional · UNAD · TEC</div>
<h1 class="vh-title">Predicción de valencia desde el rostro</h1>
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
s1, s2 = st.columns(2, gap="medium", vertical_alignment="bottom")
with s1:
    fps_obj = st.number_input("Cuadros por segundo", 1, 30, 3, step=1,
                              help="Cuántos cuadros por segundo se analizan. Se convierte a "
                                   "1 de cada N según los FPS del video. Más alto = curvas más "
                                   "suaves y más cómputo.")
with s2:
    max_seg = st.number_input("Segundos máximos analizados", 5, 300, 160, step=5,
                              help="Acota el cómputo en videos largos.")

# Controles vivos: si cambia un parámetro de proceso, exige reprocesar
# (el toggle de puntos NO entra aquí: solo cambia visibilidad, no reprocesa)
sig = (fps_obj, max_seg)
if st.session_state.get("sig") != sig:
    for key in ("preds", "aus", "tiempos", "fps", "anot_path", "t_proc"):
        st.session_state.pop(key, None)
    st.session_state.sig = sig

st.markdown('<div class="vh-sec" style="margin-top:1.7rem">Video</div>', unsafe_allow_html=True)
video = st.file_uploader("Video", type=["mp4", "mov", "avi", "mkv"],
                         key=f"up_{st.session_state.k}", label_visibility="collapsed")

st.markdown(
    '<div class="vh-note"><b>Privacidad:</b> los videos del dataset no se suben ni se '
    'almacenan aquí. Solo la <b>valencia real</b> está incluida por nombre de archivo, '
    'para poder comparar las predicciones contra la verdad. Si subes un video cuyo nombre '
    'coincide con uno del dataset, se grafica también su valencia real.</div>',
    unsafe_allow_html=True)
with st.popover("Ver nombres del dataset"):
    st.caption("Sube un video con uno de estos nombres para superponer su valencia real:")
    st.text("   ".join(sorted(GROUND_TRUTH.keys())))

# ── Empty state: mínimo y calmo ───────────────────────────────────
if video is None:
    st.caption("Un primer plano del rostro funciona mejor.")
    ref = AQUI / "comparacion_modelos.csv"
    if ref.exists():
        with st.expander("Precisión de referencia de los modelos"):
            dfr = pd.read_csv(ref)
            dfr["Modelo"] = dfr["Modelo"].str.replace(r"NN\d\s+", "", regex=True)
            st.dataframe(dfr, width="stretch", hide_index=True)
            st.caption("CCC / MSE / R² sobre las etiquetas reales del dataset (validación K-Fold). "
                       "No es recalculable sobre un video sin etiquetar.")
    st.stop()

# ── Guarda el archivo subido (barato, sin procesar) ───────────────
vid_key = f"{st.session_state.k}-{video.name}-{video.size}"
if st.session_state.get("vid_key") != vid_key:
    tmp = Path(tempfile.gettempdir()) / f"in_{st.session_state.k}_{video.name}"
    tmp.write_bytes(video.getbuffer())
    st.session_state.vid_key = vid_key
    st.session_state.in_path = str(tmp)
    for key in ("preds", "aus", "tiempos", "fps", "anot_path", "t_proc"):
        st.session_state.pop(key, None)

# ── Reproducción (disponible apenas se sube) ──────────────────────
# El toggle de puntos solo aparece una vez procesado (ya existe el video anotado).
procesado = "preds" in st.session_state
if procesado:
    rep1, rep2 = st.columns([3, 2], gap="small", vertical_alignment="center")
    with rep1:
        st.markdown('<div class="vh-sec" style="margin-bottom:0">Reproducción</div>', unsafe_allow_html=True)
    with rep2:
        ver_puntos = st.toggle("Puntos faciales", value=False, key="ver_puntos",
                               help="Muestra u oculta la malla facial. No reprocesa el video.")
else:
    st.markdown('<div class="vh-sec">Reproducción</div>', unsafe_allow_html=True)
    ver_puntos = False

if ver_puntos and st.session_state.get("anot_path"):
    st.video(st.session_state.anot_path)
else:
    st.video(st.session_state.in_path)

# ── Procesar (paso explícito) ─────────────────────────────────────
if "preds" not in st.session_state:
    pc1, pc2 = st.columns([1, 2], gap="medium", vertical_alignment="center")
    with pc1:
        procesar = st.button("Procesar video", type="primary", width="stretch")
    with pc2:
        st.caption("Analiza el video con los tres modelos y calcula la malla facial. "
                   "Puede tardar según los segundos analizados.")
    if not procesar:
        st.stop()
    scaler, mlp, lstm, bigru = cargar_modelos()
    try:
        detector = cargar_detector()
    except Exception:
        st.error("No se pudo iniciar MediaPipe en el servidor (falta una librería del sistema "
                 "para extraer Action Units). Revisa los logs de despliegue.")
        st.stop()
    anot_path = str(Path(tempfile.gettempdir()) / f"anot_{st.session_state.k}.mp4")
    prog = st.progress(0.0, text="Extrayendo Action Units…")
    t0 = time.time()
    fps, tiempos, aus, wrote = procesar_video(
        st.session_state.in_path, fps_obj, True, max_seg, detector, prog, anot_path)
    prog.empty()
    if len(aus) <= TIMESTEPS:
        st.error(f"Solo se detectó rostro en {len(aus)} muestras (se necesitan más de "
                 f"{TIMESTEPS}). Baja el N de muestreo o usa un video más largo o nítido.")
        st.stop()
    preds = predecir(aus, tiempos, scaler, mlp, lstm, bigru)
    stem = Path(video.name).stem
    if stem in GROUND_TRUTH:
        gv = np.array(GROUND_TRUTH[stem], dtype=float)
        gt_times = (DATASET_SKIP // 2 + np.arange(len(gv)) * DATASET_SKIP) / fps
        preds[GT_NAME] = (gt_times, gv)
    st.session_state.update(preds=preds, aus=aus, tiempos=tiempos, fps=fps,
                            anot_path=(anot_path if wrote else None), t_proc=time.time()-t0)
    gc.collect()
    st.rerun()

preds = st.session_state.preds
n_muestras = len(st.session_state.tiempos)
dur = float(st.session_state.tiempos[-1]) if n_muestras else 0.0
st.caption(f"{n_muestras} muestras · {dur:.1f}s analizados · {st.session_state.t_proc:.1f}s de proceso")

# ── Gráfica ───────────────────────────────────────────────────────
st.markdown("<hr/>", unsafe_allow_html=True)
gc1, gc2 = st.columns([1, 1], gap="small", vertical_alignment="center")
with gc1:
    st.markdown('<div class="vh-sec" style="margin-bottom:0">Valencia vs tiempo</div>', unsafe_allow_html=True)
with gc2:
    opciones = list(COL.keys()) + ([GT_NAME] if GT_NAME in preds else [])
    modelos_sel = st.segmented_control(
        "Modelos", opciones, selection_mode="multi",
        default=opciones, label_visibility="collapsed")

fig = go.Figure()
for m in (modelos_sel or []):
    if m in preds:
        t, v = preds[m]
        es_gt = (m == GT_NAME)
        fig.add_trace(go.Scatter(x=t, y=v, name=m,
                      mode="lines+markers" if es_gt else "lines",
                      line=dict(color=GT_COL if es_gt else COL[m], width=2.4,
                                dash="dash" if es_gt else "solid"),
                      marker=dict(size=5, color=GT_COL) if es_gt else None,
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
st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
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
    st.dataframe(pd.DataFrame(filas), width="stretch", hide_index=True,
                 column_config={"% positiva": st.column_config.NumberColumn(format="%.1f%%")})

# Exactitud vs valencia real (solo si el video tiene ground truth)
if GT_NAME in preds:
    gt_t, gv = preds[GT_NAME]
    ex = []
    for m in COL:
        if m in preds and (not modelos_sel or m in modelos_sel) and len(gv) >= 2:
            mt, mv = preds[m]
            mv_i = np.interp(gt_t, mt, mv)   # predicción remuestreada a los instantes reales
            ex.append({"Modelo": m, "CCC vs real": round(ccc(mv_i, gv), 3),
                       "Pearson": round(float(np.corrcoef(mv_i, gv)[0, 1]), 3)})
    if ex:
        st.markdown('<div class="vh-sec" style="margin-top:1.6rem">Exactitud vs valencia real</div>',
                    unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(ex), width="stretch", hide_index=True)
        st.caption("CCC y Pearson de cada modelo contra la valencia real del dataset "
                   "(predicciones remuestreadas a los instantes etiquetados). Mayor = mejor.")

st.markdown('<div class="vh-sec" style="margin-top:1.6rem">Acuerdo entre modelos</div>', unsafe_allow_html=True)
nombres = [n for n in (modelos_sel or list(COL.keys())) if n in preds and n in COL]
ag = []
for i in range(len(nombres)):
    for j in range(i + 1, len(nombres)):
        _, va = preds[nombres[i]]; _, vb = preds[nombres[j]]
        n = min(len(va), len(vb))
        if n >= 2:
            va2, vb2 = va[-n:], vb[-n:]
            ag.append({"Par": f"{nombres[i]} ↔ {nombres[j]}",
                       "CCC": round(ccc(va2, vb2), 3),
                       "Pearson": round(float(np.corrcoef(va2, vb2)[0, 1]), 3)})
if ag:
    st.dataframe(pd.DataFrame(ag), width="stretch", hide_index=True)
st.caption("Concordancia entre las predicciones de los modelos (no su exactitud).")

# ── Reiniciar (al final, después de los resultados) ───────────────
st.markdown("<hr/>", unsafe_allow_html=True)
_, rc, _ = st.columns([2, 1, 2])
with rc:
    if st.button("Reiniciar análisis", width="stretch", type="secondary"):
        for key in list(st.session_state.keys()):
            if key != "k": del st.session_state[key]
        st.session_state.k += 1
        st.rerun()
