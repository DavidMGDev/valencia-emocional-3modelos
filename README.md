# Predicción de Valencia Emocional — comparador de 3 modelos

App Streamlit que predice la **valencia emocional** (rango −1 a +1) a partir de
**Action Units faciales** extraídas con MediaPipe, comparando tres modelos:

| Modelo | Arquitectura | Contexto temporal |
|--------|--------------|-------------------|
| **NN1 MLP** | MLP (60-12-2), sklearn | frame-a-frame |
| **NN2 LSTM** | LSTM 1 capa, 64 u. | 5 frames |
| **NN3 BiGRU+Attn** | BiGRU + Self-Attention | 5 frames |

## Funcionalidad
1. Sube un video → extrae AUs → grafica **valencia vs tiempo** para los 3 modelos.
2. Toggle de **puntos de tracking facial** (genera video anotado).
3. **Reproducción** del video.
4. **Reset** para analizar otro video.
5. Comparación viva: estadísticas por modelo + acuerdo inter-modelo (CCC/Pearson).

> El video subido **no** se almacena ni se sube a ningún repositorio. El dataset
> original (caras de menores) **no** forma parte de este repositorio por privacidad.

## Ejecutar localmente
```bash
pip install -r requirements.txt
streamlit run app.py
```
Requiere Python ≤ 3.12 (MediaPipe).

## Despliegue en Streamlit Community Cloud
1. share.streamlit.io → login con GitHub.
2. *Create app* → este repo → archivo principal `app.py`.
3. *Advanced settings* → Python **3.12**.
4. Deploy.
