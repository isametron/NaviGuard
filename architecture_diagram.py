"""
architecture_diagram.py — NaviGuard
Run: python architecture_diagram.py
Saves: outputs/architecture_diagram.png
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe
import os

os.makedirs("outputs", exist_ok=True)

fig, ax = plt.subplots(figsize=(16, 9))
fig.patch.set_facecolor('#ffffff')
ax.set_facecolor('#ffffff')
ax.set_xlim(0, 16); ax.set_ylim(0, 9); ax.axis('off')

def box(cx, cy, w, h, face, edge, title, subtitle, marker=''):
    ax.add_patch(FancyBboxPatch((cx-w/2, cy-h/2), w, h,
                  boxstyle="round,pad=0.15", lw=2.5,
                  edgecolor=edge, facecolor=face, zorder=3))
    ax.add_patch(FancyBboxPatch((cx-w/2-0.05, cy-h/2-0.05), w+0.10, h+0.10,
                  boxstyle="round,pad=0.15", lw=5,
                  edgecolor=edge, facecolor='none', alpha=0.12, zorder=2))
    if marker:
        ax.text(cx, cy+0.5, marker, ha='center', va='center',
                fontsize=13, color=edge, fontweight='bold', zorder=5)
    ax.text(cx, cy+(0.0 if marker else 0.15), title,
            ha='center', va='center', fontsize=10.5, fontweight='bold',
            color='#1a1a1a', zorder=5, multialignment='center',
            path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    ax.text(cx, cy-0.42, subtitle, ha='center', va='center',
            fontsize=8, color='#555555', style='italic', zorder=5)

def arrow(x1, y1, x2, y2, col='#1565c0', label=''):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=col, lw=2.2,
                                mutation_scale=22), zorder=4)
    if label:
        ax.text((x1+x2)/2, (y1+y2)/2+0.18, label, ha='center',
                fontsize=7.5, color=col, style='italic',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor='none', alpha=0.8), zorder=6)

ax.text(8, 8.6, "System Architecture — NaviGuard",
        ha='center', fontsize=15, fontweight='bold', color='#0d1b2a')
ax.text(8, 8.15, "Satellite Clock & Ephemeris Error Prediction Using LSTM | NavIC/GNSS",
        ha='center', fontsize=9.5, color='#555555', style='italic')

pipeline = [
    (1.5,  5.6, '#e3f8fc', '#00bcd4', 'NavIC / GNSS\nData Source',  'satellite_telemetry.csv', '[DATA]'),
    (4.5,  5.6, '#f1f8e9', '#4caf50', 'Preprocessing',              'Scale + Window (20 steps)', '[PREP]'),
    (7.8,  5.6, '#f3e5f5', '#9c27b0', 'LSTM Model',                 '2-Layer + Dropout',          '[ML]'),
    (11.0, 5.6, '#ffebee', '#f44336', 'Prediction\nOutput',         'MAE ≤ 50 ns | μs units',    '[OUT]'),
    (14.0, 5.6, '#fff8e1', '#ff9800', 'Dashboard',                  'Streamlit localhost:8501',  '[UI]'),
]
for (cx, cy, face, edge, title, sub, mk) in pipeline:
    box(cx, cy, 2.4, 1.8, face, edge, title, sub, mk)

arrows = [(2.72,5.6,3.28,5.6,'#00bcd4','Raw CSV'),
          (5.72,5.6,6.58,5.6,'#4caf50','Sequences\n×20 steps'),
          (9.02,5.6,9.78,5.6,'#9c27b0','h5 + scaler.pkl'),
          (12.22,5.6,12.78,5.6,'#f44336','prediction_plot.png')]
for (x1,y1,x2,y2,col,lbl) in arrows:
    arrow(x1,y1,x2,y2,col,lbl)

details = [
    (1.5,  3.1, '#e3f8fc', '#00bcd4', 'clock_bias_s\nclock_drift_s/s\nephemeris_error_m', '3 Features | 200 rows'),
    (4.5,  3.1, '#f1f8e9', '#4caf50', 'MinMaxScaler\nWindow=20  Split=80/20\nSaves: X_seq.npy  y_seq.npy', 'sklearn'),
    (7.8,  3.1, '#f3e5f5', '#9c27b0', 'LSTM(64) Dropout(0.2)\nLSTM(32) Dropout(0.2)\nDense(16)->Dense(1)', 'TensorFlow / Keras'),
    (11.0, 3.1, '#ffebee', '#f44336', 'Inverse Transform\nActual vs Predicted\nResidual Chart', 'matplotlib'),
    (14.0, 3.1, '#fff8e1', '#ff9800', 'Sidebar Controls\nTime-series Plots\nPrediction Image', 'streamlit'),
]
for (cx, cy, face, edge, title, sub) in details:
    box(cx, cy, 2.4, 2.0, face, edge, title, sub)
    arrow(cx, 4.68, cx, 4.12, col='#aaaaaa')

files = [
    (1.5,  1.72, 'satellite_telemetry.csv', '#00bcd4'),
    (4.5,  1.72, 'preprocess.py',           '#4caf50'),
    (7.8,  1.72, 'train_lstm.py',           '#9c27b0'),
    (11.0, 1.72, 'predict.py',              '#f44336'),
    (14.0, 1.72, 'dashboard.py',            '#ff9800'),
]
for (cx, cy, fname, col) in files:
    arrow(cx, 2.08, cx, 1.92, col=col)
    ax.text(cx, cy, fname, ha='center', va='center', fontsize=8,
            color=col, fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.28', facecolor='#ffffff',
                      edgecolor=col, lw=1.2, alpha=0.95), zorder=6)

handles = [
    mpatches.Patch(color='#00bcd4', label='Data Layer'),
    mpatches.Patch(color='#4caf50', label='Preprocessing'),
    mpatches.Patch(color='#9c27b0', label='ML Model (LSTM)'),
    mpatches.Patch(color='#f44336', label='Output / Evaluation'),
    mpatches.Patch(color='#ff9800', label='Dashboard (Streamlit)'),
]
ax.legend(handles=handles, loc='lower center', ncol=5, fontsize=9,
          framealpha=0.4, facecolor='#f9f9f9', edgecolor='#cccccc',
          labelcolor='#1a1a1a', bbox_to_anchor=(0.5, 0.0))

plt.tight_layout(pad=0.4)
plt.savefig("outputs/architecture_diagram.png", dpi=160,
            bbox_inches='tight', facecolor='#ffffff')
plt.close()
print("Saved: outputs/architecture_diagram.png")
