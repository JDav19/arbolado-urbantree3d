"""Genera tablas LaTeX y figuras a partir de los resultados_*.json."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datos import cargar_crudo, ESPECIES

SAL = "../"
FIG = "../figuras/"

R = {}
for n in ["PointNet", "PointNet++", "DGCNN"]:
    R[n] = json.load(open(f"resultados_{n}.json"))["semillas"]
rf = json.load(open("resultados_RF.json"))
R["RF con escala"] = rf["RF con escala"]["semillas"]
R["RF sin escala"] = rf["RF sin escala"]["semillas"]
ORDEN = ["RF con escala", "RF sin escala", "PointNet", "PointNet++", "DGCNN"]


def serie(nombre, parte, k):
    return np.array([r[parte][k] for r in R[nombre].values()])


def ms(v):
    return f"{v.mean():.3f} $\\pm$ {v.std(ddof=1):.3f}" if len(v) > 1 else f"{v.mean():.3f}"


# ---- Tabla principal
filas = []
for n in ORDEN:
    par = "--" if n.startswith("RF") else f"{next(iter(R[n].values()))['parametros']/1e6:.2f}"
    filas.append(" & ".join([n, ms(serie(n, "test", "acc")), ms(serie(n, "test", "f1_macro")),
                             ms(serie(n, "test", "acc_balanceada")), ms(serie(n, "val", "acc")),
                             ms(serie(n, "val", "f1_macro")), par]) + " \\\\")
with open(SAL + "tabla_principal.tex", "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(filas) + "\n\\bottomrule\n")

# ---- Tabla por clase (F1 de prueba, promedio de semillas)
filas = []
for n in ORDEN:
    f1c = np.mean([r["test"]["f1_clase"] for r in R[n].values()], axis=0)
    filas.append(" & ".join([n] + [f"{v:.2f}" for v in f1c]) + " \\\\")
with open(SAL + "tabla_clases.tex", "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(filas) + "\n\\bottomrule\n")

# ---- Mejor red por F1 de validacion
redes = ["PointNet", "PointNet++", "DGCNN"]
mejor = max(redes, key=lambda n: serie(n, "val", "f1_macro").mean())
print("mejor red por validacion:", mejor)

# ---- Matriz de confusion de la mejor red (suma de semillas)
cm = np.sum([np.array(r["test"]["cm"]) for r in R[mejor].values()], axis=0)
cmn = cm / cm.sum(1, keepdims=True)
fig, ax = plt.subplots(figsize=(5.2, 4.4))
im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(6)); ax.set_yticks(range(6))
ax.set_xticklabels([e.replace(" ", "\n", 1) for e in ESPECIES], fontsize=6.5)
ax.set_yticklabels(ESPECIES, fontsize=7)
for i in range(6):
    for j in range(6):
        ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=8,
                color="white" if cmn[i, j] > 0.5 else "black")
ax.set_xlabel("Especie predicha"); ax.set_ylabel("Especie real")
plt.colorbar(im, fraction=0.046, label="Proporción por fila")
plt.tight_layout(); plt.savefig(FIG + "confusion.pdf"); plt.close()

# ---- Curvas de entrenamiento (semilla 0)
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
for n, c in zip(redes, ["#1f77b4", "#2ca02c", "#d62728"]):
    cur = np.array(R[n]["0"]["curva"])
    ax[0].plot(cur[:, 0], cur[:, 1], color=c, label=n)
    ax[1].plot(cur[:, 0], cur[:, 3], color=c, label=n)
ax[0].set_xlabel("Época"); ax[0].set_ylabel("Pérdida de entrenamiento")
ax[1].set_xlabel("Época"); ax[1].set_ylabel("F1 macro de validación")
ax[0].legend(fontsize=7); ax[0].grid(alpha=.3); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig(FIG + "curvas.pdf"); plt.close()

# ---- Ejemplos cualitativos: un arbol de prueba por especie
meta, nubes = cargar_crudo()
idx_test = np.where(meta["train_val"].values == "test")[0]
ytest = meta["class"].values[idx_test]
pred_dl = np.array(R[mejor]["0"]["pred_test"])
pred_rf = np.array(R["RF sin escala"]["0"]["pred_test"])
fig, axs = plt.subplots(2, 3, figsize=(7.4, 5.6))
for c, ax in enumerate(axs.ravel()):
    k = np.where(ytest == c)[0][0]
    n = nubes[idx_test[k]]
    x = n[:, 0] - n[:, 0].mean(); z = n[:, 2] - n[:, 2].min()
    ax.scatter(x, z, c=z, cmap="viridis", s=2)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.tick_params(labelsize=6)
    ok_dl, ok_rf = pred_dl[k] == c, pred_rf[k] == c
    ax.set_title(f"Real: {ESPECIES[c]}\n{mejor}: {ESPECIES[pred_dl[k]]} ({'ok' if ok_dl else 'error'})\n"
                 f"RF: {ESPECIES[pred_rf[k]]} ({'ok' if ok_rf else 'error'})", fontsize=6.5)
    ax.set_ylabel("Altura (m)", fontsize=6.5)
plt.tight_layout(); plt.savefig(FIG + "cualitativo.png", dpi=200); plt.close()

# ---- Cifras para el texto
res = {"mejor": mejor}
for n in ORDEN:
    res[n] = dict(test_acc=serie(n, "test", "acc").mean(), test_f1=serie(n, "test", "f1_macro").mean(),
                  val_acc=serie(n, "val", "acc").mean(), val_f1=serie(n, "val", "f1_macro").mean(),
                  tiempo=float(np.mean([r.get("tiempo_s", 0) for r in R[n].values()])))
json.dump(res, open("resumen.json", "w"), indent=1, default=float)
print(json.dumps(res, indent=1, default=float))
