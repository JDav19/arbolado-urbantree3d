"""Entrena y evalua un modelo (PointNet, PointNet++ o DGCNN) sobre UrbanTree3D.

Uso  python entrenar.py PointNet
"""
import json
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from datos import cargar_crudo, forma_unitaria
from modelos import crear

N_PUNTOS = 1024
EPOCAS = int(sys.argv[2]) if len(sys.argv) > 2 else 80
LOTE = 16
LR = 1e-3
SEMILLAS = [0, 5, 95]
HILOS = 4


def metricas(pred, real, n_clases=6):
    cm = np.zeros((n_clases, n_clases), dtype=int)
    for r, p in zip(real, pred):
        cm[r, p] += 1
    tp = np.diag(cm).astype(float)
    prec = tp / np.maximum(cm.sum(0), 1)
    rec = tp / np.maximum(cm.sum(1), 1)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-9), 0.0)
    return dict(acc=tp.sum() / cm.sum(), f1_macro=f1.mean(), acc_balanceada=rec.mean(),
                f1_clase=f1.tolist(), cm=cm.tolist())


def aumentar(x, rng):
    """x (B,N,3) numpy. Rotacion en Z, escala, jitter y descarte de puntos."""
    B, N, _ = x.shape
    t = rng.uniform(0, 2 * np.pi, B)
    c, s = np.cos(t), np.sin(t)
    R = np.zeros((B, 3, 3)); R[:, 0, 0] = c; R[:, 0, 1] = -s; R[:, 1, 0] = s; R[:, 1, 1] = c; R[:, 2, 2] = 1
    x = np.einsum("bij,bnj->bni", R, x)
    x = x * rng.uniform(0.85, 1.15, (B, 1, 3))
    x = x + rng.normal(0, 0.01, x.shape)
    descarte = rng.random((B, N)) < rng.uniform(0, 0.2, (B, 1))
    x[descarte] = x[:, :1, :].repeat(N, axis=1)[descarte]
    return x.astype(np.float32)


def preparar():
    meta, nubes = cargar_crudo()
    rng = np.random.default_rng(1234)
    sets = {}
    for nombre in ["train", "val", "test"]:
        idx = np.where(meta["train_val"].values == nombre)[0]
        sets[nombre] = dict(idx=idx, y=meta["class"].values[idx],
                            nubes=[nubes[i] for i in idx],
                            x_fijo=np.stack([forma_unitaria(nubes[i], N_PUNTOS, rng) for i in idx]))
    return meta, sets


def predecir(modelo, x):
    modelo.eval()
    with torch.no_grad():
        return torch.cat([modelo(torch.from_numpy(x[i:i + 32]).transpose(1, 2).contiguous()).argmax(1)
                          for i in range(0, len(x), 32)]).numpy()


def correr(nombre, semilla, sets):
    torch.manual_seed(semilla)
    rng = np.random.default_rng(semilla)
    modelo = crear(nombre)
    n_par = sum(p.numel() for p in modelo.parameters())
    ytr = sets["train"]["y"]
    frec = np.bincount(ytr, minlength=6)
    pesos = torch.tensor((frec.sum() / (6 * frec)) ** 0.5, dtype=torch.float32)
    perdida = nn.CrossEntropyLoss(weight=pesos)
    opt = torch.optim.Adam(modelo.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCAS)
    mejor, estado, curva, t0 = (-1, -1), None, [], time.time()
    for ep in range(EPOCAS):
        modelo.train()
        orden = rng.permutation(len(ytr))
        total = 0.0
        for i in range(0, len(orden), LOTE):
            b = orden[i:i + LOTE]
            if len(b) < 2:
                continue
            x = np.stack([forma_unitaria(sets["train"]["nubes"][j], N_PUNTOS, rng) for j in b])
            x = torch.from_numpy(aumentar(x, rng)).transpose(1, 2).contiguous()
            opt.zero_grad()
            loss = perdida(modelo(x), torch.from_numpy(ytr[b]))
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        sched.step()
        mv = metricas(predecir(modelo, sets["val"]["x_fijo"]), sets["val"]["y"])
        curva.append((ep + 1, total / len(ytr), mv["acc"], mv["f1_macro"]))
        clave = (mv["f1_macro"], mv["acc"])
        if clave > mejor:
            mejor, estado = clave, {k: v.clone() for k, v in modelo.state_dict().items()}
        if (ep + 1) % 5 == 0:
            print(f"{nombre} s{semilla} ep{ep+1:03d} loss={total/len(ytr):.3f} val_acc={mv['acc']:.3f} "
                  f"val_f1={mv['f1_macro']:.3f} t={time.time()-t0:.0f}s", flush=True)
    modelo.load_state_dict(estado)
    pv = predecir(modelo, sets["val"]["x_fijo"])
    pt = predecir(modelo, sets["test"]["x_fijo"])
    res = dict(val=metricas(pv, sets["val"]["y"]), test=metricas(pt, sets["test"]["y"]),
               parametros=n_par, tiempo_s=time.time() - t0, curva=curva,
               pred_test=pt.tolist())
    return res


if __name__ == "__main__":
    nombre = sys.argv[1]
    torch.set_num_threads(HILOS)
    meta, sets = preparar()
    salida = {"epocas": EPOCAS, "n_puntos": N_PUNTOS, "semillas": {}}
    for s in SEMILLAS:
        salida["semillas"][str(s)] = correr(nombre, s, sets)
        with open(f"resultados_{nombre}.json", "w") as f:
            json.dump(salida, f, default=float)
    print("listo", nombre)
