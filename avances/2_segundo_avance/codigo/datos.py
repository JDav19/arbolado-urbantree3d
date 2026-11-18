"""Carga de UrbanTree3D (152 arboles individuales en formato LAS) y preparacion de entradas.

Fuente: https://doi.org/10.5281/zenodo.19475932 (CC BY 4.0).
Se usa la particion oficial train/val/test del archivo tree_metadata_training_model.csv.
"""
import os
import numpy as np
import pandas as pd
import laspy

RAIZ = os.path.join(os.path.dirname(__file__), "..", "datos_raw")
CACHE = os.path.join(RAIZ, "cache.npz")
ESPECIES = ["Aesculus hippocastanum", "Prunus serrulata", "Pyrus calleryana",
            "Salix alba", "Tilia platyphyllos", "Tilia x europeana"]


def cargar_crudo():
    """Lee los LAS y devuelve listas por arbol. Guarda un cache npz para acelerar."""
    meta = pd.read_csv(os.path.join(RAIZ, "tree_metadata_training_model.csv"))
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        return meta, list(z["nubes"])
    nubes = []
    for f in meta["filename"]:
        las = laspy.read(os.path.join(RAIZ, "extraido", f))
        xyz = np.stack([las.x, las.y, las.z], axis=1).astype(np.float64)
        inten = np.asarray(las.intensity, dtype=np.float64)
        ret = np.asarray(las.return_number, dtype=np.float64)
        nret = np.asarray(las.number_of_returns, dtype=np.float64)
        nubes.append(np.column_stack([xyz, inten, ret, nret]))
    np.savez(CACHE, nubes=np.array(nubes, dtype=object))
    return meta, nubes


def forma_unitaria(nube, n, rng, remuestreo_aleatorio=True):
    """XYZ centrado en el centroide y escalado a la esfera unitaria, con n puntos.

    Si el arbol tiene menos de n puntos se muestrea con reemplazo.
    """
    p = nube[:, :3] - nube[:, :3].mean(axis=0, keepdims=True)
    p = p / np.linalg.norm(p, axis=1).max()
    idx = rng.choice(len(p), n, replace=len(p) < n)
    return p[idx].astype(np.float32)


def caracteristicas_manuales(nube):
    """Descriptores geometricos clasicos para el baseline de bosque aleatorio.

    Devuelve (con_escala, sin_escala). Las de escala incluyen altura, ancho de copa y conteo de puntos.
    """
    xyz, inten = nube[:, :3], nube[:, 3]
    z = xyz[:, 2] - xyz[:, 2].min()
    xy = xyz[:, :2] - xyz[:, :2].mean(axis=0)
    altura = z.max()
    ancho_x, ancho_y = np.ptp(xy[:, 0]), np.ptp(xy[:, 1])
    radio = np.linalg.norm(xy, axis=1)
    zn = z / max(altura, 1e-6)
    pct = np.percentile(zn, [10, 25, 50, 75, 90])
    hist, _ = np.histogram(zn, bins=10, range=(0, 1))
    hist = hist / len(zn)
    cov = np.cov((xyz - xyz.mean(0)).T)
    ev = np.sort(np.linalg.eigvalsh(cov))[::-1]
    ev = ev / ev.sum()
    sin_escala = np.concatenate([
        pct, hist, ev,
        [radio.mean() / max(radio.max(), 1e-6), radio.std() / max(radio.max(), 1e-6),
         min(ancho_x, ancho_y) / max(ancho_x, ancho_y, 1e-6), zn.mean(), zn.std(),
         inten.mean() / 1000.0, inten.std() / 1000.0, nube[:, 4].mean(), nube[:, 5].mean()],
    ])
    escala = np.array([altura, max(ancho_x, ancho_y), altura / max(ancho_x, ancho_y, 1e-6), np.log(len(z))])
    return np.concatenate([escala, sin_escala]), sin_escala
