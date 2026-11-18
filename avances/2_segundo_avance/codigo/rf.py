"""Baseline clasico: bosque aleatorio sobre descriptores geometricos de cada arbol."""
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from datos import cargar_crudo, caracteristicas_manuales
from entrenar import metricas

SEMILLAS = [0, 5, 95]


def main():
    meta, nubes = cargar_crudo()
    F = [caracteristicas_manuales(n) for n in nubes]
    y = meta["class"].values
    sp = meta["train_val"].values
    salida = {}
    for nombre, col in [("RF con escala", 0), ("RF sin escala", 1)]:
        X = np.stack([f[col] for f in F])
        salida[nombre] = {"semillas": {}, "n_descriptores": X.shape[1]}
        for s in SEMILLAS:
            rf = RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=s, n_jobs=2)
            rf.fit(X[sp == "train"], y[sp == "train"])
            salida[nombre]["semillas"][str(s)] = dict(
                val=metricas(rf.predict(X[sp == "val"]), y[sp == "val"]),
                test=metricas(rf.predict(X[sp == "test"]), y[sp == "test"]),
                pred_test=rf.predict(X[sp == "test"]).tolist())
    with open("resultados_RF.json", "w") as f:
        json.dump(salida, f, default=float)
    for k, v in salida.items():
        a = [r["test"]["acc"] for r in v["semillas"].values()]
        f1 = [r["test"]["f1_macro"] for r in v["semillas"].values()]
        print(k, v["n_descriptores"], "test acc", np.mean(a).round(3), "f1", np.mean(f1).round(3))


if __name__ == "__main__":
    main()
