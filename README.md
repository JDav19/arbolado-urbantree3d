# Clasificación de especies de árboles urbanos con LiDAR y aprendizaje profundo

Proyecto de la Electiva en Ingeniería de Software de la Universidad de San Buenaventura Cali, hecho por Julian David Fuentes Granja. El proyecto entrena redes neuronales para reconocer la especie de un árbol a partir de un escaneo 3D tomado desde un avión, usando los 152 árboles del dataset público UrbanTree3D.

## Cómo está organizado el repositorio

El repositorio guarda todo el proceso del proyecto, desde el primer avance hasta la entrega final.

| Carpeta o archivo | Contenido |
|---|---|
| `avances/1_primer_avance` | Documento del primer avance, con la definición del problema y la elección del dataset |
| `avances/2_segundo_avance` | Documento del segundo avance con su código LaTeX, los scripts de Python con los que se entrenaron los primeros modelos, sus registros, sus resultados y sus figuras |
| `Proyecto_Final_Arbolado_UrbanTree3D.ipynb` | Notebook final ejecutado en Google Colab con GPU T4, con todo el entrenamiento, las métricas y las figuras |
| `entrenamiento` | Resultados de cada entrenamiento del notebook final y las figuras que generó |
| `informe_final` | Informe técnico final en PDF y su código LaTeX |

## Cómo ejecutar el notebook

Se abre en Google Colab, se elige la GPU T4 en el tipo de entorno de ejecución y se le da a Ejecutar todo. El notebook descarga el dataset desde Zenodo y tarda unas dos horas.

## Datos

El dataset no está en el repositorio porque el notebook lo descarga solo. Es UrbanTree3D de Hamdani et al. (2026), publicado en la revista Data con el DOI 10.5281/zenodo.19475932 y licencia CC BY 4.0.
