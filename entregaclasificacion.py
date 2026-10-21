# -*- coding: utf-8 -*-
"""
entregaclasificacion.py
=======================
Entrega final: clasificación supervisada (Random Forest) de una imagen Sentinel-2 L2A.

Basado en las notebooks del repo:
  - 02c  -> lectura de bandas desde el .SAFE.zip, recorte con poligono y remuestreo 20 m -> 10 m
  - 03a  -> generación de muestras (CSV) a partir de vectores, con extraer_muestras()
  - 03b  -> entrenamiento, matriz de confusión, métricas, búsqueda de max_depth,
            mapa clasificado y validación externa con un segundo recorte ("Yapa")

Imagen: S2A_MSIL2A_20250109T151721_N0511_R125_T19NBH_20250109T183248.SAFE

-------------------------------------------------------------------------------
USO EN GOOGLE COLAB (el script se ejecuta en DOS pasadas)
-------------------------------------------------------------------------------
Celda 1:   !pip install -q rasterio geopandas scikit-learn scipy matplotlib seaborn gdown
Celda 2:   !wget -q -O entregaclasificacion.py https://raw.githubusercontent.com/AlexandraUD/ClasificacionSupervisada/main/entregaclasificacion.py
Celda 3:   %run entregaclasificacion.py

Carpeta de trabajo en Drive (CARPETA, más abajo). Entradas (si no están, el script las
descarga solo: el zip con gdown y los vectores desde el repo de GitHub):
  - el .zip de la Sentinel-2 (S2A_MSIL2A_..._T19NBH_....SAFE.zip), sin descomprimir
  - recorte1.gpkg / recorte2.gpkg                      -> polígonos de recorte (hechos en QGIS)
  - muestras_recorte1.gpkg / muestras_recorte2.gpkg    -> polígonos de muestras con campos
                                                          Clase (entero) y DESC (texto)
  (también se aceptan .shp con los mismos nombres)

PASADA 1: si faltan los vectores de muestras, el script genera los dos recortes
          (salidas/s2_recorte1.tif y salidas/s2_recorte2.tif), te avisa y se detiene.
          Abrís esos recortes en QGIS y dibujás las muestras.
PASADA 2: con las muestras ya guardadas, vuelve a correr el script y hace todo el resto:
          CSV de muestras, firmas espectrales, entrenamiento, métricas, mapas GeoTIFF y
          validación con el recorte 2.
-------------------------------------------------------------------------------
"""

import glob
import os
import re
import shutil
import sys
import textwrap
import time
import urllib.request
from zipfile import ZipFile

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from rasterio.enums import Resampling
from rasterio.features import shapes
from rasterio.mask import mask
from rasterio.windows import from_bounds
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, cohen_kappa_score,
                             confusion_matrix)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from shapely.geometry import shape

# =============================================================================
# CONFIGURACIÓN  (lo único que normalmente hay que tocar)
# =============================================================================
# Carpeta de Google Drive donde están el zip y los vectores
CARPETA = "/content/drive/MyDrive/ClasificacionSupervisada"

# Nombre del producto (sin extensión). El script busca <nombre>*.zip dentro de CARPETA
PRODUCTO = "S2A_MSIL2A_20250109T151721_N0511_R125_T19NBH_20250109T183248"

# Nombres base de los vectores de entrada (sin extensión; se busca .gpkg y luego .shp)
RECORTE1 = "recorte1"
RECORTE2 = "recorte2"
MUESTRAS1 = "muestras_recorte1"
MUESTRAS2 = "muestras_recorte2"

# Si el zip no está en CARPETA, se descarga con gdown desde este archivo de Drive
# (el archivo debe estar compartido como "Cualquier persona con el enlace")
ID_DRIVE_ZIP = "1B-jaWC9GyF4SwkIw90Z7f_6Wc3--Hw9L"
# Si faltan los vectores en CARPETA, se bajan del repo de GitHub
URL_REPO_RAW = "https://raw.githubusercontent.com/AlexandraUD/ClasificacionSupervisada/main/"

# Copiar el zip del Drive al disco local de Colab (lee mucho más rápido)
COPIAR_ZIP_LOCAL = True
# Volver a generar los recortes aunque ya existan
REGENERAR_RECORTES = False

# Bandas: primero las de 10 m, después las de 20 m (remuestreadas a 10 m).
# El orden da Band_0 ... Band_5, igual que en las notebooks 03a/03b.
BANDAS_10M = ["B02", "B03", "B04", "B08"]
BANDAS_20M = ["B11", "B12"]
BANDAS = BANDAS_10M + BANDAS_20M
COLUMNAS_BANDAS = [f"Band_{i}" for i in range(len(BANDAS))]
# Longitudes de onda centrales (nm) de B02, B03, B04, B08, B11, B12
LONGITUDES_ONDA_NM = np.array([490, 560, 665, 842, 1610, 2190])

VALOR_RELLENO = 65535                    # relleno para mask() en extraer_muestras()
NODATA_CLASIF = np.iinfo(np.uint16).max  # nodata del mapa clasificado (65535)
SEMILLA = 42

SALIDAS = os.path.join(CARPETA, "salidas")

# Colores por clase (tomados de simblogia.qml, la simbología de QGIS), indexados por DESC
COLORES_DESC = {
    "Bosques y montes": "#269E08",
    "Construcciones": "#CC0A17",
    "Cuerpos de agua": "#2482D5",
    "Cultivos y parcelas": "#EDEA24",
    "Vegetacion herbacea descubierta": "#4CF12B",
    "Tierras eriales": "#4A131D",
    "Nube": "#FFFFFF",
}
COLORES_CLASE = {}  # se completa en main() con asignar_colores()


# =============================================================================
# Funciones de visualización (Notebooks 02c / 03a / 03b)
# =============================================================================
def escalar_imagen(imagen, p, nodata=None):
    # Calcular los percentiles p y 100-p
    valor_min = np.percentile(imagen[imagen != nodata], p)
    valor_max = np.percentile(imagen[imagen != nodata], 100 - p)

    # Truncar la imagen al rango [p, 100-p]
    ecualizada = np.clip(imagen, valor_min, valor_max)

    # Normalizar la imagen al rango 0-255
    ecualizada = (ecualizada - valor_min) / (valor_max - valor_min) * 255

    # Convertir a entero de 8 bits
    return np.uint8(ecualizada)


def escalar_imagen_multibanda(imagen, p, nodata=None):
    # imagen: (filas, columnas, bandas)
    imagen_escalada = np.empty_like(imagen)
    canales = imagen_escalada.shape[-1]
    for i in range(canales):
        imagen_escalada[:, :, i] = escalar_imagen(imagen[:, :, i], p, nodata=nodata)
    return np.uint8(imagen_escalada)


def guardar_figura(nombre_png):
    """Guarda la figura actual en SALIDAS, la muestra y la cierra."""
    plt.tight_layout()
    plt.savefig(os.path.join(SALIDAS, nombre_png), dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


# =============================================================================
# Lectura de Sentinel-2 L2A (Notebook 02c)
# =============================================================================
def _ruta_banda(nombres_zip, banda, res):
    """Busca dentro del zip el .jp2 de una banda a la resolución indicada."""
    patron = re.compile(rf"/R{res}m/[^/]+_{banda}_{res}m\.jp2$")
    for nombre in nombres_zip:
        if patron.search(nombre):
            return nombre
    raise FileNotFoundError(f"No encontré la banda {banda} a {res} m dentro del zip.")


def leer_bandas_l2a(zipfilename, bands_to_extract, res, poligono=None):
    """Lee imagen S2 Nivel 2A desde el ZIP (versión de la Notebook 02c).

    zipfilename: ruta del zip
    bands_to_extract: lista de bandas, ej. ['B02', 'B03', 'B04', 'B08']
    res: resolución (carpeta R{res}m), ej. 10
    poligono: GeoDataFrame para recortar (se reproyecta al CRS de la imagen). None = imagen completa

    Mejoras respecto de la notebook: las bandas salen SIEMPRE en el orden pedido
    y el polígono se reproyecta solo al CRS de la imagen (sirve para cualquier zona/huso).
    """
    with ZipFile(zipfilename, "r") as zfile:
        nombres = zfile.namelist()

    stack_bandas = []
    metadatos = None
    for banda in bands_to_extract:
        bnd = _ruta_banda(nombres, banda, res)
        fname = f"/vsizip/{zipfilename}/{bnd}"
        with rasterio.open(fname) as ds:
            metadatos = ds.meta.copy()
            if poligono is not None:
                geometrias = poligono.to_crs(ds.crs).geometry
                clip, clip_transform = mask(ds, geometrias, crop=True)
                stack_bandas.append(clip[0])
                metadatos["transform"] = clip_transform
                metadatos["height"] = clip.shape[1]
                metadatos["width"] = clip.shape[2]
            else:
                stack_bandas.append(ds.read(1))

    stack_bandas = np.array(stack_bandas)
    metadatos["count"] = stack_bandas.shape[0]
    metadatos["driver"] = "GTiff"
    metadatos["nodata"] = 0  # en L2A el relleno es 0
    return stack_bandas, metadatos


def leer_bandas_remuestreadas(zipfilename, bands_to_extract, res, meta_ref):
    """Lee bandas de menor resolución (ej. 20 m) y las remuestrea a la grilla de meta_ref (10 m).

    Equivale a "2.2 Remuestreo a 10 metros" de la Notebook 02c, pero leyendo la ventana
    exacta del recorte de referencia (misma extensión y mismo tamaño en píxeles), así los
    stacks de 10 m y de 20 m siempre se pueden concatenar sin diferencias de 1 píxel.
    """
    with ZipFile(zipfilename, "r") as zfile:
        nombres = zfile.namelist()

    alto, ancho, t = meta_ref["height"], meta_ref["width"], meta_ref["transform"]
    izquierda, arriba = t.c, t.f
    derecha, abajo = t.c + t.a * ancho, t.f + t.e * alto

    stack = []
    for banda in bands_to_extract:
        bnd = _ruta_banda(nombres, banda, res)
        with rasterio.open(f"/vsizip/{zipfilename}/{bnd}") as ds:
            ventana = from_bounds(izquierda, abajo, derecha, arriba, transform=ds.transform)
            stack.append(ds.read(1, window=ventana, out_shape=(alto, ancho),
                                 resampling=Resampling.bilinear))
    return np.array(stack)


def generar_recorte(zipfilename, poligono, salida_tif):
    """Genera el stack B02,B03,B04,B08,B11,B12 a 10 m recortado con 'poligono' y lo guarda."""
    print(f"  Leyendo bandas de 10 m {BANDAS_10M} ...")
    stack10, meta = leer_bandas_l2a(zipfilename, BANDAS_10M, 10, poligono=poligono)
    print(f"  Leyendo bandas de 20 m {BANDAS_20M} y remuestreando a 10 m ...")
    stack20 = leer_bandas_remuestreadas(zipfilename, BANDAS_20M, 20, meta)

    stack_final = np.concatenate((stack10, stack20), axis=0).astype("uint16")
    print(f"  Dimensiones del stack final: {stack_final.shape}")

    meta.update(count=stack_final.shape[0], dtype="uint16", nodata=0, driver="GTiff",
                height=stack_final.shape[1], width=stack_final.shape[2])
    with rasterio.open(salida_tif, "w", **meta) as dst:
        dst.write(stack_final)
        for i, b in enumerate(BANDAS):
            dst.set_band_description(i + 1, b)
    print(f"  Guardado: {salida_tif}")


# =============================================================================
# Muestras (Notebook 03a)
# =============================================================================
def extraer_muestras(imagen, poligonos_externos, valor_relleno, columnas=COLUMNAS_BANDAS):
    """Función de la Notebook 03a. Espera columnas 'Clase' y 'DESC' en los polígonos.

    Agregados: se saltea (con aviso) un polígono que no toca la imagen y se descartan
    los píxeles sin dato (0) además del valor de relleno.
    """
    muestras_extraidas = pd.DataFrame()
    for i, pol in poligonos_externos.iterrows():
        try:
            clip, _ = mask(dataset=imagen, shapes=[pol.geometry], crop=True, nodata=valor_relleno)
        except ValueError:
            print(f"  AVISO: el polígono {i} (Clase={pol['Clase']}) no se superpone con la imagen. Se omite.")
            continue
        d, x, y = clip.shape
        clip = clip.reshape([d, x * y]).T
        clip = pd.DataFrame(clip, columns=columnas)
        clip = clip[(clip != valor_relleno).all(axis=1) & (clip != 0).all(axis=1)]
        clip["Clase"] = pol["Clase"]
        clip["DESC"] = pol["DESC"]
        clip["Rect_ID"] = i + 1
        muestras_extraidas = pd.concat([muestras_extraidas, clip], ignore_index=True)
    return muestras_extraidas


def leer_vector_muestras(ruta_vector, crs_destino):
    """Lee el vector de muestras, verifica los campos y lo pasa al CRS de la imagen."""
    pols = gpd.read_file(ruta_vector)
    faltan = [c for c in ("Clase", "DESC") if c not in pols.columns]
    if faltan:
        raise ValueError(f"A {os.path.basename(ruta_vector)} le faltan los campos {faltan}. "
                         "Debe tener 'Clase' (entero) y 'DESC' (texto).")
    pols = pols[pols.geometry.notna() & ~pols.geometry.is_empty].reset_index(drop=True)
    pols["Clase"] = pols["Clase"].astype(int)
    pols["DESC"] = pols["DESC"].astype(str)
    return pols.to_crs(crs_destino)


def dict_de_clases(*dataframes):
    """{Clase: DESC} a partir de uno o más DataFrames/GeoDataFrames."""
    clases = {}
    for df in dataframes:
        for c, nombre in df[["Clase", "DESC"]].drop_duplicates().itertuples(index=False):
            clases[int(c)] = str(nombre)
    return dict(sorted(clases.items()))


def asignar_colores(dict_clases):
    """Asigna a cada código de clase el color de QGIS según su DESC (o un color tab10 si no está)."""
    for clase, desc in dict_clases.items():
        COLORES_CLASE[clase] = COLORES_DESC.get(desc, plt.cm.tab10(clase % 10))


def color_clase(clase):
    """Color de una clase (el mismo de la simbología de QGIS)."""
    return COLORES_CLASE.get(clase, plt.cm.tab10(clase % 10))


def graficar_firmas_espectrales(dfs, dict_clases, nombre_png):
    """Firmas espectrales medias +- desvío por clase (Notebook 03a, 1.1).

    dfs: {título: DataFrame de muestras}. Se hace un panel por cada DataFrame (todas sus muestras),
    con el mismo eje Y para poder comparar el recorte 1 con el recorte 2.
    """
    fig, axes = plt.subplots(1, len(dfs), figsize=(7.5 * len(dfs), 6), sharey=True, squeeze=False)
    etiquetas_x = [f"{b}\n{int(w)}" for b, w in zip(BANDAS, LONGITUDES_ONDA_NM)]
    for ax, (titulo, df) in zip(axes[0], dfs.items()):
        for clase, nombre in dict_clases.items():
            sub = df.loc[df["Clase"] == clase, COLUMNAS_BANDAS]
            if len(sub) == 0:
                continue
            media = sub.mean().values
            desvio = sub.std().fillna(0).values
            ax.plot(LONGITUDES_ONDA_NM, media, "o-", lw=2, ms=5, color=color_clase(clase))
            ax.fill_between(LONGITUDES_ONDA_NM, media - desvio, media + desvio,
                            color=color_clase(clase), alpha=0.12)
        ax.set_xticks(LONGITUDES_ONDA_NM)
        ax.set_xticklabels(etiquetas_x, fontsize=9)
        ax.set_xlabel("Banda y longitud de onda central (nm)")
        ax.set_title(f"{titulo} ({len(df):,} píxeles)".replace(",", " "))
        ax.grid(True, alpha=0.4)
    axes[0][0].set_ylabel("Reflectancia media (valor de la banda)")
    fig.suptitle("Firmas espectrales por clase (media; banda = ± 1 desvío estándar)", fontsize=13)
    fig.legend(handles=[plt.Line2D([0], [0], color=color_clase(c), lw=3, marker="o", label=n)
                        for c, n in dict_clases.items()],
               loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.06))
    guardar_figura(nombre_png)


def graficar_pairplot(df, dict_clases, titulo, nombre_png, max_por_clase=300):
    """Dispersión entre pares de bandas con igual cantidad máxima de píxeles por clase (03a, 1.1).

    Con una muestra al azar simple, las clases grandes tapan a las pequeñas; por eso se toman hasta
    'max_por_clase' píxeles de cada clase.
    """
    nombres_banda = {c: f"{b} ({int(w)} nm)" for c, b, w in zip(COLUMNAS_BANDAS, BANDAS, LONGITUDES_ONDA_NM)}
    partes = [g.sample(min(max_por_clase, len(g)), random_state=SEMILLA)
              for _, g in df.groupby("Clase")]
    sub = pd.concat(partes)[COLUMNAS_BANDAS + ["DESC"]].rename(columns=nombres_banda)
    paleta = {desc: color_clase(c) for c, desc in dict_clases.items()}
    orden = [dict_clases[c] for c in sorted(dict_clases) if dict_clases[c] in set(sub["DESC"])]
    g = sns.pairplot(sub, hue="DESC", hue_order=orden, palette=paleta, corner=True,
                     diag_kind="kde", height=1.9,
                     plot_kws={"alpha": 0.5, "s": 14, "edgecolor": "none"},
                     diag_kws={"common_norm": False, "fill": True, "alpha": 0.25})
    sns.move_legend(g, "upper right", bbox_to_anchor=(0.97, 0.97), title="Clase", frameon=True)
    g.figure.suptitle(f"{titulo} (hasta {max_por_clase} píxeles por clase)", y=1.01, fontsize=13)
    plt.savefig(os.path.join(SALIDAS, nombre_png), dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


def graficar_recorte_con_poligonos(ruta_tif, poligonos, titulo, nombre_png):
    """Imagen en falso color (NIR, R, G) con los polígonos de muestras encima (03a, 1.2)."""
    with rasterio.open(ruta_tif) as src:
        img = src.read()
        left, bottom, right, top = src.bounds
    esc = escalar_imagen_multibanda(img.transpose(1, 2, 0), 1, nodata=0)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(esc[:, :, [3, 2, 1]], extent=[left, right, bottom, top], origin="upper")
    ax.ticklabel_format(style="plain", axis="both")
    if poligonos is not None:
        for clase in sorted(poligonos["Clase"].unique()):
            sub = poligonos[poligonos["Clase"] == clase]
            sub.plot(ax=ax, facecolor="none", edgecolor=color_clase(int(clase)), linewidth=2)
        ax.legend(handles=[Patch(facecolor="none", edgecolor=color_clase(int(c)),
                                 label=poligonos.loc[poligonos["Clase"] == c, "DESC"].iloc[0])
                           for c in sorted(poligonos["Clase"].unique())], loc="upper right")
    ax.set_title(titulo)
    guardar_figura(nombre_png)


# =============================================================================
# Evaluación y clasificación (Notebook 03b)
# =============================================================================
def evaluar(modelo, X, y, dict_clases, titulo, nombre_png):
    """Matriz de confusión + classification_report + exactitud global."""
    y_pred = modelo.predict(X)
    etiquetas = sorted({int(e) for e in set(y) | set(y_pred)})
    nombres = [dict_clases.get(e, str(e)) for e in etiquetas]

    m = confusion_matrix(y, y_pred, labels=etiquetas)
    nombres_cortos = [textwrap.fill(n, 14) for n in nombres]  # evita que las etiquetas se pisen
    fig, ax = plt.subplots(figsize=(10, 9))
    disp = ConfusionMatrixDisplay(confusion_matrix=m, display_labels=nombres_cortos)
    disp.plot(ax=ax, cmap="Greens", xticks_rotation=45)
    plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")
    ax.set_title(f"Matriz de confusión - {titulo}")
    guardar_figura(nombre_png)

    print(f"\n----- {titulo} -----")
    print(classification_report(y, y_pred, labels=etiquetas, target_names=nombres, zero_division=0))
    acc = accuracy_score(y, y_pred)
    print(f"Precisión global: {acc:.3f}")
    print(f"Índice kappa: {cohen_kappa_score(y, y_pred):.3f}")
    return acc


def clasificar_imagen(modelo, ruta_tif, salida_tif, dict_clases, titulo, nombre_png):
    """Aplica el clasificador a toda la imagen, grafica y guarda el mapa como GeoTIFF."""
    with rasterio.open(ruta_tif) as src:
        bandas = src.read()
        meta = src.meta.copy()

    d, x, y = bandas.shape
    # Romper la estructura espacial -> (x*y, d)
    bandas_flat = bandas.reshape([d, x * y]).T
    # Píxeles con dato (fuera del polígono de recorte / borde de tile = 0)
    valido = (bandas_flat != 0).all(axis=1)

    pred = np.full(x * y, NODATA_CLASIF, dtype=np.uint16)
    if valido.any():
        entrada = pd.DataFrame(bandas_flat[valido], columns=COLUMNAS_BANDAS)
        pred[valido] = modelo.predict(entrada)
    pred = pred.reshape(x, y)
    print(f"  Valores únicos del mapa: {np.unique(pred)}")

    # Gráfico con leyenda
    fig, ax = plt.subplots(figsize=(10, 10))
    n_cl = max(int(c) for c in dict_clases) + 1
    cmap = ListedColormap([color_clase(i) for i in range(n_cl)])
    ax.imshow(np.ma.masked_equal(pred, NODATA_CLASIF), cmap=cmap, vmin=0, vmax=max(n_cl - 1, 1),
              interpolation="nearest")
    presentes = [int(c) for c in np.unique(pred) if c != NODATA_CLASIF]
    ax.legend(handles=[Patch(color=color_clase(c), label=dict_clases.get(c, str(c)))
                       for c in presentes], loc="lower right")
    ax.set_title(titulo)
    ax.axis("off")
    guardar_figura(nombre_png)

    # Guardar GeoTIFF de una banda (como en la Notebook 03b, 3.4)
    meta.update(count=1, dtype="uint16", nodata=NODATA_CLASIF, driver="GTiff")
    with rasterio.open(salida_tif, "w", **meta) as dst:
        dst.write(np.expand_dims(pred, axis=0))
    print(f"  Guardado: {salida_tif}")

    # También como vector GeoPackage (polígonos por clase) para abrir en QGIS
    filas = [{"Clase": int(v), "DESC": dict_clases.get(int(v), str(int(v))), "geometry": shape(g)}
             for g, v in shapes(pred, mask=(pred != NODATA_CLASIF), transform=meta["transform"])]
    salida_gpkg = os.path.splitext(salida_tif)[0] + ".gpkg"
    gpd.GeoDataFrame(filas, geometry="geometry", crs=meta["crs"]).to_file(salida_gpkg, driver="GPKG")
    print(f"  Guardado: {salida_gpkg}")


# =============================================================================
# Utilidades de archivos / Colab
# =============================================================================
def montar_drive():
    """En Colab monta Drive si la carpeta de trabajo está allí."""
    if CARPETA.startswith("/content/drive") and not os.path.isdir("/content/drive/MyDrive"):
        try:
            from google.colab import drive
            drive.mount("/content/drive")
        except ImportError:
            pass


def buscar_vector(nombre_base):
    """Devuelve la ruta de <nombre_base>.gpkg, siempre con la versión más reciente del repo.

    Primero descarga el .gpkg de GitHub y reemplaza la copia de CARPETA (así se usan las muestras
    recién subidas al repo). Si la descarga falla, usa lo que haya en CARPETA (.gpkg o .shp).
    """
    ruta = os.path.join(CARPETA, nombre_base + ".gpkg")
    temporal = ruta + ".descarga"
    try:
        # el parámetro evita que se use una copia vieja guardada en caché
        urllib.request.urlretrieve(f"{URL_REPO_RAW}{nombre_base}.gpkg?v={int(time.time())}", temporal)
        shutil.move(temporal, ruta)
        print(f"Actualizado desde GitHub: {nombre_base}.gpkg")
        return ruta
    except Exception:
        if os.path.exists(temporal):
            os.remove(temporal)
    for ext in (".gpkg", ".shp"):
        ruta = os.path.join(CARPETA, nombre_base + ext)
        if os.path.exists(ruta):
            print(f"No se pudo descargar {nombre_base}.gpkg; se usa la copia de {CARPETA}.")
            return ruta
    return None


def preparar_zip():
    """Ubica el zip de la Sentinel-2 (o lo descarga) y lo copia al disco local de Colab."""
    candidatos = sorted(glob.glob(os.path.join(CARPETA, PRODUCTO + "*.zip")))
    if not candidatos:
        import gdown
        destino = os.path.join("/content" if os.path.isdir("/content") else CARPETA,
                               PRODUCTO + ".SAFE.zip")
        if not os.path.exists(destino) or not os.path.getsize(destino):
            print("Descargando el zip de la Sentinel-2 desde Drive (~1 GB, puede tardar)...")
            gdown.download(id=ID_DRIVE_ZIP, output=destino, quiet=False)
        if not os.path.exists(destino) or not os.path.getsize(destino):
            sys.exit("No pude descargar el zip. Revisa que el archivo esté compartido con "
                     f"'Cualquier persona con el enlace' o súbelo a {CARPETA}.")
        return destino
    origen = candidatos[0]
    if COPIAR_ZIP_LOCAL and origen.startswith("/content/drive"):
        destino = os.path.join("/content", os.path.basename(origen))
        if not os.path.exists(destino) or os.path.getsize(destino) != os.path.getsize(origen):
            print(f"Copiando el zip al disco local de Colab ({os.path.getsize(origen) / 1e9:.2f} GB)...")
            shutil.copy(origen, destino)
        return destino
    return origen


# =============================================================================
# PROGRAMA PRINCIPAL
# =============================================================================
def main():
    montar_drive()
    os.makedirs(SALIDAS, exist_ok=True)

    # ------------------------------------------------------------------
    # PASO 1. Recortes de la Sentinel-2 (Notebook 02c)
    # ------------------------------------------------------------------
    print("=" * 70, "\nPASO 1: Recortes de la imagen Sentinel-2\n" + "=" * 70)
    rec1_vec, rec2_vec = buscar_vector(RECORTE1), buscar_vector(RECORTE2)
    tif1 = os.path.join(SALIDAS, "s2_recorte1.tif")
    tif2 = os.path.join(SALIDAS, "s2_recorte2.tif")

    for nombre, vec, tif in ((RECORTE1, rec1_vec, tif1), (RECORTE2, rec2_vec, tif2)):
        if os.path.exists(tif) and not REGENERAR_RECORTES:
            print(f"Ya existe {os.path.basename(tif)}, no se vuelve a generar.")
            continue
        if vec is None:
            sys.exit(f"Falta el polígono de recorte '{nombre}.gpkg' (o .shp) en {CARPETA}. "
                     "Crealo en QGIS (un polígono por archivo) y volvé a correr.")
        print(f"Generando {os.path.basename(tif)} con {os.path.basename(vec)}")
        generar_recorte(preparar_zip(), gpd.read_file(vec), tif)

    # Vista previa en color natural (B04, B03, B02) de ambos recortes
    for tif, titulo, png in ((tif1, "Recorte 1 (entrenamiento)", "vista_recorte1.png"),
                             (tif2, "Recorte 2 (validación)", "vista_recorte2.png")):
        with rasterio.open(tif) as src:
            img = src.read()
            left, bottom, right, top = src.bounds
        esc = escalar_imagen_multibanda(img.transpose(1, 2, 0), 1, nodata=0)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(esc[:, :, [2, 1, 0]], extent=[left, right, bottom, top], origin="upper")
        ax.ticklabel_format(style="plain", axis="both")
        ax.set_title(titulo)
        guardar_figura(png)

    # ------------------------------------------------------------------
    # ¿Ya están las muestras? Si no, se corta acá (fin de la PASADA 1)
    # ------------------------------------------------------------------
    mue1_vec, mue2_vec = buscar_vector(MUESTRAS1), buscar_vector(MUESTRAS2)
    if mue1_vec is None or mue2_vec is None:
        print("\n" + "-" * 70)
        print("Recortes listos. Ahora, en QGIS:")
        print(f"  1) Abrí {tif1} y {tif2}")
        print(f"  2) Dibujá las muestras de cada recorte y guardalas en {CARPETA} como")
        print(f"     {MUESTRAS1}.gpkg  y  {MUESTRAS2}.gpkg (campos: Clase entero, DESC texto)")
        print("  3) Volvé a correr este script para la clasificación.")
        print("-" * 70)
        return

    # ------------------------------------------------------------------
    # PASO 2. Muestras del recorte 1 (Notebook 03a)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70, "\nPASO 2: Muestras de entrenamiento (recorte 1)\n" + "=" * 70)
    with rasterio.open(tif1) as s2_rec1:
        pols1 = leer_vector_muestras(mue1_vec, s2_rec1.crs)
        muestras1 = extraer_muestras(s2_rec1, pols1, VALOR_RELLENO)

    with rasterio.open(tif2) as s2_rec2:
        pols2 = leer_vector_muestras(mue2_vec, s2_rec2.crs)
        muestras2 = extraer_muestras(s2_rec2, pols2, VALOR_RELLENO)

    if muestras1.empty or muestras2.empty:
        sys.exit("No se extrajeron muestras de alguno de los recortes. "
                 "Revisá que los polígonos de muestras estén dentro del recorte correspondiente.")

    dict_clases = dict_de_clases(muestras1, muestras2)
    print("Clases:", dict_clases)
    asignar_colores(dict_clases)
    solo_en_1 = set(muestras1["Clase"]) - set(muestras2["Clase"])
    solo_en_2 = set(muestras2["Clase"]) - set(muestras1["Clase"])
    if solo_en_1:
        print(f"AVISO: las clases {sorted(solo_en_1)} no tienen muestras en el recorte 2 (normal si no existen allí).")
    if solo_en_2:
        print(f"AVISO: las clases {sorted(solo_en_2)} están en el recorte 2 pero NO en el 1: el modelo no puede predecirlas.")

    muestras1.to_csv(os.path.join(SALIDAS, "muestras_recorte1.csv"), index=False)
    muestras2.to_csv(os.path.join(SALIDAS, "muestras_recorte2.csv"), index=False)
    print("\nMuestras recorte 1 por clase:")
    print(muestras1.groupby(["Clase", "DESC"]).size().to_string())
    print("\nMuestras recorte 2 por clase:")
    print(muestras2.groupby(["Clase", "DESC"]).size().to_string())

    # Firmas con TODAS las muestras de cada recorte (un panel por recorte, mismo eje Y)
    graficar_firmas_espectrales({"Recorte 1 (entrenamiento)": muestras1,
                                 "Recorte 2 (validación)": muestras2},
                                dict_clases, "firmas_espectrales.png")
    # Dispersión entre pares de bandas de las muestras de entrenamiento, balanceada por clase
    graficar_pairplot(muestras1, dict_clases, "Dispersión entre bandas - recorte 1",
                      "pairplot_recorte1.png")

    graficar_recorte_con_poligonos(tif1, pols1, "Recorte 1 con polígonos de muestras",
                                   "poligonos_recorte1.png")
    graficar_recorte_con_poligonos(tif2, pols2, "Recorte 2 con polígonos de muestras",
                                   "poligonos_recorte2.png")

    # ------------------------------------------------------------------
    # PASO 3. Entrenamiento y evaluación (Notebook 03b)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70, "\nPASO 3: Random Forest\n" + "=" * 70)
    X = muestras1[COLUMNAS_BANDAS]
    Y = muestras1["Clase"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, stratify=Y, test_size=0.25, random_state=SEMILLA, shuffle=True)
    print(f"Tamaño de y_train: {y_train.shape} | Tamaño de y_test: {y_test.shape}")

    # 3.1 Modelo base (max_depth=10, 100 árboles)
    rf_base = RandomForestClassifier(max_depth=10, n_estimators=100, random_state=SEMILLA, n_jobs=-1)
    rf_base.fit(X_train, y_train)
    evaluar(rf_base, X_test, y_test, dict_clases, "RF base (max_depth=10) - test recorte 1",
            "confusion_rf_base_test.png")

    # 3.2 Búsqueda de la profundidad óptima con validación cruzada
    print("\nBuscando el max_depth óptimo (2 a 19)...")
    lmd = np.arange(2, 20)
    res_test = np.zeros(len(lmd))
    res_train = np.zeros(len(lmd))
    for idx, md in enumerate(lmd):
        clf = RandomForestClassifier(max_depth=md, random_state=SEMILLA, n_estimators=100, n_jobs=-1)
        res_test[idx] = cross_val_score(clf, X_test, y_test, cv=5).mean()
        res_train[idx] = cross_val_score(clf, X_train, y_train, cv=5).mean()
        print(f"  max_depth = {md:2d} | CV test = {res_test[idx]:.3f} | CV train = {res_train[idx]:.3f}")

    imx = int(np.argmax(res_test))
    best_prof, best_acc = int(lmd[imx]), res_test[imx]
    print(f"\nMejor resultado: profundidad {best_prof}, exactitud {best_acc:.3f}")

    plt.plot(lmd, res_test, label="Test")
    plt.plot(lmd, res_train, label="Train")
    plt.axvline(x=best_prof, label="Mejor profundidad", c="r", linestyle="--")
    plt.title("Exactitud en función de la profundidad")
    plt.ylabel("Exactitud")
    plt.xlabel("Máxima profundidad")
    plt.legend()
    guardar_figura("rf_exactitud_vs_profundidad.png")

    # 3.3 Modelo final con la mejor profundidad
    rf_mejor = RandomForestClassifier(max_depth=best_prof, n_estimators=100,
                                      random_state=SEMILLA, n_jobs=-1)
    rf_mejor.fit(X_train, y_train)
    evaluar(rf_mejor, X_test, y_test, dict_clases,
            f"RF max_depth={best_prof} - test recorte 1", "confusion_rf_mejor_test.png")

    # 3.4 Comparación con KNN (3 vecinos), sin escalar y con RobustScaler
    knn = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)
    knn_esc = make_pipeline(RobustScaler(), KNeighborsClassifier(n_neighbors=3)).fit(X_train, y_train)
    evaluar(knn, X_test, y_test, dict_clases, "KNN sin escalar - test recorte 1",
            "confusion_knn_test.png")
    evaluar(knn_esc, X_test, y_test, dict_clases, "KNN escalado - test recorte 1",
            "confusion_knn_esc_test.png")

    # 3.5 Mapa clasificado del recorte 1
    print("\nClasificando el recorte 1...")
    clasificar_imagen(rf_mejor, tif1, os.path.join(SALIDAS, "clasificacion_recorte1_rf.tif"),
                      dict_clases, f"Mapa clasificado - Recorte 1 - RF (max_depth={best_prof})",
                      "mapa_recorte1_rf.png")

    # ------------------------------------------------------------------
    # PASO 4. Validación externa con el recorte 2 ("Yapa" de la Notebook 03b)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70, "\nPASO 4: Validación con el recorte 2 y muestras independientes\n" + "=" * 70)
    X_val = muestras2[COLUMNAS_BANDAS]
    Y_val = muestras2["Clase"]

    evaluar(rf_base, X_val, Y_val, dict_clases, "RF base (max_depth=10) - VALIDACIÓN recorte 2",
            "confusion_rf_base_validacion.png")
    evaluar(rf_mejor, X_val, Y_val, dict_clases, f"RF max_depth={best_prof} - VALIDACIÓN recorte 2",
            "confusion_rf_mejor_validacion.png")

    evaluar(knn, X_val, Y_val, dict_clases, "KNN sin escalar - VALIDACIÓN recorte 2",
            "confusion_knn_validacion.png")
    evaluar(knn_esc, X_val, Y_val, dict_clases, "KNN escalado - VALIDACIÓN recorte 2",
            "confusion_knn_esc_validacion.png")

    print("\nClasificando el recorte 2...")
    clasificar_imagen(rf_mejor, tif2, os.path.join(SALIDAS, "clasificacion_recorte2_rf.tif"),
                      dict_clases, f"Mapa clasificado - Recorte 2 - RF (max_depth={best_prof})",
                      "mapa_recorte2_rf.png")

    print("\n" + "=" * 70)
    print(f"LISTO. Todos los resultados (CSV, GeoTIFF y PNG) están en: {SALIDAS}")
    print("=" * 70)


if __name__ == "__main__":
    main()
