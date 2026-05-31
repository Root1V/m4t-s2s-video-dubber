"""
orquestar.py — Orquestador de traducción de videos

Uso:
  python orquestar.py               → traduce TODOS los videos de CARPETA_ENTRADA
  python orquestar.py tutorial.mp4  → traduce solo ese video

El script ejecuta en orden: traducir.py → ensamblar_video.py por cada video.
Los resultados se guardan en CARPETA_SALIDA (configurada en cada script).
"""

import sys
import os
import glob
import shutil
import traceback
from datetime import datetime

# Importar los módulos (sin ejecutar main())
import traducir
import ensamblar_video

# ─── Configuración ────────────────────────────────────────────────
# Se toma directamente de traducir.py para no duplicar rutas
CARPETA_ENTRADA    = traducir.CARPETA_ENTRADA
CARPETA_PROCESADOS = os.path.join(os.path.dirname(CARPETA_ENTRADA), "procesados")
EXTENSIONES_VIDEO  = (".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v")
# ──────────────────────────────────────────────────────────────────


def banner(titulo: str, ancho: int = 62):
    print(f"\n{'═' * ancho}")
    print(f"  {titulo}")
    print(f"{'═' * ancho}")


def obtener_videos(nombre_especifico: str | None) -> list[str]:
    """Devuelve lista de nombres de archivo a procesar."""
    if nombre_especifico:
        ruta = (
            os.path.join(CARPETA_ENTRADA, nombre_especifico)
            if not os.path.isabs(nombre_especifico)
            else nombre_especifico
        )
        if not os.path.exists(ruta):
            print(f"❌ No se encontró el archivo: {ruta}")
            sys.exit(1)
        return [os.path.basename(ruta)]

    # Buscar todos los videos en CARPETA_ENTRADA
    encontrados = []
    for ext in EXTENSIONES_VIDEO:
        encontrados.extend(glob.glob(os.path.join(CARPETA_ENTRADA, f"*{ext}")))
        encontrados.extend(glob.glob(os.path.join(CARPETA_ENTRADA, f"*{ext.upper()}")))

    nombres = sorted(set(os.path.basename(v) for v in encontrados))

    if not nombres:
        print(f"❌ No se encontraron videos en: {CARPETA_ENTRADA}")
        print(f"   Extensiones soportadas: {', '.join(EXTENSIONES_VIDEO)}")
        sys.exit(1)

    return nombres


def procesar_video(nombre_video: str) -> str:
    """Traduce y ensambla un video. Devuelve 'ok' o 'error'."""

    # ── Paso 1: Traducir audio ─────────────────────────────────────
    banner(f"[TRADUCIENDO] {nombre_video}")

    # Override del global antes de llamar main() — Python busca
    # variables de módulo en tiempo de ejecución, así que esto funciona.
    traducir.ARCHIVO_ENTRADA = nombre_video
    traducir.main()

    # ── Paso 2: Ensamblar video ────────────────────────────────────
    banner(f"[ENSAMBLANDO] {nombre_video}")

    ensamblar_video.ARCHIVO_VIDEO = nombre_video
    ensamblar_video.ARCHIVO_AUDIO = None   # auto: toma el WAV más reciente de CARPETA_SALIDA
    ensamblar_video.main()


def mover_a_procesados(nombre_video: str):
    """Mueve el video original a CARPETA_PROCESADOS tras traducirlo."""
    os.makedirs(CARPETA_PROCESADOS, exist_ok=True)
    origen  = os.path.join(CARPETA_ENTRADA, nombre_video)
    destino = os.path.join(CARPETA_PROCESADOS, nombre_video)

    # Si ya existe un archivo con el mismo nombre en procesados, añade timestamp
    if os.path.exists(destino):
        base, ext = os.path.splitext(nombre_video)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        destino = os.path.join(CARPETA_PROCESADOS, f"{base}_{ts}{ext}")

    shutil.move(origen, destino)
    print(f"📦 Video movido a procesados: {os.path.basename(destino)}")


def main():
    nombre_especifico = sys.argv[1] if len(sys.argv) > 1 else None
    videos = obtener_videos(nombre_especifico)

    inicio_total = datetime.now()

    banner(f"ORQUESTADOR — {len(videos)} video(s) en cola")
    for i, v in enumerate(videos, 1):
        print(f"  {i}. {v}")

    exitosos = []
    fallidos  = []

    for i, nombre in enumerate(videos, 1):
        print(f"\n{'─' * 62}")
        print(f"  [{i}/{len(videos)}]  {nombre}")
        print(f"{'─' * 62}")
        inicio = datetime.now()

        try:
            procesar_video(nombre)
            mover_a_procesados(nombre)
            duracion = datetime.now() - inicio
            exitosos.append((nombre, str(duracion).split(".")[0]))
            print(f"\n✅  {nombre}  completado en {duracion}".split(".")[0])
        except Exception as e:
            duracion = datetime.now() - inicio
            fallidos.append((nombre, str(e)))
            print(f"\n❌  Error procesando '{nombre}':")
            traceback.print_exc()

    # ── Resumen final ──────────────────────────────────────────────
    duracion_total = datetime.now() - inicio_total
    banner("RESUMEN FINAL")
    print(f"  Tiempo total: {str(duracion_total).split('.')[0]}")
    print(f"\n  ✅ Exitosos ({len(exitosos)}):")
    for nombre, t in exitosos:
        print(f"     • {nombre}  ({t})")
    if fallidos:
        print(f"\n  ❌ Fallidos ({len(fallidos)}):")
        for nombre, err in fallidos:
            print(f"     • {nombre}: {err}")
    print()


if __name__ == "__main__":
    main()
