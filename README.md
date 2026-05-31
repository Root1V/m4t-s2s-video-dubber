# m4t-s2s-video-dubber

Dobla videos a cualquier idioma usando [SeamlessM4T v2 Large](https://huggingface.co/facebook/seamless-m4t-v2-large) de Meta — traduce habla en un solo paso (Speech-to-Speech), sin pasar por texto. Por defecto traduce al **español**. Soporte nativo para Apple Silicon (MPS).

## Características actuales

- **Speech-to-Speech (S2S)** con SeamlessM4T v2: traduce habla en un solo paso, sin texto intermedio
- **Multi-idioma**: cualquier combinación de idiomas soportados por SeamlessM4T (100+) vía `--src-lang` / `--tgt-lang`
- **Stem separation** (Demucs htdemucs): separa voz del fondo musical antes de traducir — preserva música y efectos intactos
- **VAD** (Voice Activity Detection) por energía RMS: preserva los silencios originales del video
- **Phase vocoder** (stretch): ajusta cada segmento traducido a la duración original sin artefactos graves
- **Procesamiento en lote**: procesa todos los videos de una carpeta automáticamente
- **Configuración por variables de entorno** (`M4T_*`): sin hardcodeo de rutas ni parámetros
- **CLI** con argparse: procesa un video específico o toda la cola con flags de idioma
- **Gestión de paquetes con UV**: entorno reproducible vía `pyproject.toml` + `uv.lock`

## Hardware probado

| Hardware | Backend | Velocidad (1:53 video) |
|---|---|---|
| Apple M4 MAX, 128 GB | MPS | ~32 segundos |

## Requisitos

- Python 3.10+
- [UV](https://docs.astral.sh/uv/) (`brew install uv`)
- macOS con Apple Silicon (MPS) — también funciona en CPU/CUDA con ajustes mínimos

## Instalación

```bash
git clone https://github.com/Root1V/m4t-s2s-video-dubber.git
cd m4t-s2s-video-dubber

# Instalar todas las dependencias desde el lockfile
uv sync

# (Opcional) copiar y ajustar configuración
cp .env.example .env
```

## Uso

```bash
# Traducir todos los videos en videos/ → español (default)
uv run python main.py

# Traducir un video específico → español
uv run python main.py "mi_video.mp4"

# Especificar idioma destino
uv run python main.py "mi_video.mp4" --tgt-lang fra    # → francés
uv run python main.py "mi_video.mp4" --tgt-lang por    # → portugués
uv run python main.py "mi_video.mp4" --tgt-lang deu    # → alemán

# Especificar ambos idiomas
uv run python main.py "mi_video.mp4" --src-lang spa --tgt-lang eng   # español → inglés
uv run python main.py "mi_video.mp4" --src-lang fra --tgt-lang jpn   # francés → japonés

# Generar subtítulos .srt además del video (opt-in)
uv run python main.py "mi_video.mp4" --srt
uv run python main.py "mi_video.mp4" --tgt-lang fra --srt

# Separar voz de fondo antes de traducir (Demucs htdemucs)
uv run python main.py "mi_video.mp4" --stem
uv run python main.py "mi_video.mp4" --stem --srt   # combinar con subtítulos
```

> **Stem separation**: `--stem` usa Demucs (htdemucs) para aislar la voz de la música y efectos de fondo. La música se preserva intacta en el video final. Descarga ~80 MB de modelo en el primer uso.

El archivo de salida incluye el idioma destino en el nombre: `mi_video_spa_20260531_120000.mp4`

Los videos se buscan en `../videos/` (relativo al repo), los resultados se guardan en `../resultados/` y los procesados se mueven a `../procesados/`.

### Idiomas soportados

| Código | Idioma | Código | Idioma |
|---|---|---|---|
| `eng` | Inglés | `spa` | **Español** (default) |
| `fra` | Francés | `por` | Portugués |
| `deu` | Alemán | `ita` | Italiano |
| `jpn` | Japonés | `cmn` | Chino mandarín |
| `ara` | Árabe | `rus` | Ruso |
| `hin` | Hindi | `kor` | Coreano |

Lista completa en [SeamlessM4T supported languages](https://huggingface.co/facebook/seamless-m4t-v2-large#supported-languages).

### Carpetas de trabajo

```
video_translate/
├── videos/          ← coloca aquí los videos de entrada (.mp4, .mkv, etc.)
├── resultados/      ← WAV + MP4 traducidos se generan aquí
├── procesados/      ← videos fuente se mueven aquí tras procesarse
└── traductor_m4/    ← este repositorio
```

Todas las rutas son sobreescribibles con variables de entorno (ver `.env.example`).

## Estructura del proyecto

```
traductor_m4/
├── main.py                    # CLI entrypoint
├── pyproject.toml             # Dependencias y metadatos (UV)
├── uv.lock                    # Lockfile reproducible
├── .env.example               # Plantilla de variables de entorno
├── m4t_dubber/
│   ├── __init__.py
│   ├── config.py              # Configuración centralizada
│   ├── pipeline.py            # DubbingPipeline — orquestador principal
│   └── audio/
│       ├── __init__.py
│       ├── translator.py      # AudioTranslator — carga modelo, segmenta, traduce
│       ├── assembler.py       # VideoAssembler — mezcla audio traducido con video
│       ├── separator.py       # StemSeparator — separa voz de fondo con Demucs
│       └── subtitler.py       # write_srt — genera archivo .srt
└── tools/
    ├── probar_voces.py        # Prueba speaker IDs 0-199 para elegir voz
    ├── extraer_audio.py       # Extrae audio de un video a WAV
    └── debug_audio.py         # Diagnóstico del pipeline de traducción
```

## Configuración

Copia `.env.example` como `.env` y ajusta según necesites:

| Variable | Default | Descripción |
|---|---|---|
| `M4T_INPUT_DIR` | `../videos` | Carpeta de videos de entrada |
| `M4T_OUTPUT_DIR` | `../resultados` | Carpeta de salida |
| `M4T_PROCESSED_DIR` | `../procesados` | Carpeta de videos procesados |
| `M4T_MODEL_ID` | `facebook/seamless-m4t-v2-large` | Modelo HuggingFace |
| `M4T_SRC_LANG` | `eng` | Idioma fuente (el modelo auto-detecta si se omite) |
| `M4T_TGT_LANG` | `spa` | **Idioma destino (español por defecto)** |
| `M4T_SPEAKER_ID` | `4` | Voz del locutor (0-199) |
| `M4T_MAX_CHUNK_S` | `15` | Máximo segundos por chunk (límite MPS) |
| `M4T_NUM_BEAMS` | `2` | Beams para generación |
| `M4T_REPETITION_PENALTY` | `1.3` | Penalización de repetición |

### Flags CLI

| Flag | Descripción |
|---|---|
| `VIDEO` | Nombre del video en `INPUT_DIR` (opcional; sin él procesa todos) |
| `--src-lang LANG` | Idioma fuente (código SeamlessM4T, ej. `eng`). Override de `M4T_SRC_LANG` |
| `--tgt-lang LANG` | Idioma destino (código SeamlessM4T, ej. `spa`). Override de `M4T_TGT_LANG` |
| `--srt` | Genera un `.srt` de subtítulos sincronizado (desactivado por defecto) |
| `--stem` | Separa voz de fondo con Demucs antes de traducir; preserva música/efectos (desactivado por defecto) |

> Los flags `--src-lang` y `--tgt-lang` de la CLI tienen prioridad sobre las variables de entorno.

## Herramientas de diagnóstico

```bash
# Probar voces 0-9 con los primeros 10s de un video
uv run python tools/probar_voces.py mi_video.mp4 --ids 0-9 --segundos 10

# Extraer audio de un video
uv run python tools/extraer_audio.py mi_video.mp4 -o audio.wav

# Debug completo del pipeline de traducción
uv run python tools/debug_audio.py mi_video.mp4
```

## Pipeline interno

```
Video MP4
  └─ torchaudio.load()          → tensor [C, T] @ 44100Hz
       └─ resample → 16kHz mono
            └─ VAD (RMS energy) → segmentos de voz + silencios
                 └─ por cada segmento de voz:
                      ├─ sub-chunks ≤ 15s (límite MPS)
                      ├─ SeamlessM4Tv2ForSpeechToSpeech.generate()
                      └─ phase vocoder stretch → duración original
                           └─ concatenar + normalizar → WAV
                                └─ MoviePy → MP4 final

# Con --stem:
Video MP4
  └─ Demucs htdemucs → vocals.wav + no_vocals.wav (44.1 kHz stereo)
       └─ vocals.wav → [pipeline anterior] → vocals_traducido.wav
            └─ mix(vocals_traducido, no_vocals) → audio_final.wav
                 └─ MoviePy → MP4 final
```

## Dependencias clave

| Paquete | Versión | Uso |
|---|---|---|
| `torch` | 2.12.0 | Backend MPS / CUDA / CPU |
| `torchaudio` | 2.11.0 | Carga y resampleo de audio |
| `torchcodec` | 0.13.0 | Backend de decodificación de video/audio |
| `transformers` | 5.9.0 | Modelo SeamlessM4Tv2 + AutoProcessor |
| `sentencepiece` | 0.2.1 | Tokenizer del modelo |
| `protobuf` | 7.35.0 | Requerido por sentencepiece |
| `demucs` | 4.0.1 | Separación de stems (vocals / no_vocals) |
| `moviepy` | 2.2.1 | Ensamblado de video final |
| `numpy` | 2.2.6 | DSP (VAD, stretch, normalización) |

## Desarrollo

```bash
# Activar entorno
source .venv/bin/activate

# O usar uv directamente sin activar
uv run python main.py

# Agregar dependencia
uv add <paquete>

# Linter
uv run ruff check .
```

## Versiones

| Tag | Descripción |
|---|---|
| `v1.0.0` | Pipeline funcional inicial (scripts planos) |
| `v1.1.0` | Reestructuración enterprise + UV + dependencias corregidas |
| `v1.2.0` | Soporte multi-idioma CLI (`--src-lang` / `--tgt-lang`) |
| `v1.3.0` | Subtítulos SRT opt-in (`--srt`) |
| `v1.4.0` | Separación de voz de fondo con Demucs (`--stem`) |
