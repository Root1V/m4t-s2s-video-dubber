# SPEC — Especificaciones y Roadmap de Funcionalidades

Este documento describe el estado actual del proyecto y las funcionalidades planificadas,
ordenadas por prioridad de impacto. Cada ítem se implementa en su propia rama de `develop`
y se mergea con tag de versión al completarse.

---

## Estado actual — v1.1.0

### Pipeline implementado

```
Video MP4 → VAD → SeamlessM4T v2 S2S → Phase Vocoder → WAV + MP4
```

| Componente | Archivo | Estado |
|---|---|---|
| Configuración centralizada | `m4t_dubber/config.py` | ✅ |
| Carga de audio (torchaudio + torchcodec) | `m4t_dubber/audio/translator.py` | ✅ |
| VAD por energía RMS | `m4t_dubber/audio/translator.py` | ✅ |
| Sub-chunking ≤ 15s (límite MPS) | `m4t_dubber/audio/translator.py` | ✅ |
| Traducción S2S (SeamlessM4Tv2) | `m4t_dubber/audio/translator.py` | ✅ |
| Stretch por phase vocoder | `m4t_dubber/audio/translator.py` | ✅ |
| Ensamblado video + audio (MoviePy) | `m4t_dubber/audio/assembler.py` | ✅ |
| Orquestador / procesamiento en lote | `m4t_dubber/pipeline.py` | ✅ |
| CLI con argparse | `main.py` | ✅ |
| Gestión de paquetes (UV) | `pyproject.toml` + `uv.lock` | ✅ |
| Herramientas de diagnóstico | `tools/` | ✅ |

### Limitaciones conocidas

- La voz traducida se mezcla con música/efectos de fondo del original
- Sin soporte para elegir idioma fuente/destino desde CLI
- Sin subtítulos generados
- Sin reanudación si el proceso se interrumpe a mitad
- Voz sintética fija (speaker_id=4), no clona la voz del hablante original
- Sin interfaz web ni API

---

## Funcionalidades pendientes

### F-01 — Separación de voz de fondo (stem separation)
**Prioridad:** Alta  
**Impacto:** Muy alto — elimina el problema de la voz mezclada con música  
**Rama:** `feat/stem-separation`

Integrar [Demucs](https://github.com/facebookresearch/demucs) (htdemucs model) para separar:
- `vocals` — voz del hablante (se traduce)
- `no_vocals` — música, efectos, ambiente (se preserva intacta)

Pipeline resultante:
```
Video MP4
  └─ Demucs → vocals.wav + no_vocals.wav
       └─ vocals.wav → [pipeline actual] → vocals_esp.wav
            └─ mix(vocals_esp.wav, no_vocals.wav) → audio_final.wav
                 └─ MoviePy → MP4
```

**Archivos a crear/modificar:**
- `m4t_dubber/audio/separator.py` — clase `StemSeparator`
- `m4t_dubber/pipeline.py` — integrar paso de separación
- `pyproject.toml` — agregar `demucs`

---

### F-02 — Soporte multi-idioma desde CLI
**Prioridad:** Alta  
**Impacto:** Alto — el modelo ya soporta 100+ idiomas, solo falta exponerlo  
**Rama:** `feat/multilang-cli`

SeamlessM4T v2 soporta traducción entre docenas de idiomas. Actualmente solo inglés→español.

Cambios:
```bash
# Nuevos flags en main.py
uv run python main.py video.mp4 --src-lang eng --tgt-lang fra
uv run python main.py video.mp4 --tgt-lang por  # portugués
```

**Archivos a modificar:**
- `main.py` — agregar `--src-lang` (default: `eng`) y `--tgt-lang` (default: `spa`)
- `m4t_dubber/config.py` — `SRC_LANG` como nueva variable de entorno `M4T_SRC_LANG`
- `m4t_dubber/audio/translator.py` — pasar `src_lang` al procesador

Idiomas SeamlessM4T soportados: `eng, spa, fra, deu, por, ita, jpn, cmn, ara, rus, ...`

---

### F-03 — Generación de subtítulos (SRT/VTT)
**Prioridad:** Media  
**Impacto:** Alto — el modelo genera texto internamente, es un subproducto gratuito  
**Rama:** `feat/subtitles`

El `SeamlessM4Tv2Processor` puede devolver tokens de texto además del audio.
Capturar esos tokens permite generar subtítulos sincronizados con los timestamps del VAD.

Salidas adicionales:
- `{stem}_esp_{ts}.srt` — subtítulos en español
- `{stem}_esp_{ts}_bilingual.srt` — bilingüe (inglés + español) opcional

**Archivos a crear/modificar:**
- `m4t_dubber/audio/subtitler.py` — clase `SubtitleGenerator`
- `m4t_dubber/audio/translator.py` — capturar tokens de texto del modelo
- `m4t_dubber/pipeline.py` — incluir generación de SRT en el flujo

---

### F-04 — Reanudación de trabajos (checkpoint por segmento)
**Prioridad:** Media  
**Impacto:** Alto para videos largos — evita reprocesar desde cero si hay un corte  
**Rama:** `feat/resume-checkpoint`

Para videos de 1+ hora, un fallo a mitad significa perder todo el trabajo. Implementar:
- Cache de segmentos traducidos en disco (archivos `.npy` o `.pkl` por segmento)
- Al reiniciar, cargar los segmentos ya procesados y continuar desde el último pendiente
- Limpiar el cache al finalizar correctamente

**Archivos a crear/modificar:**
- `m4t_dubber/audio/checkpoint.py` — lógica de cache por segmento
- `m4t_dubber/audio/translator.py` — integrar checkpoint en `_translate_segments()`
- `m4t_dubber/config.py` — `CHECKPOINT_DIR` (default: `.cache/`)

---

### F-05 — API REST (FastAPI)
**Prioridad:** Media  
**Impacto:** Convierte la herramienta local en un servicio deployable  
**Rama:** `feat/rest-api`

Endpoints mínimos:
```
POST /translate          — sube video, devuelve job_id
GET  /jobs/{job_id}      — status del trabajo (pending/running/done/error)
GET  /jobs/{job_id}/download  — descarga el MP4 traducido
DELETE /jobs/{job_id}    — cancela y limpia
```

**Archivos a crear:**
- `api/main.py` — app FastAPI
- `api/routes/jobs.py` — endpoints
- `api/models.py` — schemas Pydantic
- `pyproject.toml` — agregar `fastapi`, `uvicorn`, `python-multipart`

---

### F-06 — Clonación de voz del hablante
**Prioridad:** Baja  
**Impacto:** Muy alto en calidad perceptual — la voz traducida sonaría como el original  
**Rama:** `feat/voice-cloning`  
**Estado:** ✅ Implementado en v1.5.0

Usa **F5-TTS v1 Base** (zero-shot, ~1.5 GB descarga única) para sintetizar la
traducción con la voz del hablante original extraída de los primeros 15 s de voz
real del video (saltando intros musicales con `_first_speech_s`).

Pipeline:
```
Video MP4
  ├─ SeamlessM4T (modo texto) → subtítulos traducidos
  ├─ extract_reference(start=primer_segmento_habla) → reference.wav (15 s)
  └─ F5-TTS.infer(ref=reference.wav, text=subtítulos_filtrados) → voz clonada
       └─ MoviePy → MP4 final
```

Flag CLI: `--clone-voice` (combina con `--stem` para preservar música de fondo)

**Limitación conocida:** Se usa una sola referencia de voz para todos los segmentos.
Si el video tiene múltiples hablantes, todos sonarán igual (ver F-08).

---

### F-08 — Diarización multi-hablante
**Prioridad:** Media  
**Impacto:** Alto — videos con más de un hablante clonados correctamente por voz  
**Rama:** `feat/speaker-diarization`  
**Depende de:** F-06 (voice cloning)

Actualmente `--clone-voice` extrae una sola referencia de voz y la aplica a todos
los segmentos. Si el video tiene múltiples hablantes, todos suenan igual.

Esta feature añade diarización automática con [pyannote.audio](https://github.com/pyannote/pyannote-audio)
para detectar qué tramos de audio pertenecen a cada hablante, extraer una
referencia individual por hablante y asignar la referencia correcta a cada
segmento de síntesis.

Pipeline:
```
Video MP4
  ├─ pyannote.audio → diarización {speaker_id: [(t_start, t_end), ...]}
  ├─ extract_reference × N_speakers → {speaker_id: reference.wav}
  └─ F5-TTS.infer(ref=reference[speaker_id], text=segmento) × segmentos
       └─ MoviePy → MP4 final
```

**Archivos a crear/modificar:**
- `m4t_dubber/audio/diarizer.py` — clase `SpeakerDiarizer` (pyannote.audio)
- `m4t_dubber/audio/voice_cloner.py` — `synthesize_from_segments` acepta
  `speaker_refs: dict[str, Path]` para referencia por hablante
- `m4t_dubber/pipeline.py` — integrar diarización antes de clonación
- `pyproject.toml` — agregar `pyannote.audio`

**Consideraciones:**
- pyannote.audio requiere token de HuggingFace (aceptar términos del modelo)
- Primer uso descarga ~1 GB de modelos adicionales
- En videos con un solo hablante, se comporta igual que F-06 sin overhead

---

### F-07 — Interfaz web (Gradio)
**Prioridad:** Baja  
**Impacto:** Accesibilidad para usuarios no técnicos  
**Rama:** `feat/gradio-ui`

UI mínima con Gradio:
- Subida de video (drag & drop)
- Selector de idioma destino
- Barra de progreso por segmento
- Preview y descarga del resultado

**Archivos a crear:**
- `app.py` — interfaz Gradio
- `pyproject.toml` — agregar `gradio`

---

## Orden de implementación sugerido

| # | Feature | Versión objetivo | Rama | Estado |
|---|---|---|---|---|
| 1 | F-02 — Multi-idioma CLI | v1.2.0 | `feat/multilang-cli` | ✅ |
| 2 | F-03 — Subtítulos SRT | v1.3.0 | `feat/subtitles` | ✅ |
| 3 | F-01 — Separación de voz | v1.4.0 | `feat/stem-separation` | ✅ |
| 4 | F-06 — Clonación de voz | v1.5.0 | `feat/voice-cloning` | ✅ |
| 5 | F-08 — Diarización multi-hablante | v1.6.0 | `feat/speaker-diarization` | ⬜ |
| 6 | F-04 — Checkpoint/reanudación | v1.7.0 | `feat/resume-checkpoint` | ⬜ |
| 7 | F-05 — API REST | v2.0.0 | `feat/rest-api` | ⬜ |
| 8 | F-07 — UI Gradio | v2.1.0 | `feat/gradio-ui` | ⬜ |

---

## Convención de ramas y versiones

```
main          ← solo releases estables (merges desde develop con tag)
develop       ← integración de features
feat/<name>   ← una rama por feature de este SPEC
fix/<name>    ← hotfixes
```

Versioning: `vMAJOR.MINOR.PATCH`
- MAJOR: cambio de arquitectura (e.g. añadir API, cambiar modelo base)
- MINOR: nueva funcionalidad (cada feature de este SPEC)
- PATCH: bugfix o dependencias
