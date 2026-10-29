# AGRO MIRAI — Project Deep Dive

A complete, pin-to-pin explanation of the system: what it does, how it is built, where
every dataset came from, how every model was trained, and how a request actually flows
through the app end to end. Written for anyone picking up the project cold.

---

## 1. What the project is

AGRO MIRAI is a mobile app for farmers that gives three kinds of advice from a phone:

1. **What crop to grow** — given soil (N/P/K, pH), temperature, humidity and rainfall.
2. **How much to irrigate, and when** — given soil moisture context, weather and crop
   stage.
3. **Whether a crop leaf is diseased, and what it is** — from a photo taken on the phone.

On top of that it layers field mapping (GPS pin + acreage-based field boundary),
multilingual voice interaction (English / Hindi / Kannada / Telugu), offline-first local
storage with sync, and push notifications — built for low-end Android hardware and
patchy rural connectivity.

---

## 2. Tech stack

| Layer | Technology |
|---|---|
| Backend API | Python 3.12, Flask, Pydantic-style request/response models |
| Backend DB | SQLite (local/dev), Supabase (managed Postgres, production) |
| Mobile app | React Native 0.86, Expo SDK 57, TypeScript (strict mode) |
| Maps | MapLibre GL (`@maplibre/maplibre-react-native`) — vector tiles, no Google Maps dependency |
| Local storage | `expo-secure-store` for tokens, AsyncStorage (SQLite-backed) for app state/offline cache |
| Camera / media | `expo-camera`, `expo-image-picker`, `expo-image-manipulator` |
| Push notifications | `expo-notifications` → real Android OS notification channel (no in-app notification center) |
| Crop recommendation model | scikit-learn RandomForestClassifier |
| Irrigation urgency model | scikit-learn RandomForestClassifier |
| Leaf disease detection | Google Gemini (vision) — two-stage gate + analyze pipeline, described in §6 |
| Voice (STT/TTS/translation) | Sarvam AI (`saaras` STT, `bulbul` TTS, `sarvam-translate`), with an in-house AI4Bharat stack as the heavier, self-hosted alternative tier |
| Hosting | Render (API), Supabase (DB/auth), Vercel (APK download page / landing) |
| CI | GitHub Actions — includes a scheduled keep-alive workflow so the free-tier backend never cold-starts on a farmer |

---

## 3. System architecture

```
                         ┌───────────────────────────┐
                         │   Mobile App (Expo/RN)    │
                         │  ScanTab / MapTab / MeTab  │
                         └─────────────┬──────────────┘
                                       │ REST (JSON)
                                       ▼
                         ┌───────────────────────────┐
                         │     Flask API (/v1,/v2)    │
                         │  routing, auth, orchestration
                         └──────┬───────────────┬──────┘
                                │               │
                    domain      │               │ domain
                    records     ▼               ▼ records
                     ┌────────────────┐  ┌─────────────────────┐
                     │  DataStore      │  │  Data-acquisition    │
                     │  repository     │  │  adapters             │
                     │  interface      │  │  (weather/soil/NDVI)  │
                     └───┬────────┬────┘  └───┬──────┬───────┬────┘
                         │        │            │      │       │
                    ┌────▼──┐ ┌───▼─────┐ ┌────▼─┐ ┌──▼───┐ ┌─▼──────────┐
                    │SQLite │ │Supabase │ │Open- │ │Soil- │ │Earth Engine │
                    │(dev)  │ │Postgres │ │Meteo │ │Grids │ │(Sentinel-2) │
                    └───────┘ └─────────┘ └──────┘ └──────┘ └─────────────┘
```

**Why it's shaped this way:** every arrow that crosses a module boundary carries only a
plain data record (a `WeatherReading`, a `SoilSample`, a `FieldInput` — never a raw HTTP
response or a vendor-specific object). That means the three external data adapters
(weather, soil, satellite NDVI) and the database layer (`DataStore`) are the *only* two
places in the codebase that know which vendor or engine sits behind them. Swapping
Supabase for something else, or Open-Meteo for a different weather API, never touches
the recommendation/advisory logic that consumes the data.

### 3.1 Account isolation

Every single database query is scoped to the account that owns it (`account_id`). There
is no cross-account read path anywhere in the API — this is treated as a hard,
non-negotiable rule throughout the codebase, because the app holds real farmer location
and field data for multiple independent users.

### 3.2 Why 3 services became 1

Early on, the backend was deliberately split into three separate Render services (main
API, a CNN-inference microservice, a tabular-ML microservice) specifically to fit inside
Render's free-tier 512MB-per-service RAM ceiling. Once the tabular models turned out to
be cheap enough to run in-process, the separate tabular-ML service was folded back into
the main API — the recommendation logic for crop and irrigation now runs as part of the
main Flask service directly, simplifying the deployment back down.

### 3.3 Keeping the free tier awake

Render's free tier spins a service down after 15 minutes of no traffic, which otherwise
means a 50–60 second cold-start delay the next time a farmer opens the app. A scheduled
GitHub Actions workflow pings the backend's `/health` endpoint on a fixed interval,
every day, forever, at zero cost — so the service is never actually asleep when a real
user hits it.

---

## 4. Data acquisition layer

Three adapters sit behind one shared interface (`fetch(field) -> list[dict]`), each
responsible for exactly one external source:

| Adapter | Source | What it returns | API key needed |
|---|---|---|---|
| Weather | Open-Meteo (current + forecast + historical) | temperature, humidity, rainfall | No |
| Soil | ISRIC SoilGrids v2 (global soil-property grid) | N/P/K-adjacent properties, pH, texture | No |
| NDVI (vegetation index) | Google Earth Engine, Sentinel-2 satellite imagery | NDVI = (NIR−Red)/(NIR+Red), computed from the most recent low-cloud scene over the field | Yes (GEE service account) |

**Soil depth choice:** SoilGrids predicts several depth bands. The system always uses the
topsoil band (0–5cm) — the layer closest to where a farmer's own soil test would
actually sample, and where fertiliser application and pH most directly matter.

**NDVI fallback design (a deliberate, documented decision, not an accident):** a live
Earth Engine query is attempted first, bounded by a timeout. If it fails or times out,
the system falls back to a local NDVI cache seeded ahead of time from real GEE pulls
(never fabricated numbers) and keyed by field location. If even the cache misses, the
system returns an honest "unavailable" error rather than ever inventing a plausible-looking
NDVI value. Every `NDVIReading` carries a `source` field (`gee_live` vs `cache`) so any
caller can tell exactly which path served it, without needing to dig through logs.

---

## 5. The two trained machine-learning models

### 5.1 Crop recommendation model

- **Algorithm:** Random Forest classifier (scikit-learn).
- **Dataset:** Kaggle "Crop Recommendation Dataset" (`atharvaingle/crop-recommendation-dataset`,
  Apache-2.0 licensed) — 2,200 rows, 22 balanced crop labels, 7 input features:
  nitrogen (N), phosphorus (P), potassium (K), temperature, humidity, soil pH, rainfall.
- **Training:** fixed random seed (42) for full reproducibility; trained via a standalone
  script (`tools/train_crop_model.py`) that is re-runnable from scratch at any time.
- **Result:** ~99.5% held-out accuracy. Explainability is provided through SHAP values,
  so a prediction can be traced back to which input feature drove it.
- **Why Random Forest and not a neural net:** with only 7 clean, already-numeric features
  and 2,200 rows, a tree ensemble reaches near-ceiling accuracy, trains in seconds, needs
  no GPU, and — critically for farmer-facing advice — is directly explainable feature-by-feature,
  which a deep model would not give for free.

### 5.2 Irrigation urgency model

- **Algorithm:** Random Forest classifier.
- **Dataset:** Kaggle "Irrigation Water Requirement Prediction Dataset"
  (`miadul/irrigation-water-requirement-prediction-dataset`, MIT licensed) — 10,000 rows,
  zero missing values. Chosen after an explicit search across the available irrigation
  datasets on Kaggle because it was the cleanest, most-used dataset with a usable
  classification framing, and its `Season` column already uses the Kharif / Rabi / Zaid
  vocabulary the rest of the system's season handling is built around — a genuine fit,
  not a coincidence that was forced.
- **Target:** the dataset's label is a 3-class irrigation urgency (`Low` / `Medium` /
  `High`; 5,864 / 3,800 / 336 rows respectively) — not a continuous "how many millimetres
  of water" number.
- **How the actual depth-in-mm recommendation is produced:** rather than inventing a fake
  regression target to make the model output a number it was never actually trained to
  predict, the model's only job is to classify urgency. The literal depth-in-mm figure
  shown to the farmer is a separate, transparent rule-based lookup (an FAO-56-style water
  balance calculation) keyed off that predicted urgency tier. This keeps the "what did
  the model actually learn" question honest — it was trained to answer "how urgent," and
  that's exactly what it is used for.
- **Result:** 72.4% accuracy, 0.58 macro-F1 on held-out data. The macro-F1 gap versus
  accuracy reflects the `High` class's small size in the source data (336 of 10,000 rows)
  even with class-balanced training weights — a known, documented property of the
  dataset, not a training bug.

Both models' trained binary files are intentionally **not committed to the repository** —
only the training scripts and the evaluation reports (accuracy, macro-F1, confusion
matrix, as JSON) are committed. This means the model files are always exactly
reproducible from the same Kaggle data and the same training script, and the repository
stays diffable and reviewable instead of carrying opaque binaries.

---

## 6. Leaf disease detection pipeline

This is the most iterated-on part of the system, and it is worth explaining the final
design plus the reasoning behind it.

**What a farmer does:** opens the Scan tab, photographs a crop leaf (or picks a photo
from the gallery), and gets back: is this diseased, what's the likely disease, how
confident is the system, and what should they do about it.

**How it actually works, end to end:**

1. **Cheap local pre-check.** Before any network call, a fast pixel-level check on the
   device-side image rules out obviously-wrong input (near-solid-color images, etc.)
   without spending an API call on something that was never going to be a leaf photo.
2. **AI gate ("is this actually a plant leaf?").** The image is sent to a vision model
   (Gemini) with a strict prompt whose only job is to answer yes/no on whether the photo
   genuinely shows a plant leaf — rejecting blurry photos, random objects, people, screens,
   or a plain background that happens to be green. This gate exists because an earlier
   version of the pipeline was too permissive and would occasionally return a confident
   diagnosis for a photo of literally anything (a bedsheet, a laptop) — the gate is the
   fix for that, and it is deliberately strict.
3. **Disease identification/analysis.** Once a photo passes the gate, a second,
   purpose-built vision prompt identifies the plant, decides whether it shows disease
   symptoms, names the probable disease if so, gives a confidence level, a short
   description, and a recommended action.
4. **Species/crop cross-check (Kindwise).** As an additional identification layer, the
   system also calls Kindwise's plant-identification and crop-health APIs — a specialist
   plant-ID service separate from the general-purpose vision model — to corroborate the
   species guess and add a second opinion on plant-health status.
5. **Honest failure, not a fake answer.** If the identification step cannot produce a
   real result (model unavailable, API outage, etc.), the API returns a clear
   "analysis unavailable" response rather than ever returning a cached or guessed
   diagnosis dressed up as a real one. This was a deliberate, explicit design decision:
   a wrong "no disease" is far worse for a farmer than an honest "try again."

**A CNN model (MobileNetV2, trained on the public PlantVillage dataset, 38 disease
classes, ~99% accuracy on its own held-out test set) exists in the codebase as an
earlier, self-contained disease-classification path.** It remains available as an
alternate/offline-capable classifier, but the current live Scan-tab flow described above
is the vision-model pipeline — chosen for production because it generalizes to leaf
photos and conditions well outside PlantVillage's own curated, clean-background training
images, which is closer to what a real farmer's phone camera actually captures in a real
field.

---

## 7. Voice & multilingual support

A single internal voice interface sits in front of whichever backend actually does the
work, so the rest of the app never talks to a vendor SDK directly. In production this is
backed by **Sarvam AI** (`saaras` for speech-to-text, `bulbul` for text-to-speech,
`sarvam-translate` for translation) via REST, rotating across multiple API keys for
resilience. If the cloud tier is ever exhausted or unreachable, the system degrades to a
clear, localized "voice unavailable, here's the text" message in the farmer's own
language — never a silent failure or a wrong-language response.

A heavier, fully self-hosted alternative stack was also built end-to-end on **AI4Bharat's**
open models — IndicTrans2 (distilled 200M, for translation), IndicConformer (600M
multilingual ASR), Whisper-tiny.en (for English-specific ASR, since AI4Bharat's own ASR
line is Indic-languages-only), and Piper / `vits_rasa_13` for text-to-speech. This stack
runs entirely offline/self-hosted and was validated to fit a low-spec (4GB RAM) target
machine, with the one caveat that the ASR model takes roughly two minutes to load into
memory the first time a process starts (after which inference is a few seconds per
utterance) — intentionally loaded once at startup rather than per request, so the first
real request of a session is never mistaken for the system being broken.

---

## 8. Mapping & field boundaries

The map tab shows an exact GPS pin for a farmer's field (not just an approximate area),
styled like a familiar consumer maps app. During registration, a farmer sets their
field's coordinates and enters how many acres it covers; the system then draws a
circular boundary on the map centered on that pin, sized so its area matches the
farmer-entered acreage — giving a visual sense of the field's extent without requiring
the farmer to trace an actual polygon by hand.

---

## 9. Offline-first & reliability choices

- **Local-first storage:** app state and cached data live in on-device storage so the
  app remains usable with a flaky connection; writes sync to the backend when
  connectivity is available.
- **Graceful degradation everywhere:** any external dependency (weather, soil, NDVI,
  voice, disease analysis) that becomes unavailable returns an explicit "unavailable"
  signal rather than silently substituting a guess. This is a deliberate project-wide
  principle: it is always better for the app to say "I don't know right now" than to
  present a fabricated answer as if it were real.
- **Real OS notifications only:** the app relies entirely on genuine Android system
  notifications (via a proper notification channel) for alerts, rather than maintaining
  a separate in-app notification inbox — one less piece of state that can drift out of
  sync with reality.

---

## 10. Summary of what's learned vs. what's rules

It's worth being explicit about this, because it's easy to blur the line:

| Feature | Learned (ML/AI) | Rule-based |
|---|---|---|
| Crop recommendation | ✅ Random Forest | — |
| Irrigation urgency (Low/Med/High) | ✅ Random Forest | — |
| Irrigation depth-in-mm | — | ✅ FAO-56-style water-balance lookup, keyed off the learned urgency |
| Leaf disease identification | ✅ Vision model (Gemini) + Kindwise cross-check | Pixel/colour pre-check gate before the AI call |
| NDVI source selection | — | ✅ Live-fetch-with-cache-fallback policy |
| Voice tier selection | — | ✅ Cloud-first-with-degrade policy |

Nothing in the system fabricates a number or a diagnosis to fill a gap where a model or
API genuinely has no answer — every "I don't know" path returns an honest unavailable
state instead.
