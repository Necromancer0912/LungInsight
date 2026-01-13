# Lung Insight Diagnostic Console

Modernized desktop console that fuses lung audio and chest X-ray models with LLM-driven clinical reasoning. Goal: rapid intake, focused follow-up, and structured clinician-ready reports—local-first, Electron-wrapped for Windows.

---

## Table of Contents

1. Motivation & Goals
2. High-Level Overview
3. Architecture Layers
4. Directory Structure
5. Detailed Architecture Narrative
6. Frontend Deep Dive
7. Backend Deep Dive
8. LLM Usage (Model, Prompts, Samples)
9. Severity Logic
10. Data Flow & State Flow
11. API Reference
12. Models Snapshot
13. Dataset Policy & Data Hygiene
14. Security & Privacy
15. Performance Notes
16. UX & Styling Notes
17. Setup & Run (Quickstart + Detailed)
18. Environment Variables
19. Testing Strategy
20. Troubleshooting (Expanded)
21. FAQ
22. Glossary
23. Contribution Guidelines
24. Release & Packaging
25. Logging & Error Handling
26. Known Limitations & Roadmap Ideas
27. Change Log (Recent Highlights)
28. License

---

## 1) Motivation & Goals

- Bridge model outputs to clinician-ready decisions (red flags, meds with cautions, lifestyle/diet, escalation triggers).
- Standardize follow-up with detailed, non-vague, non-repetitive questions that surface missing clinical signal.
- Keep data local-first; LLM calls run via local Ollama for privacy.
- Provide a native-like experience with Electron + a modern React UI and structured parsing.

---

## 2) High-Level Overview

- User uploads audio (breath sounds) or chest X-ray.
- Backend model predicts disease label.
- If non-healthy, LLM generates 16 detailed follow-up questions.
- User answers; responses sent to LLM for a structured clinical report.
- Frontend parses the report into sections, applies severity badge, and renders styled cards.
- All files remain local; uploads go to `flask_server/uploaded_files/` (ignored by git).

---

## 3) Architecture Layers

| Layer | Stack | Entry Point | Purpose |
| --- | --- | --- | --- |
| UI shell | Electron + React + styled-components | `electron_app/src/App.js` | Intake, upload, follow-ups, report parsing/rendering. |
| API | Flask + Ollama client | `flask_server/app.py` | Predictions + LLM prompts (questions/reports). |
| Models | Python (torch/sklearn/etc.) | `detection_model/Audio_model.py`, `detection_model/Image_model.py` | Existing model code/weights (unchanged). |

---

## 4) Directory Structure

| Path | Purpose |
| --- | --- |
| `electron_app/` | Electron + React frontend (builds to `dist/bundle.js`). |
| `electron_app/src/App.js` | Main UI flow, styling, parsing, severity logic. |
| `electron_app/main.js` | Electron main process (window lifecycle). |
| `electron_app/preload.js` | Safe IPC bridge. |
| `electron_app/index.html` | Loads the bundled renderer. |
| `electron_app/webpack.config.js` | Build config (DefinePlugin for API base). |
| `flask_server/` | Flask API + LLM prompt orchestration. |
| `flask_server/app.py` | Endpoints: health, audio_prediction, image_prediction, generate_questions, analyze_responses. |
| `flask_server/uploaded_files/` | Runtime uploads (ignored). |
| `detection_model/` | Audio & image model code/weights (untouched). |
| `archive/` | Legacy/experimental assets. |
| `.gitignore` | Ignores venv, node_modules, dist, uploads, env, logs, OS cruft. |

---

## 5) Detailed Architecture Narrative

- **Electron shell** hosts the React renderer; frameless window with custom title bar.
- **Renderer** handles UX, file selection, stepper states (Intake → Follow-up → Report), axios calls to Flask.
- **Backend Flask** routes: health, audio_prediction, image_prediction, generate_questions, analyze_responses.
- **LLM** via Ollama handles language tasks: follow-up question generation and clinical report synthesis.
- **Models**: audio and image inference stay as provided; orchestrated by Flask endpoints.
- **Parsing layer**: renderer parses markdown-ish LLM output into structured sections, preserving bold.

---

## 6) Frontend Deep Dive

- Stack: React 17, styled-components, axios, Webpack/Babel.
- Styling: Space Grotesk typography, gradients, custom bullets, uppercase section titles, 15px base body, 14px controls.
- Steps:
	- Step 1 Intake: demographics, modality select, file drop.
	- Step 2 Follow-up: up to 16 deduped detailed questions from LLM; progress bar; skip option.
	- Step 3 Report: severity badge, parsed sections; fallback to raw formatted analysis if parsing empty.
- Parsing: headings and bullets extracted; bold (`**...**`) preserved with `formatInline`.
- Severity badge: explicit label preferred; otherwise keyword heuristic (high vs moderate vs low).

---

## 7) Backend Deep Dive

- Stack: Flask, CORS, Ollama client.
- Endpoints:
	- `GET /health`
	- `POST /audio_prediction` → `audio_prediction`
	- `POST /image_prediction` → `predict_image`
	- `POST /generate_questions` → LLM prompt for 16 specific follow-ups.
	- `POST /analyze_responses` → LLM prompt for structured clinical report.
- Files:
	- `flask_server/app.py`: routing + prompt construction.
	- `detection_model/Audio_model.py`: existing audio pipeline.
	- `detection_model/Image_model.py`: existing image pipeline.

---

## 8) LLM Usage (Model, Prompts, Samples)

- Model: `qwen2.5:3b` by default (configurable via `OLLAMA_MODEL`).
- Question prompt goals: 16 numbered, detailed, non-vague, include red flags, comorbidities/meds, exposures, onset/duration/severity; no repetition.
- Report prompt goals: structured markdown with summary, likely conditions, severity, red flags, tests (with rationale), meds (dosing ranges, contraindications, interactions, when to avoid), supportive care, diet do/avoid, home remedies with cautions, escalation triggers, differential, plan reminder.
- Sample (question prompt fragment): “Generate 16 numbered clinical follow-up questions for suspected {disease}; avoid vagueness; include red flags, comorbidities, current meds, exposures, onset/duration/progression, triggers, relieving factors, vitals.”
- Sample (report prompt fragment): “Analyze answers {…}. Return sections: Summary; Likely conditions; Severity; Red flags; Tests (why); Medications (dosing ranges, cautions, when not to use); Supportive care; Diet do/avoid; Home remedies (with cautions); Escalation triggers; Differential; Plan reminder.”

---

## 9) Severity Logic

- Prefers explicit “Severity: …” in LLM output.
- Otherwise keywords:
	- High: hypoxia, SpO2 < 94, cyanosis, shock, hemodynamic, respiratory distress, altered mental/confusion, severe chest pain, rapidly worsening.
	- Moderate: fever, cough, wheeze, shortness, chest pain, infection, pneumonia, bronchitis, COPD, asthma.
- Colors: urgent `#ef4444`; moderate `#fbbf24`; low `#22c55e`.

---

## 10) Data Flow & State Flow

- Data path: Renderer → Flask → Models → Renderer → LLM → Renderer.
- States: Intake (step 1) → Follow-up (step 2) → Report (step 3).
- Files saved to `flask_server/uploaded_files/` (ignored).

---

## 11) API Reference

| Endpoint | Method | Body | Response |
| --- | --- | --- | --- |
| `/health` | GET | none | `{ status: "ok" }` |
| `/audio_prediction` | POST | multipart `file` | `{ prediction: str }` |
| `/image_prediction` | POST | multipart `file` | `{ prediction: str }` |
| `/generate_questions` | POST | `{ disease: str }` | `{ questions: [str], warning? }` |
| `/analyze_responses` | POST | `{ answers: { idx: str } }` | `{ analysis: str }` |

---

## 12) Models Snapshot

- Audio: `detection_model/Audio_model.py` (kept as provided).
- Image: `detection_model/Image_model.py` (kept as provided).
- Weights: unchanged; ensure present locally.

---

## 13) Dataset Policy & Data Hygiene

- Keep only ~5 audio and ~5 image samples per class in git; store full datasets externally.
- `.gitignore` excludes uploads, venv, node_modules, dist, env, logs, OS cruft.
- Do not commit `flask_server/uploaded_files/` or large binaries.

---

## 14) Security & Privacy

- Data stays local; Ollama is local; no external calls assumed.
- Renderer/preload uses contextIsolation; nodeIntegration disabled in renderer.
- Avoid committing PHI; keep uploads ignored.

---

## 15) Performance Notes

- Webpack bundle ~375 KiB; acceptable for Electron.
- Questions/report are synchronous axios calls; consider batching if needed.
- Models run in Python; ensure optimized dependencies if you tune performance.

---

## 16) UX & Styling Notes

- Space Grotesk typography (body 15px; controls 14px).
- Gradients on report card, custom bullets, uppercase section titles, severity badge.
- Stepper shows Intake → Follow-up → Report.

---

## 17) Setup & Run

### Backend (Flask)
```powershell
cd flask_server
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# optional LLM override
$env:OLLAMA_MODEL="qwen2.5:7b"
python app.py
```

### Frontend (Electron + React)
```powershell
cd electron_app
npm install
npm start   # build + launch Electron

# Dev build-only
npm run dev
# Tests
npm test
```

### Runtime Layout
- Flask: `http://localhost:5000`
- Electron: loads `dist/bundle.js` via `index.html`
- Uploads: `flask_server/uploaded_files/`

---

## 18) Environment Variables

| Variable | Default | Scope | Description |
| --- | --- | --- | --- |
| `OLLAMA_MODEL` | `qwen2.5:3b` | Backend | LLM for questions/reports. |
| `REACT_APP_API_BASE` | `http://localhost:5000` | Frontend build | API base for axios (DefinePlugin). |

---

## 19) Testing Strategy

- Renderer: `npm test` (jest + testing-library setup).
- Manual: run Flask + Electron; exercise upload, questions, report parsing.
- Future: add API contract tests for Flask endpoints.

---

## 20) Troubleshooting (Expanded)

| Symptom | Likely Cause | Fix |
| --- | --- | --- |
| Blank Electron window | Renderer JS error | Run `npm start` with `ELECTRON_ENABLE_LOGGING=1`; check console; fix `src/App.js`; rebuild. |
| 500 on predictions | Missing model assets | Ensure `detection_model` files/weights exist; check Flask traceback. |
| Empty LLM output | Ollama not running/model missing | Start Ollama; set `OLLAMA_MODEL` to available model. |
| API unreachable | Flask not running | Start `python app.py`; confirm `REACT_APP_API_BASE`. |
| Severity always urgent | Keyword set too strict | Adjust lists in `src/App.js` deriveReport. |
| Large bundle warning | Webpack perf hint | Acceptable for Electron; or apply code splitting if desired. |

---

## 21) FAQ

- **Can I swap the LLM?** Yes, set `OLLAMA_MODEL` to any local Ollama model.
- **Can I change question count?** Yes, adjust prompt or frontend cap (currently 16).
- **Where are uploads stored?** `flask_server/uploaded_files/` (git-ignored).
- **Do you retrain models here?** No, only orchestrate existing weights.
- **How to relax severity?** Edit keyword lists in `deriveReport`.

---

## 22) Glossary

- **LLM**: Large Language Model, here via Ollama.
- **Renderer**: Electron web context hosting React UI.
- **Preload**: Secure bridge exposing limited APIs to renderer.
- **Severity badge**: UI indicator of risk (urgent/moderate/low).

---

## 23) Contribution Guidelines

- Keep detection models untouched unless intentional.
- Favor small styled-components and memoized hooks.
- Rebuild after parsing/severity changes (`npm run build`).
- Do not commit large binaries or PHI.

---

## 24) Release & Packaging

- Current workflow: dev/run via `npm start` (build + launch).
- Packaging (not wired): would use Electron Builder; add config if needed.

---

## 25) Logging & Error Handling

- Electron: run with `ELECTRON_ENABLE_LOGGING=1` for console logs.
- Flask: prints errors to console; return JSON errors.
- No remote telemetry; local-only.

---

## 26) Known Limitations & Roadmap Ideas

- No offline packaging script yet (could add electron-builder).
- Severity heuristic is simple; could incorporate numeric vitals.
- No persisted session/history; could add local storage of reports.
- Tests limited to renderer jest; could add API tests.

---

## 27) Change Log (Recent Highlights)

- Added structured report parsing with bold preservation and styled sections.
- Expanded follow-up generation to 16 detailed questions.
- Tuned severity logic to honor explicit labels and tighten high-risk keywords.
- Modernized UI styling (Space Grotesk, gradients, custom bullets, larger fonts).
- Detailed README with data hygiene guidance.

---

## 28) License

MIT
