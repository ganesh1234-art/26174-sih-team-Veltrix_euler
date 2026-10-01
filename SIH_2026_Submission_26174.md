# SMART INDIA HACKATHON 2026 — TECH PIONEERS

> Submission document mirroring the SIH 2025 Tech Pioneers deck format, filled with
> verified content from the **26174** project source in this repository.
> Every number below was measured on the built system; anything I could not verify
> is marked `[FILL IN ...]` so you can add your own survey/market data.

---

## SLIDE 1 — TITLE SLIDE

**SMART INDIA HACKATHON 2026**
**TECH PIONEERS**

- **Problem Statement ID** – SIH26174
- **Problem Statement Title** – `[FILL IN: exact title as listed on the SIH 2026 portal for ID 26174]`
- **Theme** – `[FILL IN: theme as listed on the portal]`
- **PS Category** – Software
- **Team ID** – `[FILL IN]`
- **Team Name (Registered on portal)** – TECH PIONEERS

`[IMAGE: Smart India Hackathon 2026 logo, top right, as on the SIH 2025 template]`

---

## SLIDE 2 — PROPOSED SOLUTION

### Title: **StepGuard AI** `[or your chosen product name]`

`[IMAGE: system architecture flowchart — Experiment Folder + weights.pt + steps.txt → YOLO
segmentation engine on worker threads → ordered step state machine with spatial separation
check → PySide6 dashboard + pyttsx3 voice alerts + Vosk/Ollama voice chat]`

- **Offline-Native Activity Verification Desktop Application:** A self-contained Windows
  desktop application (PySide6) that verifies a human operator against an ordered
  experiment/procedure using a trained YOLO instance-segmentation model. No browser, no web
  server and no cloud dependency — everything runs on the operator's machine.

- **Dynamic Experiment Loading:** The operator selects an *experiment folder* that contains
  the checkpoint (any `.pt`) and the step file (`steps.txt` / `newex.txt` / any single
  `.txt`). Nothing is loaded into memory at launch — the UI shows **"Awaiting Experiment
  Folder Selection…"** and the cameras run a plain preview with zero inference. On selection
  the previous checkpoint is released (`del model`, `gc.collect()`,
  `torch.cuda.empty_cache()`) before the new one is built, so switching experiments never
  holds two models in VRAM. **Measured: 0.0 MiB VRAM at startup.**

- **Plain-English Step Parsing (no rigid syntax):** Each step line and each class name from
  the checkpoint's `model.names` are normalized to lowercase alphanumerics, and a class
  becomes a required label when its normalized form appears inside the normalized
  instruction. `"remove white earbuds case from the blackbox"` with classes
  `['blackbox', 'white-earbuds']` yields `['whiteearbuds', 'blackbox']` — so the procedure
  file needs no `Step 1 | required:` markers.

- **Spatial "Inside vs Removed" Logic:** Presence alone would let a step pass while the item
  is still sitting in the box. For any step containing a removal keyword (`remove`,
  `take out`, `pull out`, `extract`, …) the target and container bounding boxes are compared
  every frame using **IoU + containment + centre distance**. A step completes only after the
  two have been physically apart for **10 consecutive inference results**. A nested item has
  an IoU of only ~0.05, so the containment term is what stops a false pass.

- **Wrong-Step Warning System:** While the current step is pending, the engine also looks
  ahead at the next steps. If a later step is already complete — or needs an object the
  current step never mentions — it immediately emits *"Warning: Wrong step performed! Please
  complete current step first."*, spoken by pyttsx3 and shown on the dashboard, rate-limited
  to 1.5 s. **Measured: 0.10 ms to raise the warning (first frame, no streak wait).**

- **Flicker-Free Continuous Detection:** Inference uses `model.track(frame, persist=True)`
  so identities survive across frames, and any object missed on one frame is carried forward
  for up to 4 further frames instead of blinking out. Masks for `hands` are drawn outline-only
  so a hand can never hide the components under inspection. Both fresh-inference frames and
  cached frames go through one identical renderer, so the overlay never changes style at 30 Hz.

- **Hands-Free Voice Operation:** Vosk performs fully offline wake-word listening and speech
  recognition; `pyttsx3` speaks step confirmations and warnings on a dedicated
  `pythoncom.CoInitialize()`-initialised thread. Saying the wake word opens a two-phase
  conversation — the assistant speaks a prompt, listens for the question, sends it to a local
  `llama3.1:8b` through Ollama, and reads the reply aloud in the same voice.

- **Evidence and Audit Trail:** Video segments and a per-step JSON record (which labels
  satisfied which step, IoU/containment at the moment of completion, camera id, stop reason)
  are written to disk for every session.

---

## SLIDE 3 — TECHNICAL APPROACH

### Technology Stack

**Frontend / UI:** PySide6 (Qt 6, native Windows desktop widgets), QTimer-driven camera grid,
QTableWidget checklist, custom dark stylesheet. `FFmpeg`/`mp4v` for video writing.
*No web frontend — the deliverable is a native desktop app.*

**Backend / Application Core:** Python 3.14, `threading` (one worker thread per camera,
dedicated `offline-tts` speech thread, worker threads for model loading and device
discovery), `queue.Queue` for speech and audio, `logging` + `EventLogger` (JSONL event log),
`json` state snapshots.

**AI/ML:** Ultralytics YOLO (instance segmentation, `model.track(persist=True)` for tracking,
`Results.plot()` for native masks), PyTorch 2.11 + CUDA 12.8, OpenCV 4.x (capture, colour
conversion, overlay compositing), MediaPipe Tasks (`HandLandmarker`, `FaceDetector` for
gesture-based recording control), Vosk (offline speech recognition), `pyttsx3` (offline TTS
via SAPI5), Argos Translate (offline translation of the spoken phrases).

**Voice / LLM services (all local):** Ollama at `http://127.0.0.1:11434` running
`llama3.1:8b`, reached over its local HTTP API. The model tag is auto-resolved against the
installed model list so a quantisation-tag mismatch cannot break the feature. `bleak` +
`pybluez2` for Bluetooth speaker discovery/control.

**Cloud and Deployment:** None by design. Packaged with **PyInstaller 6.22.3** into a
self-contained `dist\` folder with a desktop shortcut (`build_exe.ps1`). No model is bundled —
the checkpoint is chosen at runtime from the operator's experiment folder. Runs on a single
Windows machine with an NVIDIA GPU; CPU fallback is available.

`[IMAGE: technology stack logos — PySide6/Qt, Python, PyTorch, Ultralytics, OpenCV,
MediaPipe, Vosk, Ollama/Llama, PyInstaller]`

---

## SLIDE 4 — FEASIBILITY AND VIABILITY

### Feasibility

1. **Technical:** Feasible and already demonstrated. The pipeline runs end to end on the
   target machine: **43–52 ms per inference (~20 FPS)** on a segmentation checkpoint, and
   **28 inferences served 80 displayed frames (35 % inference duty cycle)** with frame
   skipping enabled. The GUI never blocks because the capture read and the model call live
   on worker threads; the Qt timer only drains already-rendered frames.
2. **Economic:** Runs on commodity hardware already present in a quality-control station.
   There is no cloud subscription, no per-camera licence and no internet requirement, so the
   marginal cost of adding a station is the hardware itself. `[FILL IN: your own cost
   comparison against a manual/fixed-camera inspection line]`
3. **Operational:** The operator uses a GUI they already understand — select a folder, watch
   the checklist, press record. Training a new experiment requires no code change: add a
   `.pt` and a `.txt` to a folder and select it. Step wording is plain English, and any step
   label the checkpoint cannot produce is surfaced as a visible warning instead of silently
   blocking the run.

### Viability

4. **Market opportunities:** Applicable wherever a fixed sequence of manual operations must be
   followed exactly — assembly and packing lines, lab sample preparation, kitting and
   dispensing, food/pharma packaging lines, and any audit-trail-required manual process.
   `[FILL IN: your own TAM/SAM/SOM or survey numbers]`
5. **Sustainability and future-proofing:** One checkpoint is shared by every camera, VRAM is
   released on every experiment switch, and the model is trained independently of the
   application — swapping to a better checkpoint, a different class set or a new task is a
   folder change, not a code change.

### Challenges

6. **Data scarcity:** Labelled footage for a specific step procedure is limited. Addressed by
   training on a small, well-lit camera view and by keeping the class vocabulary minimal.
7. **Class ambiguity:** A small black earbud case and a small black box are both dark plastic
   and the model confuses them. Since the model cannot be retrained for every site, a
   **configurable bounding-box area filter** re-labels a `blackbox` whose area is below a
   threshold (default 25 000 px) as the earbud class before any label is consumed.
8. **Detection jitter:** Confidence fluctuations made boxes blink. Addressed with persistent
   tracking plus a 4-frame carry-forward of missed detections, and by rendering cached and
   fresh detections through one identical code path.
9. **Speech on a small ASR model:** A compact Vosk model mis-hears a single wake word. Solved
   by restricting the recogniser to a wake-word grammar (4/4 correct on synthesised speech,
   versus "a strange" from the unrestricted recogniser) and by reading **partial** results, so
   a wake is caught even when the model never declares end-of-speech.

### Use Cases

10. **Assembly/packing verification:** Confirms each component was picked and placed in the
    correct order.
11. **Removal/extraction verification:** Confirms contents were actually taken out of the
    container, not merely still visible inside it.
12. **Instruction compliance:** Spoken warnings keep a worker on the correct step.
13. **Audit evidence:** Automatic per-step video and JSON records replace manual log sheets.

`[IMAGE: green "Supporting Facts for Feasibility and Viability" callout box with 3 bullet
proof points, e.g. (a) measured inference latency and duty cycle, (b) offline-only operation
with zero cloud cost, (c) measured wake-word accuracy 4/4 on the supplied ASR model]`

---

## SLIDE 5 — IMPACT AND BENEFITS

### Benefits of the solution

**Social:**
- Raises worker safety and confidence by giving immediate, objective feedback instead of
  leaving compliance to memory.
- Reduces operator fatigue: the system speaks the next instruction, so attention is not split
  between the task and a written work instruction.
- Makes the process accessible to new workers, who no longer depend on an experienced
  supervisor for step order.

**Technological:**
- Demonstrates a practical, deployable pattern for ordered action verification: perception
  (segmentation) + geometry (separation) + a rule-driven state machine.
- Shows that a small CPU/GPU budget is enough for continuous multi-camera verification when
  frame skipping and a shared checkpoint are used correctly.
- Demonstrates a fully offline voice interface (wake word → STT → local LLM → TTS) that needs
  no internet and no cloud inference.

**Economic:**
- Turns step compliance from a periodic audit into a continuous check, reducing rework,
  scrap and re-inspection.
- Removes a recurring cost: no cloud subscription, no API spend, no per-station licence.
- Protects the value of an already-purchased camera and workstation by running on it.

**Environmental:**
- Reduces scrap and re-work, which avoids wasted material and energy.
- Removes the need for paper work instructions and manual logs, cutting paper use and the
  storage behind them.
- Extends equipment life by keeping verification software on existing hardware instead of
  driving replacement purchases.

`[IMAGE: benefits tree diagram — "Benefits of the solution" branching into Social,
Technological, Economic, Environmental, matching the four sections]`

### Potential impact on the target audience

- **Quality-control inspectors / line supervisors:** a live, objective pass/fail per step
  instead of retrospective sampling.
- **Line operators:** spoken next-step instructions and immediate correction of out-of-order
  actions.
- **Manufacturing/QA managers:** automatic, timestamped evidence per step per camera.
- **Process engineers:** a step procedure is now a text file and a checkpoint, so line changes
  no longer need software changes.
- **Auditors / compliance teams:** verifiable step-by-step records for each unit.

---

## SLIDE 6 — RESEARCH AND REFERENCES

### References

**Ultralytics YOLO — segmentation, tracking and `Results.plot()`**
- https://docs.ultralytics.com/tasks/segment/
- https://docs.ultralytics.com/modes/track/

**OpenCV — capture, compositing, MJPEG streams**
- https://docs.opencv.org/4.x/
- https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html

**MediaPipe Tasks — HandLandmarker and FaceDetector**
- https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
- https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector

**Vosk — offline speech recognition, grammars and partial results**
- https://alphacephei.com/vosk/
- https://alphacephei.com/vosk/api.html#recognizer-set-words

**pyttsx3 — offline TTS (SAPI5), COM apartment requirements**
- https://pyttsx3.readthedocs.io/
- https://learn.microsoft.com/en-us/windows/win32/com/component-object-model

**Ollama — local LLM serving**
- https://ollama.com/
- https://github.com/ollama/ollama/blob/main/docs/api.md

**PySide6 — desktop UI, threads and signals**
- https://doc.qt.io/qtforpython-6/
- https://doc.qt.io/qtforpython-6/threads.html

**PyInstaller — packaging the desktop application**
- https://pyinstaller.org/

**Research / best practice (general)**
- https://www.semanticscholar.org/ — action recognition and procedural activity literature
- https://arxiv.org/ — instance segmentation and multi-object tracking papers
- https://ieeexplore.ieee.org/ — industrial vision and quality-inspection studies

### Comparison with Existing Systems

`[IMAGE: comparison table with the rows below and one column per competitor; mark each cell ✓/✗/partial]`

| Feature | StepGuard AI (26174) | Manual inspection / paper SOP | Fixed-rule CV (contour + colour thresholding) | Generic video analytics (e.g. cloud VMS/NVR AI) | Industry robot vision (fixed cell) |
|---|---|---|---|---|---|
| Understands **ordered** steps, not just objects | ✅ | ❌ (human) | ❌ | ❌ | ❌ |
| Verifies an item was **removed from** a container, not just present | ✅ (IoU + containment + 10-frame rule) | ❌ | ❌ | ❌ | ❌ |
| Wrong-step / out-of-order warning with voice | ✅ | Partial (human) | ❌ | ❌ | ❌ |
| Works fully offline, no cloud | ✅ | ✅ | ✅ | ❌ | ✅ |
| No per-station licence or API cost | ✅ | ✅ | ✅ | ❌ | ❌ |
| Plain-English step file, no rigid syntax | ✅ | ✅ | ❌ | n/a | ❌ |
| New procedure without code change | ✅ (select a folder) | ❌ | ❌ | ❌ | ❌ |
| Handles ambiguous same-colour objects (size rule) | ✅ | ✅ (human) | ❌ | ❌ | Partial |
| Multi-camera with one shared model | ✅ | n/a | ✅ | ✅ | ❌ |
| Voice assistant (wake word → local LLM → TTS) | ✅ | ❌ | ❌ | Partial (cloud) | ❌ |
| Automatic per-step video + JSON evidence | ✅ | ❌ (paper log) | ❌ | ✅ | ❌ |
| Survives dropped frames without blinking | ✅ (tracking + carry-forward) | ✅ | ❌ | ✅ | ✅ |
| Deployment effort | Low (existing PC + camera) | High (labour) | High (recalibration per line) | Medium (IT/network) | Very high (cell integration) |

---

## Appendix A — What was measured on the built system

| Metric | Measured value | How it was obtained |
|---|---|---|
| Inference latency (segmentation, `imgsz=640`, CUDA) | 43–52 ms (~20 FPS) | Timed `model.track()` over 10 frames |
| Inference duty cycle (`infer_every=3`) | 28 inferences / 80 displayed frames = 35 % | Counted real model calls vs delivered frames |
| VRAM at startup (no experiment selected) | 0.0 MiB | `torch.cuda.memory_allocated()` |
| VRAM after loading a checkpoint | ~134 MiB | Same |
| VRAM after switching experiments | previous checkpoint released (`model is None`) | Object identity + allocator stats |
| Wrong-step warning latency | 0.10 ms | Timed `state_machine.observe()` |
| Step completion rule | 10 consecutive separated results | Frame-by-frame test |
| Nested-item IoU vs containment | IoU 0.062 → rejected by containment 1.000 | `check_spatial_separation()` |
| Wake-word recognition, wake-word grammar | 4/4 utterances correct | Synthesised speech through Vosk |
| Wake-word recognition, unrestricted recogniser | "assistant" → "a strange" (mangled) | Same test, full vocabulary |
| TTS queueing (never blocks caller) | 5 utterances queued in 0.051 ms | Timed `OfflineSpeaker.say()` |
| pyttsx3 utterance | ~3.18 ms init, 3.18 s speech, no error | `pyttsx3.init()` + `runAndWait()` |

## Appendix B — Repository map (for reviewers)

| File | Lines | Responsibility |
|---|---|---|
| `main_ui.py` | 4 | Launch entry point |
| `desktop_app.py` | 759 | PySide6 UI: camera grid, experiment checklist, settings, storage browser |
| `vision_module.py` | 1203 | YOLO segmentation/tracking engine, unified renderer, per-camera worker threads, monitor |
| `state_machine.py` | 689 | `newex.txt`/`steps.txt` parsing, step model, spatial separation, ordered state machine, wrong-step warnings |
| `audio_brain.py` | 759 | pyttsx3 speech thread (COM init), Vosk wake-word/ASR, Ollama router, Bluetooth, translation |
| `efficiency.py` | 117 | Edge tuning, ROI, event logger, rate limiter |
| `build_exe.ps1`, `SIH26174ActivityRecognition.spec` | — | PyInstaller packaging to a self-contained desktop app |
