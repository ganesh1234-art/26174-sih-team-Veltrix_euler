# SIH 26174 — Offline Desktop Activity Recognition

This project is a native Windows desktop application. It does not start a web
server or require a browser. Object recognition is performed entirely by a
trained Ultralytics YOLO instance-segmentation checkpoint, and the ordered
experiment procedure is verified from the class names that checkpoint emits.
Monitoring, speech recognition, translation, vision inference, experiment
verification, storage, Wi-Fi camera ingestion, and camera reconnect loop run
locally. Voice Chat is the sole optional local REST call and targets
ip address (Ollama on the same computer).

## Run during development

```powershell
python -m pip install -r requirements.txt
python desktop_app.py
```

The app automatically starts with `CAM-0`, the laptop/default camera. Use
**Refresh Local Cameras** to discover additional USB cameras, select several,
then choose **Start Selected** to arrange them in a live grid. Enter an RTSP or
MJPEG URL to add a Wi-Fi camera. Each camera retains only its latest rendered
frame and reconnects with capped exponential backoff after a read failure.

## Startup state: no model is loaded

On launch nothing is read into memory. `MultiCameraMonitor` starts with
`segmenter = None` and the UI reports **Awaiting Experiment Folder Selection...**
while the cameras run a plain live preview with no inference at all. The
checkpoint comes only from the folder the user picks, so switching experiments
never leaves the previous `.pt` in VRAM.

Press **SELECT EXPERIMENT FOLDER** and pick a folder that contains:

- any `*.pt` — the YOLO checkpoint. Nothing is hard-coded: `weights.pt`,
  `best.pt`, `last.pt` or any other single `.pt` in the folder is used, so an
  `Ultralytics` run folder works unchanged. Its `model.names` become the only
  vocabulary the pipeline understands, and `Results.plot()` draws its native
  instance masks onto the camera frame.
- one `.txt` step file — the ordered procedure. `steps.txt` wins, then
  `newex.txt`, then the folder's only `.txt`, so any name works. Every numbered
  line is a step; unnumbered lines are shown as the description.

`MultiCameraMonitor.load_experiment()` performs the swap: pause the camera
threads so nothing infers against a dying model, release the old checkpoint
(`del`, `gc.collect()`, `torch.cuda.empty_cache()`), build `YOLO(path)`, warm it
up, re-parse the steps against the checkpoint's own class names, then resume. It
runs on a worker thread, so the UI stays responsive.

Example step file:

```
new ordered experiment
1. open the blackbox which consists of white earbuds case and black earbuds case.
2. take out the white earbuds case first from the blackbox.
3. and then take out the black earbuds case from the blackbox.
```

## Step parsing without strict syntax

No `Step 1 | required: label` markers are needed. Each step line and each
`model.names` entry are normalized to lowercase alphanumerics (spaces, hyphens
and underscores removed), and a class becomes a required label when its
normalized form is a substring of the normalized instruction. `"remove white
earbuds case from the blackbox"` with classes `['blackbox', 'white-earbuds']`
therefore yields `['whiteearbuds', 'blackbox']`, ordered by where each name
appears in the sentence. Detected labels are normalized the same way before the
comparison, so `black-earbuds` always satisfies a step requiring `blackearbuds`.
Because this is substring matching, a class name that is a fragment of an
unrelated word (`case` inside `showcase`) can match unintentionally; keep class
names distinctive or raise `match_frames`.

## Spatial removal logic ("inside the box" vs "removed from the box")

A step whose instruction contains a removal keyword (`remove`, `take out`,
`pull out`, `extract`, `empty`, ...) does not pass on presence alone. The first
required label is the object that must move and the second is its container, and
both must be detected *and* measured apart:

- `check_spatial_separation()` in `state_machine.py` reports `overlap_iou`,
  `containment` (shared area over the **smaller** box) and `center_distance`.
  Separation requires `IoU <= 0.15` **and** containment below 0.30.
- The containment term is what makes the check work. A small item nested inside
  a large box has an IoU of only ~0.05, so an IoU-only test would call it
  "removed" while it is still in the box.
- The target must stay clear of every detected container box for
  `EdgeTuning.separation_frames` (default 10) consecutive inference results
  before the step passes; any overlap resets the streak. If no boxes are supplied
  the step is never satisfied, so a missing measurement cannot pass by accident.

A plain step still passes on the first frame where all its labels are present
(`EdgeTuning.match_frames`, default 1; raise to 2-3 to demand consecutive
matching frames). On a pass the console prints `Step X Completed!`, the step is
checked off, the state machine advances, and `pyttsx3` reads out the **next**
step's instruction from its own thread.

## Threading

## Frame rate, tracking and memory

Inference uses `model.track(frame, persist=True)`, so the tracker keeps its
association state between calls: an object missed on one frame is still reported
with the same track id instead of vanishing and re-appearing, which is what made
boxes blink. The checkpoint is shared by all cameras, so the tracker is reset when
a camera (re)connects to keep IDs from being matched across streams. The tracker
is fed a lower confidence floor than the display threshold (ByteTrack needs
low-confidence candidates to hold an identity); detections below the display
threshold are filtered out afterwards, so the state machine only ever sees
confident objects.

`VisionSettings.infer_every` (default `3`) runs the model on every Nth captured
frame. Both paths - a fresh inference and a frame that reused the cache - go
through the **same** `render_frame()` routine, so masks, colours, line thickness
and captions are identical and the overlay never flickers at 30 Hz. Cached
detections (boxes, masks, labels, track ids) survive up to `HELD_FRAME_GRACE`
skipped frames, so a single dropped detection does not make an object
disappear; a held frame is marked in the HUD. `VisionSettings.imgsz` (default
`640`) bounds the inference cost. The camera card shows capture FPS and inference
FPS separately because they differ whenever frames are skipped.

Every camera runs on its own `threading.Thread` (`vision-<camera>`) that does
the capture read, `results = model.track(frame)`, rendering, the area filter and
the state machine observation; the Qt timer only drains already-annotated frames.
`pyttsx3` runs on a further thread (`offline-tts`) fed by a `queue.Queue`, calls
`pythoncom.CoInitialize()` at thread start because SAPI5 refuses to speak on a
thread without a COM apartment, and creates the engine on that same thread.
Loading a checkpoint and its warm-up inference also run on a worker thread.

## Wrong-step warnings

While the current step is unsatisfied, the state machine also looks a few steps
ahead. If a later step's labels are all present, it emits
`{"event": "step_warning", "message": "Warning: Wrong step performed! Please complete current step first."}`,
spoken by `pyttsx3` and shown in the experiment tab. A 5 second cooldown keeps
one mistake from producing a burst of speech. A frame that satisfies the current
step always advances it, even when later objects are in view.

## Supplied local assets

No model is bundled: the checkpoint comes from the experiment folder at runtime.

- `hand_landmarker.task` (MediaPipe Tasks `HandLandmarker`)
- `face_detector.tflite` (MediaPipe Tasks `FaceDetector`)
- `vosk_model/` (offline speech recognition)

Video and verification JSON are saved in the current user's
`%LOCALAPPDATA%\SIH26174ActivityRecognition\storage` folder. The hand and face
tasks drive the recording gestures (thumbs-up toggles recording, an open palm
stops it) and are independent of object recognition.

## Build the EXE and Desktop icon

```powershell
.\build_exe.ps1
```

This produces a self-contained `dist\SIH26174ActivityRecognition` application
folder and adds **SIH 26174 Activity Recognition** to the Windows Desktop as a
shortcut. The onedir format is deliberate: model assets remain available without
per-launch extraction latency. The resulting EXE can open offline from that icon.

Bluetooth speakers must be paired and selected as the Windows default A2DP audio
device. The desktop app can scan/connect compatible BLE control channels, while
Windows owns the A2DP audio transport that `pyttsx3` uses.

## Additional features that can be implemented and our team is trying to do so

1. We can directly connect the ai to chemical valves control so that there is no wastage of resource constraint chemicals .
2. Currently working on fitting the entire project with ai offline chat model llama 3.1 8b within jetson nano , providing totally automated and headless monitoring and alerting.
