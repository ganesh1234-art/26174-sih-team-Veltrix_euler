"""Native offline desktop UI for SIH 26174 Activity Recognition.

This is a Qt application, not a web server: no browser, HTTP dashboard, or
internet connection is needed for monitoring.  Only the optional *local*
Ollama endpoint is used when Voice Chat is switched on.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import sys
from threading import Thread
from typing import Any

import cv2
from PySide6.QtCore import QObject, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QColor, QDesktopServices, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QTableWidget,
    QTableWidgetItem,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from audio_brain import AudioBrain, BluetoothAudioManager
from efficiency import EdgeTuning, OPTIMIZATION_NAMES
from state_machine import ExperimentFolder, ExperimentStep, required_labels_satisfied
from vision_module import (
    FrameResult,
    MultiCameraMonitor,
    VisionSettings,
    discover_cameras,
    normalize_stream_source,
)


APP_NAME = "SIH 26174 Activity Recognition"


def resource_path(relative_path: str) -> Path:
    """Resolve source-tree assets and PyInstaller one-file bundled assets."""

    root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return root / relative_path


def user_data_path(relative_path: str) -> Path:
    root = Path(os.getenv("LOCALAPPDATA", Path.home())) / "SIH26174ActivityRecognition"
    root.mkdir(parents=True, exist_ok=True)
    return root / relative_path


class SignalBridge(QObject):
    message = Signal(str)
    devices = Signal(list)
    cameras = Signal(list)
    monitor_updates = Signal(object)
    experiment_loaded = Signal(object, object)


class CameraScanner(QThread):
    found = Signal(list)

    def run(self) -> None:
        self.found.emit(discover_cameras())


class CameraCard(QFrame):
    def __init__(self, camera_id: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.camera_id = camera_id
        self.setObjectName("cameraCard")
        self.setMinimumSize(310, 255)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        self.header = QLabel(camera_id)
        self.header.setObjectName("cameraHeader")
        self.image = QLabel("Waiting for camera frame…")
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image.setMinimumHeight(210)
        self.image.setStyleSheet("background:#02050b; border:1px solid #154261;")
        self.status = QLabel("CONNECTING")
        self.status.setObjectName("cameraStatus")
        layout.addWidget(self.header)
        layout.addWidget(self.image, 1)
        layout.addWidget(self.status)

    def update_frame(self, result: FrameResult | None, status: dict[str, Any]) -> None:
        if result is not None:
            frame = result.frame
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888).copy()
            pixmap = QPixmap.fromImage(image).scaled(
                self.image.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            )
            self.image.setPixmap(pixmap)
            trigger = "REC" if result.recording else "LIVE"
            if status.get("awaiting_experiment"):
                self.status.setText(f"{trigger} • {status.get('fps', 0)} FPS • {MultiCameraMonitor.AWAITING_EXPERIMENT}")
                return
            state = ", ".join(result.labels[:2]) or "no object detected"
            rate = f"{status.get('fps', 0)} FPS capture / {status.get('inference_fps', 0)} FPS infer"
            self.status.setText(f"{trigger} • {result.instances} mask(s) • {rate} • {state}")
        elif not status.get("connected"):
            self.status.setText(f"RECONNECTING • {status.get('error') or 'waiting'}")


class DesktopApplication(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1500, 940)
        self.bridge = SignalBridge()
        self.bridge.message.connect(self._set_status)
        self.bridge.devices.connect(self._show_bluetooth_devices)
        self.bridge.cameras.connect(self._set_camera_choices)
        self.bridge.experiment_loaded.connect(self._apply_experiment)
        self.scanner: CameraScanner | None = None
        self.cards: dict[str, CameraCard] = {}
        self.experiment_folder: ExperimentFolder | None = None
        self.experiment_steps: list[ExperimentStep] = []
        self._pending_steps: list[ExperimentStep] = []
        self._detected_labels: list[str] = []
        self._discovered_cameras: list[dict[str, str | int]] = []
        self.network_sources: dict[str, str] = {}
        self.audio = AudioBrain(user_data_path("phrase_cache.json"))
        self.bluetooth = BluetoothAudioManager()
        self.monitor = self._create_monitor()
        self._build_ui()
        self._apply_theme()
        # Laptop camera 0 is selected by default. The worker reconnects if it
        # is unavailable instead of blocking application startup.
        self.monitor.add_camera(0, "CAM-0")
        self._rebuild_grid()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._poll_monitor)
        self.timer.start(33)
        self._scan_cameras()

    def _default_settings(self) -> VisionSettings:
        """No checkpoint is referenced at startup; a folder supplies it later."""

        return VisionSettings(
            storage_dir=str(user_data_path("storage")),
            hand_model_path=str(resource_path("hand_landmarker.task")),
            face_model_path=str(resource_path("face_detector.tflite")),
        )

    def _create_monitor(self) -> MultiCameraMonitor:
        return MultiCameraMonitor(self._default_settings(), EdgeTuning(), [])

    def _build_ui(self) -> None:
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(14, 10, 14, 12)
        title = QLabel("◈ SIH 26174  /  ACTIVITY RECOGNITION")
        title.setObjectName("title")
        subtitle = QLabel("NATIVE OFFLINE DESKTOP • YOLO26 INSTANCE SEGMENTATION • ORDERED STEP VERIFICATION • MULTI-CAMERA")
        subtitle.setObjectName("subtitle")
        root_layout.addWidget(title)
        root_layout.addWidget(subtitle)
        self.tabs = QTabWidget()
        self.tabs.addTab(self._make_monitor_tab(), "MONITOR")
        self.tabs.addTab(self._make_experiment_tab(), "EXPERIMENT")
        self.tabs.addTab(self._make_storage_tab(), "STORAGE")
        self.tabs.addTab(self._make_settings_tab(), "SETTINGS")
        root_layout.addWidget(self.tabs, 1)
        self.status_label = QLabel(MultiCameraMonitor.AWAITING_EXPERIMENT)
        self.status_label.setObjectName("appStatus")
        root_layout.addWidget(self.status_label)
        self.setCentralWidget(root)
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        self.addAction(exit_action)

    def _make_monitor_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        controls = QHBoxLayout()
        self.camera_choices = QListWidget()
        self.camera_choices.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.camera_choices.setMaximumHeight(76)
        self.camera_choices.setMinimumWidth(280)
        controls.addWidget(self.camera_choices, 1)
        scan = QPushButton("REFRESH LOCAL CAMERAS")
        scan.clicked.connect(self._scan_cameras)
        start = QPushButton("START SELECTED")
        start.clicked.connect(self._start_selected_cameras)
        stop = QPushButton("STOP SELECTED")
        stop.clicked.connect(self._stop_selected_cameras)
        reconnect = QPushButton("RECONNECT SELECTED")
        reconnect.clicked.connect(self._reconnect_selected_cameras)
        start_recording = QPushButton("START RECORDING")
        start_recording.clicked.connect(self._start_recording)
        stop_recording = QPushButton("STOP RECORDING")
        stop_recording.clicked.connect(self._stop_recording)
        for button in (scan, start, stop, reconnect, start_recording, stop_recording):
            controls.addWidget(button)
        layout.addLayout(controls)
        wifi = QHBoxLayout()
        self.stream_url = QLineEdit()
        self.stream_url.setPlaceholderText(
            "Phone IP or stream URL — e.g. 192.168.1.50, 192.168.1.50:8080/video, rtsp://…"
        )
        add_stream = QPushButton("ADD WI-FI CAMERA")
        add_stream.clicked.connect(self._add_network_camera)
        wifi.addWidget(self.stream_url, 1)
        wifi.addWidget(add_stream)
        layout.addLayout(wifi)
        local_camera = QHBoxLayout()
        self.local_camera_index = QLineEdit("1")
        self.local_camera_index.setMaximumWidth(70)
        self.local_camera_index.setToolTip("Local webcam index. The usual second webcam is index 1.")
        add_local = QPushButton("ADD USB WEBCAM")
        add_local.clicked.connect(self._add_local_camera)
        local_camera.addWidget(QLabel("USB index"))
        local_camera.addWidget(self.local_camera_index)
        local_camera.addWidget(add_local)
        local_camera.addStretch(1)
        layout.addLayout(local_camera)
        self.grid_holder = QWidget()
        self.grid = QGridLayout(self.grid_holder)
        self.grid.setSpacing(10)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.grid_holder)
        layout.addWidget(scroll, 1)
        self.map_label = QLabel(MultiCameraMonitor.AWAITING_EXPERIMENT)
        self.map_label.setWordWrap(True)
        self.map_label.setObjectName("mapLabel")
        layout.addWidget(self.map_label)
        return page

    def _make_experiment_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        description = QLabel(
            "Select an experiment folder. It must contain a YOLO .pt checkpoint and a .txt "
            "file with the ordered steps (steps.txt or any single .txt). Nothing is loaded "
            "into memory until you pick one."
        )
        description.setWordWrap(True)
        buttons = QHBoxLayout()
        select_folder = QPushButton("SELECT EXPERIMENT FOLDER")
        select_folder.setToolTip("Pick the folder that contains the .pt model and the .txt step file.")
        select_folder.clicked.connect(self._select_experiment_folder)
        reload_folder = QPushButton("RELOAD FOLDER")
        reload_folder.clicked.connect(self._reload_experiment_folder)
        reset = QPushButton("RESET PROGRESS")
        reset.clicked.connect(self._reset_experiment)
        buttons.addWidget(select_folder)
        buttons.addWidget(reload_folder)
        buttons.addWidget(reset)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        self.experiment_path = QLineEdit(MultiCameraMonitor.AWAITING_EXPERIMENT)
        self.experiment_path.setReadOnly(True)
        layout.addWidget(self.experiment_path)
        self.experiment_view = QPlainTextEdit()
        self.experiment_view.setReadOnly(True)
        self.experiment_view.setMaximumHeight(110)
        self.experiment_view.setPlainText(
            "The step file's description appears here once a folder is selected."
        )
        layout.addWidget(self.experiment_view)
        self.checklist = QTableWidget(0, 6)
        self.checklist.setHorizontalHeaderLabels(
            ["Done", "Step", "Instruction", "Required YOLO Labels", "Detected Now", "Status"]
        )
        self.checklist.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.checklist.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.checklist, 1)
        self.model_label = QLabel(f"Model: {MultiCameraMonitor.AWAITING_EXPERIMENT}")
        self.model_label.setWordWrap(True)
        self.model_label.setObjectName("mapLabel")
        layout.addWidget(self.model_label)
        self.warning_label = QLabel("")
        self.warning_label.setWordWrap(True)
        self.warning_label.setObjectName("warningLabel")
        layout.addWidget(self.warning_label)
        return page

    def _make_storage_tab(self) -> QWidget:
        page = QWidget()
        layout = QHBoxLayout(page)
        self.storage_list = QListWidget()
        self.storage_list.currentItemChanged.connect(self._show_storage_metadata)
        refresh = QPushButton("REFRESH RECORDINGS")
        refresh.clicked.connect(self._refresh_storage)
        left = QVBoxLayout()
        left.addWidget(refresh)
        left.addWidget(self.storage_list, 1)
        self.storage_metadata = QPlainTextEdit()
        self.storage_metadata.setReadOnly(True)
        self.storage_preview = QLabel("Select a recording to view its offline metadata and preview.")
        self.storage_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.storage_preview.setMinimumSize(500, 320)
        open_video = QPushButton("OPEN SELECTED VIDEO")
        open_video.clicked.connect(self._open_selected_video)
        right = QVBoxLayout()
        right.addWidget(self.storage_preview, 1)
        right.addWidget(open_video)
        right.addWidget(self.storage_metadata, 1)
        left_widget, right_widget = QWidget(), QWidget()
        left_widget.setLayout(left)
        right_widget.setLayout(right)
        splitter = QSplitter()
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([400, 900])
        layout.addWidget(splitter)
        self._refresh_storage()
        return page

    def _make_settings_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        model_group = QGroupBox("LOCAL MODEL ASSETS")
        model_layout = QGridLayout(model_group)
        self.hand_path = QLineEdit(str(resource_path("hand_landmarker.task")))
        self.face_path = QLineEdit(str(resource_path("face_detector.tflite")))
        self.yolo_path = QLineEdit(MultiCameraMonitor.AWAITING_EXPERIMENT)
        self.yolo_path.setReadOnly(True)
        self.yolo_path.setToolTip("Set by SELECT EXPERIMENT FOLDER: the .pt found in that folder.")
        self.vosk_path = QLineEdit(str(resource_path("vosk_model")))
        for row, (name, field) in enumerate((("HandLandmarker", self.hand_path), ("FaceDetector", self.face_path), ("Experiment model (.pt)", self.yolo_path), ("Vosk model folder", self.vosk_path))):
            model_layout.addWidget(QLabel(name), row, 0)
            model_layout.addWidget(field, row, 1)
        select_folder = QPushButton("SELECT EXPERIMENT FOLDER")
        select_folder.clicked.connect(self._select_experiment_folder)
        reload_models = QPushButton("APPLY PATHS / RESTART VISION")
        reload_models.clicked.connect(self._restart_monitor)
        model_layout.addWidget(select_folder, 4, 0)
        model_layout.addWidget(reload_models, 4, 1)
        layout.addWidget(model_group)
        voice_group = QGroupBox("OFFLINE VOICE AND BLUETOOTH")
        voice_layout = QGridLayout(voice_group)
        self.language = QLineEdit("en")
        self.wake_word = QLineEdit("assistant")
        self.voice_toggle = QCheckBox("Voice Chat (wake-word starts local Ollama, 120-second keep-alive)")
        self.voice_toggle.toggled.connect(self._set_voice_chat)
        speak = QPushButton("TEST OFFLINE ALERT")
        speak.clicked.connect(lambda: self.audio.announce("Activity recognition is ready", self.language.text().strip() or "en"))
        scan_ble = QPushButton("SCAN BLUETOOTH")
        scan_ble.clicked.connect(self._scan_bluetooth)
        self.ble_devices = QListWidget()
        self.ble_address = QLineEdit()
        self.ble_address.setPlaceholderText("BLE speaker/control address")
        connect_ble = QPushButton("CONNECT BLE CONTROL")
        connect_ble.clicked.connect(self._connect_bluetooth)
        voice_layout.addWidget(QLabel("Alert language"), 0, 0)
        voice_layout.addWidget(self.language, 0, 1)
        voice_layout.addWidget(QLabel("Wake word"), 1, 0)
        voice_layout.addWidget(self.wake_word, 1, 1)
        voice_layout.addWidget(self.voice_toggle, 2, 0, 1, 2)
        voice_layout.addWidget(speak, 3, 0)
        voice_layout.addWidget(scan_ble, 3, 1)
        voice_layout.addWidget(self.ble_devices, 4, 0, 1, 2)
        voice_layout.addWidget(self.ble_address, 5, 0)
        voice_layout.addWidget(connect_ble, 5, 1)
        layout.addWidget(voice_group)
        profile = QGroupBox("EDGE EFFICIENCY PROFILE")
        profile_layout = QVBoxLayout(profile)
        profile_layout.addWidget(QLabel("\n".join(f"• {item}" for item in OPTIMIZATION_NAMES)))
        layout.addWidget(profile)
        layout.addStretch(1)
        return page

    def _apply_theme(self) -> None:
        self.setStyleSheet("""
            QMainWindow, QWidget { background:#050913; color:#d8edff; font-family:'Cascadia Mono','Consolas',monospace; }
            #title { color:#00e5ff; font-size:23px; font-weight:700; letter-spacing:3px; }
            #subtitle { color:#7599b8; font-size:11px; letter-spacing:1px; padding-bottom:6px; }
            QTabBar::tab { background:#0a1628; border:1px solid #1a5274; color:#8aa6c1; padding:10px 20px; margin-right:3px; }
            QTabBar::tab:selected { color:#00e5ff; border-color:#00e5ff; background:#0b253a; }
            QPushButton { background:#071c2d; border:1px solid #00bcd4; color:#dcfbff; padding:8px 12px; font-weight:700; }
            QPushButton:hover { border-color:#ff3dcc; color:#ffb3ec; background:#13283d; }
            QLineEdit, QListWidget, QPlainTextEdit { background:#06111f; border:1px solid #1d5576; color:#d8edff; padding:6px; selection-background-color:#0d6d88; }
            QGroupBox { border:1px solid #245d7e; margin-top:12px; padding:10px; color:#00e5ff; font-weight:700; }
            QGroupBox::title { subcontrol-origin:margin; left:10px; padding:0 5px; }
            #cameraCard { border:1px solid #12749a; background:#071322; }
            #cameraHeader { color:#00e5ff; font-weight:700; }
            #cameraStatus, #appStatus { color:#9ec9e6; padding:4px; }
            #mapLabel { border:1px solid #5b2873; background:#130b21; color:#f1b6e3; padding:8px; }
            #warningLabel { color:#ffb347; padding:2px; }
            QTableWidget { background:#06111f; border:1px solid #1d5576; gridline-color:#123a52; }
            QHeaderView::section { background:#0b253a; color:#00e5ff; border:1px solid #1d5576; padding:4px; }
        """)

    def _scan_cameras(self) -> None:
        if self.scanner and self.scanner.isRunning():
            return
        # Do not bury the "awaiting folder" prompt while scanning.
        if self.monitor.model_loaded:
            self._set_status("Scanning local cameras without interrupting active feeds…")
        self.scanner = CameraScanner(self)
        self.scanner.found.connect(self._set_camera_choices)
        self.scanner.start()

    def _set_camera_choices(self, cameras: list[dict[str, str | int]]) -> None:
        self._discovered_cameras = list(cameras)
        self._render_camera_choices()
        if not self.monitor.model_loaded:
            self._set_status(MultiCameraMonitor.AWAITING_EXPERIMENT)
        elif not cameras and not self.network_sources:
            self._set_status("No camera confirmed during scan. CAM-0 will continue automatic reconnect attempts.")
        else:
            self._set_status(
                f"Found {len(cameras)} local camera(s) and {len(self.network_sources)} Wi-Fi stream(s)."
            )

    def _render_camera_choices(self) -> None:
        """Rebuild the selectable camera list from local scan and Wi-Fi streams.

        Both local webcams (integer index) and Wi-Fi streams (URL string) are
        listed so START/STOP/RECONNECT SELECTED operate on either kind.
        """

        active = {str(worker.source) for worker in self.monitor.workers.values()}
        self.camera_choices.clear()
        for camera in self._discovered_cameras:
            item = QListWidgetItem(f"{camera['label']}  [index {camera['index']}]")
            item.setData(Qt.ItemDataRole.UserRole, camera["index"])
            self.camera_choices.addItem(item)
            if str(camera["index"]) in active:
                item.setSelected(True)
        for camera_id, source in self.network_sources.items():
            item = QListWidgetItem(f"Wi-Fi stream {camera_id}  [{source}]")
            item.setData(Qt.ItemDataRole.UserRole, source)
            self.camera_choices.addItem(item)
            if str(source) in active:
                item.setSelected(True)

    def _selected_sources(self) -> list[str | int]:
        return [item.data(Qt.ItemDataRole.UserRole) for item in self.camera_choices.selectedItems()]

    def _start_selected_cameras(self) -> None:
        sources = self._selected_sources()
        if not sources:
            self._set_status("Select one or more local cameras first.")
            return
        for source in sources:
            self.monitor.add_camera(source)
        self._rebuild_grid()
        self._set_status(f"Started {len(sources)} selected camera worker(s).")

    def _stop_selected_cameras(self) -> None:
        sources = {str(source) for source in self._selected_sources()}
        ids = [camera_id for camera_id, worker in self.monitor.workers.items() if str(worker.source) in sources]
        for camera_id in ids:
            self.monitor.remove_camera(camera_id)
        self._rebuild_grid()
        self._set_status("Selected camera workers stopped.")

    def _reconnect_selected_cameras(self) -> None:
        sources = {str(source) for source in self._selected_sources()}
        count = 0
        for camera_id, worker in self.monitor.workers.items():
            if str(worker.source) in sources:
                self.monitor.reconnect(camera_id)
                count += 1
        self._set_status(f"Requested reconnect for {count} camera(s).")

    def _start_recording(self) -> None:
        print("[UI] Start Recording button pressed", flush=True)
        count = self.monitor.start_recording()
        self._set_status(f"Manual recording requested for {count} camera(s).")

    def _stop_recording(self) -> None:
        print("[UI] Stop Recording button pressed", flush=True)
        count = self.monitor.stop_recording()
        self._set_status(f"Manual recording stopped for {count} camera(s).")

    def _add_network_camera(self) -> None:
        raw = self.stream_url.text().strip()
        if not raw:
            QMessageBox.warning(self, APP_NAME, "Enter a camera IP address or stream URL.")
            return
        # Shared normaliser expands a bare IP (e.g. 192.168.1.50) to the MJPEG
        # endpoint that the Android IP Webcam app serves on port 8080.
        source = normalize_stream_source(raw)
        camera_id = self.monitor.add_camera(source)
        self.network_sources[camera_id] = source
        self.stream_url.clear()
        self._rebuild_grid()
        self._render_camera_choices()
        self._set_status(f"Added {camera_id} at {source}; it will auto-reconnect if the stream drops.")

    def _add_local_camera(self) -> None:
        try:
            index = int(self.local_camera_index.text().strip())
            if index < 0:
                raise ValueError
        except ValueError:
            QMessageBox.warning(self, APP_NAME, "USB camera index must be a non-negative number.")
            return
        camera_id = self.monitor.add_camera(index, f"CAM-{index}")
        self._rebuild_grid()
        self._render_camera_choices()
        self._set_status(f"Added USB webcam {index} as {camera_id}.")

    def _rebuild_grid(self) -> None:
        while self.grid.count():
            item = self.grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.cards = {}
        for index, camera_id in enumerate(self.monitor.workers):
            card = CameraCard(camera_id)
            self.cards[camera_id] = card
            self.grid.addWidget(card, index // 2, index % 2)

    def _poll_monitor(self) -> None:
        """Drain rendered frames from the worker threads. Never infers, never blocks."""

        try:
            updates = self.monitor.poll()
        except Exception as exc:
            self._set_status(f"Vision poll error: {exc}")
            return
        detected: list[str] = []
        for camera_id, (result, status) in updates.items():
            card = self.cards.get(camera_id)
            if card:
                card.update_frame(result, status)
            if result is None:
                continue
            detected.extend(result.labels)
            self._update_checklist()
            if result.state_event:
                self._announce_step(result.state_event)
        self._detected_labels = list(dict.fromkeys(detected))
        self._update_checklist()
        if self.monitor.model_loaded:
            self.map_label.setText(
                "DETECTED CLASSES: " + (", ".join(self._detected_labels) or "nothing detected yet")
            )
        else:
            self.map_label.setText(MultiCameraMonitor.AWAITING_EXPERIMENT)
        self._refresh_audio_status()

    def _refresh_audio_status(self) -> None:
        """Keep the audio state visible in the status bar.

        A windowed build has no console, so a silent TTS or wake-word failure
        would otherwise look like the feature simply does not work.
        """

        speaker = self.audio.speaker
        problems = []
        if speaker.last_error:
            problems.append(speaker.last_error)
        if self.audio.voice_chat_enabled and not self.audio.wake_chat_healthy():
            problems.append(self.audio.voice_chat_status())
        if not problems:
            return
        message = "AUDIO: " + " | ".join(problems)
        if not self.status_label.text().startswith("AUDIO:"):
            self._set_status(message)
        else:
            self.status_label.setText(message)

    def _announce_step(self, event: dict[str, Any]) -> None:
        """Speak a state event through pyttsx3 on the speaker's own thread.

        ``AudioBrain.announce`` only enqueues onto the ``offline-tts`` thread
        (which calls ``pythoncom.CoInitialize()`` for SAPI5), so this returns
        immediately and the frame loop never stalls on speech.
        """

        language = self.language.text().strip() or "en"
        kind = str(event.get("event") or "")
        if kind == "experiment_complete":
            self.audio.announce("Experiment complete. All steps are verified.", language)
            self.warning_label.setText("")
            self._set_status(str(event.get("message") or "Experiment Completed Successfully"))
            self._update_checklist()
            return
        if kind == "step_warning":
            message = str(event.get("message") or "Warning: wrong step performed")
            expected = event.get("expected_step")
            observed = event.get("observed_step")
            self.warning_label.setText(
                f"{message}  (expected step {expected}, saw objects for step {observed})"
            )
            self._set_status(message)
            self.audio.announce(message, language)
            return
        number = event.get("next_step")
        instruction = event.get("next_instruction")
        if number and instruction:
            self.audio.announce(f"Step {number}. {instruction}", language)
        else:
            self.audio.announce("All steps completed.", language)

    def _select_experiment_folder(self) -> None:
        folder_name = QFileDialog.getExistingDirectory(
            self,
            "Select experiment folder (.pt model + .txt step file)",
            str(Path.home()),
        )
        if not folder_name:
            return
        self.experiment_path.setText(folder_name)
        self._load_experiment_folder(folder_name)

    def _reload_experiment_folder(self) -> None:
        if self.experiment_folder is None:
            self._select_experiment_folder()
            return
        self._load_experiment_folder(str(self.experiment_folder.path))

    def _load_experiment_folder(self, folder: str) -> None:
        """Validate the folder, load its checkpoint and parse its steps.

        Reading a ``.pt`` and its warm-up inference take seconds, so the whole
        operation runs on a worker thread: the Qt event loop and any live camera
        preview keep running while the model is loaded.
        """

        self._set_status(f"Loading experiment folder {Path(folder).name}…")
        self.warning_label.setText("")

        def task() -> None:
            try:
                bundle = self.monitor.load_experiment(folder)
                self.bridge.experiment_loaded.emit(bundle, self.monitor.classes)
            except Exception as exc:
                self.bridge.message.emit(f"Could not load experiment folder: {exc}")

        Thread(target=task, name="experiment-load", daemon=True).start()

    def _apply_experiment(self, bundle: ExperimentFolder, classes: tuple[str, ...]) -> None:
        self.experiment_folder = bundle
        self.experiment_steps = list(bundle.steps)
        self.experiment_path.setText(f"{bundle.path}   [{bundle.weights_path.name} + {bundle.steps_path.name}]")
        self.yolo_path.setText(str(bundle.weights_path))
        self.experiment_view.setPlainText(bundle.description or f"{bundle.steps_path.name} has no description lines.")
        self.model_label.setText(f"Model: {self.monitor.segmenter.describe() if self.monitor.segmenter else 'not loaded'}")
        self.warning_label.setText("\n".join(bundle.warnings))
        self._populate_checklist()
        self._set_status(
            f"Experiment '{bundle.name}' activated: {bundle.weights_path.name} + {bundle.steps_path.name}, "
            f"{len(bundle.steps)} step(s)."
        )

    def _populate_checklist(self) -> None:
        steps = self.experiment_steps
        self.checklist.setRowCount(len(steps))
        for row, step in enumerate(steps):
            done = QTableWidgetItem("")
            done.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            done.setCheckState(Qt.CheckState.Unchecked)
            self.checklist.setItem(row, 0, done)
            for column, value in enumerate(
                (str(step.number), step.instruction, ", ".join(step.required_labels) or "—"), start=1
            ):
                self.checklist.setItem(row, column, QTableWidgetItem(value))
            self.checklist.setItem(row, 4, QTableWidgetItem("—"))
            self.checklist.setItem(row, 5, QTableWidgetItem("Pending"))
            self._set_step_status(row, "Pending")
        self.checklist.resizeColumnsToContents()
        self._update_checklist()

    def _update_checklist(self) -> None:
        """Reflect the state machine index and the live detections in the table."""

        if not self.experiment_steps or self.checklist.rowCount() != len(self.experiment_steps):
            return
        state = self.monitor.state_machine
        current_index = state.index
        finished = state.complete
        current_step = self.experiment_steps[current_index] if not finished else None
        for row, step in enumerate(self.experiment_steps):
            if finished or row < current_index:
                # Every row is ticked and marked complete once the last step passes.
                self._set_step_status(row, "Completed")
                self._set_checked(row, True)
            elif row == current_index:
                self._set_step_status(row, "In Progress")
                self._set_checked(row, False)
            else:
                self._set_checked(row, False)
            detected = QTableWidgetItem("—")
            if step is current_step:
                satisfied = required_labels_satisfied(self._detected_labels, step.required_labels)
                detached = ""
                if step.requires_separation and step.separation_container:
                    done, needed = state.separation_progress
                    detached = f" ({done}/{needed} apart)"
                detected.setText(("yes" if satisfied else "no") + detached)
                detected.setBackground(QColor("#1f6f43" if satisfied else "#6f1f1f"))
            self.checklist.setItem(row, 4, detected)

    def _set_checked(self, row: int, checked: bool) -> None:
        item = self.checklist.item(row, 0)
        if item is not None:
            item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)

    def _set_step_status(self, row: int, status: str) -> None:
        item = self.checklist.item(row, 5)
        if item is None:
            item = QTableWidgetItem()
            self.checklist.setItem(row, 5, item)
        item.setText(status)
        colors = {"Pending": "#3a4653", "In Progress": "#8a6d1f", "Completed": "#1f6f43"}
        item.setBackground(QColor(colors.get(status, "#3a4653")))
        item.setForeground(Qt.GlobalColor.white)
        item.setToolTip(status)

    def _reset_experiment(self) -> None:
        self.monitor.configure_experiment(self.experiment_steps)
        self._detected_labels = []
        self._populate_checklist()
        self._set_status("Experiment progress reset.")

    def _refresh_storage(self) -> None:
        self.storage_list.clear()
        folder = user_data_path("storage")
        folder.mkdir(parents=True, exist_ok=True)
        for video in sorted(folder.glob("*.mp4"), key=lambda value: value.stat().st_mtime, reverse=True):
            item = QListWidgetItem(video.name)
            item.setData(Qt.ItemDataRole.UserRole, str(video))
            self.storage_list.addItem(item)

    def _show_storage_metadata(self, item: QListWidgetItem | None, _: QListWidgetItem | None) -> None:
        if not item:
            return
        video = Path(str(item.data(Qt.ItemDataRole.UserRole)))
        metadata_path = video.with_suffix(".json")
        try:
            self.storage_metadata.setPlainText(json.dumps(json.loads(metadata_path.read_text(encoding="utf-8")), indent=2))
        except (OSError, json.JSONDecodeError):
            self.storage_metadata.setPlainText("Metadata is missing or invalid.")
        capture = cv2.VideoCapture(str(video))
        ok, frame = capture.read()
        capture.release()
        if ok and frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888).copy()
            self.storage_preview.setPixmap(QPixmap.fromImage(image).scaled(self.storage_preview.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def _open_selected_video(self) -> None:
        item = self.storage_list.currentItem()
        if item:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(item.data(Qt.ItemDataRole.UserRole))))

    def _restart_monitor(self) -> None:
        """Re-create the monitor from the currently configured asset paths."""

        sources = [(worker.source, camera_id) for camera_id, worker in self.monitor.workers.items()]
        previous = self.monitor
        self.monitor = MultiCameraMonitor(self._default_settings(), EdgeTuning(), [])
        for source, camera_id in sources or [(0, "CAM-0")]:
            self.monitor.add_camera(source, camera_id)
        previous.close()
        self._rebuild_grid()
        self._set_status("Vision restarted. Select an experiment folder to load a model.")
        if self.experiment_folder is not None:
            self._load_experiment_folder(str(self.experiment_folder.path))

    def _set_voice_chat(self, enabled: bool) -> None:
        vosk = self.vosk_path.text().strip()
        wake_word = self.wake_word.text().strip() or "assistant"

        def task() -> None:
            # Without this guard any failure inside set_voice_chat kills the
            # thread silently and the toggle looks permanently dead.
            try:
                report = self.audio.set_voice_chat(enabled, vosk, wake_word)
            except Exception as exc:
                report = f"Voice chat could not start: {type(exc).__name__}: {exc}"
            self.bridge.message.emit(report)

        Thread(target=task, name="voice-toggle", daemon=True).start()

    def _scan_bluetooth(self) -> None:
        def task() -> None:
            self.bridge.devices.emit(asyncio.run(self.bluetooth.discover_async()))
        Thread(target=task, name="ble-scan", daemon=True).start()
        self._set_status("Scanning Bluetooth in the background…")

    def _show_bluetooth_devices(self, devices: list[dict[str, str]]) -> None:
        self.ble_devices.clear()
        for device in devices:
            self.ble_devices.addItem(f"{device['name']}  [{device['address']}]")
        self._set_status(f"Bluetooth scan complete: {len(devices)} device(s). Pair/select an A2DP speaker in Windows for audio routing.")

    def _connect_bluetooth(self) -> None:
        address = self.ble_address.text().strip()
        if not address:
            item = self.ble_devices.currentItem()
            if item and "[" in item.text():
                address = item.text().rsplit("[", 1)[1].rstrip("]")
        if not address:
            self._set_status("Choose a discovered device or enter its Bluetooth address.")
            return
        def task() -> None:
            connected = asyncio.run(self.bluetooth.connect(address))
            self.bridge.message.emit("BLE control connected." if connected else "Could not connect the BLE control channel.")
        Thread(target=task, name="ble-connect", daemon=True).start()

    def _set_status(self, message: str) -> None:
        self.status_label.setText(message)

    def closeEvent(self, event: Any) -> None:
        self.timer.stop()
        self.monitor.close()
        self.audio.close()
        event.accept()


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    window = DesktopApplication()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
