"""Ordered experiment steps, experiment-folder parsing and the step state machine.

An *experiment folder* is chosen by the user and holds:

* the trained Ultralytics YOLO segmentation checkpoint (``.pt``) whose class
  names are the only vocabulary the pipeline understands, and
* the ``.txt`` step file that describes the ordered procedure in plain text.

The parser extracts the object labels each numbered step requires.  During
inference the detected YOLO class names are compared against the labels of the
current step; when they match, the state machine advances and reports the event
so the UI can check the step off and speak it with pyttsx3.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import json
import math
import re
import time


#: The checkpoint expected inside a selected experiment folder. Any ``.pt`` in
#: the folder is accepted, so an ``Ultralytics`` run folder works unchanged.
WEIGHTS_FILENAME = "weights.pt"
WEIGHTS_EXTENSIONS = (".pt",)
PREFERRED_WEIGHTS = ("weights.pt", "best.pt", "last.pt")

#: Preferred names for the step file. Any single ``.txt`` in the same folder is
#: accepted as well, so a user may supply their own step file under any name.
STEPS_FILENAME = "newex.txt"
PREFERRED_STEPS = ("steps.txt", "newex.txt")

#: Files that are never step files (artifacts of a retired planner flow).
IGNORED_STEP_FILES = frozenset({"auto_planner.txt", "auto_planner.json", "prompt.txt"})

#: Verbs that mean "this object must leave that container". A step containing
#: one of these needs the spatial separation check, not mere presence, so an
#: item still sitting inside the box cannot complete the step.
REMOVAL_KEYWORDS = (
    "remove", "removes", "removed", "removing", "take out", "takeout", "taken out",
    "take off", "pull out", "pullout", "pick out", "extract", "take away", "empty",
    "unpack", "withdraw", "retrieve",
)

#: Bounding boxes are ``(x, y, width, height)`` in pixels throughout.
Box = tuple[int, int, int, int]

#: Overlap at or above this IoU means the two boxes are still touching.
DEFAULT_SEPARATION_MAX_IOU = 0.15

#: Overlap at or above this fraction of the *smaller* box means one box is
#: inside the other. A nested box has a low IoU by construction (IoU is divided
#: by the larger area), so containment has to be tested separately or an item
#: sitting inside an open box would be reported as separated.
DEFAULT_SEPARATION_CONTAINMENT = 0.30

#: Inference results the objects must stay apart before a removal step passes.
DEFAULT_SEPARATION_FRAMES = 10

#: Minimum seconds between two ``step_warning`` events, so a wrong action does
#: not produce a burst of spoken warnings.
WARNING_COOLDOWN_SECONDS = 1.5

#: How far ahead the wrong-step check looks. A future step that is satisfied
#: while the current one is not usually means the operator jumped ahead.
WARNING_LOOKAHEAD = 2


def normalize_boxes(boxes: Mapping[str, Sequence[Box]] | None) -> dict[str, list[Box]]:
    """Normalize every key of a ``{label: [box, ...]}`` mapping.

    ``observe()`` is a public entry point, so the mapping may arrive with raw
    checkpoint class names (``black-earbuds``) while the steps store normalized
    labels (``blackearbuds``). Normalizing here makes the separation lookup
    independent of which spelling the caller used.
    """

    if not boxes:
        return {}
    grouped: dict[str, list[Box]] = {}
    for label, detections in boxes.items():
        key = normalize_label(label)
        if not key:
            continue
        grouped.setdefault(key, []).extend(
            tuple(int(value) for value in box[:4]) for box in detections
        )
    return grouped


@dataclass(slots=True)
class SeparationResult:
    """Outcome of one target/container geometry comparison."""

    separated: bool
    overlap_iou: float
    containment: float
    center_distance: float
    reason: str


def box_iou(first: Box, second: Box) -> float:
    """Intersection over union of two ``(x, y, width, height)`` boxes."""

    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[0] + first[2], second[0] + second[2])
    bottom = min(first[1] + first[3], second[1] + second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    if intersection == 0:
        return 0.0
    first_area = first[2] * first[3]
    second_area = second[2] * second[3]
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def box_containment(inner: Box, outer: Box) -> float:
    """Fraction of ``inner`` covered by ``outer``; 1.0 when fully nested."""

    left = max(inner[0], outer[0])
    top = max(inner[1], outer[1])
    right = min(inner[0] + inner[2], outer[0] + outer[2])
    bottom = min(inner[1] + inner[3], outer[1] + outer[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    area = inner[2] * inner[3]
    return intersection / area if area > 0 else 0.0


def check_spatial_separation(
    box1: Box,
    box2: Box,
    max_overlap_iou: float = DEFAULT_SEPARATION_MAX_IOU,
    containment_limit: float = DEFAULT_SEPARATION_CONTAINMENT,
    min_center_distance: float = 0.0,
) -> SeparationResult:
    """Decide whether two detected objects count as physically separated.

    ``box1`` is the item being removed and ``box2`` the container it must leave.
    Three independent signals are combined, all of which must agree that the
    objects are apart:

    * **IoU** - the shared area relative to the union. A low IoU alone is not
      enough: an item nested inside a large box can reach IoU 0.05 while still
      being inside it.
    * **Containment** - the shared area relative to the *smaller* box, which is
      what actually distinguishes "inside the box" from "next to the box".
    * **Center distance** - optional pixel threshold between the two box
      centers, for objects that visually overlap but are handled apart.
    """

    intersection_iou = box_iou(box1, box2)
    smaller = box1 if box1[2] * box1[3] <= box2[2] * box2[3] else box2
    larger = box2 if smaller is box1 else box1
    containment = box_containment(smaller, larger)
    first_center = (box1[0] + box1[2] / 2.0, box1[1] + box1[3] / 2.0)
    second_center = (box2[0] + box2[2] / 2.0, box2[1] + box2[3] / 2.0)
    center_distance = math.hypot(
        first_center[0] - second_center[0], first_center[1] - second_center[1]
    )
    far_apart = min_center_distance > 0 and center_distance >= min_center_distance

    if containment > containment_limit:
        return SeparationResult(False, intersection_iou, containment, center_distance, "inside container")
    if intersection_iou > max_overlap_iou and not far_apart:
        return SeparationResult(False, intersection_iou, containment, center_distance, "boxes still overlap")
    return SeparationResult(True, intersection_iou, containment, center_distance, "separated")


#: Head nouns that make a word a usable object label.
OBJECT_NOUNS = frozenset(
    {
        "box", "boxes", "container", "bottle", "cup", "tool", "hand", "hands",
        "phone", "book", "plate", "object", "earbud", "earbuds", "bud", "buds",
        "case", "charger", "cable", "pouch", "lid", "wrapper", "packet", "sleeve",
        "tag", "label", "sticker", "card", "device", "headphone", "buds",
    }
)

#: Verbs that terminate a label phrase; nothing after them belongs to the noun.
ACTION_WORDS = frozenset(
    {
        "remove", "removes", "removed", "removing", "take", "takes", "took",
        "taking", "pick", "picks", "picked", "picking", "place", "places",
        "placed", "placing", "put", "puts", "putting", "open", "opens", "opened",
        "close", "closes", "closed", "hold", "holds", "held", "use", "used",
        "using", "separate", "separates", "separated", "move", "moves", "moved",
        "add", "adds", "added", "cover", "covers", "covered", "show", "shows",
        "shown", "showed", "display", "displays", "displayed", "insert",
        "drop", "drops", "dropped", "lift", "lifts", "lifted", "empty", "empties",
        "emptied", "start", "starts", "started", "begin", "begins", "finish",
        "finishes", "do", "does",
    }
)

#: Words that carry no object identity and are dropped from a candidate phrase.
STRUCTURAL_WORDS = frozenset(
    {
        "the", "a", "an", "of", "and", "then", "that", "this", "these", "there",
        "it", "its", "is", "are", "was", "were", "be", "been", "as", "in",
        "into", "inside", "onto", "on", "over", "from", "with", "without",
        "to", "out", "off", "at", "by", "for", "or", "infront", "front",
        "respectively", "another", "other", "each", "one", "two", "three",
        "color", "colour", "colored", "coloured", "main", "big", "large",
        "small", "little", "tiny", "huge", "camera", "frame", "view", "in",
        "shown", "camera", "visible", "seen", "appear", "appears", "first",
        "second", "last", "next", "after", "before", "finally", "order", "ordered",
    }
)

#: Maximum number of words an extracted object phrase may span.
MAX_PHRASE_WORDS = 4


@dataclass(slots=True)
class ExperimentStep:
    number: int
    instruction: str
    required_labels: list[str] = field(default_factory=list)
    #: True when the instruction asks for an object to leave a container, in
    #: which case presence alone is not enough to pass the step.
    requires_separation: bool = False
    #: Normalized label of the object that must move, and of the container it
    #: must be separated from. Empty when the step has no container.
    separation_target: str = ""
    separation_container: str = ""

    @property
    def summary(self) -> str:
        if not self.required_labels:
            return "no object label"
        if self.requires_separation and self.separation_container:
            return f"{self.separation_target} out of {self.separation_container}"
        return ", ".join(self.required_labels)


def requires_removal(instruction: str) -> str:
    """Return the removal keyword found in an instruction, or ``""``."""

    line = f" {str(instruction).casefold()} "
    return next((keyword for keyword in REMOVAL_KEYWORDS if keyword in line), "")


def assign_separation(step: ExperimentStep) -> ExperimentStep:
    """Mark a removal step and pick its target and container labels.

    The first required label is the object that moves and the second is the
    container it is taken out of, which follows the way these instructions are
    written ("remove white earbuds case from the blackbox" -> target
    ``whiteearbuds``, container ``blackbox``). A removal step that names only
    one object has no container to measure against and falls back to presence.
    """

    if not requires_removal(step.instruction) or len(step.required_labels) < 2:
        return step
    step.requires_separation = True
    step.separation_target = step.required_labels[0]
    step.separation_container = step.required_labels[1]
    return step


@dataclass(slots=True)
class StepEvidence:
    step_number: int
    instruction: str
    required_labels: list[str] = field(default_factory=list)
    matched_at: str | None = None
    labels: list[str] = field(default_factory=list)
    source: str | None = None


@dataclass(slots=True)
class ExperimentFolder:
    """A validated experiment folder and the steps parsed out of its ``.txt`` file."""

    path: Path
    weights_path: Path
    steps_path: Path
    steps: list[ExperimentStep]
    description: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def labels(self) -> list[str]:
        """Every distinct label any step requires."""

        return list(dict.fromkeys(label for step in self.steps for label in step.required_labels))


def _words(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", str(text).casefold())


def normalize_label(value: str) -> str:
    """Lowercase alphanumeric form (``black-earbuds`` → ``blackearbuds``)."""

    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def _singular(word: str) -> str:
    if word.endswith("ies") and len(word) > 4:
        return f"{word[:-3]}y"
    if word.endswith("es") and word[:-2].endswith(("s", "x", "z", "ch", "sh")):
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


def label_matches(detected: str, required: str) -> bool:
    """Return whether a YOLO class name satisfies a label required by a step.

    ``required_labels`` always holds normalized forms, so the comparison is a
    set membership test.  Normalizing both sides absorbs the spelling drift
    between the checkpoint and the instruction (``black-earbuds`` /
    ``black earbuds`` / ``blackearbuds`` all collapse to one key).
    """

    return bool(normalize_label(detected)) and normalize_label(detected) == normalize_label(required)


def required_labels_satisfied(detected: Iterable[str], required: Sequence[str]) -> bool:
    """Return whether every required label of a step is present in this frame."""

    if not required:
        return False
    observed = {normalize_label(label) for label in detected if str(label).strip()}
    return all(normalize_label(need) in observed for need in required if str(need).strip())


def classes_in_instruction(instruction: str, class_names: Sequence[str]) -> list[str]:
    """Normalized YOLO class names that occur inside one instruction line.

    This is the universal parser rule: no ``Step 1 | required:`` syntax is
    needed.  Both the instruction and every class name are normalized to
    lowercase alphanumerics (spaces, hyphens and underscores removed), and a
    class counts as required when its normalized form is a substring of the
    normalized instruction.  ``'remove white earbuds case from the blackbox'``
    with classes ``['blackbox', 'white-earbuds']`` therefore yields
    ``['whiteearbuds', 'blackbox']``, ordered by where each name appears in the
    sentence rather than by the order of the class list.
    """

    line = normalize_label(instruction)
    if not line:
        return []
    found: list[tuple[int, str]] = []
    seen: set[str] = set()
    for name in class_names:
        key = normalize_label(name)
        if not key or key in seen:
            continue
        position = line.find(key)
        if position >= 0:
            seen.add(key)
            found.append((position, key))
    return [key for _, key in sorted(found, key=lambda item: (item[0], item[1]))]


def _candidate_phrases(instruction: str) -> list[str]:
    """Fallback label extraction when no checkpoint class is named in a step.

    Keeps the checklist meaningful for a step the current checkpoint cannot
    express; such a step is reported as a warning because it can never advance.
    """

    tokens = _words(instruction)
    phrases: list[str] = []
    for start in range(len(tokens)):
        if tokens[start] in ACTION_WORDS:
            continue
        collected: list[str] = []
        for word in tokens[start : start + MAX_PHRASE_WORDS]:
            if word in ACTION_WORDS:
                break
            if word not in STRUCTURAL_WORDS:
                collected.append(word)
            if not collected:
                continue
            if any(_singular(item) in OBJECT_NOUNS for item in collected):
                phrases.append(" ".join(collected))
    if not phrases:
        return []
    # Keep only maximal phrases so "black box" wins over its own "box" fragment.
    unique = list(dict.fromkeys(phrases))
    maximal = [
        phrase
        for phrase in unique
        if not any(other != phrase and _is_subphrase(phrase, other) for other in unique)
    ]
    return [normalize_label(phrase) for phrase in maximal]


def _is_subphrase(phrase: str, other: str) -> bool:
    phrase_tokens = set(_words(phrase))
    other_tokens = set(_words(other))
    return bool(phrase_tokens) and phrase_tokens < other_tokens


def parse_newex(text: str, filename: str = STEPS_FILENAME, class_names: Sequence[str] = ()) -> tuple[list[ExperimentStep], str]:
    """Parse the step file into ordered steps plus their required object labels.

    ``class_names`` is the checkpoint's ``model.names``.  A step's required
    labels are the normalized class names that appear inside the instruction
    (see :func:`classes_in_instruction`), so plain English needs no
    ``Step 1 | required: ...`` marker.  Numbered lines are the steps, any other
    non-empty line is the description, and an ``objects: a, b`` line stays
    supported as an explicit declaration.
    """

    source = text.decode("utf-8-sig") if isinstance(text, bytes) else text
    numbered = re.compile(r"^\s*(?:step\s*)?\d+\s*[.)\-:]\s*", re.I)
    declaration = re.compile(r"^\s*(?:objects?|classes?|labels?)\s*:\s*(.+)$", re.I)
    steps: list[ExperimentStep] = []
    description_lines: list[str] = []
    declared: list[str] = []
    vocabulary = [str(name) for name in class_names]
    for raw_line in source.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = declaration.match(line)
        if match:
            declared.extend(item.strip() for item in re.split(r"[,;]", match.group(1)) if item.strip())
            continue
        if not numbered.match(line):
            description_lines.append(line)
            continue
        instruction = numbered.sub("", line).strip()
        if not instruction:
            continue
        # Vocabulary first: a named class is the most reliable requirement.
        labels = classes_in_instruction(instruction, vocabulary)
        if not labels:
            labels = [
                normalize_label(name)
                for name in declared
                if normalize_label(name) in normalize_label(instruction)
            ]
        steps.append(
            assign_separation(
                ExperimentStep(len(steps) + 1, instruction, labels or _candidate_phrases(instruction))
            )
        )
    if not steps:
        raise ValueError(
            f"No numbered steps were found in {filename}. Each step must start with '1.', '2.', ..."
        )
    return steps, "\n".join(description_lines)


def find_weights_file(folder: Path) -> Path:
    """Locate the ``.pt`` checkpoint inside a user supplied experiment folder.

    Nothing about the model is hard-coded: any ``*.pt`` in the folder works.
    ``weights.pt``, ``best.pt`` and ``last.pt`` win when present, otherwise a
    single candidate is used and several candidates are reported as ambiguous.
    """

    for name in PREFERRED_WEIGHTS:
        candidate = folder / name
        if candidate.is_file():
            return candidate
    candidates = sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.casefold() in WEIGHTS_EXTENSIONS
    )
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"No model file was found in '{folder.name}'. Place a YOLO "
            f"'.{WEIGHTS_EXTENSIONS[0][1:]}' checkpoint in the experiment folder."
        )
    names = ", ".join(path.name for path in candidates)
    raise FileNotFoundError(
        f"'{folder.name}' holds several model files ({names}). Keep only the checkpoint "
        f"for this experiment."
    )


def find_step_file(folder: Path) -> Path:
    """Locate the step file inside a user supplied experiment folder.

    ``steps.txt`` wins, then ``newex.txt``.  Otherwise the folder's only ``.txt``
    is used, so any step file name the user chooses works.  Several candidates
    are ambiguous and are reported instead of being guessed.
    """

    for name in PREFERRED_STEPS:
        candidate = folder / name
        if candidate.is_file():
            return candidate
    candidates = sorted(
        path
        for path in folder.glob("*.txt")
        if path.is_file() and path.name.casefold() not in IGNORED_STEP_FILES
    )
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"No step file was found in '{folder.name}'. Place {PREFERRED_STEPS[0]} "
            f"(or a single .txt file) next to the model."
        )
    names = ", ".join(path.name for path in candidates)
    raise FileNotFoundError(
        f"'{folder.name}' holds several .txt files ({names}). Rename the step file to "
        f"{PREFERRED_STEPS[0]} so it can be identified."
    )


def load_experiment_folder(folder: str | Path, class_names: Sequence[str] = ()) -> ExperimentFolder:
    """Validate a user selected experiment folder and parse its steps.

    The folder must contain a ``.pt`` checkpoint and a step ``.txt`` file.
    ``class_names`` is the vocabulary of the checkpoint; labels no checkpoint
    class can produce are reported as warnings so they are visible in the UI
    instead of silently blocking a step forever.
    """

    path = Path(folder).expanduser()
    if not path.is_dir():
        raise FileNotFoundError(f"Experiment folder not found: {path}")
    weights_path = find_weights_file(path)
    steps_path = find_step_file(path)
    steps, description = parse_newex(
        steps_path.read_text(encoding="utf-8-sig"), steps_path.name, class_names
    )
    warnings: list[str] = []
    vocabulary = {normalize_label(name) for name in class_names}
    for step in steps:
        for label in step.required_labels:
            if vocabulary and label not in vocabulary:
                warnings.append(
                    f"Step {step.number}: '{label}' is not a class of this checkpoint "
                    f"(classes: {', '.join(str(name) for name in class_names) or 'none'})."
                )
    return ExperimentFolder(path, weights_path, steps_path, steps, description, warnings)


class ActivityStateMachine:
    """Ordered verifier driven by detected YOLO classes and object geometry.

    A plain step passes as soon as every label it requires is present in one
    frame (``match_frames=1``). A removal step additionally needs its target and
    container to be *spatially separated* for ``separation_frames`` consecutive
    inference results, which is what stops an item still sitting inside the box
    from completing the step.
    """

    def __init__(
        self,
        steps: list[ExperimentStep] | None = None,
        match_frames: int = 1,
        separation_frames: int = DEFAULT_SEPARATION_FRAMES,
        separation_max_iou: float = DEFAULT_SEPARATION_MAX_IOU,
        separation_containment: float = DEFAULT_SEPARATION_CONTAINMENT,
    ) -> None:
        self.steps = list(steps or [])
        self.match_frames = max(1, int(match_frames))
        self.separation_frames = max(1, int(separation_frames))
        self.separation_max_iou = float(separation_max_iou)
        self.separation_containment = float(separation_containment)
        self.index = 0
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.evidence = [
            StepEvidence(step.number, step.instruction, list(step.required_labels)) for step in self.steps
        ]
        self._streak = 0
        self._separation_streak = 0
        self.last_separation: SeparationResult | None = None
        self._last_warning_at = 0.0

    @property
    def current_step(self) -> ExperimentStep | None:
        return self.steps[self.index] if 0 <= self.index < len(self.steps) else None

    @property
    def required_labels(self) -> list[str]:
        step = self.current_step
        return list(step.required_labels) if step else []

    @property
    def separation_progress(self) -> tuple[int, int]:
        """Consecutive separated results so far, and the total required."""

        return self._separation_streak, self.separation_frames

    @property
    def complete(self) -> bool:
        return bool(self.steps) and self.index >= len(self.steps)

    def reset(self) -> None:
        self.index = 0
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.evidence = [
            StepEvidence(step.number, step.instruction, list(step.required_labels)) for step in self.steps
        ]
        self._streak = 0
        self._separation_streak = 0
        self.last_separation = None
        self._last_warning_at = 0.0

    def observe(
        self,
        labels: Iterable[str],
        source: str = "vision",
        boxes: Mapping[str, Sequence[Box]] | None = None,
    ) -> dict[str, Any] | None:
        """Feed one frame's class names (and their boxes) and advance at most one step.

        Labels are normalized first, so a detection of ``black-earbuds`` always
        satisfies a step that requires ``blackearbuds``. The step passes as
        soon as *all* of its required labels are present in the same frame (or
        for ``match_frames`` consecutive frames); a removal step must also hold
        the separation condition. When the current step is *not* satisfied but a
        later one is, a rate-limited ``step_warning`` is returned instead.

        The returned event carries ``next_instruction`` so the voice can
        announce it without the caller re-reading the instruction list.
        """

        step = self.current_step
        if step is None:
            return None
        detected = sorted({normalize_label(label) for label in labels if str(label).strip()})
        normalized_boxes = normalize_boxes(boxes)
        current_satisfied = required_labels_satisfied(detected, step.required_labels)
        if not current_satisfied:
            # Checked before anything else, on every single frame, so an out of
            # order action is reported immediately instead of waiting for the
            # separation streak of the current step.
            self._streak = 0
            self._separation_streak = 0
            self.last_separation = None
            return self._wrong_step_warning(detected)
        separation = self._evaluate_separation(step, normalized_boxes)
        if step.requires_separation and step.separation_container and separation is None:
            self._separation_streak = 0
            return self._wrong_step_warning(detected)
        if separation is not None and not separation.separated:
            # The objects are present but the removal has not happened yet. If a
            # later step is already completely satisfied the operator is working
            # ahead, so warn immediately instead of waiting for the streak. Only
            # the strict test is used here, because the current step's own labels
            # are present and a partial overlap would be ambiguous.
            self._separation_streak = 0
            self.last_separation = separation
            return self._wrong_step_warning(detected)
        if step.requires_separation and step.separation_container:
            self._separation_streak += 1
            if separation is not None:
                self.last_separation = separation
            if self._separation_streak < self.separation_frames:
                return None
            self._separation_streak = 0
        self._streak += 1
        if self._streak < self.match_frames:
            return None
        self._streak = 0
        evidence = self.evidence[self.index]
        evidence.matched_at = datetime.now(timezone.utc).isoformat()
        evidence.labels = detected
        evidence.source = source
        self.index += 1
        upcoming = self.current_step
        if self.complete:
            return {
                "event": "experiment_complete",
                "message": "Experiment Completed Successfully",
                "step": evidence.step_number,
                "instruction": evidence.instruction,
                "required_labels": list(step.required_labels),
                "labels": detected,
                "next_step": None,
                "next_instruction": None,
                "next_required_labels": [],
                "complete": True,
                "total_steps": len(self.steps),
            }
        event: dict[str, Any] = {
            "event": "step_passed",
            "message": f"Step {evidence.step_number} Completed!",
            "step": evidence.step_number,
            "instruction": evidence.instruction,
            "required_labels": list(step.required_labels),
            "labels": detected,
            "separation_required": step.requires_separation,
            "separation_target": step.separation_target,
            "separation_container": step.separation_container,
            "separation": asdict(self.last_separation) if self.last_separation else None,
            "next_step": upcoming.number,
            "next_instruction": upcoming.instruction,
            "next_required_labels": list(upcoming.required_labels),
            "next_requires_separation": upcoming.requires_separation,
            "complete": False,
            "total_steps": len(self.steps),
        }
        self.last_separation = None
        return event

    def _wrong_step_warning(self, detected: list[str], strict: bool | None = None) -> dict[str, Any] | None:
        """Report work belonging to a later step while the current one is pending.

        Only called when the current step is *not* satisfied, or is stuck waiting
        for separation. A later step counts as evidence when it is completely
        satisfied, or when it needs an object the current step never mentions -
        that is a genuine jump ahead rather than a shared object.
        """

        current = self.current_step
        if current is None or not detected:
            return None
        present = set(detected)
        missing_current = [need for need in current.required_labels if need not in present]
        current_needs = set(current.required_labels)
        for future in self.steps[self.index + 1 : self.index + 1 + WARNING_LOOKAHEAD]:
            if not future.required_labels:
                continue
            future_present = [need for need in future.required_labels if need in present]
            if not future_present:
                continue
            exclusive = [need for need in future_present if need not in current_needs]
            if strict is False:
                if not exclusive:
                    continue
            elif len(future_present) != len(future.required_labels):
                continue
            now = time.monotonic()
            if now - self._last_warning_at < WARNING_COOLDOWN_SECONDS:
                return None
            self._last_warning_at = now
            return {
                "event": "step_warning",
                "message": "Warning: Wrong step performed! Please complete current step first.",
                "expected_step": current.number,
                "expected_instruction": current.instruction,
                "expected_labels": list(current.required_labels),
                "missing_labels": missing_current,
                "observed_step": future.number,
                "observed_instruction": future.instruction,
                "observed_labels": future_present,
                "labels": detected,
                "complete": False,
            }
        return None

    def _evaluate_separation(
        self, step: ExperimentStep, boxes: Mapping[str, Sequence[Box]]
    ) -> SeparationResult | None:
        """Compare the target against every container detection in this frame.

        Both sides of the lookup are normalized, so ``black-earbuds`` in the
        frame and ``blackearbuds`` in the step resolve to the same key. Returns
        ``None`` when the step needs geometry but the caller supplied no boxes,
        or when either object is absent, so an unavailable measurement can never
        be mistaken for a pass.
        """

        if not step.requires_separation or not step.separation_container:
            return None
        target_boxes = normalize_boxes(boxes).get(normalize_label(step.separation_target)) or []
        container_boxes = normalize_boxes(boxes).get(normalize_label(step.separation_container)) or []
        if not target_boxes or not container_boxes:
            return None
        results = [
            check_spatial_separation(
                target,
                container,
                self.separation_max_iou,
                self.separation_containment,
            )
            for target in target_boxes
            for container in container_boxes
        ]
        # The closest pair decides: the step only passes when the object is
        # clear of every container box currently on screen.
        return min(results, key=lambda item: (item.containment, item.overlap_iou))
    def snapshot(self) -> dict[str, Any]:
        return {
            "status": "pass" if self.complete else ("not_configured" if not self.steps else "in_progress"),
            "started_at": self.started_at,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "next_expected_step": None if self.complete else (self.current_step.instruction if self.current_step else None),
            "next_expected_labels": self.required_labels,
            "steps": [asdict(item) for item in self.evidence],
        }

    def write_log(self, output_path: str | Path, video_path: str | Path | None = None, **metadata: Any) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.snapshot()
        payload.update(metadata)
        if video_path:
            payload["video_file"] = Path(video_path).name
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path
