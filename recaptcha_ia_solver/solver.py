# Standard imports
import os
import re
import shutil
from io import BytesIO
from time import monotonic, sleep
from typing import Iterable, Optional, Set

# Third-party imports
import numpy as np
import requests
from PIL import Image
from ultralytics import YOLO
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.action_chains import ActionChains

# Primary recognizer: a vision-language model that answers "which cells contain
# the target?" directly from the composite (see recognizer.py for why a VLM
# beats the single-label classifier on cluttered multi-object reCAPTCHA tiles).
from recaptcha_ia_solver import recognizer

# Primary model: fine-tuned classifier (scripts/train_classifier.py) trained
# on the merged verytuffcat + DannyLuna reCAPTCHA datasets (~57k images).
# Covers the 14 cell categories reCAPTCHA most often shows: bicycle, bridge,
# bus, car, chimney, crosswalk, hydrant, motorcycle, mountain, other, palm,
# stair, tractor, traffic light. Override with RECAPTCHA_YOLO_MODEL env var.
DEFAULT_YOLO_MODEL = "models/recaptcha_classifier.pt"

# Fallback detector: Open Images V7-pretrained YOLOv8x. Auto-loaded when the
# primary model has no class match for the current challenge phrase, so terms
# the classifier wasn't trained on (boat, truck, taxi, parking meter, stop
# sign, train, tower, vehicle) still resolve. Override with
# RECAPTCHA_YOLO_FALLBACK; set to empty string to disable fallback. Stored
# under models/ so a project checkout that already has the file (or a fresh
# ultralytics auto-download) doesn't dump a 130MB blob in the repo root.
DEFAULT_YOLO_FALLBACK_MODEL = "models/yolov8x-oiv7.pt"

# reCAPTCHA challenge term -> Open Images V7 class names. Multi-class targets
# (e.g. "vehicle") map to several classes; absent terms (bridge/chimney/
# crosswalk/mountain/tractor) yield an empty set in stock OIV7 and trigger a
# reload until a fine-tuned model is plugged in. Ordered longest-first so
# `re.search` honors compound terms before their substrings.
RECAPTCHA_TO_OIV7 = {
    # ── Korean challenge terms (Google renders the reCAPTCHA in the account's
    # locale; Korean accounts get 자동차/버스/etc.). Values include both the
    # primary classifier's lowercase class names and the OIV7 capitalized ones
    # so whichever model is loaded resolves. Ordered longest-first like the
    # English entries below so compound terms win before substrings. ──
    "오토바이": ["motorcycle", "Motorcycle"],
    "횡단보도": ["crosswalk", "Crosswalk"],
    "소화전": ["hydrant", "Fire hydrant"],
    "신호등": ["traffic light", "Traffic light"],
    "자전거": ["bicycle", "Bicycle"],
    "자동차": ["car", "Car"],
    "트랙터": ["tractor", "Tractor"],
    "야자수": ["palm", "Palm tree"],
    "소방전": ["hydrant", "Fire hydrant"],
    "택시": ["Taxi"],
    "트럭": ["Truck"],
    "버스": ["bus", "Bus"],
    "굴뚝": ["chimney", "Chimney"],
    "보트": ["Boat"],
    "계단": ["stair", "Stairs", "stairs"],
    "다리": ["bridge", "Bridge"],
    "교각": ["bridge", "Bridge"],
    "타워": ["Tower"],
    "기차": ["Train"],
    "열차": ["Train"],
    "산": ["mountain", "Mountain"],
    "fire hydrant": ["Fire hydrant", "hydrant"],
    "parking meter": ["Parking meter"],
    "traffic light": ["Traffic light", "traffic light"],
    "palm tree": ["Palm tree", "palm"],
    "stop sign": ["Stop sign"],
    "motorcycle": ["Motorcycle", "motorcycle"],
    "bicycle": ["Bicycle", "bicycle"],
    "vehicle": [
        "Car",
        "Bus",
        "Truck",
        "Motorcycle",
        "Taxi",
        "Vehicle",
        "Land vehicle",
        "car",
        "bus",
        "motorcycle",
    ],
    "hydrant": ["Fire hydrant", "hydrant"],
    "stair": ["Stairs", "stair"],
    "tower": ["Tower"],
    "train": ["Train"],
    "truck": ["Truck"],
    "boat": ["Boat"],
    "taxi": ["Taxi"],
    "car": ["Car", "car"],
    "bus": ["Bus", "bus"],
    "bridge": ["Bridge", "bridge"],
    "chimney": ["Chimney", "chimney"],
    "crosswalk": ["Crosswalk", "crosswalk"],
    "mountain": ["Mountain", "mountain"],
    "tractor": ["Tractor"],
}


def _resolve_model_path(path: str) -> str:
    """
    Best-effort path resolution: if `path` is relative and missing from CWD,
    try resolving it against the project root (two levels up from this file).
    Returns the original `path` unchanged if neither candidate exists, so
    Ultralytics' weight-name shortcut (e.g., bare "yolov8x-oiv7.pt" → auto
    download) still works.
    """
    if not path or os.path.isabs(path) or os.path.exists(path):
        return path
    # repo root = two levels up from this file (pkg/solver.py -> pkg -> repo)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    alt = os.path.join(project_root, path)
    return alt if os.path.exists(alt) else path


def _try_load_yolo(path: str, verbose: bool = False) -> Optional[YOLO]:
    """Load a YOLO model; return None if loading fails (e.g., file not found)."""
    if not path:
        return None
    try:
        return YOLO(_resolve_model_path(path))
    except Exception as exc:
        if verbose:
            print(f"failed to load {path}: {exc}")
        return None


def _model_class_index(model: YOLO) -> dict:
    """Return a {lowercased class name -> class id} index for the loaded model."""
    raw = getattr(model, "names", {}) or {}
    if isinstance(raw, dict):
        items = raw.items()
    else:
        items = enumerate(raw)
    return {str(name).strip().lower(): int(idx) for idx, name in items}


def _resolve_target_classes(target_text: str, model: YOLO) -> Set[int]:
    """
    Map a reCAPTCHA challenge phrase to the set of class IDs the loaded model
    should detect. Returns an empty set when nothing matches — callers treat
    that as "skip and reload".
    """
    if not target_text:
        return set()
    haystack = target_text.lower()
    name_to_id = _model_class_index(model)
    resolved: Set[int] = set()
    for term, class_names in RECAPTCHA_TO_OIV7.items():
        if not re.search(rf"\b{re.escape(term)}", haystack):
            continue
        for class_name in class_names:
            cid = name_to_id.get(class_name.lower())
            if cid is not None:
                resolved.add(cid)
        if resolved:
            break
    return resolved

def find_between(s, first, last):
    """
    Find a substring between two substrings.
    :param s: string to search.
    :param first: first substring.
    :param last: last substring.
    """
    try:
        start = s.index(first) + len(first)
        end = s.index(last, start)
        return s[start:end]
    except ValueError:
        return ""


def random_delay(mu=0.3, sigma=0.1):
    """
    Random delay to simulate human behavior.
    :param mu: mean of normal distribution.
    :param sigma: standard deviation of normal distribution.
    """
    delay = np.random.normal(mu, sigma)
    delay = max(0.1, delay)
    sleep(delay)


def go_to_recaptcha_iframe1(driver):
    """
    Go to the first recaptcha iframe. (CheckBox)

    The iframe's src always contains "/recaptcha/api2/anchor" regardless of
    page locale, so matching on src is more robust than @title (which Google
    localizes to e.g. "reCAPTCHA" in English, "리캡차" / similar in Korean).
    """
    driver.switch_to.default_content()
    recaptcha_iframe1 = WebDriverWait(driver=driver, timeout=20).until(
        EC.presence_of_element_located(
            (By.XPATH, '//iframe[contains(@src, "/recaptcha/api2/anchor") or contains(@src, "/recaptcha/enterprise/anchor")]')
        )
    )
    driver.switch_to.frame(recaptcha_iframe1)


def go_to_recaptcha_iframe2(driver):
    """
    Go to the second recaptcha iframe. (Images)

    The challenge iframe's src always contains "/recaptcha/api2/bframe"
    regardless of page locale; @title is localized (English: "...challenge...",
    Korean: "...챌린지..."), so we match on src instead.
    """
    driver.switch_to.default_content()
    recaptcha_iframe2 = WebDriverWait(driver=driver, timeout=20).until(
        EC.presence_of_element_located(
            (By.XPATH, '//iframe[contains(@src, "/recaptcha/api2/bframe") or contains(@src, "/recaptcha/enterprise/bframe")]')
        )
    )
    driver.switch_to.frame(recaptcha_iframe2)


def classify_grid_cells(target_set: Iterable[int], grid_n: int, verbose, model) -> list:
    """
    Per-cell classification path used when the loaded YOLO model is a
    classifier (e.g. fine-tuned on `verytuffcat/recaptcha-dataset`).

    Slices `recaptcha_images/0.png` into `grid_n x grid_n` tiles, runs
    classification on each tile, and returns the 1-indexed cells whose top-1
    class is in `target_set`. Predictions below `RECAPTCHA_YOLO_MIN_CONF` are
    discarded so a borderline classifier guess never costs us a false click.
    """
    target_set = set(int(x) for x in target_set)
    try:
        min_conf = float(os.environ.get("RECAPTCHA_YOLO_MIN_CONF", "0.35"))
    except ValueError:
        min_conf = 0.35

    image = Image.open("recaptcha_images/0.png").convert("RGB")
    arr = np.asarray(image)
    height, width = arr.shape[:2]
    cell_h = height / grid_n
    cell_w = width / grid_n

    cells = []
    for r in range(grid_n):
        for c in range(grid_n):
            y1, y2 = int(round(r * cell_h)), int(round((r + 1) * cell_h))
            x1, x2 = int(round(c * cell_w)), int(round((c + 1) * cell_w))
            cells.append(arr[y1:y2, x1:x2])

    results = model.predict(cells, task="classify", verbose=verbose)
    answers = []
    cell_report = []
    for idx, res in enumerate(results):
        probs = getattr(res, "probs", None)
        if probs is None:
            cell_report.append(f"{idx + 1}:none")
            continue
        top1 = int(getattr(probs, "top1", -1))
        top1_conf = float(getattr(probs, "top1conf", 1.0) or 1.0)
        cell_report.append(f"{idx + 1}:cls{top1}@{top1_conf:.2f}")
        if top1 not in target_set:
            continue
        if top1_conf < min_conf:
            continue
        answers.append(idx + 1)
    if verbose:
        print(
            f"classify_grid_cells: target_set={sorted(target_set)} "
            f"min_conf={min_conf} cells=[{' '.join(cell_report)}] -> answers={answers}"
        )
    return answers


def get_target_classes(driver, model: YOLO, verbose: bool = False) -> Set[int]:
    """
    Inspect the reCAPTCHA challenge title and return the set of class IDs the
    detector should look for. An empty set signals "no supported category in
    this challenge — reload."
    """
    target = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located(
            (By.XPATH, '//div[@id="rc-imageselect"]//strong')
        )
    )
    target_text = target.text or ""
    resolved = _resolve_target_classes(target_text, model)
    if verbose:
        print(f"challenge target={target_text!r} -> class ids {sorted(resolved)}")
    return resolved


def _detect_conf() -> float:
    """Confidence floor for the detection models. reCAPTCHA serves small,
    heavily-compressed tiles, so a stock detector's boxes on a real target
    often land at 0.15-0.30 — well under the ultralytics 0.25 default, which
    silently dropped them. Tunable via RECAPTCHA_YOLO_DETECT_CONF."""
    try:
        return float(os.environ.get("RECAPTCHA_YOLO_DETECT_CONF", "0.15"))
    except ValueError:
        return 0.15


def dynamic_and_selection_solver(target_set: Iterable[int], verbose, model):
    """
    Detection-model path for a 3x3 grid: run the detector on the whole grid
    image and return the 1-indexed cells whose center a target-class box falls
    in. The 3x3 "select all images" tiles are independent photos, so a box
    maps to exactly one cell (its center) — overlap-mapping would leak false
    positives across tile seams. Cell size is derived from the actual image
    dimensions rather than a hard-coded 100px so non-300x300 grids still work.
    :param target_set: iterable of YOLO class IDs that satisfy the challenge.
    :param verbose: print verbose.
    """
    target_set = set(int(x) for x in target_set)

    image = np.asarray(Image.open("recaptcha_images/0.png").convert("RGB"))
    height, width = image.shape[:2]
    cell_h, cell_w = height / 3.0, width / 3.0
    result = model.predict(
        image, task="detect", verbose=verbose, conf=_detect_conf()
    )

    answers = set()
    hits = []
    for box in result[0].boxes:
        cls_id = int(box.cls)
        if cls_id not in target_set:
            continue
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
        xc, yc = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        row = min(2, max(0, int(yc // cell_h)))
        col = min(2, max(0, int(xc // cell_w)))
        answer = row * 3 + col + 1
        answers.add(answer)
        hits.append(f"cls{cls_id}@{float(box.conf):.2f}->cell{answer}")
    if verbose:
        print(
            f"dynamic_and_selection_solver: grid={width}x{height} "
            f"target_set={sorted(target_set)} hits=[{' '.join(hits)}] "
            f"-> answers={sorted(answers)}"
        )
    return sorted(answers)


def get_all_captcha_img_urls(driver):
    """
    Get all the image urls from the recaptcha.
    """
    try:
        urls = driver.execute_script(
            "return Array.from(document.querySelectorAll("
            "'#rc-imageselect-target img')).map(function (img) { "
            "return img.src || ''; });"
        )
        if isinstance(urls, list) and urls:
            return [str(url or "") for url in urls]
    except Exception:
        # Older/fake drivers may not expose execute_script in this frame.
        # Keep the original element path as a compatibility fallback.
        pass

    images = WebDriverWait(driver, 10).until(
        EC.presence_of_all_elements_located(
            (By.XPATH, '//div[@id="rc-imageselect-target"]//img')
        )
    )

    img_urls = []
    for img in images:
        img_urls.append(img.get_attribute("src"))

    return img_urls


def download_img(name, url):
    """
    Download the image.
    :param name: name of the image.
    :param url: url of the image.
    """

    response = requests.get(url, stream=True, timeout=15)
    with open(f"recaptcha_images/{name}.png", "wb") as out_file:
        shutil.copyfileobj(response.raw, out_file)
    del response


def _write_dynamic_grid_image(images, grid_n=3, out_path="recaptcha_images/0.png"):
    """Write a single composite grid from individual dynamic reCAPTCHA tiles."""
    expected = grid_n * grid_n
    if len(images) < expected:
        raise ValueError(f"expected at least {expected} tiles, got {len(images)}")
    tiles = [image.convert("RGB") for image in images[:expected]]
    tile_w, tile_h = tiles[0].size
    canvas = Image.new("RGB", (tile_w * grid_n, tile_h * grid_n))
    for idx, tile in enumerate(tiles):
        if tile.size != (tile_w, tile_h):
            tile = tile.resize((tile_w, tile_h))
        row, col = divmod(idx, grid_n)
        canvas.paste(tile, (col * tile_w, row * tile_h))
    canvas.save(out_path)


def _use_screenshot_compose():
    """Compose the dynamic grid from a live element screenshot rather than
    re-fetching per-cell URLs. A/B toggle; default off = legacy URL compose.

    Dynamic 3x3 shares ONE composite URL across the un-clicked cells (measured
    live: distinct=4 dominates), so the URL-fetch composite duplicates a tile
    into several positions and the VLM over-selects the repeats — which is why
    the legacy path then has to bail on distinct < n*n. Screenshotting the
    rendered grid is correct regardless of URL distinctness.
    """
    return os.environ.get("RECAPTCHA_DYNAMIC_SCREENSHOT", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _screenshot_grid_to_png(driver, out_path="recaptcha_images/0.png"):
    """Capture the rendered challenge grid as one image (what the user sees).

    Sidesteps URL semantics entirely: shared composite URLs still render as
    distinct tiles on screen, exactly the reason the static composite path
    works. Returns True on success, False so the caller can fall back to the
    URL-fetch composite if the element screenshot is unavailable.
    """
    try:
        target = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "rc-imageselect-target"))
        )
        png = target.screenshot_as_png
        Image.open(BytesIO(png)).convert("RGB").save(out_path)
        return True
    except Exception:
        return False


def download_dynamic_grid_img(img_urls, grid_n=3, driver=None):
    """Download the current 3x3 challenge as recaptcha_images/0.png.

    Static 3x3 challenges expose one composite URL repeated nine times.
    Dynamic challenges expose separate tile URLs; compose them before model
    inference so the grid math maps boxes/classes to the real cell positions.
    When the screenshot toggle is on, dynamic/4x4 grids are captured straight
    from the rendered element instead (driver required).
    """
    expected = grid_n * grid_n
    if driver is not None and _use_screenshot_compose():
        if _screenshot_grid_to_png(driver):
            return
        # screenshot failed — fall through to the legacy URL-fetch composite

    if len(img_urls) < expected:
        download_img(0, img_urls[0])
        return
    if len(set(img_urls[:expected])) <= 1:
        download_img(0, img_urls[0])
        return

    images = []
    for url in img_urls[:expected]:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        images.append(Image.open(BytesIO(response.content)).convert("RGB"))
    _write_dynamic_grid_image(images, grid_n=grid_n)


def _wait_for_new_dynamic_imgs(answers, before_img_urls, driver, max_wait_s=15):
    """
    Poll the dynamic-captcha grid until the answered cells show new image URLs,
    or until `max_wait_s` elapses. Bounded retry — without this, edge cases
    (reCAPTCHA pre-verifies, cells get removed, network hiccup) hang the
    surrounding `while True` polling loop indefinitely.

    Returns (is_new, img_urls). On timeout returns (False, last_img_urls) so
    the caller can break the dynamic-loop and let the outer success/reload
    flow take over.
    """
    deadline = monotonic() + max_wait_s
    img_urls = before_img_urls
    while monotonic() < deadline:
        try:
            is_new, img_urls = get_all_new_dynamic_captcha_img_urls(
                answers, before_img_urls, driver
            )
        except Exception:
            # cells went away mid-poll (e.g., reCAPTCHA already moved to
            # verified state) — treat as "no new images, give up gracefully"
            return False, img_urls
        if is_new:
            return True, img_urls
        sleep(0.3)
    return False, img_urls


def _wait_for_settled_dynamic_grid(driver, grid_n, max_wait_s=8):
    """Return the tile URLs only once the fade-in transition has settled.

    `_wait_for_new_dynamic_imgs` confirms the *answered* cells changed, but the
    other cells can still be mid-fade at that instant, and during a transition
    reCAPTCHA briefly serves several cells the SAME (placeholder/prior) URL. A
    snapshot taken then has distinct << n*n, so `download_dynamic_grid_img`
    composes the same tile into multiple positions and the VLM over-selects
    every repeat (measured live: distinct=4 -> a 7-of-9 over-select). Poll until
    every cell shows a distinct URL (settled) or `max_wait_s` elapses
    (fail-open: a grid that legitimately repeats a tile must not hang forever).
    """
    expected = grid_n * grid_n
    deadline = monotonic() + max_wait_s
    urls = get_all_captcha_img_urls(driver)
    while monotonic() < deadline and len(set(urls[:expected])) < expected:
        sleep(0.3)
        urls = get_all_captcha_img_urls(driver)
    return urls


def get_all_new_dynamic_captcha_img_urls(answers, before_img_urls, driver):
    """
    Get all the new image urls from the recaptcha.
    :param answers: answers from the recaptcha.
    :param before_img_urls: image urls before.
    """
    try:
        # Reuse the batched browser-script path. On forwarded remote CDP each
        # WebElement.get_attribute call is a network round-trip, and this
        # function runs repeatedly inside a 0.3-second polling loop.
        img_urls = get_all_captcha_img_urls(driver)
    except Exception:
        return False, []

    # Check if the image urls are the same as before
    index_common = []
    for answer in answers:
        if img_urls[answer - 1] == before_img_urls[answer - 1]:
            index_common.append(answer)

    # Return if the image urls are the same as before
    if len(index_common) >= 1:
        is_new = False
        return is_new, img_urls
    else:
        is_new = True
        return is_new, img_urls


def square_solver(target_set: Iterable[int], verbose, model):
    """
    Detection-model path for a 4x4 "select all squares" grid: run the detector
    on the whole composite and return every 1-indexed cell a target-class box
    overlaps. Unlike the 3x3 grid, the 4x4 is one photo cut into 16 squares, so
    an object box legitimately spans several cells and all of them must be
    selected. Cell size is derived from the actual image dimensions.
    :param target_set: iterable of YOLO class IDs that satisfy the challenge.
    :param verbose: print verbose.
    """
    target_set = set(int(x) for x in target_set)

    image = np.asarray(Image.open("recaptcha_images/0.png").convert("RGB"))
    height, width = image.shape[:2]
    cell_h, cell_w = height / 4.0, width / 4.0
    result = model.predict(
        image, task="detect", verbose=verbose, conf=_detect_conf()
    )

    answers = set()
    hits = []
    for box in result[0].boxes:
        cls_id = int(box.cls)
        if cls_id not in target_set:
            continue
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
        r1 = min(3, max(0, int(y1 // cell_h)))
        r2 = min(3, max(0, int((y2 - 1) // cell_h)))
        c1 = min(3, max(0, int(x1 // cell_w)))
        c2 = min(3, max(0, int((x2 - 1) // cell_w)))
        cells = []
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                cell = r * 4 + c + 1
                answers.add(cell)
                cells.append(cell)
        hits.append(f"cls{cls_id}@{float(box.conf):.2f}->{cells}")
    if verbose:
        print(
            f"square_solver: grid={width}x{height} "
            f"target_set={sorted(target_set)} hits=[{' '.join(hits)}] "
            f"-> answers={sorted(answers)}"
        )
    return sorted(answers)


def _recognizer_mode() -> str:
    """vlm | local | hybrid (default). hybrid = VLM first, YOLO on VLM failure."""
    mode = os.environ.get("RECAPTCHA_RECOGNIZER", "hybrid").strip().lower()
    return mode if mode in ("vlm", "local", "hybrid") else "hybrid"


def get_challenge_phrase(driver, verbose: bool = False) -> str:
    """Return the challenge target phrase (the bold word(s), e.g. 'fire hydrant',
    '버스'). The VLM consumes this directly — no phrase->class table, so any
    locale and any category resolves. Empty string if the title isn't present."""
    try:
        strong = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located(
                (By.XPATH, '//div[@id="rc-imageselect"]//strong')
            )
        )
        phrase = (strong.text or "").strip()
    except Exception:
        phrase = ""
    if verbose:
        print(f"challenge phrase={phrase!r}")
    return phrase


def _vlm_answers(phrase, grid_n, verbose, composite="recaptcha_images/0.png"):
    """VLM cell picks for the current composite, or None to fall back to YOLO.

    Returns None when the VLM is disabled (mode=local / no key) or fails after
    retries; returns a (possibly empty) list of 1-indexed cells otherwise. An
    empty list is a real answer: "this image has no more matches" — for a
    dynamic grid that's the signal to verify."""
    if _recognizer_mode() == "local" or not recognizer.vlm_enabled():
        return None
    return recognizer.recognize_cells(composite, phrase, grid_n, verbose=verbose)


def _is_dead_driver_error(exc: Exception) -> bool:
    """True for errors that mean the browser/driver is gone for good (process
    crashed, session deleted, connection refused). Retrying these just burns the
    whole deadline doing nothing — the caller aborts immediately instead."""
    text = f"{type(exc).__name__}: {exc}".lower()
    needles = (
        "max retries exceeded",
        "connection refused",
        "failed to establish a new connection",
        "chrome not reachable",
        "invalid session id",
        "session deleted",
        "no such window",
        "disconnected",
        "tab crashed",
        "cannot determine loading status",
    )
    return any(n in text for n in needles)


def _human_click(driver, element, mu=0.4, sigma=0.15):
    """Click `element` with a short curved mouse approach + variable dwell.

    reCAPTCHA scores cursor telemetry: teleport-then-click (raw .click()) reads
    as a bot and drives the trust score down, which is what escalates a session
    into the hardest, near-unsolvable challenge loops. A couple of intermediate
    moves with jittered offsets and human-scale pauses emit the mousemove
    stream a real click produces. Falls back to a plain click if ActionChains
    can't run (e.g. headless without a virtual cursor)."""
    try:
        chain = ActionChains(driver)
        for _ in range(np.random.randint(2, 4)):
            dx = int(np.random.randint(-18, 19))
            dy = int(np.random.randint(-18, 19))
            try:
                chain.move_to_element_with_offset(element, dx, dy)
            except Exception:
                chain.move_to_element(element)
            chain.pause(max(0.03, float(np.random.normal(0.12, 0.05))))
        chain.move_to_element(element)
        chain.pause(max(0.04, float(np.random.normal(mu * 0.5, sigma * 0.5))))
        chain.click()
        chain.perform()
    except Exception:
        try:
            element.click()
        except Exception:
            driver.execute_script("arguments[0].click();", element)
    random_delay(mu=mu, sigma=sigma)


def _click_cells(driver, answers):
    """Human-click selected grid cells with one remote lookup/action batch."""
    answers = list(answers)
    if not answers:
        return

    indices = [int(answer) - 1 for answer in answers]
    if any(index < 0 for index in indices):
        raise ValueError(f"cell answers must be 1-indexed: {answers!r}")

    locator = (By.XPATH, '//div[@id="rc-imageselect-target"]//td')
    required = max(indices) + 1

    def _enough_cells(current_driver):
        cells = current_driver.find_elements(*locator)
        return cells if len(cells) >= required else False

    cells = WebDriverWait(driver, 10).until(_enough_cells)
    selected = [cells[index] for index in indices]

    try:
        chain = ActionChains(driver)
        for cell in selected:
            for _ in range(np.random.randint(2, 4)):
                dx = int(np.random.randint(-18, 19))
                dy = int(np.random.randint(-18, 19))
                try:
                    chain.move_to_element_with_offset(cell, dx, dy)
                except Exception:
                    chain.move_to_element(cell)
                chain.pause(max(0.03, float(np.random.normal(0.12, 0.05))))
            chain.move_to_element(cell)
            chain.pause(max(0.04, float(np.random.normal(0.225, 0.09))))
            chain.click()
            # `_human_click(..., mu=0.45, sigma=0.18)` slept after each
            # perform. Keep the same inter-click timing inside this one batch.
            chain.pause(max(0.1, float(np.random.normal(0.45, 0.18))))
        chain.perform()
    except Exception:
        # Match _human_click's headless fallback without paying one ActionChains
        # command per cell on the normal remote-CDP path.
        for cell in selected:
            try:
                cell.click()
            except Exception:
                driver.execute_script("arguments[0].click();", cell)
            random_delay(mu=0.45, sigma=0.18)


def solve_recaptcha(driver, verbose):
    """
    Solve the recaptcha.
    :param driver: selenium driver.
    :param verbose: print verbose.
    """

    go_to_recaptcha_iframe1(driver)

    WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable(
            (By.XPATH, '//div[@class="recaptcha-checkbox-border"]')
        )
    )
    check_box = driver.find_element(
        By.XPATH, '//div[@class="recaptcha-checkbox-border"]'
    )
    _human_click(driver, check_box, mu=0.6, sigma=0.2)

    # High-trust sessions verify on the checkbox alone — no image challenge ever
    # appears. Check briefly before paying for the challenge path.
    try:
        WebDriverWait(driver, 3).until(
            EC.presence_of_element_located(
                (By.XPATH, '//span[contains(@aria-checked, "true")]')
            )
        )
        if verbose:
            print("solved on checkbox alone (no image challenge)")
        driver.switch_to.default_content()
        return
    except Exception:
        pass

    go_to_recaptcha_iframe2(driver)

    mode = _recognizer_mode()
    use_vlm = mode in ("vlm", "hybrid") and recognizer.vlm_enabled()

    # Local YOLO is the fallback (or the only path in mode=local). Skip the
    # ~150 MB model load entirely in pure-VLM mode.
    primary = None
    fallback = None
    primary_path = os.environ.get("RECAPTCHA_YOLO_MODEL", DEFAULT_YOLO_MODEL)
    fallback_path = os.environ.get(
        "RECAPTCHA_YOLO_FALLBACK", DEFAULT_YOLO_FALLBACK_MODEL
    )
    if mode != "vlm":
        primary = _try_load_yolo(primary_path, verbose=verbose)
        if primary is None:
            primary = _try_load_yolo(fallback_path, verbose=verbose)
            fallback_path = ""
    if primary is None and not use_vlm:
        raise RuntimeError(
            "could not load any reCAPTCHA recognizer "
            "(no VLM key available and no local YOLO model loaded)"
        )
    if verbose:
        print(
            f"recognizer mode={mode} use_vlm={use_vlm} "
            f"local={getattr(primary, 'task', None)} fallback={fallback_path or 'off'}"
        )

    os.makedirs("recaptcha_images", exist_ok=True)

    # Hard wall-clock bound so a pathological challenge tree (reCAPTCHA keeps
    # rejecting and re-issuing) can't hang forever; the caller decides whether
    # to retry from scratch.
    try:
        deadline_seconds = float(os.environ.get("RECAPTCHA_SOLVER_DEADLINE_SEC", "120"))
    except ValueError:
        deadline_seconds = 120.0
    overall_deadline = monotonic() + deadline_seconds
    # Bounded reloads: a string of empty answers used to reload until the
    # deadline (and excessive reloads themselves raise reCAPTCHA's suspicion).
    try:
        max_reloads = int(os.environ.get("RECAPTCHA_MAX_RELOADS", "12"))
    except ValueError:
        max_reloads = 12
    reloads = 0

    def _local_answers(grid_n):
        """YOLO fallback cell picks for the current composite (1-indexed)."""
        nonlocal fallback
        if primary is None:
            return []
        target_set = get_target_classes(driver, primary, verbose)
        model = primary
        # 4x4 cross-tile needs a detector (per-cell classification of a 1/16
        # slice is meaningless); prefer the OIV7 fallback there.
        want_detect = grid_n >= 4
        if (not target_set or want_detect) and fallback_path:
            if fallback is None:
                fallback = _try_load_yolo(fallback_path, verbose=verbose)
            if fallback is not None:
                fb_set = get_target_classes(driver, fallback, verbose)
                if fb_set:
                    target_set, model = fb_set, fallback
        if not target_set:
            return []
        if getattr(model, "task", None) == "classify":
            return classify_grid_cells(target_set, grid_n, verbose, model)
        if grid_n >= 4:
            return square_solver(target_set, verbose, model)
        return dynamic_and_selection_solver(target_set, verbose, model)

    def _answers_for_current(phrase, grid_n):
        """VLM first, YOLO fallback. List (maybe empty) of 1-indexed cells."""
        if use_vlm:
            cells = _vlm_answers(phrase, grid_n, verbose)
            if cells is not None:
                return cells
        return _local_answers(grid_n)

    while monotonic() < overall_deadline:
        try:
            captcha = None
            answers = []
            img_urls = []
            phrase = ""
            grid_n = 3
            # ---- choose a challenge we can actually answer ----
            while monotonic() < overall_deadline:
                reload = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.ID, "recaptcha-reload-button"))
                )
                title_wrapper = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.ID, "rc-imageselect"))
                )
                # <td> count is locale-proof: 16 = 4x4 "select all squares",
                # 9 = 3x3 "select all images" (title strings are localized).
                try:
                    td_count = len(
                        driver.find_elements(
                            By.XPATH, '//div[@id="rc-imageselect-target"]//td'
                        )
                    )
                except Exception:
                    td_count = -1
                grid_n = 4 if td_count >= 16 else 3
                phrase = get_challenge_phrase(driver, verbose)
                if verbose:
                    print(
                        f"challenge: td={td_count} grid={grid_n}x{grid_n} "
                        f"phrase={phrase!r} text={title_wrapper.text!r}"
                    )

                img_urls = get_all_captcha_img_urls(driver)
                # download_dynamic_grid_img composes distinct dynamic tiles into
                # one grid and falls back to the single composite URL otherwise —
                # correct for static 3x3, dynamic 3x3, and the one-image 4x4.
                # Pass the real grid_n: a 4x4 with distinct tile URLs must be
                # composed as 16 cells, not sliced to the first 9 as a 3x3.
                download_dynamic_grid_img(img_urls, grid_n=grid_n, driver=driver)
                answers = _answers_for_current(phrase, grid_n)

                # Over-selection guard for the FIRST round too (mirrors the
                # dynamic re-round guard at ~line 840 — they were asymmetric: a
                # 3x3 first round had NO guard, so a runaway VLM read of [1..9]
                # / [1,2,3,5,6,7,8,9] got clicked wholesale, guaranteeing a
                # reject and burning a verify round). reCAPTCHA never serves a
                # grid where (nearly) every cell matches, so near-full is
                # virtually always over-selection: treat >= n*n-1 (8/9 on 3x3,
                # 15-16/16 on 4x4) as unusable and reload for a fresh, answerable
                # grid. A legitimately busy 5-7 cell round still passes.
                n_cells = grid_n * grid_n
                usable = 1 <= len(answers) < n_cells - 1
                if usable:
                    captcha = "squares" if grid_n >= 4 else "dynamic"
                    break

                reloads += 1
                if reloads > max_reloads:
                    if verbose:
                        print(f"exceeded {max_reloads} reloads, giving up")
                    driver.switch_to.default_content()
                    return
                if verbose:
                    print(
                        f"no usable answers ({answers}); reload {reloads}/{max_reloads}"
                    )
                random_delay()
                reload.click()
                WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable(
                        (By.XPATH, '(//div[@id="rc-imageselect-target"]//td)[1]')
                    )
                )

            if captcha is None:
                break

            _dump_solve_diag(driver, captcha, answers, tag="initial")
            _click_cells(driver, answers)

            if captcha == "dynamic":
                # Dynamic fade-in: after each click round, re-fetch + re-compose
                # the whole grid, re-recognize, click new matches, until none
                # remain or the bounded deadline trips. Re-composing the full
                # grid each round avoids the fragile hard-coded-100px tile-paste.
                dynamic_deadline = monotonic() + min(60, deadline_seconds)
                while monotonic() < dynamic_deadline:
                    is_new, img_urls = _wait_for_new_dynamic_imgs(
                        answers, img_urls, driver
                    )
                    if not is_new:
                        break
                    shot = _use_screenshot_compose()
                    if shot:
                        # Screenshot path: shared composite URLs still render as
                        # distinct tiles, so the distinct gate is meaningless.
                        # Let the fade-in settle briefly, then snapshot below.
                        sleep(0.6)
                        fresh_urls = get_all_captcha_img_urls(driver)
                    else:
                        fresh_urls = _wait_for_settled_dynamic_grid(driver, grid_n)
                    n_distinct = len(set(fresh_urls[:grid_n * grid_n]))
                    if verbose:
                        print(
                            f"dynamic re-round: {len(fresh_urls)} img urls "
                            f"(expect {grid_n * grid_n}); distinct={n_distinct} "
                            f"mode={'shot' if shot else 'url'}"
                        )
                    # Legacy URL-compose only: a grid that never settles
                    # (duplicate placeholder tiles mid-fade) would repeat a tile
                    # and the VLM would over-select every copy (measured:
                    # distinct=4 -> 7-of-9). Don't trust this round — verify with
                    # the prior answers. The screenshot path renders correctly
                    # even at distinct < n*n, so it skips this bail and keeps
                    # re-rounding the real grid.
                    if not shot and n_distinct < grid_n * grid_n:
                        break
                    download_dynamic_grid_img(fresh_urls, grid_n=grid_n, driver=driver)
                    img_urls = fresh_urls
                    answers = _answers_for_current(phrase, grid_n)
                    # Over-selection guard for fade-in re-rounds. Selecting
                    # almost the entire grid in one re-round is the runaway
                    # over-selection seen live (a 3x3 round returning 8 of 9
                    # cells, then 4 — wildly unstable) and guarantees a reject.
                    # Use a conservative bound (>= n*n-1, i.e. 8/9 or 15/16) so a
                    # legitimately busy round of 5-7 matches still gets clicked;
                    # only the near-full pathology stops the re-round loop.
                    if not answers or len(answers) >= grid_n * grid_n - 1:
                        break
                    _dump_solve_diag(driver, captcha, answers, tag="dynamic")
                    _click_cells(driver, answers)

            verify = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.ID, "recaptcha-verify-button"))
            )
            random_delay(mu=1.2, sigma=0.3)
            if verbose:
                print(f"clicking verify (type={captcha}, answers={answers})")
            _dump_solve_diag(driver, captcha, answers, tag="pre-verify")
            _human_click(driver, verify, mu=0.5, sigma=0.2)

            try:
                go_to_recaptcha_iframe1(driver)
                WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located(
                        (By.XPATH, '//span[contains(@aria-checked, "true")]')
                    )
                )
                if verbose:
                    print("solved")
                driver.switch_to.default_content()
                break
            except Exception:
                if verbose:
                    print("verify did not solve; re-entering challenge iframe")
                go_to_recaptcha_iframe2(driver)
        except Exception as e:
            # A dead browser/driver (crash, session gone, connection refused)
            # can't be retried — bail immediately instead of burning the whole
            # deadline re-throwing the same connection error. But a
            # requests.RequestException comes from our own tile downloads (an
            # image-server hiccup), NOT the webdriver — its "Max retries" /
            # "connection" text must not trip the abort, so exclude it and let
            # it fall through to the transient re-anchor path.
            if (not isinstance(e, requests.exceptions.RequestException)
                    and _is_dead_driver_error(e)):
                if verbose:
                    print(f"driver/browser is dead, aborting: {e!r}")
                break
            # Transient errors (StaleElementReference, timeouts): re-anchor on
            # the challenge iframe and let `overall_deadline` decide when to quit.
            if verbose:
                print(f"transient error in solve loop, retrying: {e!r}")
            sleep(0.5)
            try:
                go_to_recaptcha_iframe1(driver)
                WebDriverWait(driver, 2).until(
                    EC.presence_of_element_located(
                        (By.XPATH, '//span[contains(@aria-checked, "true")]')
                    )
                )
                if verbose:
                    print("solved (verified after transient error)")
                driver.switch_to.default_content()
                break
            except Exception:
                pass
            try:
                go_to_recaptcha_iframe2(driver)
            except Exception:
                continue


def _dump_solve_diag(driver, captcha, answers, tag=""):
    """Debug aid: when RECAPTCHA_SOLVER_DIAG_DIR is set, snapshot the browser
    viewport plus the composite image the classifier just scored, so a failed
    solve can be eyeballed cell-by-cell against what the model chose. No-op
    when the env var is unset."""
    diag_dir = os.environ.get("RECAPTCHA_SOLVER_DIAG_DIR")
    if not diag_dir:
        return
    try:
        os.makedirs(diag_dir, exist_ok=True)
        stamp = f"{int(monotonic() * 1000) % 100_000_000}"
        if tag:
            stamp = f"{stamp}-{tag}"
        try:
            driver.save_screenshot(os.path.join(diag_dir, f"{stamp}-view.png"))
        except Exception:
            pass
        composite = "recaptcha_images/0.png"
        if os.path.exists(composite):
            shutil.copy(composite, os.path.join(diag_dir, f"{stamp}-grid.png"))
        with open(os.path.join(diag_dir, f"{stamp}-meta.txt"), "w") as fh:
            fh.write(f"captcha={captcha} answers={answers}\n")
    except Exception as exc:
        print(f"_dump_solve_diag failed: {exc!r}")


def is_solved(driver) -> bool:
    """
    Returns True if the reCAPTCHA checkbox iframe currently shows the verified
    state (the green checkmark with no `style="display:none"` override).
    """
    try:
        driver.switch_to.default_content()
        iframe_inner = driver.find_element(
            By.XPATH,
            "//iframe[contains(@src, '/recaptcha/api2/anchor') or contains(@src, '/recaptcha/enterprise/anchor')]",
        )
        driver.switch_to.frame(iframe_inner)
        checkmark = driver.find_element(
            By.CSS_SELECTOR, ".recaptcha-checkbox-checkmark"
        )
        attributes = checkmark.get_dom_attribute("style")
        return attributes == ""
    except Exception:
        return False
    finally:
        driver.switch_to.default_content()
