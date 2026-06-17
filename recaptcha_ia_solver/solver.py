# Standard imports
import os
import re
import shutil
from io import BytesIO
from time import monotonic, sleep
from typing import Iterable, Optional, Set

# Third-party imports
import cv2
import numpy as np
import requests
from PIL import Image
from ultralytics import YOLO
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.action_chains import ActionChains

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

    response = requests.get(url, stream=True)
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


def download_dynamic_grid_img(img_urls, grid_n=3):
    """Download the current 3x3 challenge as recaptcha_images/0.png.

    Static 3x3 challenges expose one composite URL repeated nine times.
    Dynamic challenges expose separate tile URLs; compose them before model
    inference so the grid math maps boxes/classes to the real cell positions.
    """
    expected = grid_n * grid_n
    if len(img_urls) < expected:
        download_img(0, img_urls[0])
        return
    if len(set(img_urls[:expected])) <= 1:
        download_img(0, img_urls[0])
        return

    images = []
    for url in img_urls[:expected]:
        response = requests.get(url)
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


def get_all_new_dynamic_captcha_img_urls(answers, before_img_urls, driver):
    """
    Get all the new image urls from the recaptcha.
    :param answers: answers from the recaptcha.
    :param before_img_urls: image urls before.
    """
    images = WebDriverWait(driver, 10).until(
        EC.presence_of_all_elements_located(
            (By.XPATH, '//div[@id="rc-imageselect-target"]//img')
        )
    )
    img_urls = []

    # Get all the image urls
    for img in images:
        try:
            img_urls.append(img.get_attribute("src"))
        except:
            is_new = False
            return is_new, img_urls

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


def paste_new_img_on_main_img(main, new, loc):
    """
    Paste the new image on the main image.
    :param main: main image.
    :param new: new image.
    :param loc: location of the new image.
    """
    paste = np.copy(main)

    row = (loc - 1) // 3
    col = (loc - 1) % 3

    start_row, end_row = row * 100, (row + 1) * 100
    start_col, end_col = col * 100, (col + 1) * 100

    paste[start_row:end_row, start_col:end_col] = new

    paste = cv2.cvtColor(paste, cv2.COLOR_RGB2BGR)
    cv2.imwrite("recaptcha_images/0.png", paste)


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

    action_chain = ActionChains(driver)
    check_box = driver.find_element(By.XPATH, '//div[@class="recaptcha-checkbox-border"]')
    action_chain.move_to_element(check_box).click().perform()

    go_to_recaptcha_iframe2(driver)

    primary_path = os.environ.get("RECAPTCHA_YOLO_MODEL", DEFAULT_YOLO_MODEL)
    fallback_path = os.environ.get(
        "RECAPTCHA_YOLO_FALLBACK", DEFAULT_YOLO_FALLBACK_MODEL
    )
    primary = _try_load_yolo(primary_path, verbose=verbose)
    if primary is None:
        # Primary missing — promote fallback so the solver still runs.
        primary = _try_load_yolo(fallback_path, verbose=verbose)
        fallback_path = ""
        if primary is None:
            raise RuntimeError(
                f"could not load any reCAPTCHA model "
                f"(tried RECAPTCHA_YOLO_MODEL and RECAPTCHA_YOLO_FALLBACK)"
            )
    fallback = None  # lazy-loaded only when a target term misses the primary
    if verbose:
        print(
            f"loaded primary={primary_path} task={getattr(primary, 'task', '?')}; "
            f"fallback={fallback_path or 'disabled'}"
        )

    os.makedirs("recaptcha_images", exist_ok=True)

    # Hard wall-clock bound: at this point we've already accepted that we
    # cannot solve this challenge tree (e.g., reCAPTCHA keeps failing our
    # verifies and re-issuing new challenges). Give up so the caller can
    # decide whether to retry from scratch instead of hanging forever.
    try:
        deadline_seconds = float(os.environ.get("RECAPTCHA_SOLVER_DEADLINE_SEC", "120"))
    except ValueError:
        deadline_seconds = 120.0
    overall_deadline = monotonic() + deadline_seconds

    while True:
        if monotonic() > overall_deadline:
            if verbose:
                print("solve_recaptcha overall deadline reached, giving up")
            break
        try:
            while True:
                reload = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.ID, "recaptcha-reload-button"))
                )
                title_wrapper = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.ID, "rc-imageselect"))
                )
                # Grid size + challenge type, detected structurally. The
                # upstream string checks ("squares"/"none" in title) only work
                # for an English-locale widget; IG serves reCAPTCHA in Korean
                # (hl=ko) so they never matched — every challenge fell through
                # to the 3x3-one-time branch, and 4x4 grids got sliced 3x3 into
                # garbage cells. The <td> count is locale-proof: 16 tds = 4x4
                # "select all squares", 9 = 3x3 "select all images".
                try:
                    td_count = len(
                        driver.find_elements(
                            By.XPATH, '//div[@id="rc-imageselect-target"]//td'
                        )
                    )
                except Exception:
                    td_count = -1
                grid_n = 4 if td_count >= 16 else 3
                if verbose:
                    print(
                        f"challenge wrapper: td_count={td_count} "
                        f"grid={grid_n}x{grid_n} text={title_wrapper.text!r}"
                    )

                target_set = get_target_classes(driver, primary, verbose)
                model = primary
                if not target_set and fallback_path:
                    if fallback is None:
                        if verbose:
                            print(f"loading fallback {fallback_path}")
                        fallback = _try_load_yolo(fallback_path, verbose=verbose)
                    if fallback is not None:
                        target_set = get_target_classes(driver, fallback, verbose)
                        if target_set:
                            model = fallback
                is_classifier = getattr(model, "task", None) == "classify"

                if not target_set:
                    random_delay()
                    if verbose:
                        print("skipping (no supported category in challenge)")
                    reload.click()
                elif grid_n == 4:
                    if verbose:
                        print("found a 4x4 select-all-squares captcha")
                    img_urls = get_all_captcha_img_urls(driver)
                    if verbose:
                        print(
                            f"squares: img count={len(img_urls)} "
                            f"distinct={len(set(img_urls))}"
                        )
                    download_dynamic_grid_img(img_urls, grid_n=3)
                    answers = []
                    if is_classifier and fallback_path:
                        if fallback is None:
                            if verbose:
                                print(f"loading fallback {fallback_path} for 4x4 detector")
                            fallback = _try_load_yolo(fallback_path, verbose=verbose)
                        if fallback is not None:
                            fallback_target_set = get_target_classes(
                                driver, fallback, verbose
                            )
                            if fallback_target_set:
                                if verbose:
                                    print("using fallback detector for 4x4 squares")
                                answers = square_solver(
                                    fallback_target_set, verbose, fallback
                                )
                                if verbose and not (len(answers) >= 1 and len(answers) < 16):
                                    print(
                                        "fallback detector produced no usable 4x4 answers; "
                                        "falling back to classifier"
                                    )
                    if not (len(answers) >= 1 and len(answers) < 16):
                        if is_classifier:
                            answers = classify_grid_cells(target_set, 4, verbose, model)
                        else:
                            answers = square_solver(target_set, verbose, model)
                    if len(answers) >= 1 and len(answers) < 16:
                        captcha = "squares"
                        break
                    else:
                        if verbose:
                            print("squares: no usable answers, reloading")
                        reload.click()
                else:
                    # 3x3 "select all images" — routed through the dynamic
                    # handler, which clicks the matches then re-checks reloaded
                    # tiles until none remain. If reCAPTCHA does not reload any
                    # tiles (a plain one-time grid), _wait_for_new_dynamic_imgs
                    # returns quickly and we just click verify once, so this
                    # path also covers the static 3x3 case.
                    if verbose:
                        print("found a 3x3 select-all-images captcha")
                    img_urls = get_all_captcha_img_urls(driver)
                    if verbose:
                        print(
                            f"dynamic: img count={len(img_urls)} "
                            f"distinct={len(set(img_urls))}"
                        )
                    download_img(0, img_urls[0])
                    dynamic_model = model
                    dynamic_target_set = target_set
                    dynamic_is_classifier = is_classifier
                    answers = []
                    if is_classifier and fallback_path:
                        if fallback is None:
                            if verbose:
                                print(f"loading fallback {fallback_path} for 3x3 detector")
                            fallback = _try_load_yolo(fallback_path, verbose=verbose)
                        if fallback is not None:
                            fallback_target_set = get_target_classes(
                                driver, fallback, verbose
                            )
                            if fallback_target_set:
                                if verbose:
                                    print("using fallback detector for 3x3 dynamic")
                                answers = dynamic_and_selection_solver(
                                    fallback_target_set, verbose, fallback
                                )
                                if answers:
                                    dynamic_model = fallback
                                    dynamic_target_set = fallback_target_set
                                    dynamic_is_classifier = False
                                elif verbose:
                                    print(
                                        "fallback detector produced no 3x3 answers; "
                                        "falling back to classifier"
                                    )
                    if not answers:
                        if is_classifier:
                            answers = classify_grid_cells(target_set, 3, verbose, model)
                        else:
                            answers = dynamic_and_selection_solver(target_set, verbose, model)
                    if len(answers) >= 1:
                        model = dynamic_model
                        target_set = dynamic_target_set
                        is_classifier = dynamic_is_classifier
                        captcha = "dynamic"
                        break
                    else:
                        if verbose:
                            print("dynamic: no usable answers, reloading")
                        reload.click()
                WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable(
                        (By.XPATH, '(//div[@id="rc-imageselect-target"]//td)[1]')
                    )
                )

            if captcha == "dynamic":
                for answer in answers:
                    WebDriverWait(driver, 10).until(
                        EC.element_to_be_clickable(
                            (
                                By.XPATH,
                                f'(//div[@id="rc-imageselect-target"]//td)[{answer}]',
                            )
                        )
                    ).click()
                    random_delay(mu=0.5, sigma=0.2)
                # Outer dynamic-loop deadline: hard cap so no edge case (cells
                # removed, network stall, reCAPTCHA already verified) keeps us
                # spinning forever waiting for refreshed thumbnails.
                dynamic_deadline = monotonic() + 60
                while monotonic() < dynamic_deadline:
                    before_img_urls = img_urls
                    is_new, img_urls = _wait_for_new_dynamic_imgs(
                        answers, before_img_urls, driver
                    )
                    if not is_new:
                        # No fresh thumbnails arrived — challenge likely already
                        # transitioned to "verify"; bail and let outer success
                        # check decide.
                        break

                    new_img_index_urls = [answer - 1 for answer in answers]

                    for index in new_img_index_urls:
                        download_img(index + 1, img_urls[index])
                    paste_deadline = monotonic() + 15
                    while monotonic() < paste_deadline:
                        try:
                            for answer in answers:
                                main_img = Image.open("recaptcha_images/0.png")
                                new_img = Image.open(f"recaptcha_images/{answer}.png")
                                location = answer
                                paste_new_img_on_main_img(main_img, new_img, location)
                            break
                        except Exception:
                            is_new, img_urls = _wait_for_new_dynamic_imgs(
                                answers, before_img_urls, driver
                            )
                            if not is_new:
                                break
                            for index in [answer - 1 for answer in answers]:
                                download_img(index + 1, img_urls[index])

                    if is_classifier:
                        answers = classify_grid_cells(target_set, 3, verbose, model)
                    else:
                        answers = dynamic_and_selection_solver(target_set, verbose, model)

                    if len(answers) >= 1:
                        for answer in answers:
                            WebDriverWait(driver, 10).until(
                                EC.element_to_be_clickable(
                                    (
                                        By.XPATH,
                                        f'(//div[@id="rc-imageselect-target"]//td)[{answer}]',
                                    )
                                )
                            ).click()
                            random_delay(mu=0.5, sigma=0.1)
                    else:
                        break
            elif captcha == "squares":
                if verbose:
                    print(f"clicking {len(answers)} cell(s) for {captcha}: {answers}")
                for answer in answers:
                    WebDriverWait(driver, 10).until(
                        EC.element_to_be_clickable(
                            (
                                By.XPATH,
                                f'(//div[@id="rc-imageselect-target"]//td)[{answer}]',
                            )
                        )
                    ).click()
                    random_delay()

            verify = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.ID, "recaptcha-verify-button"))
            )
            random_delay(mu=2, sigma=0.2)
            if verbose:
                print(f"clicking verify button (captcha type={captcha})")
            _dump_solve_diag(driver, captcha, answers, tag="pre-verify")
            verify.click()

            try:
                go_to_recaptcha_iframe1(driver)
                WebDriverWait(driver, 4).until(
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
                    print(
                        "verify did not yield solved state; re-entering challenge iframe"
                    )
                go_to_recaptcha_iframe2(driver)
        except Exception as e:
            # Transient errors (StaleElementReference, ElementNotInteractable,
            # WebDriverWait timeouts on a single element) used to break out of
            # the outer loop unconditionally — that returned a "solved looking"
            # state to callers even though the checkbox was never verified.
            # Now we soak up the error, re-anchor on the challenge iframe, and
            # let `overall_deadline` decide when to actually give up.
            if verbose:
                print(f"transient error in solve loop, retrying: {e!r}")
            sleep(0.5)
            try:
                go_to_recaptcha_iframe1(driver)
                # If the checkbox already shows verified, accept the success
                # even though the loop saw an error mid-flight.
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
                # iframe2 is gone too — could be either solved or completely
                # broken; fall through to the next outer-loop iteration which
                # will hit the deadline check.
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
