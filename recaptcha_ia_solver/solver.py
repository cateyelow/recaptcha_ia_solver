# Standard imports
import os
import re
import shutil
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
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
    for idx, res in enumerate(results):
        probs = getattr(res, "probs", None)
        if probs is None:
            continue
        top1 = int(getattr(probs, "top1", -1))
        if top1 not in target_set:
            continue
        top1_conf = float(getattr(probs, "top1conf", 1.0) or 1.0)
        if top1_conf < min_conf:
            continue
        answers.append(idx + 1)
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


def dynamic_and_selection_solver(target_set: Iterable[int], verbose, model):
    """
    Get the answers from the recaptcha images.
    :param target_set: iterable of YOLO class IDs that satisfy the challenge.
    :param verbose: print verbose.
    """
    target_set = set(int(x) for x in target_set)

    image = Image.open("recaptcha_images/0.png")
    image = np.asarray(image)
    result = model.predict(image, task="detect", verbose=verbose)

    target_index = [
        idx for idx, num in enumerate(result[0].boxes.cls) if int(num) in target_set
    ]

    answers = []
    boxes = result[0].boxes.data
    for i in target_index:
        target_box = boxes[i]
        x1, y1 = int(target_box[0]), int(target_box[1])
        x2, y2 = int(target_box[2]), int(target_box[3])

        xc = (x1 + x2) / 2
        yc = (y1 + y2) / 2

        row = yc // 100
        col = xc // 100
        answer = int(row * 3 + col + 1)
        answers.append(answer)

    return list(set(answers))


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


def get_occupied_cells(vertices):
    """
    Get the occupied cells from the vertices.
    :param vertices: vertices of the image.
    """
    occupied_cells = set()
    rows, cols = zip(*[((v - 1) // 4, (v - 1) % 4) for v in vertices])

    for i in range(min(rows), max(rows) + 1):
        for j in range(min(cols), max(cols) + 1):
            occupied_cells.add(4 * i + j + 1)

    return sorted(list(occupied_cells))


def square_solver(target_set: Iterable[int], verbose, model):
    """
    Get the answers from the recaptcha images.
    :param target_set: iterable of YOLO class IDs that satisfy the challenge.
    :param verbose: print verbose.
    """
    target_set = set(int(x) for x in target_set)

    image = Image.open("recaptcha_images/0.png")
    image = np.asarray(image)
    result = model.predict(image, task="detect", verbose=verbose)
    boxes = result[0].boxes.data

    target_index = [
        idx for idx, num in enumerate(result[0].boxes.cls) if int(num) in target_set
    ]

    answers = []
    count = 0
    for i in target_index:
        target_box = boxes[i]
        p1, p2 = (int(target_box[0]), int(target_box[1])), (
            int(target_box[2]),
            int(target_box[3]),
        )
        x1, y1 = p1
        x4, y4 = p2
        x2 = x4
        y2 = y1
        x3 = x1
        y3 = y4
        xys = [x1, y1, x2, y2, x3, y3, x4, y4]

        four_cells = []
        for i in range(4):
            x = xys[i * 2]
            y = xys[(i * 2) + 1]

            if x < 112.5 and y < 112.5:
                four_cells.append(1)
            if 112.5 < x < 225 and y < 112.5:
                four_cells.append(2)
            if 225 < x < 337.5 and y < 112.5:
                four_cells.append(3)
            if 337.5 < x <= 450 and y < 112.5:
                four_cells.append(4)

            if x < 112.5 and 112.5 < y < 225:
                four_cells.append(5)
            if 112.5 < x < 225 and 112.5 < y < 225:
                four_cells.append(6)
            if 225 < x < 337.5 and 112.5 < y < 225:
                four_cells.append(7)
            if 337.5 < x <= 450 and 112.5 < y < 225:
                four_cells.append(8)

            if x < 112.5 and 225 < y < 337.5:
                four_cells.append(9)
            if 112.5 < x < 225 and 225 < y < 337.5:
                four_cells.append(10)
            if 225 < x < 337.5 and 225 < y < 337.5:
                four_cells.append(11)
            if 337.5 < x <= 450 and 225 < y < 337.5:
                four_cells.append(12)

            if x < 112.5 and 337.5 < y <= 450:
                four_cells.append(13)
            if 112.5 < x < 225 and 337.5 < y <= 450:
                four_cells.append(14)
            if 225 < x < 337.5 and 337.5 < y <= 450:
                four_cells.append(15)
            if 337.5 < x <= 450 and 337.5 < y <= 450:
                four_cells.append(16)
        answer = get_occupied_cells(four_cells)
        count += 1
        for ans in answer:
            answers.append(ans)
    answers = sorted(list(answers))
    return list(set(answers))


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
    overall_deadline = monotonic() + 120

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
                elif "squares" in title_wrapper.text:
                    if verbose:
                        print("Square captcha found....")
                    img_urls = get_all_captcha_img_urls(driver)
                    download_img(0, img_urls[0])
                    if is_classifier:
                        answers = classify_grid_cells(target_set, 4, verbose, model)
                    else:
                        answers = square_solver(target_set, verbose, model)
                    if len(answers) >= 1 and len(answers) < 16:
                        captcha = "squares"
                        break
                    else:
                        reload.click()
                elif "none" in title_wrapper.text:
                    if verbose:
                        print("found a 3x3 dynamic captcha")
                    img_urls = get_all_captcha_img_urls(driver)
                    download_img(0, img_urls[0])
                    if is_classifier:
                        answers = classify_grid_cells(target_set, 3, verbose, model)
                    else:
                        answers = dynamic_and_selection_solver(target_set, verbose, model)
                    if len(answers) >= 1:
                        captcha = "dynamic"
                        break
                    else:
                        reload.click()
                else:
                    if verbose:
                        print("found a 3x3 one time selection captcha")
                    img_urls = get_all_captcha_img_urls(driver)
                    download_img(0, img_urls[0])
                    if is_classifier:
                        answers = classify_grid_cells(target_set, 3, verbose, model)
                    else:
                        answers = dynamic_and_selection_solver(target_set, verbose, model)
                    if len(answers) >= 1:
                        captcha = "selection"
                        break
                    else:
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
            elif captcha == "selection" or captcha == "squares":
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
