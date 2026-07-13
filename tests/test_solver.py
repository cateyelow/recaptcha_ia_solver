from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock


def _stub_oiv7():
    """Mimic Open Images V7 class catalog (subset)."""
    return SimpleNamespace(
        names={
            42: "Bicycle", 73: "Bus", 90: "Car", 342: "Motorcycle", 52: "Boat",
            558: "Truck", 522: "Taxi", 548: "Traffic light", 190: "Fire hydrant",
            495: "Stop sign", 370: "Parking meter", 489: "Stairs", 364: "Palm tree",
            546: "Tower", 550: "Train", 567: "Vehicle", 302: "Land vehicle",
        },
        task="detect",
    )


def _stub_classifier():
    """Mimic the fine-tuned classifier (verytuffcat layout)."""
    return SimpleNamespace(
        names={
            0: "bicycle", 1: "bridge", 2: "bus", 3: "car", 4: "chimney",
            5: "crosswalk", 6: "hydrant", 7: "motorcycle", 8: "mountain",
            9: "other", 10: "palm", 11: "stair", 12: "traffic light",
        },
        task="classify",
    )


def test_resolve_target_classes_oiv7_basic_terms():
    from recaptcha_ia_solver.solver import _resolve_target_classes

    m = _stub_oiv7()
    assert _resolve_target_classes("Select all images with bicycles", m) == {42}
    assert _resolve_target_classes("Select all images with buses", m) == {73}
    assert _resolve_target_classes("Select all images with cars", m) == {90}
    assert _resolve_target_classes("Select all images with motorcycles", m) == {342}


def test_resolve_target_classes_oiv7_compound_terms():
    from recaptcha_ia_solver.solver import _resolve_target_classes

    m = _stub_oiv7()
    assert _resolve_target_classes("Select all images with fire hydrants", m) == {190}
    assert _resolve_target_classes("Select all images with traffic lights", m) == {548}
    assert _resolve_target_classes("Select all images with palm trees", m) == {364}
    assert _resolve_target_classes("Select all images with parking meters", m) == {370}
    assert _resolve_target_classes("Select all images with stop signs", m) == {495}


def test_resolve_target_classes_oiv7_vehicle_umbrella():
    from recaptcha_ia_solver.solver import _resolve_target_classes

    m = _stub_oiv7()
    got = _resolve_target_classes("Select all images of vehicles", m)
    assert {90, 73, 558, 342, 522, 567, 302} <= got


def test_resolve_target_classes_classifier_extra_categories():
    """Categories absent from OIV7 must resolve once classifier is loaded."""
    from recaptcha_ia_solver.solver import _resolve_target_classes

    m = _stub_classifier()
    assert _resolve_target_classes("Select all images with bridges", m) == {1}
    assert _resolve_target_classes("Select all images with chimneys", m) == {4}
    assert _resolve_target_classes("Select all images with crosswalks", m) == {5}
    assert _resolve_target_classes("Select all images with mountains", m) == {8}


def test_resolve_target_classes_classifier_aliases():
    from recaptcha_ia_solver.solver import _resolve_target_classes

    m = _stub_classifier()
    # "fire hydrant" phrase should map even though classifier names it "hydrant".
    assert _resolve_target_classes("Select all images with fire hydrants", m) == {6}
    # "palm tree" -> classifier "palm".
    assert _resolve_target_classes("Select all images with palm trees", m) == {10}


def test_resolve_target_classes_oiv7_lacks_classifier_categories():
    from recaptcha_ia_solver.solver import _resolve_target_classes

    m = _stub_oiv7()
    # OIV7 has no Bridge/Chimney/Crosswalk/Mountain, so detection alone cannot
    # answer these — solve_recaptcha relies on the classifier to fill the gap.
    assert _resolve_target_classes("Select all images with bridges", m) == set()
    assert _resolve_target_classes("Select all images with chimneys", m) == set()
    assert _resolve_target_classes("Select all images with crosswalks", m) == set()
    assert _resolve_target_classes("Select all images with mountains", m) == set()


def test_resolve_target_classes_unknown_phrase_returns_empty():
    from recaptcha_ia_solver.solver import _resolve_target_classes

    m = _stub_oiv7()
    assert _resolve_target_classes("Select all squares with helicopters", m) == set()
    assert _resolve_target_classes("", m) == set()


def test_resolve_model_path_existing_relative_resolved_to_project_root(tmp_path, monkeypatch):
    from recaptcha_ia_solver import solver as M

    # Existing relative path stays as-is (it works from project root).
    if M._resolve_model_path.__module__:
        # When CWD is project root we expect verbatim or project-root-relative
        result = M._resolve_model_path("recaptcha_ia_solver/solver.py")
        assert result.endswith("recaptcha_ia_solver/solver.py")


def test_resolve_model_path_passthrough_for_bare_weight_name():
    """Ultralytics auto-downloads bare names like 'yolov8x-oiv7.pt'."""
    from recaptcha_ia_solver.solver import _resolve_model_path

    assert _resolve_model_path("yolov8x-oiv7.pt") == "yolov8x-oiv7.pt"


def test_resolve_model_path_absolute_passthrough():
    from recaptcha_ia_solver.solver import _resolve_model_path

    assert _resolve_model_path("/tmp/never.pt") == "/tmp/never.pt"


def test_classify_grid_cells_returns_one_indexed_matches(tmp_path, monkeypatch):
    """
    classify_grid_cells should slice the saved 0.png into grid_n*grid_n cells,
    feed them all to the classifier in one batch, and emit 1-indexed positions
    whose top-1 class is in target_set.
    """
    import os
    import numpy as np
    from PIL import Image

    from recaptcha_ia_solver import solver as M

    cwd = tmp_path
    monkeypatch.chdir(cwd)
    os.makedirs("recaptcha_images", exist_ok=True)
    Image.fromarray(np.zeros((300, 300, 3), dtype=np.uint8)).save("recaptcha_images/0.png")

    fake_model = MagicMock()
    fake_model.task = "classify"

    # Each call to model.predict is given a list of 9 cells; return a list of
    # results matching that length where cells 0,4,8 (1-indexed: 1,5,9) win.
    def fake_predict(cells, task=None, verbose=None):
        results = []
        for idx, _ in enumerate(cells):
            top1 = 7 if idx in (0, 4, 8) else 9
            results.append(
                SimpleNamespace(probs=SimpleNamespace(top1=top1, top1conf=0.9))
            )
        return results

    fake_model.predict.side_effect = fake_predict

    answers = M.classify_grid_cells({7}, 3, verbose=False, model=fake_model)
    assert sorted(answers) == [1, 5, 9]


def test_write_dynamic_grid_image_composes_individual_tiles(tmp_path, monkeypatch):
    """Dynamic 3x3 challenges can expose nine independent 100px tile images.
    The solver must build a 300x300 composite before running grid detection."""
    import os
    from PIL import Image

    from recaptcha_ia_solver import solver as M

    monkeypatch.chdir(tmp_path)
    os.makedirs("recaptcha_images", exist_ok=True)
    tiles = []
    for idx in range(9):
        image = Image.new("RGB", (10, 10), (idx, idx + 10, idx + 20))
        tiles.append(image)

    M._write_dynamic_grid_image(tiles, grid_n=3, out_path="recaptcha_images/0.png")

    composed = Image.open("recaptcha_images/0.png").convert("RGB")
    assert composed.size == (30, 30)
    assert composed.getpixel((5, 5)) == (0, 10, 20)
    assert composed.getpixel((15, 5)) == (1, 11, 21)
    assert composed.getpixel((25, 25)) == (8, 18, 28)


def test_download_dynamic_grid_prefers_screenshot_for_shared_urls(monkeypatch):
    """A rendered 3x3 grid can reuse one composite URL for all nine cells.

    When screenshot mode is enabled, the browser rendering is authoritative
    and must be captured before the URL-shape shortcuts.  Otherwise the solver
    bypasses the matching browser proxy and can block in a direct HTTP fetch.
    """
    from recaptcha_ia_solver import solver as M

    driver = object()
    screenshots = []
    monkeypatch.setenv("RECAPTCHA_DYNAMIC_SCREENSHOT", "1")
    monkeypatch.setattr(
        M,
        "_screenshot_grid_to_png",
        lambda actual_driver: screenshots.append(actual_driver) or True,
    )
    monkeypatch.setattr(
        M,
        "download_img",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("URL download used before browser screenshot")
        ),
    )

    M.download_dynamic_grid_img(["https://example.test/composite"] * 9, 3, driver)

    assert screenshots == [driver]


def test_get_all_captcha_img_urls_uses_one_browser_script_call():
    """Remote CDP drivers must not pay one WebDriver round-trip per tile."""
    from recaptcha_ia_solver import solver as M

    urls = [f"https://example.test/{idx}.png" for idx in range(9)]

    class FakeImage:
        def get_attribute(self, name):
            raise AssertionError(f"per-element get_attribute used for {name}")

    class FakeDriver:
        def __init__(self):
            self.scripts = []

        def execute_script(self, script):
            self.scripts.append(script)
            return urls

        def find_elements(self, *args, **kwargs):
            return [FakeImage() for _ in range(9)]

    driver = FakeDriver()

    assert M.get_all_captcha_img_urls(driver) == urls
    assert len(driver.scripts) == 1
    assert "querySelectorAll" in driver.scripts[0]


def test_get_all_new_dynamic_captcha_urls_reuses_single_browser_script_call():
    """Dynamic polling must not restore one remote round-trip per tile."""
    from recaptcha_ia_solver import solver as M

    before = [f"https://example.test/before-{idx}.png" for idx in range(9)]
    current = before.copy()
    current[1] = "https://example.test/after-1.png"
    current[4] = "https://example.test/after-4.png"

    class FakeDriver:
        def __init__(self):
            self.scripts = []

        def execute_script(self, script):
            self.scripts.append(script)
            return current

        def find_elements(self, *_args, **_kwargs):
            raise AssertionError("per-element dynamic URL lookup used")

    driver = FakeDriver()

    is_new, urls = M.get_all_new_dynamic_captcha_img_urls(
        [2, 5], before, driver
    )

    assert is_new is True
    assert urls == current
    assert len(driver.scripts) == 1


def test_get_all_new_dynamic_captcha_urls_keeps_script_fallback():
    """Drivers without script support still use the legacy element path."""
    from recaptcha_ia_solver import solver as M

    before = [f"https://example.test/before-{idx}.png" for idx in range(9)]
    current = before.copy()
    current[2] = "https://example.test/after-2.png"

    class FakeImage:
        def __init__(self, url):
            self.url = url

        def get_attribute(self, name):
            assert name == "src"
            return self.url

    class FakeDriver:
        def execute_script(self, _script):
            raise RuntimeError("script unsupported")

        def find_elements(self, *_args, **_kwargs):
            return [FakeImage(url) for url in current]

    is_new, urls = M.get_all_new_dynamic_captcha_img_urls(
        [3], before, FakeDriver()
    )

    assert is_new is True
    assert urls == current


def test_click_cells_batches_element_lookup_and_action_chain_perform(monkeypatch):
    """Remote CDP must pay one lookup and one perform for a click batch."""
    from recaptcha_ia_solver import solver as M

    class FakeElement:
        def __init__(self, index):
            self.index = index

        def is_displayed(self):
            return True

        def is_enabled(self):
            return True

        def click(self):
            raise AssertionError("plain element click fallback used")

    class FakeDriver:
        def __init__(self):
            self.cells = [FakeElement(index) for index in range(1, 10)]
            self.find_element_calls = 0
            self.find_elements_calls = 0

        def find_element(self, _by, xpath):
            self.find_element_calls += 1
            index = int(xpath.rsplit("[", 1)[1].rstrip("]"))
            return self.cells[index - 1]

        def find_elements(self, _by, _xpath):
            self.find_elements_calls += 1
            return self.cells

        def execute_script(self, *_args):
            raise AssertionError("JavaScript click fallback used")

    class FakeActionChains:
        instances = []
        perform_calls = 0

        def __init__(self, driver):
            self.driver = driver
            self.current = None
            self.click_targets = []
            self.offset_moves = 0
            self.pauses = 0
            self.__class__.instances.append(self)

        def move_to_element_with_offset(self, element, _dx, _dy):
            self.current = element
            self.offset_moves += 1
            return self

        def move_to_element(self, element):
            self.current = element
            return self

        def pause(self, _seconds):
            self.pauses += 1
            return self

        def click(self):
            self.click_targets.append(self.current.index)
            return self

        def perform(self):
            self.__class__.perform_calls += 1

    driver = FakeDriver()
    monkeypatch.setattr(M, "ActionChains", FakeActionChains)
    monkeypatch.setattr(M, "random_delay", lambda **_kwargs: None)

    M._click_cells(driver, [1, 3, 5])

    assert driver.find_element_calls == 0
    assert driver.find_elements_calls == 1
    assert len(FakeActionChains.instances) == 1
    chain = FakeActionChains.instances[0]
    assert FakeActionChains.perform_calls == 1
    assert chain.click_targets == [1, 3, 5]
    assert chain.offset_moves >= 2 * 3
    assert chain.pauses >= 3 * 3


def test_classify_grid_cells_4x4(tmp_path, monkeypatch):
    """4x4 squares-mode also works through the same code path."""
    import os
    import numpy as np
    from PIL import Image

    from recaptcha_ia_solver import solver as M

    monkeypatch.chdir(tmp_path)
    os.makedirs("recaptcha_images", exist_ok=True)
    Image.fromarray(np.zeros((450, 450, 3), dtype=np.uint8)).save("recaptcha_images/0.png")

    fake_model = MagicMock()
    fake_model.task = "classify"
    fake_model.predict.side_effect = lambda cells, task=None, verbose=None: [
        SimpleNamespace(
            probs=SimpleNamespace(
                top1=3 if idx in (0, 5, 10, 15) else 9, top1conf=0.95,
            )
        )
        for idx, _ in enumerate(cells)
    ]

    answers = M.classify_grid_cells({3}, 4, verbose=False, model=fake_model)
    assert sorted(answers) == [1, 6, 11, 16]


def test_classify_grid_cells_rejects_low_confidence(tmp_path, monkeypatch):
    """Predictions below RECAPTCHA_YOLO_MIN_CONF must be dropped — false
    clicks are far costlier than missed clicks for reCAPTCHA, so we tolerate
    a recall hit to keep precision high."""
    import os
    import numpy as np
    from PIL import Image

    from recaptcha_ia_solver import solver as M

    monkeypatch.chdir(tmp_path)
    os.makedirs("recaptcha_images", exist_ok=True)
    Image.fromarray(np.zeros((300, 300, 3), dtype=np.uint8)).save("recaptcha_images/0.png")
    monkeypatch.setenv("RECAPTCHA_YOLO_MIN_CONF", "0.6")

    fake_model = MagicMock()
    fake_model.task = "classify"
    confidences = [0.95, 0.40, 0.95, 0.10, 0.80, 0.95, 0.50, 0.95, 0.90]
    fake_model.predict.side_effect = lambda cells, task=None, verbose=None: [
        SimpleNamespace(probs=SimpleNamespace(top1=7, top1conf=confidences[i]))
        for i in range(len(cells))
    ]

    answers = M.classify_grid_cells({7}, 3, verbose=False, model=fake_model)
    # Cells with conf < 0.6 (indices 1, 3, 6 -> 1-indexed 2, 4, 7) must be
    # filtered out even though their top-1 class matched.
    assert sorted(answers) == [1, 3, 5, 6, 8, 9]


def test_recaptcha_to_oiv7_mapping_ordering_handles_compound_first():
    """
    The ordering of RECAPTCHA_TO_OIV7 must put compound terms (e.g.
    "fire hydrant") before their substrings ("hydrant"), otherwise re.search
    short-circuits on the wrong term and returns the substring's classes only.
    """
    from recaptcha_ia_solver.solver import RECAPTCHA_TO_OIV7

    keys = list(RECAPTCHA_TO_OIV7.keys())
    pairs = [
        ("fire hydrant", "hydrant"),
        ("palm tree", "palm"),  # classifier-only alias
        ("traffic light", "tower"),
    ]
    for compound, substring in pairs:
        if compound in keys and substring in keys:
            assert keys.index(compound) < keys.index(substring), (
                f"{compound!r} must appear before {substring!r} in mapping"
            )
