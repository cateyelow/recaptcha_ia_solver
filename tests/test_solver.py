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
