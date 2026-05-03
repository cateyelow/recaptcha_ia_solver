"""recaptcha_ia_solver — image-grid reCAPTCHA solver using a fine-tuned YOLO classifier."""

from recaptcha_ia_solver.solver import (
    DEFAULT_YOLO_MODEL,
    DEFAULT_YOLO_FALLBACK_MODEL,
    RECAPTCHA_TO_OIV7,
    classify_grid_cells,
    is_solved,
    solve_recaptcha,
)

__all__ = [
    "DEFAULT_YOLO_MODEL",
    "DEFAULT_YOLO_FALLBACK_MODEL",
    "RECAPTCHA_TO_OIV7",
    "classify_grid_cells",
    "is_solved",
    "solve_recaptcha",
]

__version__ = "0.1.0"
