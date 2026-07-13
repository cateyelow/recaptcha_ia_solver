"""End-to-end reCAPTCHA pass-rate test on Google's official demo page.

Flow per try: open the demo, click the checkbox, let `solve_recaptcha` walk any
image challenges (it already loops/reloads internally on a miss), submit the
form, and read the verdict from "Verification Success" / "Hooray".

This box is shared with ~20 other sessions and is memory-saturated, so a FRESH
Chrome is frequently OOM-killed the instant it launches (a Connection-refused
fires before the solver does anything). That is a *measurement* artifact, not a
solve failure. Two mitigations:

  1. Reuse ONE Chrome across tries (re-`get` the demo) so the risky launch is
     paid once, not per try — once a process survives launch it keeps running.
  2. When the browser does die mid-try, classify the try as INVALID and rebuild;
     only tries that actually reached `verify` count toward the pass rate.

`n` (argv[1]) is the target number of VALID tries. RECAPTCHA_HEADLESS=1 (default
here) runs --headless=new — lighter on the shared box and a lower-trust session,
closer to the bot-like session Gangnam-Unni hits in production.
"""

import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

# Run from the repo root so `models/*.pt` resolves, and make the package
# importable before pulling the solver in.
PROJECT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT)
sys.path.insert(0, str(PROJECT))

from recaptcha_ia_solver.solver import (  # noqa: E402
    _is_dead_driver_error,
    solve_recaptcha,
)


DEMO_URL = "https://www.google.com/recaptcha/api2/demo?hl=en"


def _resolve_chrome():
    """Pick a stable Chrome binary and its real major version.

    A box can carry several Chrome builds (e.g. a snap Chromium 149 and a deb
    google-chrome 146). undetected_chromedriver auto-picks one binary and
    separately fetches a chromedriver for ``version_main``; when those majors
    differ every launch dies with "session not created: ... only supports Chrome
    version N". So we pin an explicit binary (``RECAPTCHA_CHROME_BINARY``
    override, else the usual google-chrome paths) and read its actual major so
    the driver always matches the browser. Returns ``(path, major)`` with either
    element ``None`` to let undetected_chromedriver auto-detect that part.
    """
    candidates = [
        os.environ.get("RECAPTCHA_CHROME_BINARY"),
        shutil.which("google-chrome"),
        "/usr/bin/google-chrome",
        "/opt/google/chrome/google-chrome",
        shutil.which("google-chrome-stable"),
        shutil.which("chromium"),
    ]
    for path in candidates:
        if not path or not os.path.exists(path):
            continue
        try:
            out = subprocess.run(
                [path, "--version"], capture_output=True, text=True, timeout=10
            ).stdout
        except Exception:
            continue
        m = re.search(r"\b(\d+)\.\d+\.\d+", out)
        if m:
            return path, int(m.group(1))
    return None, None


def make_driver(headless: bool = True):
    options = uc.ChromeOptions()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--lang=en-US")
    options.add_argument("--window-size=1280,1024")
    options.add_argument("--ignore-certificate-errors")
    options.add_argument("--ignore-ssl-errors")
    options.add_argument("--allow-insecure-localhost")
    # Memory-lean flags: the box is resource-saturated (shared with ~20
    # sessions), so trim Chrome's footprint to cut OOM-on-launch crashes.
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-software-rasterizer")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-background-networking")
    binary, major = _resolve_chrome()
    kwargs = {"options": options, "headless": headless}
    if binary:
        kwargs["browser_executable_path"] = binary
    if major:
        kwargs["version_main"] = major
    driver = uc.Chrome(**kwargs)
    driver.set_page_load_timeout(60)
    return driver


def run_once(driver, verbose: bool = False) -> bool:
    """One solve on an already-live driver. Returns True iff verified.

    Raises on a dead browser (so the caller can rebuild it) or any other error;
    the caller decides whether that's an INVALID crash or a real FAIL.
    """
    driver.get(DEMO_URL)
    WebDriverWait(driver, 15).until(
        EC.presence_of_element_located((By.XPATH, '//iframe[@title="reCAPTCHA"]'))
    )
    # solve_recaptcha clicks the checkbox, walks challenges, and reloads on a
    # miss until its own deadline.
    solve_recaptcha(driver=driver, verbose=verbose)

    driver.switch_to.default_content()
    original_url = driver.current_url
    # requestSubmit() is more reliable than clicking the button: it dodges the
    # transparent post-checkbox overlay that intercepts native clicks for ~1-2s.
    driver.execute_script(
        "document.getElementById('recaptcha-demo-form').requestSubmit();"
    )
    WebDriverWait(driver, 30).until(
        lambda d: d.current_url != original_url
        or "Verification Success" in d.page_source
        or "Hooray" in d.page_source
    )
    body = driver.page_source
    return ("Verification Success" in body) or ("Hooray" in body)


def main(n: int = 8, verbose: bool = False, headless: bool = True):
    valid = []          # one bool per try that actually reached verify
    tries = 0
    crashes = 0
    max_tries = n * 5   # cap so a fully wedged box can't loop forever
    driver = None
    while len(valid) < n and tries < max_tries:
        tries += 1
        if tries > 1:
            time.sleep(8)  # ease rate-based suspicion between tries
        try:
            if driver is None:
                driver = make_driver(headless=headless)
            ok = run_once(driver, verbose=verbose)
            valid.append(ok)
            print(f"[try {tries:>2}] {'PASS' if ok else 'FAIL':<4} "
                  f"-> {sum(valid)}/{len(valid)} valid PASS")
        except Exception as e:  # noqa: BLE001
            last = traceback.format_exc(limit=6).splitlines()[-1][:80]
            if _is_dead_driver_error(e):
                crashes += 1
                print(f"[try {tries:>2}] INVALID (browser died, rebuilding) {last}")
                try:
                    driver.quit()
                except Exception:
                    pass
                driver = None  # rebuild on next loop
            else:
                # solver gave up, submit timed out, etc. — a real FAIL of a
                # valid try, not a measurement crash.
                valid.append(False)
                print(f"[try {tries:>2}] FAIL (no verify) "
                      f"-> {sum(valid)}/{len(valid)} valid PASS  {last}")
        sys.stdout.flush()
    if driver is not None:
        try:
            driver.quit()
        except Exception:
            pass
    passed = sum(valid)
    nv = len(valid)
    rate = 100 * passed / nv if nv else 0.0
    print(f"\nfinal: {passed}/{nv} valid passed = {rate:.1f}%  "
          f"({tries} tries, {crashes} crashes)")
    return passed, nv


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    verbose = "-v" in sys.argv
    headless = os.environ.get("RECAPTCHA_HEADLESS", "1") == "1"
    _passed, _nv = main(n=n, verbose=verbose, headless=headless)
    # Measurement harness: exit 0 only if the pass rate over VALID tries cleared
    # a high bar (>=90%); a low rate is a meaningful non-zero exit, not success.
    sys.exit(0 if (_nv > 0 and _passed / _nv >= 0.9) else 1)
