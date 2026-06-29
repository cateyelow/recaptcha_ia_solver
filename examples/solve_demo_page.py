"""End-to-end reCAPTCHA pass-rate test on Google's official demo page.

Flow per attempt:
  1. open https://www.google.com/recaptcha/api2/demo
  2. switch to the checkbox iframe and click it
  3. delegate to `solve_recaptcha` (which loops through any image challenges)
  4. submit the form by clicking the page's submit button
  5. verdict: if the response page contains "Verification Success" → PASS,
     "Try again" / nothing → FAIL.

We DON'T pre-judge "headless = bad" — we only run with a visible display ($DISPLAY).
The harness leaks no state between attempts (new browser per try) so the score
reflects model+solver behavior, not warmed-up cookies.

Plain undetected_chromedriver — no seleniumwire MITM, which Google's modern
fingerprint analysis catches via TLS / CONNECT-pattern signals (you see the
bot-detected NoScript-fallback iframe after a few attempts when MITM is active).
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

# Run from the repo root so `models/recaptcha_classifier.pt` resolves, and make
# the package importable before pulling the solver in.
PROJECT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT)
sys.path.insert(0, str(PROJECT))

from recaptcha_ia_solver.solver import solve_recaptcha  # noqa: E402


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


def make_driver(headless: bool = False):
    options = uc.ChromeOptions()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--lang=en-US")
    options.add_argument("--window-size=1280,1024")
    # selenium-wire MITMs HTTPS with its own root cert; without these flags
    # Chrome shows the privacy-error interstitial instead of the demo page.
    options.add_argument("--ignore-certificate-errors")
    options.add_argument("--ignore-ssl-errors")
    options.add_argument("--allow-insecure-localhost")
    binary, major = _resolve_chrome()
    kwargs = {"options": options, "headless": headless}
    if binary:
        kwargs["browser_executable_path"] = binary
    if major:
        kwargs["version_main"] = major
    driver = uc.Chrome(**kwargs)
    driver.set_page_load_timeout(60)
    return driver


def attempt(verbose: bool = False):
    driver = make_driver(headless=False)
    try:
        driver.get(DEMO_URL)
        try:
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.XPATH, '//iframe[@title="reCAPTCHA"]'))
            )
        except Exception:
            os.makedirs("/tmp/realtest_dbg", exist_ok=True)
            driver.save_screenshot("/tmp/realtest_dbg/no_iframe.png")
            with open("/tmp/realtest_dbg/no_iframe.html", "w") as f:
                f.write(driver.page_source[:20000])
            return False, "checkbox iframe never appeared (see /tmp/realtest_dbg/)"
        # Hand off to the existing solver. It clicks the checkbox, walks
        # challenges, and exits when verified.
        solve_recaptcha(driver=driver, verbose=verbose)

        # Solver returns; submit the form and read the verdict from the resulting page.
        driver.switch_to.default_content()
        # reCAPTCHA leaves a z-index 2_000_000_000 transparent overlay during
        # its post-checkbox analysis window; native .click() throws
        # ElementClickInterceptedException for ~1-2s. Submitting via the form's
        # native requestSubmit() is more reliable than clicking the button:
        # button-click can race the overlay AND can no-op if the button isn't
        # the active element.
        original_url = driver.current_url
        driver.execute_script(
            "document.getElementById('recaptcha-demo-form').requestSubmit();"
        )
        WebDriverWait(driver, 30).until(
            lambda d: d.current_url != original_url
            or "Verification Success" in d.page_source
            or "Hooray" in d.page_source
        )
        body = driver.page_source
        passed = ("Verification Success" in body) or ("Hooray" in body)
        if not passed:
            os.makedirs("/tmp/realtest_dbg", exist_ok=True)
            try:
                driver.save_screenshot(f"/tmp/realtest_dbg/postsubmit_{int(time.time())}.png")
                with open(f"/tmp/realtest_dbg/postsubmit_{int(time.time())}.html", "w") as f:
                    f.write(body[:50000])
            except Exception:
                pass
        return passed, ("OK" if passed else "no Success text in response")
    except Exception as e:
        os.makedirs("/tmp/realtest_dbg", exist_ok=True)
        try:
            driver.save_screenshot(f"/tmp/realtest_dbg/fail_{int(time.time())}.png")
        except Exception:
            pass
        tb = traceback.format_exc(limit=8)
        return False, f"exception: {e.__class__.__name__}\n{tb}"
    finally:
        try:
            driver.quit()
        except Exception:
            pass


def main(n: int = 10, verbose: bool = False):
    results = []
    for i in range(1, n + 1):
        # Light inter-attempt delay so back-to-back fresh sessions don't
        # raise the rate-based suspicion score on Google's side.
        if i > 1:
            time.sleep(8)
        t0 = time.time()
        ok, note = attempt(verbose=verbose)
        dt = time.time() - t0
        results.append(ok)
        print(f"[{i:>2d}/{n}] {'PASS' if ok else 'FAIL'}  {dt:>5.1f}s  {note}")
        sys.stdout.flush()
    passed = sum(results)
    print(f"\nfinal: {passed}/{n} passed")
    return passed


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    verbose = "-v" in sys.argv
    sys.exit(0 if main(n=n, verbose=verbose) >= 9 else 1)
