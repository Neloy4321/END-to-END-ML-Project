"""Read-only local HTTP smoke test. Uses synthetic answers only, no user data."""
import argparse
import json
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from sleep_project.behavioral_assessment import FORM_FIELDS


def request(base, path, data=None, headers=None):
    body = None if data is None else data
    req = Request(base + path, data=body, headers=headers or {})
    try:
        with urlopen(req, timeout=5) as response:
            return response.status, response.headers, response.read().decode("utf-8")
    except HTTPError as exc:
        return exc.code, exc.headers, exc.read().decode("utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    root = args.url.rstrip("/")
    for _ in range(20):
        try:
            code, _, body = request(root, "/readyz")
            if code == 200 and json.loads(body)["status"] == "ready":
                break
        except (URLError, ValueError, TimeoutError):
            time.sleep(0.5)
    else:
        raise AssertionError("Local server is not ready")

    code, headers, home = request(root, "/")
    assert code == 200 and "Sleep Survey Explorer" in home
    assert headers.get("Cache-Control") == "no-store"
    assert headers.get("X-Content-Type-Options") == "nosniff"
    artifact = json.loads(Path("models/sleep_duration_multiclass_v2.json").read_text(encoding="utf-8"))
    answers = {key: artifact["category_options"][col][0] for key, col, _ in FORM_FIELDS}
    code, _, html = request(root, "/assess", urlencode(answers).encode())
    assert code == 200 and "Model-estimated questionnaire response" in html
    rich = dict(answers, age="18-30", medical_self_report="Asthma")
    code, _, html = request(root, "/assess", urlencode(rich).encode())
    assert code == 200 and "Reported medical condition: Asthma" in html
    payload = json.dumps({"answers": answers, "bed_time": "After 2 AM", "wake_time": "Before 6 AM"}).encode()
    code, _, outcome = request(root, "/api/compare", payload, {"Content-Type": "application/json"})
    assert code == 200 and len(json.loads(outcome)["scenario"]["category_probabilities"]) == 4
    bad_answers = dict(answers, bed_time="invented")
    assert request(root, "/assess", urlencode(bad_answers).encode())[0] == 422
    assert request(root, "/assess", b"x" * 16385)[0] == 413
    assert request(root, "/predict", b"")[0] == 410
    assert request(root, "/metrics")[0] in (401, 404)
    print("PASS: ready, form, optional medical description, comparison, validation, limits, retired medical endpoint, metrics auth")


if __name__ == "__main__":
    main()
