"""PhishGuard prototype: paste a URL, get an early phishing verdict with reasons.

Run: python app.py  (then open http://127.0.0.1:5000)
"""
import os
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from flask import Flask, render_template_string, request

from features import FEATURE_NAMES, HOST_FEATURE_NAMES, extract, extract_host

_DEFAULT = Path(__file__).parent / "out" / "phishguard_v3_url_xgb.joblib"
# Under gunicorn sys.argv holds gunicorn's args, so only honour argv when run directly.
MODEL_PATH = Path(os.environ.get("MODEL_PATH") or (sys.argv[1] if __name__ == "__main__" and len(sys.argv) > 1 else _DEFAULT))
bundle = joblib.load(MODEL_PATH)
model, threshold = bundle["model"], bundle["threshold"]
HOST = bundle.get("kind") == "host"
NAMES = HOST_FEATURE_NAMES if HOST else FEATURE_NAMES
EXTRACT = extract_host if HOST else extract
explainer = shap.TreeExplainer(model)
IPV6_LITERAL = re.compile(r"^[0-9a-f:]+$")

READABLE = {
    "url_length": "URL length", "host_length": "Host name length", "path_length": "Path length",
    "query_length": "Query string length", "num_subdomains": "Number of subdomains", "path_depth": "Path depth",
    "num_dots": "Number of dots", "num_hyphens": "Number of hyphens", "num_slashes": "Number of slashes",
    "num_digits": "Number of digits", "digit_ratio": "Share of digits", "letter_ratio": "Share of letters",
    "special_char_ratio": "Share of special characters", "url_entropy": "Randomness (entropy) of URL",
    "host_entropy": "Randomness (entropy) of host", "is_ip_host": "IP address used as host", "is_https": "Uses HTTPS",
    "has_www": "Starts with www", "num_suspicious_words": "Suspicious words (login, verify, ...)",
    "tld_length": "Top-level domain length", "domain_length": "Domain name length", "num_equals": "Number of '='",
    "num_question": "Number of '?'", "longest_token_length": "Longest word in URL", "num_params": "Number of parameters",
    "risky_tld": "High-risk top-level domain", "is_shortener": "URL shortener", "domain_has_hyphen": "Hyphen in domain",
    "domain_digit_count": "Digits in domain", "host_dots": "Dots in host name", "host_hyphens": "Hyphens in host name", "host_digits": "Digits in host name", "host_digit_ratio": "Share of digits in host", "host_suspicious_words": "Suspicious words in host", "domain_entropy": "Randomness (entropy) of domain", "longest_host_token": "Longest word in host", "num_underscores": "Number of underscores", "num_percent": "Number of '%'",
}

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>PhishGuard – Early Phishing Detection</title>
<style>
body{font-family:Segoe UI,Arial,sans-serif;background:#f3f5f9;margin:0;color:#1d2433}
header{background:#1f3a5f;color:#fff;padding:18px 32px}header h1{margin:0;font-size:22px}header p{margin:4px 0 0;opacity:.8;font-size:13px}
main{max-width:760px;margin:28px auto;padding:0 16px}
.card{background:#fff;border-radius:10px;box-shadow:0 1px 4px rgba(0,0,0,.08);padding:22px;margin-bottom:18px}
input[type=text]{width:78%;padding:11px;font-size:15px;border:1px solid #c5ccd8;border-radius:6px}
button{padding:11px 20px;font-size:15px;background:#1f3a5f;color:#fff;border:0;border-radius:6px;cursor:pointer}
.verdict{font-size:22px;font-weight:700;padding:12px 16px;border-radius:8px}
.phish{background:#fdecec;color:#a61b1b;border:1px solid #f2b8b8}.legit{background:#e8f6ec;color:#1d6b35;border:1px solid #b6dfc2}
.bar{height:10px;background:#e3e7ee;border-radius:5px;overflow:hidden;margin:8px 0 2px}.fill{height:100%}
table{width:100%;border-collapse:collapse;font-size:14px}td,th{padding:7px 6px;border-bottom:1px solid #eef0f4;text-align:left}
.up{color:#a61b1b}.down{color:#1d6b35}small{color:#667}
</style></head><body>
<header><h1>PhishGuard</h1><p>Machine learning-based early detection of phishing websites — checks the URL before the page is opened</p></header>
<main>
<div class="card"><form method="post"><input type="text" name="url" placeholder="Paste a URL, e.g. http://secure-login-paypal.com.verify-account.xyz/signin" value="{{url or ''}}">
<button type="submit">Check URL</button></form></div>
{% if error %}<div class="card"><div class="verdict phish" style="font-size:16px">{{ error }}</div></div>{% endif %}
{% if result %}
<div class="card">
<div class="verdict {{ 'phish' if result.phishing else 'legit' }}">{{ '⚠ Likely PHISHING' if result.phishing else '✔ Likely LEGITIMATE' }}</div>
<p style="margin-top:14px">Phishing probability: <b>{{ '%.1f'|format(result.prob*100) }}%</b> &nbsp;<small>(threshold {{ '%.2f'|format(threshold) }})</small></p>
<div class="bar"><div class="fill" style="width:{{ result.prob*100 }}%;background:{{ '#c44e52' if result.phishing else '#3a8d55' }}"></div></div>
<p><small>Analysed in {{ '%.1f'|format(result.ms) }} ms without loading the page.</small></p>
<h3 style="margin-bottom:6px">Main reasons</h3>
<table><tr><th>Feature</th><th>Value</th><th>Effect</th></tr>
{% for r in result.reasons %}<tr><td>{{r.name}}</td><td>{{r.value}}</td><td class="{{ 'up' if r.shap>0 else 'down' }}">{{ '▲ towards phishing' if r.shap>0 else '▼ towards legitimate' }}</td></tr>{% endfor %}
</table></div>
{% endif %}
<p style="text-align:center"><small>Phase I prototype · XGBoost URL model (v3) trained on the PhiUSIIL dataset · for academic demonstration</small></p>
</main></body></html>"""

app = Flask(__name__)


class InvalidURL(ValueError):
    pass


def validate(url: str) -> None:
    """Reject input that has no usable host name (otherwise the model scores meaningless features)."""
    from urllib.parse import urlsplit
    from features import normalise
    try:
        host = urlsplit(normalise(url)).hostname
    except ValueError as e:
        raise InvalidURL(f"Could not parse this URL ({e}).") from e
    if not host or "." not in host and not IPV6_LITERAL.match(host):
        raise InvalidURL("Please enter a full URL with a valid host name, e.g. https://example.com/path")
    if len(url) > 2048:
        raise InvalidURL("URL is too long (max 2048 characters).")


def predict(url: str):
    import time
    validate(url)
    t = time.perf_counter()
    feats = EXTRACT(url)
    x = pd.DataFrame([feats], columns=NAMES)
    prob = float(model.predict_proba(x)[0, 1])
    ms = (time.perf_counter() - t) * 1000
    sv = explainer.shap_values(x)[0]
    order = np.argsort(-np.abs(sv))[:5]
    reasons = []
    for i in order:
        v = feats[NAMES[i]]
        reasons.append({"name": READABLE.get(NAMES[i], NAMES[i]), "value": round(v, 2) if isinstance(v, float) else v, "shap": float(sv[i])})
    return {"prob": prob, "phishing": prob >= threshold, "reasons": reasons, "ms": ms}


@app.route("/", methods=["GET", "POST"])
def index():
    url = request.form.get("url", "").strip() if request.method == "POST" else ""
    result, error = None, None
    if url:
        try:
            result = predict(url)
        except InvalidURL as e:
            error = str(e)
    return render_template_string(PAGE, url=url, result=result, error=error, threshold=threshold)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=False)
