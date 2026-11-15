"""URL feature extractor for early phishing detection.

All features are computed from the URL string alone, so a verdict can be
produced before the web page is requested (no network calls).
"""
import math
import re
from collections import Counter
from urllib.parse import urlsplit

import tldextract

# Offline extractor: uses the bundled public-suffix snapshot, never the network.
_EXTRACT = tldextract.TLDExtract(suffix_list_urls=())

SUSPICIOUS_WORDS = ["login", "signin", "verify", "account", "update", "secure", "bank",
                    "confirm", "password", "webscr", "ebayisapi", "wallet", "billing", "support"]
SHORTENERS = {"bit.ly", "goo.gl", "tinyurl.com", "t.co", "ow.ly", "is.gd", "buff.ly", "rebrand.ly",
              "cutt.ly", "shorturl.at", "tiny.cc", "rb.gy"}
RISKY_TLDS = {"tk", "ml", "ga", "cf", "gq", "xyz", "top", "zip", "click", "link", "work", "rest", "fit"}
IPV4 = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")

FEATURE_NAMES = [
    "url_length", "host_length", "path_length", "query_length", "num_subdomains", "path_depth",
    "num_dots", "num_hyphens", "num_underscores", "num_slashes", "num_digits", "num_at", "num_question",
    "num_equals", "num_ampersand", "num_percent", "num_tilde", "digit_ratio", "letter_ratio",
    "special_char_ratio", "url_entropy", "host_entropy", "is_ip_host", "has_port", "is_https",
    "has_www", "https_token_in_host_or_path", "double_slash_in_path", "num_suspicious_words",
    "is_shortener", "risky_tld", "tld_length", "domain_length", "domain_has_hyphen",
    "domain_digit_count", "longest_token_length", "num_params",
]


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in counts.values())


def normalise(url: str) -> str:
    url = url.strip()
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        url = "http://" + url
    return url


def extract(url: str) -> dict:
    url = normalise(url)
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    path, query = parts.path or "", parts.query or ""
    ext = _EXTRACT(host)
    sub = ext.subdomain
    sub_parts = [p for p in sub.split(".") if p and p != "www"] if sub else []
    lower = url.lower()
    letters = sum(ch.isalpha() for ch in url)
    digits = sum(ch.isdigit() for ch in url)
    specials = sum(not ch.isalnum() for ch in url)
    tokens = re.split(r"[/\.\?=&\-_:]+", lower)
    try:
        port = parts.port
    except ValueError:
        port = None
    return {
        "url_length": len(url),
        "host_length": len(host),
        "path_length": len(path),
        "query_length": len(query),
        "num_subdomains": len(sub_parts),
        "path_depth": len([p for p in path.split("/") if p]),
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_underscores": url.count("_"),
        "num_slashes": url.count("/"),
        "num_digits": digits,
        "num_at": url.count("@"),
        "num_question": url.count("?"),
        "num_equals": url.count("="),
        "num_ampersand": url.count("&"),
        "num_percent": url.count("%"),
        "num_tilde": url.count("~"),
        "digit_ratio": digits / len(url),
        "letter_ratio": letters / len(url),
        "special_char_ratio": specials / len(url),
        "url_entropy": _entropy(lower),
        "host_entropy": _entropy(host),
        "is_ip_host": int(bool(IPV4.match(host))),
        "has_port": int(port is not None),
        "is_https": int(parts.scheme == "https"),
        "has_www": int(host.startswith("www.")),
        "https_token_in_host_or_path": int("https" in host or "https" in path.lower()),
        "double_slash_in_path": int("//" in path),
        "num_suspicious_words": sum(w in lower for w in SUSPICIOUS_WORDS),
        "is_shortener": int(ext.registered_domain in SHORTENERS),
        "risky_tld": int(ext.suffix.split(".")[-1] in RISKY_TLDS if ext.suffix else 0),
        "tld_length": len(ext.suffix),
        "domain_length": len(ext.domain),
        "domain_has_hyphen": int("-" in ext.domain),
        "domain_digit_count": sum(ch.isdigit() for ch in ext.domain),
        "longest_token_length": max((len(t) for t in tokens), default=0),
        "num_params": len([q for q in query.split("&") if q]),
    }


def registered_domain(url: str) -> str:
    host = (urlsplit(normalise(url)).hostname or "").lower()
    ext = _EXTRACT(host)
    return ext.registered_domain or host


HOST_FEATURE_NAMES = [
    "host_length", "num_subdomains", "host_dots", "host_hyphens", "host_digits", "host_digit_ratio",
    "host_entropy", "is_ip_host", "host_suspicious_words", "is_shortener", "risky_tld", "tld_length",
    "domain_length", "domain_has_hyphen", "domain_digit_count", "domain_entropy", "longest_host_token",
]


def extract_host(url: str) -> dict:
    """Features from the host name only (scheme, 'www.' and path removed).

    Used to avoid the structural shortcut in PhiUSIIL, where every legitimate
    URL is a bare 'https://www.<domain>' homepage.
    """
    host = (urlsplit(normalise(url)).hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    ext = _EXTRACT(host)
    sub_parts = [p for p in ext.subdomain.split(".") if p] if ext.subdomain else []
    digits = sum(ch.isdigit() for ch in host)
    return {
        "host_length": len(host),
        "num_subdomains": len(sub_parts),
        "host_dots": host.count("."),
        "host_hyphens": host.count("-"),
        "host_digits": digits,
        "host_digit_ratio": digits / max(len(host), 1),
        "host_entropy": _entropy(host),
        "is_ip_host": int(bool(IPV4.match(host))),
        "host_suspicious_words": sum(w in host for w in SUSPICIOUS_WORDS),
        "is_shortener": int(ext.registered_domain in SHORTENERS),
        "risky_tld": int(ext.suffix.split(".")[-1] in RISKY_TLDS if ext.suffix else 0),
        "tld_length": len(ext.suffix),
        "domain_length": len(ext.domain),
        "domain_has_hyphen": int("-" in ext.domain),
        "domain_digit_count": sum(ch.isdigit() for ch in ext.domain),
        "domain_entropy": _entropy(ext.domain),
        "longest_host_token": max((len(t) for t in re.split(r"[.\-]+", host)), default=0),
    }
