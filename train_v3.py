"""PhishGuard v3: URL-string model trained with realistic legitimate URLs.

Problem in v1/v2: every legitimate URL in PhiUSIIL is a bare 'https://www.<domain>'
homepage, so URL models learn "has a path / no www => phishing" and flag real
pages (github.com/.../issues, docs.python.org/3/...) as phishing.

Fix: augment legitimate rows with realistic variants of the SAME real domains
(sub-domains, http/https, with/without www, typical paths, query strings,
including login/account pages that real sites also have). Domain-grouped split;
domains used in the hand-written sanity set are excluded from training.

Usage: python train_v3.py <PhiUSIIL csv> <out dir>
"""
import json, random, sys, warnings
from pathlib import Path

import joblib, numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier

from features import FEATURE_NAMES, extract, registered_domain

warnings.filterwarnings("ignore")
SEED = 42
rng = random.Random(SEED)
csv, out = Path(sys.argv[1]), Path(sys.argv[2])

HOLDOUT_DOMAINS = {"github.com", "wikipedia.org", "amazon.in", "google.com", "python.org", "irctc.co.in",
                   "stackoverflow.com", "onlinesbi.sbi", "youtube.com", "linkedin.com", "microsoft.com",
                   "bbc.co.uk", "nytimes.com", "flipkart.com", "paypal.com", "apple.com", "reddit.com"}

SUBS = ["www", "www", "www", "mail", "docs", "support", "blog", "shop", "help", "app", "m", "news", "en", "accounts", "api", "login"]
SEG = ["products", "product", "category", "blog", "news", "articles", "docs", "help", "support", "about", "contact", "login",
       "signin", "account", "settings", "profile", "users", "user", "wiki", "questions", "tagged", "issues", "pull", "search",
       "watch", "cart", "checkout", "orders", "pricing", "download", "en", "us", "in", "uk", "store", "forum", "topic", "post",
       "2023", "2024", "library", "reference", "guide", "api", "v1", "v2", "3", "dashboard", "home", "careers", "jobs", "events"]
WORDS = ["python", "data", "release", "notes", "getting-started", "how-to-reset", "annual-report", "summer-sale", "new-arrivals",
         "mobile-phones", "contact-us", "privacy-policy", "terms", "faq", "overview", "installation", "tutorial", "changelog",
         "team", "press", "index", "view", "list", "details", "inbox", "train-search", "flight-status", "my-orders"]
EXT = ["", "", "", ".html", ".htm", ".php", ".aspx", "/"]
QKEYS = ["id", "q", "page", "ref", "lang", "v", "utm_source", "sort", "tab", "p", "cat", "session"]


def rand_path():
    n = rng.choice([0, 1, 2, 2, 3, 4])
    segs = []
    for _ in range(n):
        r = rng.random()
        if r < 0.5: segs.append(rng.choice(SEG))
        elif r < 0.85: segs.append(rng.choice(WORDS))
        else: segs.append(str(rng.randint(1, 99999999)) if rng.random() < .6 else "".join(rng.choices("abcdef0123456789", k=rng.choice([8, 12, 16]))))
    p = "/" + "/".join(segs)
    if segs and rng.random() < .4: p += rng.choice(EXT)
    q = ""
    if rng.random() < 0.35:
        q = "?" + "&".join(f"{rng.choice(QKEYS)}={rng.choice(WORDS + [str(rng.randint(0, 9999))])}" for _ in range(rng.choice([1, 1, 2, 3])))
    frag = "#" + rng.choice(["top", "section-2", "comments", "inbox", ""]) if rng.random() < .08 else ""
    return p + q + frag


def legit_variant(url):
    reg = registered_domain(url)
    if not reg:
        return None
    sub = rng.choice(SUBS)
    host = reg if sub is None or (sub == "www" and rng.random() < .2) else f"{sub}.{reg}"
    scheme = "https" if rng.random() < 0.9 else "http"
    return f"{scheme}://{host}{rand_path()}"


raw = pd.read_csv(csv, usecols=["URL", "label"]).drop_duplicates("URL").reset_index(drop=True)
raw["y"] = (raw.label == 0).astype(int)
raw["domain"] = raw.URL.map(registered_domain)
raw = raw[~raw.domain.isin(HOLDOUT_DOMAINS)].reset_index(drop=True)
raw["augmented"] = 0

legit = raw[raw.y == 0]
aug_rows = []
for u in legit.URL:
    for _ in range(2 if rng.random() < 0.5 else 1):
        v = legit_variant(u)
        if v:
            aug_rows.append({"URL": v, "label": 1, "y": 0, "domain": registered_domain(v), "augmented": 1})
df = pd.concat([raw, pd.DataFrame(aug_rows)], ignore_index=True)
print(f"rows: original={len(raw):,} augmented-legit={len(aug_rows):,} total={len(df):,}; phishing={int(df.y.sum()):,} legit={int((df.y==0).sum()):,}")

print("extracting features...", flush=True)
X = pd.DataFrame([extract(u) for u in df.URL], columns=FEATURE_NAMES)
y = df.y.values
tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=SEED).split(X, y, df.domain))
print(f"train={len(tr):,} test={len(te):,} shared_domains={len(set(df.domain.iloc[tr]) & set(df.domain.iloc[te]))}")

params = dict(n_estimators=600, max_depth=6, learning_rate=0.08, subsample=0.8, colsample_bytree=0.7, min_child_weight=3,
              reg_lambda=2.0, tree_method="hist", n_jobs=-1, random_state=SEED, eval_metric="logloss", early_stopping_rounds=40)
model = XGBClassifier(**params)
model.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[te], y[te])], verbose=False)


def metrics(mask_idx, name):
    xa, ya = X.iloc[mask_idx], y[mask_idx]
    p = model.predict(xa); s = model.predict_proba(xa)[:, 1]
    tn, fp, fn, tp = confusion_matrix(ya, p).ravel()
    m = dict(accuracy=accuracy_score(ya, p), precision=precision_score(ya, p), recall=recall_score(ya, p), f1=f1_score(ya, p),
             roc_auc=roc_auc_score(ya, s), fpr=fp / (fp + tn), cm=[[int(tn), int(fp)], [int(fn), int(tp)]])
    print(f"{name:34s} " + " ".join(f"{k}={v:.4f}" for k, v in m.items() if k != "cm"))
    return m


R = {"rows": len(df), "augmented": len(aug_rows), "best_iteration": int(model.best_iteration)}
R["test_all"] = metrics(te, "test (grouped, mixed)")
te_s = pd.Series(te)
R["test_original_only"] = metrics(te[df.augmented.iloc[te].values == 0], "test (original rows only)")
R["test_augmented_legit_only"] = metrics(te[(df.augmented.iloc[te].values == 1)].tolist() + list(te[(df.augmented.iloc[te].values == 0) & (y[te] == 1)]),
                                         "test (aug legit + all phishing)")

# hand-written held-out sanity set (domains never seen in training)
legit = [
    "https://github.com/scikit-learn/scikit-learn/issues", "https://en.wikipedia.org/wiki/Phishing", "https://www.amazon.in/gp/cart/view.html",
    "https://mail.google.com/mail/u/0/#inbox", "https://docs.python.org/3/library/urllib.parse.html", "https://www.irctc.co.in/nget/train-search",
    "https://stackoverflow.com/questions/tagged/python", "https://onlinesbi.sbi/", "https://www.youtube.com/watch?v=abc123",
    "https://www.linkedin.com/in/someone/", "https://www.google.com", "https://www.microsoft.com/en-us/windows",
    "https://www.bbc.co.uk/news/world-asia-india-12345678", "https://www.nytimes.com/2024/05/01/world/index.html",
    "https://www.flipkart.com/mobiles/pr?sid=tyy,4io", "https://www.paypal.com/signin", "https://appleid.apple.com/sign-in",
    "https://www.reddit.com/r/python/comments/abc123/title/",
]
phish = [
    "http://secure-login-paypal.com.verify-account.xyz/signin", "http://192.168.10.5/hdfc/netbanking/login.php",
    "http://amaz0n-offers-update.top/account/verify", "https://sbi-kyc-update.online/index.html",
    "http://appleid.apple.com.reset-password.info/", "http://bit.ly/3xYzAbC", "http://free-crypto-giveaway.tk/claim",
    "http://paypa1-secure-update.ml/login.php", "https://netflix-billing-update.com/login/confirm?id=88213",
    "http://microsoft-support-alert.work/windows/security-warning.html", "http://www.g00gle-docs-share.link/view?doc=2291",
    "https://hdfcbank.com.secure-verify.cf/netbanking/login", "http://dhl-parcel-track-8841.top/pay",
    "https://login.facebook.com.account-recovery.xyz/checkpoint",
]
sane = []
ok = 0
for u, lab in [(u, 0) for u in legit] + [(u, 1) for u in phish]:
    pr = float(model.predict_proba(pd.DataFrame([extract(u)], columns=FEATURE_NAMES))[0, 1])
    good = (pr >= .5) == bool(lab); ok += good
    sane.append({"url": u, "true": "phishing" if lab else "legitimate", "prob": pr, "correct": bool(good)})
    print(f"  {'OK ' if good else 'BAD'} p={pr:.3f} {'phish' if lab else 'legit'} {u}")
print(f"sanity: {ok}/{len(sane)}")
R["sanity"] = sane; R["sanity_acc"] = ok / len(sane)

joblib.dump({"model": model, "features": FEATURE_NAMES, "threshold": 0.5, "kind": "url"}, out / "phishguard_v3_url_xgb.joblib")
json.dump(R, open(out / "results_v3.json", "w"), indent=1, default=float)
print("saved phishguard_v3_url_xgb.joblib, results_v3.json")
