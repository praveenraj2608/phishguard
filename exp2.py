"""Experiment 2: host-only features (shortcut removed) + sanity test on realistic legitimate URLs."""
import json, sys, time
import joblib, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier
from features import FEATURE_NAMES, HOST_FEATURE_NAMES, extract, extract_host, registered_domain
SEED = 42
csv, out = sys.argv[1], sys.argv[2]
df = pd.read_csv(csv, usecols=["URL", "label"]).drop_duplicates("URL").reset_index(drop=True)
y = (df.label == 0).astype(int).values
groups = df.URL.map(registered_domain)
X = pd.DataFrame([extract_host(u) for u in df.URL], columns=HOST_FEATURE_NAMES)
tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=SEED).split(X, y, groups))
models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, random_state=SEED)),
    "Decision Tree": DecisionTreeClassifier(max_depth=20, min_samples_leaf=5, random_state=SEED),
    "Random Forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=SEED),
    "XGBoost": XGBClassifier(n_estimators=400, max_depth=8, learning_rate=0.1, subsample=0.9, colsample_bytree=0.8, tree_method="hist", n_jobs=-1, random_state=SEED),
}
R = {"host_models": {}}
for n, m in models.items():
    m.fit(X.iloc[tr], y[tr]); p = m.predict(X.iloc[te]); s = m.predict_proba(X.iloc[te])[:, 1]
    tn, fp, fn, tp = confusion_matrix(y[te], p).ravel()
    R["host_models"][n] = dict(accuracy=accuracy_score(y[te], p), precision=precision_score(y[te], p), recall=recall_score(y[te], p),
                               f1=f1_score(y[te], p), roc_auc=roc_auc_score(y[te], s), fpr=fp / (fp + tn), cm=[[int(tn), int(fp)], [int(fn), int(tp)]])
    print(n, {k: round(v, 4) if isinstance(v, float) else v for k, v in R["host_models"][n].items()}, flush=True)
joblib.dump({"model": models["XGBoost"], "features": HOST_FEATURE_NAMES, "threshold": 0.5, "kind": "host"}, f"{out}/phishguard_host_xgb.joblib")

# sanity test: realistic legitimate URLs with paths (hand-written examples) and look-alike phishing patterns
legit = ["https://github.com/scikit-learn/scikit-learn/issues", "https://en.wikipedia.org/wiki/Phishing", "https://www.amazon.in/gp/cart/view.html",
         "https://mail.google.com/mail/u/0/#inbox", "https://docs.python.org/3/library/urllib.parse.html", "https://www.irctc.co.in/nget/train-search",
         "https://stackoverflow.com/questions/tagged/python", "https://onlinesbi.sbi/", "https://www.youtube.com/watch?v=abc123", "https://www.linkedin.com/in/someone/"]
phish = ["http://secure-login-paypal.com.verify-account.xyz/signin", "http://192.168.10.5/hdfc/netbanking/login.php", "http://amaz0n-offers-update.top/account/verify",
         "https://sbi-kyc-update.online/index.html", "http://appleid.apple.com.reset-password.info/", "http://bit.ly/3xYzAbC"]
full = joblib.load(f"{out}/phishguard_xgb.joblib")["model"]
R["sanity"] = []
for u, lab in [(u, 0) for u in legit] + [(u, 1) for u in phish]:
    pf = float(full.predict_proba(pd.DataFrame([extract(u)], columns=FEATURE_NAMES))[0, 1])
    ph = float(models["XGBoost"].predict_proba(pd.DataFrame([extract_host(u)], columns=HOST_FEATURE_NAMES))[0, 1])
    R["sanity"].append({"url": u, "true": "phishing" if lab else "legitimate", "full_prob": pf, "host_prob": ph})
for r in R["sanity"]: print(f"{r['true']:10s} full={r['full_prob']:.3f} host={r['host_prob']:.3f} {r['url']}")
json.dump(R, open(f"{out}/results_exp2.json", "w"), indent=1, default=float)
