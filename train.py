"""Phase I experiments: feature extraction, model comparison, evaluation, SHAP.

Usage: python train.py <path to PhiUSIIL CSV> <output dir>
"""
import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from features import FEATURE_NAMES, extract, registered_domain

SEED = 42
csv, out = Path(sys.argv[1]), Path(sys.argv[2])
(out / "figs").mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10})
R = {}

# ---------------- data ----------------
t0 = time.time()
raw = pd.read_csv(csv, usecols=["URL", "label"])
R["raw_rows"] = len(raw)
df = raw.drop_duplicates("URL").reset_index(drop=True)
R["after_dedup"] = len(df)
df["phishing"] = (df["label"] == 0).astype(int)  # dataset: 1 = legitimate, 0 = phishing
R["class_counts"] = {"legitimate": int((df.phishing == 0).sum()), "phishing": int((df.phishing == 1).sum())}
X = pd.DataFrame([extract(u) for u in df["URL"]], columns=FEATURE_NAMES)
groups = df["URL"].map(registered_domain)
y = df["phishing"].values
R["unique_domains"] = int(groups.nunique())
R["feature_extraction_seconds"] = round(time.time() - t0, 1)
print("features done", R, flush=True)

# dataset bias check
R["bias"] = {
    "legit_with_www": float(X.has_www[y == 0].mean()), "phish_with_www": float(X.has_www[y == 1].mean()),
    "legit_https": float(X.is_https[y == 0].mean()), "phish_https": float(X.is_https[y == 1].mean()),
    "legit_path_gt1": float((X.path_length[y == 0] > 1).mean()), "phish_path_gt1": float((X.path_length[y == 1] > 1).mean()),
    "legit_median_len": float(X.url_length[y == 0].median()), "phish_median_len": float(X.url_length[y == 1].median()),
}

# ---------------- EDA figures ----------------
fig, ax = plt.subplots(figsize=(5, 3.2))
ax.bar(["Legitimate", "Phishing"], [R["class_counts"]["legitimate"], R["class_counts"]["phishing"]], color=["#4C72B0", "#C44E52"])
for i, v in enumerate([R["class_counts"]["legitimate"], R["class_counts"]["phishing"]]):
    ax.text(i, v, f"{v:,}", ha="center", va="bottom")
ax.set_ylabel("Number of URLs"); ax.set_title("Class distribution after de-duplication")
ax.spines[["top", "right"]].set_visible(False); plt.tight_layout(); plt.savefig(out / "figs/class_dist.png", dpi=200); plt.close()

fig, axes = plt.subplots(1, 3, figsize=(10, 3))
for ax, col, lim in zip(axes, ["url_length", "num_subdomains", "url_entropy"], [150, 6, 6]):
    for lab, c, name in [(0, "#4C72B0", "Legitimate"), (1, "#C44E52", "Phishing")]:
        v = X[col][y == lab].clip(upper=lim)
        ax.hist(v, bins=30, alpha=0.55, color=c, label=name, density=True)
    ax.set_title(col); ax.spines[["top", "right"]].set_visible(False)
axes[0].legend(); plt.tight_layout(); plt.savefig(out / "figs/feature_dists.png", dpi=200); plt.close()

# ---------------- split ----------------
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
tr, te = next(gss.split(X, y, groups))
Xtr, Xte, ytr, yte, gtr = X.iloc[tr], X.iloc[te], y[tr], y[te], groups.iloc[tr]
R["split"] = {"train": len(tr), "test": len(te), "shared_domains": int(len(set(gtr) & set(groups.iloc[te])))}

models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED)),
    "Decision Tree": DecisionTreeClassifier(max_depth=20, min_samples_leaf=5, random_state=SEED),
    "Random Forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=SEED),
    "XGBoost": XGBClassifier(n_estimators=400, max_depth=8, learning_rate=0.1, subsample=0.9, colsample_bytree=0.8,
                             tree_method="hist", n_jobs=-1, random_state=SEED, eval_metric="logloss"),
    "Linear SVM": make_pipeline(StandardScaler(), LinearSVC(C=1.0, max_iter=5000, random_state=SEED)),
}


def evaluate(m, Xa, ya):
    pred = m.predict(Xa)
    score = m.predict_proba(Xa)[:, 1] if hasattr(m, "predict_proba") else m.decision_function(Xa)
    tn, fp, fn, tp = confusion_matrix(ya, pred).ravel()
    return {"accuracy": accuracy_score(ya, pred), "precision": precision_score(ya, pred), "recall": recall_score(ya, pred),
            "f1": f1_score(ya, pred), "roc_auc": roc_auc_score(ya, score), "fpr": fp / (fp + tn),
            "cm": [[int(tn), int(fp)], [int(fn), int(tp)]]}, score


R["models"], scores = {}, {}
for name, m in models.items():
    t = time.time(); m.fit(Xtr, ytr); train_s = time.time() - t
    met, sc = evaluate(m, Xte, yte)
    t = time.time(); m.predict(Xte); pred_s = time.time() - t
    met.update(train_seconds=round(train_s, 1), batch_pred_ms_per_url=1000 * pred_s / len(Xte))
    R["models"][name], scores[name] = met, sc
    print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in met.items()}, flush=True)

# ---------------- cross-validation (grouped) for the two ensembles ----------------
R["cv"] = {}
gkf = GroupKFold(n_splits=5)
for name in ["Random Forest", "XGBoost"]:
    accs, f1s = [], []
    for a, b in gkf.split(Xtr, ytr, gtr):
        m = models[name].__class__(**models[name].get_params())
        m.fit(Xtr.iloc[a], ytr[a]); p = m.predict(Xtr.iloc[b])
        accs.append(accuracy_score(ytr[b], p)); f1s.append(f1_score(ytr[b], p))
    R["cv"][name] = {"acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)), "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s))}
    print("CV", name, R["cv"][name], flush=True)

# ---------------- leakage comparison: random split vs domain split ----------------
a, b = train_test_split(np.arange(len(X)), test_size=0.2, stratify=y, random_state=SEED)
mr = XGBClassifier(**models["XGBoost"].get_params()); mr.fit(X.iloc[a], y[a])
R["random_split_xgb"] = evaluate(mr, X.iloc[b], y[b])[0]
R["random_split_shared_domains"] = int(len(set(groups.iloc[a]) & set(groups.iloc[b])))

# ---------------- shortcut check: remove scheme/www/path-presence cues ----------------
drop = ["is_https", "has_www"]
mx = XGBClassifier(**models["XGBoost"].get_params()); mx.fit(Xtr.drop(columns=drop), ytr)
R["without_www_https"] = evaluate(mx, Xte.drop(columns=drop), yte)[0]
hp = (Xte.path_length > 1).values
R["subset_with_path"] = {"n": int(hp.sum()), "n_legit": int((yte[hp] == 0).sum()), "n_phish": int((yte[hp] == 1).sum())}
if (yte[hp] == 0).sum() > 20:
    R["subset_with_path"].update(evaluate(models["XGBoost"], Xte[hp], yte[hp])[0])

# ---------------- single-URL latency (extraction + prediction) ----------------
best = models["XGBoost"]
sample = df["URL"].iloc[te[:500]].tolist()
t = time.time()
for u in sample:
    best.predict_proba(pd.DataFrame([extract(u)], columns=FEATURE_NAMES))
R["single_url_latency_ms"] = 1000 * (time.time() - t) / len(sample)

# ---------------- figures: comparison, ROC, confusion ----------------
names = list(models)
fig, ax = plt.subplots(figsize=(9, 3.6))
metrics = ["accuracy", "precision", "recall", "f1"]
w = 0.18
for i, mtr in enumerate(metrics):
    vals = [R["models"][n][mtr] * 100 for n in names]
    ax.bar(np.arange(len(names)) + (i - 1.5) * w, vals, w, label=mtr.capitalize() if mtr != "f1" else "F1-score")
ax.set_xticks(np.arange(len(names))); ax.set_xticklabels(names); ax.set_ylabel("Score (%)")
lo = min(R["models"][n][m] for n in names for m in metrics) * 100
ax.set_ylim(max(0, lo - 3), 100.5); ax.legend(ncol=4, loc="lower center", fontsize=8)
ax.spines[["top", "right"]].set_visible(False); plt.tight_layout(); plt.savefig(out / "figs/model_comparison.png", dpi=200); plt.close()

fig, ax = plt.subplots(figsize=(5.2, 4.2))
for n in names:
    fpr, tpr, _ = roc_curve(yte, scores[n]); ax.plot(fpr, tpr, label=f"{n} (AUC={R['models'][n]['roc_auc']:.4f})", lw=1.4)
ax.plot([0, 1], [0, 1], "k--", lw=0.8); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.set_xlim(0, 0.02); ax.set_ylim(0.97, 1.0); ax.legend(fontsize=7.5, loc="lower right"); ax.set_title("ROC curves (zoomed)")
plt.tight_layout(); plt.savefig(out / "figs/roc.png", dpi=200); plt.close()

fig, axes = plt.subplots(1, 5, figsize=(13, 2.9))
for ax, n in zip(axes, names):
    cm = np.array(R["models"][n]["cm"]); ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Legit", "Phish"]); ax.set_yticks([0, 1]); ax.set_yticklabels(["Legit", "Phish"])
    ax.set_title(n, fontsize=9); ax.set_xlabel("Predicted")
axes[0].set_ylabel("Actual"); plt.tight_layout(); plt.savefig(out / "figs/confusion.png", dpi=200); plt.close()

# ---------------- SHAP ----------------
expl = shap.TreeExplainer(best)
sx = Xte.sample(2000, random_state=SEED)
sv = expl.shap_values(sx)
plt.figure(); shap.summary_plot(sv, sx, max_display=15, show=False, plot_size=(8, 5.5))
plt.tight_layout(); plt.savefig(out / "figs/shap_summary.png", dpi=200, bbox_inches="tight"); plt.close()
imp = pd.Series(np.abs(sv).mean(0), index=FEATURE_NAMES).sort_values(ascending=False)
R["shap_top10"] = {k: float(v) for k, v in imp.head(10).items()}

joblib.dump({"model": best, "features": FEATURE_NAMES, "threshold": 0.5}, out / "phishguard_xgb.joblib")
R["total_seconds"] = round(time.time() - t0, 1)
json.dump(R, open(out / "results.json", "w"), indent=1, default=float)
print("DONE", R["total_seconds"])
