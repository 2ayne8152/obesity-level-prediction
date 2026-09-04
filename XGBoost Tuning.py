"""
XGBoost Classifier — Estimation of Obesity Levels Based on Eating Habits
and Physical Condition (UCI ML Repository, dataset id 544)
DOI: https://doi.org/10.24432/C5H31Z

Pipeline:
  1. Load data (via ucimlrepo, with a CSV fallback)
  2. Preprocessing (encode categoricals, encode target, train/test split)
  3. Baseline XGBoost model
  4. Hyperparameter tuning (RandomizedSearchCV)
  5. Evaluation (accuracy, classification report, ROC-AUC, confusion matrix, feature importance)

Install requirements:
    pip install ucimlrepo xgboost scikit-learn pandas numpy matplotlib seaborn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
)

from xgboost import XGBClassifier

RANDOM_STATE = 42

# --------------------------------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------------------------------
def load_data():
    try:
        from ucimlrepo import fetch_ucirepo

        dataset = fetch_ucirepo(id=544)
        X = dataset.data.features
        y = dataset.data.targets.squeeze()  # Series
        return X, y
    except Exception as e:
        print(f"ucimlrepo fetch failed ({e}); falling back to local CSV.")
        df = pd.read_csv("csv/ObesityDataSet_raw_and_data_sinthetic.csv")
        y = df["NObeyesdad"]
        X = df.drop(columns=["NObeyesdad"])
        return X, y

X, y = load_data()
print("Feature matrix shape:", X.shape)
print("Target distribution:\n", y.value_counts())

# --------------------------------------------------------------------------
# 2. PREPROCESSING
# --------------------------------------------------------------------------
# Identify column types.
#   - binary_cols:  2-category columns -> single 0/1 column
#   - ordinal_cols: genuinely ORDERED categories -> single integer column,
#                   in their real-world order (no/Sometimes/Frequently/Always)
#   - nominal_cols: unordered multi-category columns -> one-hot encoded,
#                   with one category dropped to avoid the dummy trap
#   - numeric_cols: continuous features -> standardized
binary_cols = ["Gender", "family_history_with_overweight", "FAVC", "SMOKE", "SCC"]
ordinal_cols = ["CAEC", "CALC"]                    # truly ordered categories
nominal_cols = ["MTRANS"]                          # unordered, multi-category
numeric_cols = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]

# Keep only columns that actually exist (robust to minor naming differences)
binary_cols = [c for c in binary_cols if c in X.columns]
ordinal_cols = [c for c in ordinal_cols if c in X.columns]
nominal_cols = [c for c in nominal_cols if c in X.columns]
numeric_cols = [c for c in numeric_cols if c in X.columns]

# Explicit category orders. For binary columns the order just fixes which
# label maps to 0 vs 1; for CAEC/CALC the order is the real severity order.
binary_categories = [["Female", "Male"], ["no", "yes"], ["no", "yes"], ["no", "yes"], ["no", "yes"]]
binary_categories = binary_categories[: len(binary_cols)]

ordinal_categories = [["no", "Sometimes", "Frequently", "Always"]] * len(ordinal_cols)

# Encode target labels in their natural CLINICAL order (kept consistent with
# LogisticRegression.py, even though XGBoost itself doesn't require an
# ordered target — this keeps class_names/report ordering identical across
# both scripts for easy side-by-side comparison).
class_order = [
    "Insufficient_Weight",
    "Normal_Weight",
    "Overweight_Level_I",
    "Overweight_Level_II",
    "Obesity_Type_I",
    "Obesity_Type_II",
    "Obesity_Type_III",
]

target_encoder = LabelEncoder()
target_encoder.classes_ = np.array(class_order)
y_encoded = target_encoder.transform(y)
print("\nClasses (in ordinal order):", list(target_encoder.classes_))

# ColumnTransformer: ordinal-encode binary/ordinal columns, one-hot encode
# only the genuinely nominal column (MTRANS, drop="first"), and scale
# numeric columns. Same preprocessing as LogisticRegression.py.
preprocessor = ColumnTransformer(
    transformers=[
        ("bin", OrdinalEncoder(categories=binary_categories), binary_cols),
        ("ord", OrdinalEncoder(categories=ordinal_categories), ordinal_cols),
        (
            "nom",
            OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False),
            nominal_cols,
        ),
        ("num", StandardScaler(), numeric_cols),
    ]
)

# Train/test split (stratified to preserve class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=RANDOM_STATE, stratify=y_encoded
)

# --------------------------------------------------------------------------
# 3. BASELINE MODEL
# --------------------------------------------------------------------------
n_classes = len(np.unique(y_encoded))

baseline_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            XGBClassifier(
                objective="multi:softprob",
                num_class=n_classes,
                eval_metric="mlogloss",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    ]
)

baseline_pipeline.fit(X_train, y_train)
baseline_preds = baseline_pipeline.predict(X_test)
baseline_acc = accuracy_score(y_test, baseline_preds)
print(f"\nBaseline XGBoost accuracy: {baseline_acc:.4f}")

# --------------------------------------------------------------------------
# 4. HYPERPARAMETER TUNING (RandomizedSearchCV)
# --------------------------------------------------------------------------
param_distributions = {
    "classifier__n_estimators": [100, 200, 300, 400, 500, 600],
    "classifier__max_depth": [3, 4, 5, 6, 7, 8, 9],
    "classifier__learning_rate": [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3],
    "classifier__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "classifier__colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "classifier__min_child_weight": [1, 2, 3, 5, 7],
    "classifier__gamma": [0, 0.1, 0.2, 0.3, 0.5],
    "classifier__reg_alpha": [0, 0.01, 0.1, 1, 5],
    "classifier__reg_lambda": [0.5, 1, 1.5, 2, 5],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

search_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            XGBClassifier(
                objective="multi:softprob",
                num_class=n_classes,
                eval_metric="mlogloss",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    ]
)

random_search = RandomizedSearchCV(
    estimator=search_pipeline,
    param_distributions=param_distributions,
    n_iter=100,             # number of parameter combinations to try
    scoring="accuracy",
    cv=cv,
    verbose=2,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

print("\nStarting hyperparameter search (this can take a few minutes)...")
random_search.fit(X_train, y_train)
results = pd.DataFrame(random_search.cv_results_)

# ==========================================================
# Mean CV Accuracy for Each Hyperparameter Value
# ==========================================================

parameters = [
    "param_classifier__n_estimators",
    "param_classifier__max_depth",
    "param_classifier__learning_rate",
    "param_classifier__subsample",
    "param_classifier__colsample_bytree",
    "param_classifier__min_child_weight",
    "param_classifier__gamma",
    "param_classifier__reg_alpha",
    "param_classifier__reg_lambda"
]

for param in parameters:
    
    summary = (
        results.groupby(param)["mean_test_score"]
        .mean()
        .reset_index()
        .rename(columns={
            "mean_test_score": "Mean_CV_Accuracy"
        })
        .sort_values("Mean_CV_Accuracy", ascending=False)
    )

    filename = param.replace("param_classifier__", "")

    summary.to_csv(
        f"csv/{filename}_mean_cv_accuracy.csv",
        index=False
    )

print("\nBest parameters found:")
for k, v in random_search.best_params_.items():
    print(f"  {k}: {v}")
print(f"Best CV accuracy: {random_search.best_score_:.4f}")

best_model = random_search.best_estimator_

# --------------------------------------------------------------------------
# 5. FINAL EVALUATION ON TEST SET
# --------------------------------------------------------------------------
final_preds = best_model.predict(X_test)
final_acc = accuracy_score(y_test, final_preds)
print(f"\nTuned model test accuracy: {final_acc:.4f}")
print(f"(Baseline test accuracy was: {baseline_acc:.4f})")

print("\nClassification report:\n")
print(
    classification_report(
        y_test, final_preds, target_names=target_encoder.classes_, digits=4
    )
)

# --------------------------------------------------------------------------
# ROC-AUC (multiclass, one-vs-rest)
# --------------------------------------------------------------------------
final_probs = best_model.predict_proba(X_test)

macro_roc_auc = roc_auc_score(
    y_test, final_probs, multi_class="ovr", average="macro"
)
weighted_roc_auc = roc_auc_score(
    y_test, final_probs, multi_class="ovr", average="weighted"
)
per_class_roc_auc = roc_auc_score(
    y_test, final_probs, multi_class="ovr", average=None
)

print(f"\nMacro-average ROC-AUC (OvR): {macro_roc_auc:.4f}")
print(f"Weighted-average ROC-AUC (OvR): {weighted_roc_auc:.4f}")

roc_auc_df = pd.DataFrame(
    {
        "Class": target_encoder.classes_,
        "ROC-AUC": per_class_roc_auc,
    }
).sort_values("ROC-AUC", ascending=False)

print("\nPer-class ROC-AUC (OvR):")
print(roc_auc_df.to_string(index=False))

# Confusion matrix
cm = confusion_matrix(y_test, final_preds)
fig, ax = plt.subplots(figsize=(9, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_encoder.classes_)
disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
plt.title("Confusion Matrix — Tuned XGBoost")
plt.tight_layout()
plt.savefig("image/confusion_matrix.png", dpi=150)
plt.close()
print("\nSaved image/confusion_matrix.png")

# --------------------------------------------------------------------------
# 6. FEATURE IMPORTANCE
# --------------------------------------------------------------------------
# Recover feature names in the same order ColumnTransformer concatenates
# them: bin_cols -> ordinal_cols -> one-hot(nominal_cols) -> numeric_cols.
# (Previously this pulled from named_transformers_["cat"] / categorical_cols,
# which belonged to the old single-block one-hot preprocessing and no
# longer exist now that binary/ordinal/nominal columns are split out.)
ohe = best_model.named_steps["preprocessor"].named_transformers_["nom"]
nom_feature_names = list(ohe.get_feature_names_out(nominal_cols))
all_feature_names = binary_cols + ordinal_cols + nom_feature_names + numeric_cols

importances = best_model.named_steps["classifier"].feature_importances_
feat_imp = (
    pd.Series(importances, index=all_feature_names)
    .sort_values(ascending=False)
    .head(20)
)

plt.figure(figsize=(8, 8))
sns.barplot(x=feat_imp.values, y=feat_imp.index, color="steelblue")
plt.title("Top 20 Feature Importances — Tuned XGBoost")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("image/feature_importance.png", dpi=150)
plt.close()
print("Saved image/feature_importance.png")

# --------------------------------------------------------------------------
# 6.5 SAMPLE DECISION TREE (XGBOOST) — sklearn-plot_tree-style rendering
# --------------------------------------------------------------------------
import json
import matplotlib

def _entropy(counts):
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log2(p)).sum())

def _parse_xgb_tree(tree_json):
    nodes = {}
    def walk(node):
        nid = node["nodeid"]
        if "children" in node:
            nodes[nid] = {
                "feature": node["split"],
                "threshold": node["split_condition"],
                "yes": node["yes"],  # branch taken when condition is TRUE (feature < threshold)
                "no": node["no"],
                "leaf": False,
            }
            for child in node["children"]:
                walk(child)
        else:
            nodes[nid] = {"leaf": True}
    walk(tree_json)
    return nodes

def _compute_stats(nodes, root_id, X_df, y_enc, class_names):
    stats = {}
    def walk(nid, mask):
        counts = np.array([int(np.sum((y_enc == i) & mask)) for i in range(len(class_names))])
        samples = int(mask.sum())
        stats[nid] = {
            "samples": samples,
            "value": counts,
            "entropy": _entropy(counts),
            "class": class_names[int(np.argmax(counts))] if samples > 0 else "N/A",
        }
        node = nodes[nid]
        if not node["leaf"]:
            feat, thr = node["feature"], node["threshold"]
            left_mask = mask & (X_df[feat].values < thr)
            right_mask = mask & ~(X_df[feat].values < thr)
            walk(node["yes"], left_mask)
            walk(node["no"], right_mask)
    walk(root_id, np.ones(len(y_enc), dtype=bool))
    return stats


def plot_xgb_tree_sklearn_style(
    booster, X_transformed, y_encoded, feature_names, class_names,
    tree_index=0, max_display_depth=3, figsize=(34, 18),
    title="Sample Decision Tree from Tuned XGBoost (Tree 0)",
    savepath=None,
    x_spacing=4.0,     # horizontal gap between sibling leaves
    y_spacing=3.0,      # vertical gap between depth levels
):
    X_df = pd.DataFrame(X_transformed, columns=feature_names)
    tree_json = json.loads(booster.get_dump(dump_format="json")[tree_index])
    nodes = _parse_xgb_tree(tree_json)
    stats = _compute_stats(nodes, 0, X_df, y_encoded, class_names)

    positions = {}
    leaf_counter = [0]
    def assign_x(nid, depth):
        node = nodes[nid]
        if node["leaf"] or depth >= max_display_depth:
            x = leaf_counter[0] * x_spacing
            leaf_counter[0] += 1
            positions[nid] = [x, -depth * y_spacing]
            return x, x
        lo, _ = assign_x(node["yes"], depth + 1)
        _, hi = assign_x(node["no"], depth + 1)
        x = (lo + hi) / 2
        positions[nid] = [x, -depth * y_spacing]
        return lo, hi
    assign_x(0, 0)

    palette = matplotlib.colormaps["tab10"].resampled(max(len(class_names), 10))
    class_color = {c: palette(i) for i, c in enumerate(class_names)}

    fig, ax = plt.subplots(figsize=figsize)

    def draw(nid, depth, parent_xy=None, edge_label=None):
        x, y = positions[nid]
        s, node = stats[nid], nodes[nid]
        truncated = (not node["leaf"]) and depth >= max_display_depth

        if truncated:
            text, facecolor = "(...)", "0.6"
        else:
            lines = []
            if not node["leaf"]:
                lines.append(f"{node['feature']} < {node['threshold']:.3f}")
            lines += [f"entropy = {s['entropy']:.3f}", f"samples = {s['samples']}",
                      f"value = {list(s['value'])}", f"class = {s['class']}"]
            text = "\n".join(lines)
            purity = s["value"].max() / s["samples"] if s["samples"] else 0
            base = class_color.get(s["class"], (0.9, 0.9, 0.9, 1))
            facecolor = (base[0], base[1], base[2], 0.25 + 0.6 * purity)

        # connector line drawn first, straight center-to-center; the box (higher zorder) will sit on top of its endpoints
        if parent_xy is not None:
            ax.plot([parent_xy[0], x], [parent_xy[1], y], color="black", linewidth=0.8, zorder=1)
            if edge_label:
                ax.text((parent_xy[0] + x) / 2, (parent_xy[1] + y) / 2 + 0.15, edge_label,
                        fontsize=10, ha="center", zorder=4,
                        bbox=dict(boxstyle="round,pad=0.1", facecolor="white", edgecolor="none"))

        ax.text(x, y, text, ha="center", va="center", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.4", facecolor=facecolor, edgecolor="black"), zorder=3)

        if not (node["leaf"] or truncated):
            draw(node["yes"], depth + 1, (x, y), "True" if depth == 0 else None)
            draw(node["no"], depth + 1, (x, y), "False" if depth == 0 else None)

    draw(0, 0)
    ax.set_xlim(-x_spacing, leaf_counter[0] * x_spacing)
    ax.set_ylim(-max_display_depth * y_spacing - y_spacing, y_spacing)
    ax.axis("off")
    ax.set_title(title, fontsize=16)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=200)
    plt.show()

# --- run it ---
xgb_estimator = best_model.named_steps["classifier"]
class_names = list(target_encoder.classes_)
X_train_transformed = best_model.named_steps["preprocessor"].transform(X_train)

booster = xgb_estimator.get_booster()
booster.feature_names = list(all_feature_names)   # <-- add this line

plot_xgb_tree_sklearn_style(
    booster=booster,
    X_transformed=X_train_transformed,
    y_encoded=y_train,
    feature_names=all_feature_names,
    class_names=class_names,
    tree_index=0,
    max_display_depth=3,
    savepath="image/xgboost_sample_tree.png",
)
print("Saved image/xgboost_sample_tree.png")

# --------------------------------------------------------------------------
# 7. SAVE THE FINAL MODEL
# --------------------------------------------------------------------------
import joblib

joblib.dump(best_model, "csv/xgboost_obesity_model.pkl")
joblib.dump(target_encoder, "csv/target_label_encoder.pkl")
print("\nSaved trained pipeline to csv/xgboost_obesity_model.pkl")
print("Saved target label encoder to csv/target_label_encoder.pkl")

# --------------------------------------------------------------------------
# LEARNING CURVE
# --------------------------------------------------------------------------
train_sizes, train_scores, val_scores = learning_curve(
    estimator=best_model,
    X=X_train,
    y=y_train,
    cv=5,
    scoring="accuracy",
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1,
    shuffle=True,
    random_state=RANDOM_STATE,
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

plt.figure(figsize=(8,6))

plt.plot(train_sizes, train_mean, marker='o', label="Training Accuracy")
plt.plot(train_sizes, val_mean, marker='s', label="Validation Accuracy")

plt.fill_between(
    train_sizes,
    train_mean-train_std,
    train_mean+train_std,
    alpha=0.2
)

plt.fill_between(
    train_sizes,
    val_mean-val_std,
    val_mean+val_std,
    alpha=0.2
)

plt.xlabel("Training Samples")
plt.ylabel("Accuracy")
plt.title("Learning Curve - Tuned XGBoost")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("tuning result/learning_curve.png", dpi=150)
plt.show()

# --------------------------------------------------------------------------
# NUMBER OF TREES
# --------------------------------------------------------------------------
plt.figure(figsize=(7,5))

results.groupby("param_classifier__n_estimators")["mean_test_score"].mean().plot(
    marker='o'
)

plt.title("Accuracy vs Number of Trees")
plt.xlabel("n_estimators")
plt.ylabel("Mean CV Accuracy")
plt.grid(True)

plt.tight_layout()
plt.savefig("tuning result/xgb_n_estimators.png", dpi=150)
plt.show()

# --------------------------------------------------------------------------
# MAX DEPTH
# --------------------------------------------------------------------------
plt.figure(figsize=(7,5))

results.groupby("param_classifier__max_depth")["mean_test_score"].mean().plot(
    marker='o'
)

plt.title("Accuracy vs Max Depth")
plt.xlabel("Max Depth")
plt.ylabel("Mean CV Accuracy")
plt.grid(True)

plt.tight_layout()
plt.savefig("tuning result/xgb_max_depth.png", dpi=150)
plt.show()

# --------------------------------------------------------------------------
# LEARNING RATE
# --------------------------------------------------------------------------
plt.figure(figsize=(7,5))

results.groupby("param_classifier__learning_rate")["mean_test_score"].mean().plot(
    marker='o'
)

plt.title("Accuracy vs Learning Rate")
plt.xlabel("Learning Rate")
plt.ylabel("Mean CV Accuracy")
plt.grid(True)

plt.tight_layout()
plt.savefig("tuning result/xgb_learning_rate.png", dpi=150)
plt.show()

# --------------------------------------------------------------------------
# SUBSAMPLE
# --------------------------------------------------------------------------
plt.figure(figsize=(7,5))

results.groupby("param_classifier__subsample")["mean_test_score"].mean().plot(
    marker='o'
)

plt.title("Accuracy vs Subsample")
plt.xlabel("Subsample")
plt.ylabel("Mean CV Accuracy")
plt.grid(True)

plt.tight_layout()
plt.savefig("tuning result/xgb_subsample.png", dpi=150)
plt.show()

# --------------------------------------------------------------------------
# COLUMN SAMPLE
# --------------------------------------------------------------------------
plt.figure(figsize=(7,5))

results.groupby("param_classifier__colsample_bytree")["mean_test_score"].mean().plot(
    marker='o'
)

plt.title("Accuracy vs Colsample By Tree")
plt.xlabel("colsample_bytree")
plt.ylabel("Mean CV Accuracy")
plt.grid(True)

plt.tight_layout()
plt.savefig("tuning result/xgb_colsample.png", dpi=150)
plt.show()

# --------------------------------------------------------------------------
# GAMMA
# --------------------------------------------------------------------------
plt.figure(figsize=(7,5))

results.groupby("param_classifier__gamma")["mean_test_score"].mean().plot(
    marker='o'
)

plt.title("Accuracy vs Gamma")
plt.xlabel("Gamma")
plt.ylabel("Mean CV Accuracy")
plt.grid(True)

plt.tight_layout()
plt.savefig("tuning result/xgb_gamma.png", dpi=150)
plt.show()

# --------------------------------------------------------------------------
# MINIMUM CHILD WEIGHT
# --------------------------------------------------------------------------
plt.figure(figsize=(7,5))

results.groupby("param_classifier__min_child_weight")["mean_test_score"].mean().plot(
    marker='o'
)

plt.title("Accuracy vs Min Child Weight")
plt.xlabel("Min Child Weight")
plt.ylabel("Mean CV Accuracy")
plt.grid(True)

plt.tight_layout()
plt.savefig("tuning result/xgb_child_weight.png", dpi=150)
plt.show()

# --------------------------------------------------------------------------
# ALPHA (L1 REGULARIZATION)
# --------------------------------------------------------------------------
plt.figure(figsize=(7,5))

results.groupby("param_classifier__reg_alpha")["mean_test_score"].mean().plot(
    marker='o'
)

plt.title("Accuracy vs Alpha (L1)")
plt.xlabel("reg_alpha")
plt.ylabel("Mean CV Accuracy")
plt.grid(True)

plt.tight_layout()
plt.savefig("tuning result/xgb_alpha.png", dpi=150)
plt.show()

# --------------------------------------------------------------------------
# LAMBDA (L2 REGULARIZATION)
# --------------------------------------------------------------------------
plt.figure(figsize=(7,5))

results.groupby("param_classifier__reg_lambda")["mean_test_score"].mean().plot(
    marker='o'
)

plt.title("Accuracy vs Lambda (L2)")
plt.xlabel("reg_lambda")
plt.ylabel("Mean CV Accuracy")
plt.grid(True)

plt.tight_layout()
plt.savefig("tuning result/xgb_lambda.png", dpi=150)
plt.show()

# --------------------------------------------------------------------------
# Example: how to load and use the saved model later
# --------------------------------------------------------------------------
# best_model = joblib.load("xgboost_obesity_model.pkl")
# target_encoder = joblib.load("target_label_encoder.pkl")
# preds = best_model.predict(new_data_df)
# predicted_labels = target_encoder.inverse_transform(preds)