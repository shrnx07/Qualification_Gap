"""
model.py
--------
Model training and evaluation for the Qualification Gap project.

Trains two classifiers on the applicants table:
  - Logistic Regression  (smooth decision boundary; needs feature scaling)
  - Decision Tree        (stepwise boundary; interpretable; no scaling needed)

Both models predict outcome: 1 = approved, 0 = denied.

Usage:
    from src.model import load_features, train_models, evaluate_models
    X_train, X_test, y_train, y_test = load_features(con)
    lr, dt, scaler = train_models(X_train, y_train)
    evaluate_models(lr, dt, scaler, X_test, y_test)
"""

import duckdb
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics         import (
    classification_report, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay,
)


FEATURES = ["loan_amnt", "annual_inc", "dti", "emp_length_yrs"]
TARGET   = "outcome"


# ── Data loading ───────────────────────────────────────────────────────────────

def load_features(
    con: duckdb.DuckDBPyConnection,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Pull feature matrix and target from DuckDB, return train/test split.

    Excludes rows where emp_length_yrs = -1 (missing employment length).

    Returns
    -------
    X_train, X_test, y_train, y_test : pandas DataFrames / Series
    """
    df = con.execute(f"""
        SELECT {', '.join(FEATURES)}, {TARGET}
        FROM applicants
        WHERE emp_length_yrs >= 0
    """).df()

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,        # preserves class ratio in both splits
    )

    print(f"Train: {len(X_train):,} rows  |  Test: {len(X_test):,} rows")
    print(f"Denial rate — train: {1 - y_train.mean():.3f}  |  test: {1 - y_test.mean():.3f}")

    return X_train, X_test, y_train, y_test


# ── Training ───────────────────────────────────────────────────────────────────

def train_models(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Fit logistic regression and decision tree on training data.

    Logistic regression requires scaled features; scaler is fit on train only
    to prevent data leakage.

    Returns
    -------
    lr      : Fitted LogisticRegression
    dt      : Fitted DecisionTreeClassifier
    scaler  : Fitted StandardScaler (apply to LR inputs only)
    """
    # Logistic Regression — scale first
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_sc, y_train)
    print("Logistic Regression trained.")

    # Decision Tree — no scaling needed
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt.fit(X_train, y_train)
    print("Decision Tree trained (max_depth=5).")

    # Sanity check: coefficients should be positive for income, negative for DTI
    coef_df = pd.DataFrame({"feature": FEATURES, "coefficient": lr.coef_[0]})
    print("\nLogistic Regression coefficients:")
    print(coef_df.sort_values("coefficient", ascending=False).to_string(index=False))

    return lr, dt, scaler


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate_models(
    lr, dt, scaler,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_figures: bool = True,
) -> pd.DataFrame:
    """
    Print classification reports and return a summary comparison DataFrame.

    Parameters
    ----------
    save_figures : If True, saves confusion matrices to figures/

    Returns
    -------
    summary_df : DataFrame with columns [model, accuracy, precision, recall, auc]
    """
    results = []

    model_inputs = [
        ("Logistic Regression", lr,  scaler.transform(X_test)),
        ("Decision Tree",       dt,  X_test),
    ]

    for name, model, X_t in model_inputs:
        preds = model.predict(X_t)
        proba = model.predict_proba(X_t)[:, 1]

        report = classification_report(
            y_test, preds,
            target_names=["Denied", "Approved"],
            output_dict=True,
        )

        print(f"\n{'─'*40}")
        print(f"  {name}")
        print(f"{'─'*40}")
        print(classification_report(y_test, preds, target_names=["Denied", "Approved"]))
        auc = roc_auc_score(y_test, proba)
        print(f"  AUC: {auc:.4f}")

        if save_figures:
            cm = confusion_matrix(y_test, preds)
            disp = ConfusionMatrixDisplay(cm, display_labels=["Denied", "Approved"])
            fig, ax = plt.subplots()
            disp.plot(ax=ax, colorbar=False)
            ax.set_title(f"Confusion Matrix — {name}")
            fig.savefig(
                f"figures/confusion_matrix_{name.lower().replace(' ', '_')}.png",
                dpi=150, bbox_inches="tight",
            )
            plt.close(fig)

        results.append({
            "model":     name,
            "accuracy":  round(report["accuracy"], 4),
            "precision": round(report["weighted avg"]["precision"], 4),
            "recall":    round(report["weighted avg"]["recall"], 4),
            "auc":       round(auc, 4),
        })

    summary_df = pd.DataFrame(results)
    print("\nModel Comparison Table:")
    print(summary_df.to_string(index=False))
    return summary_df


def plot_decision_tree(dt, save_path: str = "figures/decision_tree.png") -> None:
    """Save a visual of the decision tree (first 3 levels)."""
    fig, ax = plt.subplots(figsize=(20, 8))
    plot_tree(
        dt,
        feature_names=FEATURES,
        class_names=["Denied", "Approved"],
        max_depth=3,
        filled=True,
        ax=ax,
    )
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Decision tree saved to {save_path}")


# ── Quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import duckdb
    con = duckdb.connect("db/lending.db")
    X_train, X_test, y_train, y_test = load_features(con)
    lr, dt, scaler = train_models(X_train, y_train)
    evaluate_models(lr, dt, scaler, X_test, y_test)
    plot_decision_tree(dt)
