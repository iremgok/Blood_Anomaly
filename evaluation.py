import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, auc,
    classification_report,confusion_matrix,
    f1_score, precision_score, recall_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score

def run_evaluation(model, X_test, y_test, model_name, feature_names, X_train=None, y_train=None):

    print(f"\n{'=' * 45}\n{model_name.upper()}  ANALİZ \n{'=' * 45}")
    if X_train is not None and y_train is not None:
        print("\nCross Validation (5-Fold, F1)...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
        print(f" F1 skorları: {cv_scores}")
        print(f" Ortalama F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    y_pred = model.predict(X_test)
    y_probs = (model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else None)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("\n Test Sınıflandırma Raporu")
    print(classification_report(y_test, y_pred))

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[0])
    axes[0].set_title(f"Hata Matrisi\n(Recall: {rec:.2f})")
    axes[0].set_xlabel("Tahmin")
    axes[0].set_ylabel("Gerçek")

    metrics_summary = pd.DataFrame({"Metrik": ["Acc", "Prec", "Rec", "F1"], "Skor": [acc, prec, rec, f1]})
    sns.barplot(data=metrics_summary, x="Metrik", y="Skor", hue="Metrik", palette="magma", ax=axes[1], legend=False)
    axes[1].set_ylim(0, 1.1)
    axes[1].set_title("Test Performans Özeti")
    for i, v in enumerate(metrics_summary["Skor"]):
        axes[1].text(i, v + 0.02, f"{v:.2f}", ha="center", fontweight="bold")

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[-10:]
        axes[2].barh(range(len(indices)), importances[indices], color="teal")
        axes[2].set_yticks(range(len(indices)))
        axes[2].set_yticklabels([feature_names[i] for i in indices])
        axes[2].set_title("En yüksek 10 özellik önemi")
    else:
        axes[2].text(0.5, 0.5, "Özellik önemi yok\n(SVM)", ha="center", va="center", transform=axes[2].transAxes)

    if y_probs is not None:
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        roc_auc = auc(fpr, tpr)
        axes[3].plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.2f}")
        axes[3].plot([0, 1], [0, 1], color="navy", linestyle="--")
        axes[3].set_title("ROC (Test)")
        axes[3].legend(loc="lower right")

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 7))
    x_val = X_test['cell_diameter_um'] if isinstance(X_test, pd.DataFrame) else X_test[:, feature_names.index('cell_diameter_um')]
    y_val = X_test['nucleus_area_pct'] if isinstance(X_test, pd.DataFrame) else X_test[:, feature_names.index('nucleus_area_pct')]

    sns.scatterplot(x=x_val, y=y_val, hue=y_pred, style=y_test, palette="coolwarm", s=100, alpha=0.6, edgecolor='black')

    plt.title(f"{model_name} Karar Dağılım Analizi\n(Nokta Rengi: Tahmin, Şekil: Gerçek Durum)", fontsize=15)
    plt.xlabel("Hücre Çapı (um)", fontsize=12)
    plt.ylabel("Çekirdek Alan Oranı (%)", fontsize=12)
    plt.legend(title="Tahmin (Renk) / Gerçek (Şekil)", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    return {
        "Model": model_name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1}
