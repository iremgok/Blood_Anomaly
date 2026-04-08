import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score, roc_curve, auc
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

def run_evaluation(model, X_test, y_test, model_name, feature_names, X_train=None, y_train=None):
    """
    Tüm metrikleri hesaplar, Cross-Validation yapar ve görselleştirir.
    """
    print(f"\n{'=' * 45}\n{model_name.upper()} DETAYLI ANALİZ RAPORU\n{'=' * 45}")

    # 1. CROSS VALIDATION (Hatalar Düzeltildi)
    if X_train is not None and y_train is not None:
        print("\n🔁 Cross Validation (5-Fold) Başlatılıyor...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1') # Tıbbi veride F1 daha güvenilir
        print(f"CV F1-Scores: {cv_scores}")
        print(f"CV Mean F1-Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # 2. TAHMİNLER VE METRİKLER
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("\n--- Sınıflandırma Raporu ---")
    print(classification_report(y_test, y_pred))

    # 3. GÖRSELLEŞTİRME
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    # --- A. Karmaşıklık Matrisi ---
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[0])
    axes[0].set_title(f"Hata Matrisi\n(Recall: {rec:.2f})")
    axes[0].set_xlabel("Tahmin Edilen")
    axes[0].set_ylabel("Gerçek Durum")

    # --- B. Metrik Karşılaştırma ---
    metrics_summary = pd.DataFrame({'Metrik': ['Acc', 'Prec', 'Rec', 'F1'], 'Skor': [acc, prec, rec, f1]})
    # evaluation.py - 49. satır güncel hali
    sns.barplot(data=metrics_summary, x='Metrik', y='Skor', hue='Metrik', palette="magma", ax=axes[1], legend=False)
    axes[1].set_ylim(0, 1.1)
    axes[1].set_title("Performans Özeti")
    for i, v in enumerate(metrics_summary['Skor']):
        axes[1].text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')

    # --- C. Feature Importance (En Kritik 10) ---
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[-10:]
        axes[2].barh(range(len(indices)), importances[indices], color='teal')
        axes[2].set_yticks(range(len(indices)))
        axes[2].set_yticklabels([feature_names[i] for i in indices])
        axes[2].set_title("Karar Verici 10 Parametre")
    else:
        axes[2].text(0.5, 0.5, "Özellik Önemi\n(SVM/Linear modellerde katsayılar kullanılır)", ha='center')

    # --- D. ROC Curve ---
    if y_probs is not None:
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        roc_auc = auc(fpr, tpr)
        axes[3].plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
        axes[3].plot([0, 1], [0, 1], color='navy', linestyle='--')
        axes[3].set_title("ROC Eğrisi (Model Ayırt Etme Gücü)")
        axes[3].legend(loc="lower right")

    plt.tight_layout()
    plt.show()

    return {"Model": model_name, "Accuracy": acc, "Recall": rec, "F1": f1}