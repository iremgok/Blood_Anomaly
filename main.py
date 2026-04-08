import pandas as pd
from sklearn.model_selection import train_test_split
from preprocessor import preprocess_data
from model import get_models
from evaluation import run_evaluation

def main():
    print("Kan Hücresi Anormallik Tespiti Projesi Başlatılıyor...")

    # 1. VERİYİ HAZIRLA (EDA preprocessor içinde otomatik çalışacak)
    X, y = preprocess_data("blood_cell_anomaly_detection.csv")

    # 2. EĞİTİM VE TEST AYIRIMI
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. MODELLERİ ÇAĞIR (Sınıf ağırlığı hesaplanarak)
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    models = get_models(y_train)

    # 4. EĞİTİM VE DEĞERLENDİRME DÖNGÜSÜ
    all_results = []
    feature_names = X.columns.tolist()

    for name, model in models.items():
        print(f"\n--- {name} İşleniyor ---")
        # Modeli eğit
        model.fit(X_train, y_train)

        # Sonuçları hesapla ve görselleştir (evaluation.py)
        metrics = run_evaluation(
            model=model,
            X_test=X_test,
            y_test=y_test,
            model_name=name,
            feature_names=feature_names,
            X_train=X_train,
            y_train=y_train
        )
        all_results.append(metrics)

    # 5. ÖZET TABLO
    print("\n" + "=" * 50)
    print("FİNAL PERFORMANS KARŞILAŞTIRMASI")
    print("=" * 50)
    results_df = pd.DataFrame(all_results).sort_values(by="Recall", ascending=False)
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()