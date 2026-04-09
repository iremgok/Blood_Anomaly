import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from evaluation import run_evaluation
from model import get_models
from preprocessor import preprocess_data


def main():
    print("Kan Hücresi Anormallik Tespiti 2025 Projesi")

    X, y = preprocess_data("blood_cell_anomaly_detection.csv")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Eğitim kümesine göre doldurma ve ölçekleme böylelikle teste bilgi sızmaz
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_imp)
    X_test_s = scaler.transform(X_test_imp)

    X_train_df = pd.DataFrame(X_train_s, columns=X.columns, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_s, columns=X.columns, index=X_test.index)

    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    models = get_models(scale_pos_weight=float(pos_weight))

    feature_names = X.columns.tolist()
    all_results = []

    for name, model in models.items():
        print(f"\n--- {name} İşleniyor ---")
        model.fit(X_train_df, y_train)
        metrics = run_evaluation(model=model, X_test=X_test_df, y_test=y_test, model_name=name, feature_names=feature_names,
                                 X_train=X_train_df,  y_train=y_train)
        all_results.append(metrics)

    print("\n" + "=" * 50)
    print("Performans Karşılaştırmaları")
    print("=" * 50)
    results_df = pd.DataFrame(all_results).sort_values(by="F1", ascending=False)
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    main()
