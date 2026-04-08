import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

def get_models(y_train):
    # 1. VERİYİ YÜKLE
    # Zaten preprocessor.py tarafında encode edilmiş ve scale edilmiş veriyi alıyoruz
    df = pd.read_csv("processed_blood_data.csv")

    # Hedef ve Özellik ayrımı
    y = df['anomaly_label']
    X = df.drop(columns=['anomaly_label'])

    # 2. TRAIN-TEST SPLIT
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 3. SINIF AĞIRLIĞI HESAPLAMA (Tıbbi veri dengesizliği için)
    # Eğer anomali (1) sayısı çok azsa modellerin buna odaklanmasını sağlarız.
    scale_pos_weight_val = (y_train == 0).sum() / (y_train == 1).sum()

    # 4. MODELLERİN TANIMLANMASI VE İYİLEŞTİRİLMESİ
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight='balanced',  # Otomatik ağırlık dengeleme
            random_state=42
        ),
        # model.py içinde XGBoost tanımı:
        "XGBoost": XGBClassifier(
            scale_pos_weight=scale_pos_weight_val,
            # use_label_encoder=False,  <-- BU SATIRI SİL VEYA YORUMA AL
            eval_metric='logloss',
            random_state=42
        ),
        "SVM": SVC(
            kernel='rbf',
            probability=True,
            class_weight='balanced',  # Dengesiz veriye karşı koruma
            random_state=42
        )
    }

    trained_models = {}
    predictions = {}
    probabilities = {}  # Evaluation aşaması (ROC-AUC) için gerekli

    for name, model in models.items():
        print(f"\n>>> {name} eğitiliyor...")

        # Tüm modelleri artık preprocessor'dan gelen hazır veriyle eğitiyoruz
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]  # Olasılık değerlerini alıyoruz

        trained_models[name] = model
        predictions[name] = y_pred
        probabilities[name] = y_prob

    print("\nTüm modeller eğitildi. Evaluation aşamasına geçilebilir.")

    return models