import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from eda import run_eda

def preprocess_data(csv_file="blood_cell_anomaly_detection.csv"):
    # 1. VERİ YÜKLEME
    try:
        df = pd.read_csv(csv_file)
        print("Veri başarıyla yüklendi.")
    except FileNotFoundError:
        print(f"Hata: '{csv_file}' dosyası bulunamadı!")
        exit()

    # 2. İLK BAKIŞ VE KONTROLLER
    print("\n--- Veri Seti Özeti ---")
    print("Veri Setinin İlk 5 Örneği: \n", df.head(5))
    print("\nVeri Setinin Şekli: ", df.shape)
    print("\nVeri Seti Genel Bilgileri:")
    print(df.info())
    print("\nİstatistiksel Özet:")
    print(df.describe())
    print("\nEksik Değer Sayısı:")
    print(df.isnull().sum())

    # 3. MODELİN KOPYA ÇEKMESİNİ ENGELLE (Data Leakage Prevention)
    # disease_category: Hastalık adı zaten anomali olup olmadığını söyler, bunu atmalıyız.
    # dataset_source: Verinin hangi hastaneden geldiği tahmini etkilememeli.
    drop_cols = ['cell_id', 'disease_category', 'dataset_source']

    print(f"\n[INFO] Gereksiz ve sızıntı riski taşıyan sütunlar atılıyor: {drop_cols}")
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # 3. Eksik ve 0 değerleri medyan ile doldurma
    cols_to_fix = ['cell_diameter_um', 'nucleus_area_pct', 'hemoglobin_g_dl']
    for col in cols_to_fix:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)
            df[col] = df[col].fillna(df[col].median())

    # 4. Outlier log transform
    skewed_features = ['wbc_count_per_ul', 'platelet_count_per_ul']
    for col in skewed_features:
        if col in df.columns:
            df[col] = np.log1p(df[col])

    # 5. Genel eksik değer tamamlama
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    object_cols = df.select_dtypes(include=['object', 'string']).columns
    df[object_cols] = df[object_cols].fillna('Unknown')

    # 6. EDA
    run_eda(df)

    # 7. Kategorik encode
    target_col = 'anomaly_label'
    cat_features = ['patient_sex', 'dataset_source', 'staining_protocol',
                    'microscope_model', 'disease_category', 'cell_type']
    df = pd.get_dummies(df, columns=[col for col in cat_features if col in df.columns], drop_first=True)

    age_map = {'Child': 0, 'Adult': 1, 'Elderly': 2}
    if 'patient_age_group' in df.columns:
        df['patient_age_group'] = df['patient_age_group'].map(age_map).fillna(1)

    # 8. Scaling
    # preprocessor.py dosyasındaki 6. adımın sonu:
    scaler = StandardScaler()
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # HATA BURADAYDI: Sadece array döndürüyordu. Şimdi DataFrame'e geri çeviriyoruz.
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # 9. Kaydet
    df.to_csv("processed_blood_data.csv", index=False)
    print("İşleme tamamlandı. Kaydedilen dosya: 'processed_blood_data.csv'")

    return X_scaled, y

# Doğrudan çalıştırıldığında test amaçlı
if __name__ == "__main__":
    X, y = preprocess_data()