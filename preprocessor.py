import numpy as np
import pandas as pd
from eda import run_eda

# Başka model / etiket süreci çıktıları (tahmin anında genelde olmaz)
LEAKY_FEATURE_COLS = [
    "cytodiffusion_anomaly_score",
    "cytodiffusion_classification_confidence",
    "labeller_confidence_score",]

# disease_category: modelde sızıntı riski; EDA için önce tutulur, grafiklerden sonra silinir
DROP_COLS = [
    "cell_id",
    "dataset_source",]

# Bu veri setinde her hücre tipi ya hep sağlıklı ya hep anomali ile etiketli;
# cell_type kullanılırsa etiket neredeyse deterministik çözülür (%100 skor).
# Morfoloji/klinik özelliklerden öğrenmek için modelden çıkarılır (EDA'da kalır).
CELL_TYPE_EXCLUDE_AFTER_EDA = "cell_type"

def preprocess_data(csv_file="blood_cell_anomaly_detection.csv", run_eda_plots=True):
    """
    Veriyi yükler, temizler ve kodlar. Ölçekleme ve eğitim-kümesine özgü doldurma
    yapılmaz; bunlar train/test ayrımından sonra main içinde uygulanır (sızıntı önlemi).
    """
    try:
        df = pd.read_csv(csv_file)
        print("Veri başarıyla yüklendi.")
    except FileNotFoundError:
        print(f"Hata: '{csv_file}' dosyası bulunamadı!")
        raise

    print("\n--- Veri Seti Özeti ---")
    print("İlk 5 örnek:\n", df.head(5))
    print("\nŞekil:", df.shape)
    print("\nEksik değer sayısı:\n", df.isnull().sum())

    drop_now = [c for c in DROP_COLS + LEAKY_FEATURE_COLS if c in df.columns]
    if drop_now:
        print(f"\n[INFO] Atılan sütunlar: {drop_now}")
        df = df.drop(columns=drop_now)

    cols_to_fix = ["cell_diameter_um", "nucleus_area_pct", "hemoglobin_g_dl"]
    for col in cols_to_fix:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)

    skewed_features = ["wbc_count_per_ul", "platelet_count_per_ul"]
    for col in skewed_features:
        if col in df.columns:
            df[col] = np.log1p(df[col])

    object_cols = df.select_dtypes(include=["object", "string"]).columns
    df[object_cols] = df[object_cols].fillna("Unknown")

    if run_eda_plots:
        run_eda(df)

    post_eda_drop = []
    if "disease_category" in df.columns:
        post_eda_drop.append("disease_category")
    if CELL_TYPE_EXCLUDE_AFTER_EDA in df.columns:
        post_eda_drop.append(CELL_TYPE_EXCLUDE_AFTER_EDA)
    if post_eda_drop:
        print(
            f"\n[INFO] Model öncesi çıkarılan sütunlar: {post_eda_drop} "
            "EDA tamamlandı."
        )
        df = df.drop(columns=post_eda_drop)

    target_col = "anomaly_label"
    cat_features = [
        "patient_sex",
        "staining_protocol",
        "microscope_model",
    ]
    df = pd.get_dummies(
        df, columns=[c for c in cat_features if c in df.columns], drop_first=True
    )

    age_map = {"Child": 0, "Adult": 1, "Elderly": 2}
    if "patient_age_group" in df.columns:
        df["patient_age_group"] = df["patient_age_group"].map(age_map)
        df["patient_age_group"] = df["patient_age_group"].fillna(1)

    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Kayıt: ölçeklenmemiş, kodlanmış matris (denetim için)
    out = pd.concat([X, y], axis=1)
    out.to_csv("processed_blood_data.csv", index=False)
    print("\nİşlenmiş veri kaydedildi: processed_blood_data.csv")

    return X, y
