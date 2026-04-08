import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Grafik ayarları
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["figure.dpi"] = 100

def run_eda(df):
    print("\n--- EDA Süreci Başlatılıyor ---")
    sns.set_theme(style="whitegrid")

    # 1. KRİTİK DEĞİŞKENLERİN SEÇİMİ
    # Morfolojik, Klinik ve Renksel özellikleri gruplayalım
    important_numeric = [
        'cell_diameter_um', 'nucleus_area_pct', 'chromatin_density',
        'hemoglobin_g_dl', 'wbc_count_per_ul', 'stain_intensity'
    ]

    # UNIVARIATE ANALYSIS
    for col in important_numeric:
        if col in df.columns:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
            sns.histplot(df[col], kde=True, color="skyblue", ax=ax1)
            ax1.set_title(f"{col} Dağılımı")

            sns.boxplot(x=df[col], color="lightcoral", ax=ax2)
            ax2.set_title(f"{col} Aykırı Değer Analizi")
            plt.tight_layout()
            plt.show()

    # 2. HÜCRE TİPİ VE ANOMALİ İLİŞKİSİ (YENİ - Stacked Bar)
    ct = pd.crosstab(df['cell_type'], df['anomaly_label'], normalize='index')
    ct.plot(kind='bar', stacked=True, color=['#A2D2FF', '#FFAFB0'], figsize=(12, 6))
    plt.title("Hücre Tiplerine Göre Anomali Oranları")
    plt.ylabel("Oran")
    plt.xticks(rotation=45)
    plt.legend(title="Anomali", labels=["Sağlıklı (0)", "Anormal (1)"])
    plt.show()

    # 2. HEDEF ANALİZİ (Uyarılar Düzeltildi)
    plt.figure(figsize=(8, 5))
    # eda.py içindeki countplot satırı:
    sns.countplot(data=df, x='anomaly_label', hue='anomaly_label', palette="Set2", legend=False)
    plt.title("Hedef Değişken Dağılımı")
    plt.show()

    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='cell_type', hue='cell_type', palette="viridis", legend=False)
    plt.title("Hücre Türü Dağılımı")
    plt.show()

    # 4. ÖZELLİK İLİŞKİLERİ (Bivariate)
    # Örnek: Çap ve Çekirdek Oranı arasındaki ilişki
    if 'cell_diameter_um' in df.columns and 'nucleus_area_pct' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x='cell_diameter_um', y='nucleus_area_pct', hue='anomaly_label', alpha=0.6)
        plt.title("Hücre Çapı vs Çekirdek Alanı Yüzdesi")
        plt.show()

    # 3. KORELASYON MATRİSİ
    plt.figure(figsize=(16, 12))
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    # Sadece anlamlı korelasyonları vurgulamak için annot=True
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title("Özellikler Arası Tam Korelasyon Matrisi")
    plt.show()

    # 4. PAIRPLOT (YENİ - En Önemli Özelliklerin Etkileşimi)
    # Veri çok büyükse ilk 500-1000 satırı seçmek mantıklı olabilir
    top_features = ['cell_diameter_um', 'nucleus_area_pct', 'chromatin_density', 'anomaly_label']
    if all(col in df.columns for col in top_features):
        print("Pairplot oluşturuluyor (Bu işlem veri boyutuna göre zaman alabilir)...")
        sns.pairplot(df[top_features], hue='anomaly_label', palette="husl", diag_kind="kde", plot_kws={'alpha': 0.5})
        plt.suptitle("Kritik Morfolojik Özelliklerin Birbirleriyle İlişkisi", y=1.02)
        plt.show()

    # 5. JOINTPLOT (Hücre Çapı vs Çekirdek Alanı)
    if 'cell_diameter_um' in df.columns and 'nucleus_area_pct' in df.columns:
        g = sns.jointplot(
            data=df, x='cell_diameter_um', y='nucleus_area_pct',
            hue='anomaly_label', kind="kde", fill=True, palette="magma"
        )
        g.fig.suptitle("Hücre Çapı ve Çekirdek Alanı Yoğunluk Analizi", y=1.02, fontsize=14)
        g.fig.subplots_adjust(top=0.92)
        plt.show()

    # 6. RENK VE BOYAMA ANALİZİ (YENİ)
    # Hücrenin rengi (R) ve boyanma yoğunluğu anomaliyi nasıl etkiliyor?
    if 'mean_r' in df.columns and 'stain_intensity' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data=df, x='mean_r', y='stain_intensity', hue='anomaly_label', fill=True, alpha=0.4)
        plt.title("Renk (Kırmızı Kanalı) vs Boyama Yoğunluğu İlişkisi")
        plt.show()

    # 7. HASTALIK KATEGORİSİ VE KLİNİK VERİ (Violin Plot)
    if 'disease_category' in df.columns and 'hemoglobin_g_dl' in df.columns:
        plt.figure(figsize=(12, 6))
        sns.violinplot(data=df, x='disease_category', y='hemoglobin_g_dl', hue='anomaly_label',
                       split=True, palette="muted", inner="quart")
        plt.xticks(rotation=45)
        plt.title("Hastalık Kategorileri ve Anomali Bazında Hemoglobin Dağılımı")
        plt.show()

    print("--- EDA Süreci Başarıyla Tamamlandı ---")
