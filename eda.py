import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

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

    # 2. HEDEF ANALİZİ
    plt.figure(figsize=(8, 5))
    label_order = sorted(df["anomaly_label"].dropna().unique())
    sns.countplot(
        data=df,
        x="anomaly_label",
        hue="anomaly_label",
        order=label_order,
        hue_order=label_order,
        palette=["#2563EB", "#7C3AED"][: len(label_order)],
        legend=False,
        dodge=False,
    )
    plt.title("Hedef Değişken Dağılımı")
    plt.xlabel("anomaly_label")
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

    corr_cmap = sns.diverging_palette(250, 330, s=88, l=58, as_cmap=True)
    sns.heatmap(
        corr,
        annot=True,
        cmap=corr_cmap,
        center=0,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Özellikler Arası Tam Korelasyon Matrisi")
    plt.show()

    # 4. PAIRPLOT (büyük veride hız + tabakalı örnek; başlık için üst boşluk)
    top_features = ["cell_diameter_um", "nucleus_area_pct", "chromatin_density", "anomaly_label"]
    if all(col in df.columns for col in top_features):
        pairplot_max = 800
        plot_df = df[top_features].dropna()
        if len(plot_df) > pairplot_max:
            plot_df, _ = train_test_split(
                plot_df,
                train_size=pairplot_max,
                stratify=plot_df["anomaly_label"],
                random_state=42,
            )
            print(
                f"Pairplot: {pairplot_max} satırlık tabakalı alt örnek (toplam {len(df)} satır)."
            )
        else:
            print("Pairplot oluşturuluyor...")
        g = sns.pairplot(
            plot_df,
            hue="anomaly_label",
            palette="husl",
            diag_kind="kde",
            plot_kws={"alpha": 0.5},
        )
        g.fig.subplots_adjust(top=0.92)
        g.fig.suptitle(
            "Kritik Morfolojik Özelliklerin Birbirleriyle İlişkisi",
            fontsize=14,
            y=0.98,
        )
        plt.show()

    # 5. JOINTPLOT (Hücre Çapı vs Çekirdek Alanı)
    if 'cell_diameter_um' in df.columns and 'nucleus_area_pct' in df.columns:
        g = sns.jointplot(
            data=df, x='cell_diameter_um', y='nucleus_area_pct',
            hue='anomaly_label', kind="kde", fill=True, palette="magma"
        )
        # Üst başlık: top çok yüksekse suptitle figür dışında kalır / kırpılır.
        g.fig.subplots_adjust(top=0.88)
        g.fig.suptitle(
            "Hücre Çapı ve Çekirdek Alanı Yoğunluk Analizi",
            fontsize=14,
            y=0.99,
        )
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
