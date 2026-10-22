# Tugas Kelompok: Linear Regression dan Logistic Regression - FSD

## Anggota
| Nama | NIM |
|---|---|
| Ahmad Azzuri Pakalessy | 25523120 |
| Muhammad Pramudya Aldiansyah | 25523082 |
| Danis Setiyawan | 25523072 |

## Sumber Dataset
| Bagian | Dataset | Sumber |
|---|---|---|
| A | Coffee Shop Sales (Maven Roasters), 149.116 baris x 11 kolom, transaksi 3 toko di New York (Jan–Jun 2023) | https://www.kaggle.com/datasets/ahmedabbas757/coffee-sales |
| B | Titanic, 891 baris x 15 kolom, data penumpang nyata | https://github.com/mwaskom/seaborn-data (versi asli: https://www.kaggle.com/c/titanic/data) |

Untuk menjalankan Bagian A, unduh `Coffee Shop Sales.xlsx` dari tautan Kaggle lalu upload ke `/content/` di Colab (atau taruh di folder `data/`).

## Bagian A: Linear Regression
**Target:** `revenue` = `transaction_qty` x `unit_price` (USD per transaksi), numerik kontinu. Target awal `transaction_qty` tidak dipakai karena berupa bilangan bulat kecil (1–8) sehingga tidak memenuhi ketentuan target numerik kontinu.
**Fitur kandidat:** `unit_price`, `hour`, `is_weekend`, `month`, `store_location` (one-hot), `product_category` (one-hot). `transaction_qty` dikeluarkan karena merupakan komponen target (data leakage); kolom ID, `product_type`, dan `product_detail` tidak dipakai.
**Alur:** statistik deskriptif dan 4 visualisasi (A2) → cek missing/duplikat, outlier (IQR, dipertahankan karena produk mahal itu valid), one-hot encoding (A3) → seleksi fitur dengan korelasi, uji F, dan VIF (A4) → split 80:20 `random_state=42`, scaling dengan `StandardScaler` yang di-fit pada data latih saja (A5) → `LinearRegression` (A6) → MAE, MSE, RMSE, R² latih vs uji, plot aktual vs prediksi, residual (A7) → analisis performa (A8).

**Hasil:** 

| Metrik | Latih | Uji |
|---|---|---|
| MAE | 1.3803 | 1.3520 |
| MSE | 10.4070 | 2.6776 |
| RMSE | 3.2260 | 1.6363 |
| R² | 0.4809 | 0.7071 |

**Keterbatasan:** hubungan linear saja; target condong ke kanan dan outlier produk mahal menarik garis regresi; residual tidak berdistribusi normal dan variansnya tidak konstan; tidak ada data pelanggan/promo; hanya 6 bulan dan 3 toko; prediksi bisa bernilai negatif.

## Bagian B: Logistic Regression

**Target:** `survived` (0 = tidak selamat, 549 baris / 61,6%; 1 = selamat, 342 baris / 38,4%).

**Fitur:** `pclass`, `sex`, `age`, `sibsp`, `parch`, `fare`, `embarked`. Kolom `alive` (sama dengan target) dan kolom turunan (`class`, `who`, `adult_male`, `alone`, `embark_town`) dibuang; `deck` dibuang karena sekitar 77% kosong.

**Alur:** statistik dan 4 visualisasi (B2) → imputasi (median untuk `age`, modus untuk `embarked`) di dalam pipeline, `log1p` pada `fare`, one-hot, scaling, `class_weight='balanced'` untuk ketidakseimbangan kelas ringan (B3) → seleksi fitur dengan uji ANOVA F, p < 0,05, menyisakan 6 dari 9 fitur hasil encoding (B4) → split 80:20 dengan `stratify` (B5) → `LogisticRegression` dengan koefisien dan odds ratio (B6) → evaluasi (B7) → analisis (B8).

**Hasil (data uji, 179 baris):**

| Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---|---|---|---|
| 0,7430 | 0,6494 | 0,7246 | 0,6849 | 0,8149 |

Confusion matrix [[TN, FP], [FN, TP]] = [[83, 27], [19, 50]]. Accuracy latih 0,7711 vs uji 0,7430, tidak ada overfitting berarti.

**Temuan utama:** jenis kelamin laki-laki adalah faktor terkuat penurun peluang selamat (odds ratio 0,09), diikuti tiket kelas 3 (OR 0,33) dan naik dari Southampton (OR 0,60); tarif tiket yang lebih tinggi menaikkan peluang (OR 1,47 per simpangan baku log-fare).

**Metrik paling relevan:** accuracy (74,3%) hanya sedikit di atas tebakan kelas mayoritas (61,6%), sehingga recall, F1, dan AUC lebih informatif. False positive (27) lebih banyak daripada false negative (19) karena `class_weight='balanced'` mendorong model memprediksi kelas "selamat".

**Keterbatasan:** model linear pada log-odds sehingga interaksi (mis. jenis kelamin x kelas) tidak tertangkap; `age` memiliki sekitar 20% data kosong yang diimputasi; data uji kecil (179 baris); seleksi fitur bersifat eksploratif pada seluruh data; tidak ada faktor situasional seperti lokasi kabin.

## Pembagian Kontribusi

| Anggota | Kontribusi |
|---|---|
| Ahmad Azzuri Pakalessy | Mencari Dataset dan mengerjakan A.1 sampai A.5 |
| Muhammad Pramudya Aldiansyah | Mengerjakan B.4 - B.8 |
| Danis Setiyawan | Membuat repo & readme & mengerjakan A.6 - A.8 |

## Cara Menjalankan

1. Buka notebook di Google Colab.
2. Bagian A: upload `Coffee Shop Sales.xlsx` ke Colab. Bagian B: tidak perlu file tambahan (dataset dimuat dari seaborn, cadangan `data/titanic.csv`).
3. Jalankan **Runtime > Run all**, lalu **File > Save a copy in GitHub** agar output ikut tersimpan.
