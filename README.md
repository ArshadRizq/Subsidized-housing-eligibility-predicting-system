# Sistem Prediksi Eligibilitas Penerima Rumah Subsidi

Aplikasi **Streamlit** untuk memprediksi kelayakan penerima subsidi rumah menggunakan **Fuzzy Logic (Mamdani & Sugeno)** dari _scratch_, diintegrasikan dengan **Machine Learning** dan **Deep Learning** sebagai bonus pengembangan.

> **Tujuan:** Memenuhi syarat pengerjaan tugas besar mata kuliah **Dasar Kecerdasan Artifisial (DKA)** — Program Studi Teknik Informatika.

---

## 📋 Daftar Isi

- [Deskripsi Masalah](#deskripsi-masalah)
- [Dataset](#dataset)
- [Fitur Aplikasi](#fitur-aplikasi)
- [Struktur Proyek](#struktur-proyek)
- [Cara Menjalankan](#cara-menjalankan)
- [Teknologi yang Digunakan](#teknologi-yang-digunakan)
- [Anggota Kelompok](#anggota-kelompok)
- [Pemenuhan Ketentuan Tugas Besar](#pemenuhan-ketentuan-tugas-besar)

---

## Deskripsi Masalah

Pemerintah menyediakan program subsidi rumah bagi masyarakat berpenghasilan rendah. Namun, proses penentuan kelayakan penerima seringkali tidak transparan dan kurang objektif. Sistem ini dibangun untuk membantu memprediksi kelayakan seseorang menerima subsidi rumah berdasarkan **5 variabel input**:

1. **Income** — Pendapatan per bulan (USD)
2. **Household Size** — Jumlah anggota keluarga
3. **Work Experience** — Pengalaman kerja (tahun)
4. **Age** — Usia (tahun)
5. **Number of Dependents** — Jumlah tanggungan

Output sistem berupa **skor eligibilitas (0–100)** dan **label klasifikasi**:
- **Layak** — skor ≥ 55
- **Tidak Layak** — skor < 55

---

## Dataset

- **Sumber:** [Tautan dataset](https://www.kaggle.com/datasets/stealthtechnologies/regression-dataset-for-household-income-analysis) *(Subsidy Eligibility Prediction — Kaggle)*
- **Jumlah data:** 10.000 baris
- **Jumlah variabel:** 14 kolom (5 variabel input + variabel lain)
- **Tipe data:** Data nyata *(real data)*, bukan sintetis
- **Variabel input yang digunakan:**
  - `Income` — Pendapatan
  - `Household_Size` — Jumlah anggota keluarga
  - `Work_Experience` — Pengalaman kerja
  - `Age` — Usia
  - `Number_of_Dependents` — Jumlah tanggungan
- **Output:** Skor eligibilitas (0–100) dengan label `Layak` / `Tidak Layak`

---

## Fitur Aplikasi

### 🧠 Fuzzy Logic (Core) — From Scratch

Implementasi **Fuzzy Logic** tanpa menggunakan library fuzzy eksternal (murni _from scratch_):

| Komponen | Detail |
|----------|--------|
| **Variabel Linguistik** | 5 variabel input + 1 output |
| **Fungsi Keanggotaan** | Trapesium & Segitiga |
| **Rule Base** | 33 aturan IF-THEN |
| **Fuzzifikasi** | Konversi input numerik → derajat keanggotaan |
| **Inferensi** | Mamdani (min–max composition) & Sugeno (weighted average) |
| **Defuzzifikasi** | Centroid of Area (Mamdani) & Weighted Average (Sugeno) |

### 📊 Tampilan Aplikasi (6 Tab)

| Tab | Deskripsi |
|-----|-----------|
| **Predictor** | Prediksi interaktif untuk satu data input |
| **Rule Base** | Tabel 33 aturan fuzzy beserta distribusi output |
| **Membership Functions** | Visualisasi fungsi keanggotaan setiap variabel |
| **Defuzzification** | Grafik proses defuzzifikasi Mamdani & Sugeno |
| **Comparison** | Perbandingan semua metode (boxplot, scatter, segment) |
| **Batch Evaluation** | Evaluasi menyeluruh 10.000 baris + metrik + confusion matrix |

### 🤖 Integrasi Machine Learning (Bonus)

| Model | Keterangan |
|-------|-----------|
| Random Forest | ✅ |
| XGBoost | ✅ |
| SVR | ✅ |
| Decision Tree | ✅ |
| Ridge Regression | ✅ |
| KNN | ✅ |

### 🧬 Integrasi Deep Learning (Bonus)

| Model | Keterangan |
|-------|-----------|
| MLP Regressor | ✅ (3 hidden layers: 64→32→16) |

---

## Struktur Proyek

```
fuzzy_eligibility_app/
├── app.py                          # 🎯 Main aplikasi Streamlit
├── setup.sh                        # Script deploy (train model)
├── requirements.txt                # Dependensi Python
├── README.md                       # Dokumentasi ini
│
├── fuzzy_logic/                    # 🔧 Fuzzy Logic dari scratch
│   ├── __init__.py
│   ├── core.py                     # Fungsi keanggotaan (trimf, trapmf)
│   ├── rules.py                    # 33 aturan IF-THEN
│   └── inference.py                # Inferensi Mamdani & Sugeno
│
├── utils/                          # 📐 Utility functions
│   ├── __init__.py
│   ├── metrics.py                  # Ground truth score, confusion matrix
│   └── plots.py                    # Visualisasi (membership, defuzz, dll)
│
├── dl_model/                       # 🧠 Model ML/DL
│   ├── __init__.py
│   ├── train_all.py                # Training semua model
│   ├── train.py                    # Training MLP (legacy)
│   ├── model_registry.py           # Registry & lazy loading model
│   └── *.joblib                    # Model terlatih (dihasilkan setup.sh)
│
├── data/
│   └── data.csv                    # Dataset (10.000 baris)
│
└── tests/                          # 🧪 Unit tests
    ├── __init__.py
    ├── test_core.py
    ├── test_inference.py
    ├── test_app.py
    ├── test_batch.py
    ├── test_metrics.py
    └── test_inference.py
```

---

## Cara Menjalankan

### Secara Lokal

```bash
# 1. Clone repositori
git clone <repository-url>
cd fuzzy_eligibility_app

# 2. Install dependensi
pip install -r requirements.txt

# 3. Train model ML (opsional, untuk fitur bonus ML)
python dl_model/train_all.py

# 4. Jalankan aplikasi
streamlit run app.py
```

### Deploy ke Streamlit Cloud

1. Push repositori ke GitHub
2. Buka [share.streamlit.io](https://share.streamlit.io)
3. Pilih repositori → branch `main` → file `fuzzy_eligibility_app/app.py`
4. Klik **Deploy**

Streamlit Cloud akan otomatis menjalankan `setup.sh` (yang melatih model ML) sebelum aplikasi berjalan.

---

## Teknologi yang Digunakan

| Teknologi | Kegunaan |
|-----------|----------|
| **Python 3.10+** | Bahasa pemrograman |
| **Streamlit** | Framework web interaktif |
| **NumPy** | Komputasi numerik & vektorisasi |
| **Pandas** | Manipulasi data |
| **Matplotlib** | Visualisasi grafik |
| **scikit-learn** | Model ML (Random Forest, SVR, dll) & scaler |
| **XGBoost** | Model gradient boosting |
| **joblib** | Serialisasi model |

---

## Pemenuhan Ketentuan Tugas Besar

### ✅ Ketentuan Kelompok
- [x] Tugas dikerjakan secara berkelompok (maks. 3 orang)
- [x] Topik permasalahan: Prediksi eligibilitas penerima subsidi rumah
- [x] Dataset berbeda dan telah melalui approval dosen

### ✅ Ketentuan Dataset
- [x] **Data nyata** — bersumber dari Kaggle
- [x] **Minimal 5.000 baris** — 10.000 baris
- [x] **Minimal 5 variabel input** — 5 variabel (Income, Household Size, Work Experience, Age, Number of Dependents)
- [x] **1 output** — Skor eligibilitas (0–100)
- [x] **Sumber dataset** — Tercantum di bagian Dataset

### ✅ Ketentuan Fuzzy Logic
- [x] **Variabel linguistik** — 5 variabel input + 1 output
- [x] **Fungsi keanggotaan** — Trapesium & Segitiga (from scratch)
- [x] **Rule base minimal 15** — 33 aturan IF-THEN
- [x] **Fuzzy Mamdani** — Inferensi min–max + defuzzifikasi centroid
- [x] **Fuzzy Sugeno** — Inferensi min + defuzzifikasi weighted average
- [x] **From scratch** — Tidak menggunakan library fuzzy
- [x] **Proses fuzzy:** Fuzzifikasi → Inferensi → Defuzzifikasi ✅

### ✅ Analisis & Evaluasi
- [x] **Perbandingan Mamdani vs Sugeno** — Tersedia di tab Comparison & Batch Evaluation
- [x] **Evaluasi performa** — Accuracy, MAE, RMSE
- [x] **Confusion matrix** — Perbandingan dengan ground truth
- [x] **Interpretasi** — Analisis kelebihan & kekurangan setiap metode di laporan

### ✅ Bonus
- [x] **Aplikasi web Streamlit** (+5) — ✅ Terimplementasi
- [x] **Integrasi Fuzzy + Machine Learning** (+10) — ✅ 6 model ML
- [x] **Integrasi Fuzzy + Deep Learning** (+20) — ✅ MLP Regressor

### ✅ Pengumpulan
- [x] **Laporan (.pdf)** — Disertakan terpisah
- [x] **Source code (.ipynb)** — `Fuzzy_Eligibilitas_RumahSubsidi.ipynb`
- [x] **Aplikasi Streamlit** — `app.py`

---

## Anggota Kelompok

| Nama | NIM |
|------|-----
| Muhammad Arshad Rizqullah | 103012400228 |
| Zona Putra Pribadi | 103012430010 |


> ⚠️ **Sesuaikan nama, NIM, dan kontribusi anggota kelompok.**

---

## License

Tugas Besar — Mata Kuliah Dasar Kecerdasan Artifisial
