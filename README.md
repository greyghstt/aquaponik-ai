# Grand Design AI Smart Greenhouse Aquaponik

Simulasi end-to-end **Smart Greenhouse Aquaponik ikan nila + selada**
berbasis **Random Forest Classifier** untuk kebutuhan poster akademik.

Pipeline ini mensimulasikan pembacaan sensor realistis, melakukan labeling
hybrid berbasis domain aquaponik, melatih model AI, membuat visualisasi, dan
menghasilkan dashboard HTML statis.

## Alur Sistem

```mermaid
flowchart LR
    A[Sensor] --> B[Preprocessing]
    B --> C[Random Forest AI]
    C --> D[Klasifikasi Status]
    D --> E[Rekomendasi]
    E --> F[Tindakan Semi-Otomatis]
    F --> G[Dashboard]
```

## Sensor yang Disimulasikan

- pH air
- Suhu air
- DO atau dissolved oxygen
- Amonia
- Nitrit
- Nitrat
- EC/TDS
- Level air
- Suhu udara
- Kelembapan udara
- Intensitas cahaya

## Output AI

Model mengklasifikasikan kondisi sistem ke dalam:

- `Normal`
- `DO rendah`
- `Amonia tinggi`
- `Nitrit tinggi`
- `Nutrisi rendah`
- `pH tidak stabil`
- `Level air rendah`
- `Suhu air tinggi`
- `Risiko stres ikan`

## Cara Menjalankan

Install dependensi jika diperlukan:

```powershell
rtk pip install -r requirements.txt
```

Jalankan pipeline:

```powershell
rtk python main.py
```

Semua hasil akan dibuat ulang di folder `outputs/`.

## Hasil Utama

- Dashboard: `outputs/dashboard.html`
- Dataset poster: `outputs/sensor_dataset.csv`
- Dataset training: `outputs/training_dataset.csv`
- Model Random Forest: `outputs/rf_model.joblib`
- Evaluasi model: `outputs/metrics.json`
- Verifikasi simulasi: `outputs/verification.json`
- Grafik poster: `outputs/plots/`

## Visualisasi Poster

Pipeline menghasilkan visual berikut:

- Tren sensor 14 hari
- Status AI terhadap waktu
- Feature importance Random Forest
- Distribusi kelas
- Confusion matrix

## Metrik Terakhir

Hasil dari eksekusi terakhir:

- Accuracy: `0.9926`
- Macro F1-score: `0.9798`
- Verifikasi simulasi: `passed`
- Horizon poster: `14 hari`, interval `1 jam`

## Catatan Realisme Simulasi

Data dummy tidak dibuat acak murni. Generator memasukkan pola aquaponik yang
masuk akal:

- suhu air dan suhu udara naik saat siang,
- intensitas cahaya mengikuti pola pagi-siang-sore,
- DO turun ketika suhu air naik,
- amonia meningkat setelah pemberian pakan,
- nitrit mengikuti proses biologis nitrifikasi,
- nitrat dan EC/TDS berubah lebih lambat,
- level air turun bertahap karena evaporasi dan penggunaan sistem.

Dashboard juga memakai safety guardrail sederhana agar kondisi domain-risk yang
jelas, seperti pH tidak stabil, tidak tersembunyi sebagai kondisi normal.

## Referensi Domain

- Oklahoma State University Extension: prinsip small-scale aquaponics.
- UF/IFAS Extension: kualitas air, DO, amonia, dan nitrit pada sistem ikan.
- FAO: parameter utama monitoring small-scale aquaponic food production.
