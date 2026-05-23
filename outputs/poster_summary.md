# Ringkasan Poster Grand Design AI Smart Greenhouse Aquaponik

## Konsep Sistem
sensor -> preprocessing -> AI -> klasifikasi -> rekomendasi -> tindakan semi-otomatis -> dashboard

## Output AI Terakhir
- Status: Normal
- Confidence RF: 90.8%
- Skor risiko: 0.0/100
- Rekomendasi: Pertahankan pemantauan; sistem berada dalam rentang aman untuk ikan nila dan selada.
- Tindakan semi-otomatis: monitoring normal

## Evaluasi Model
- Accuracy: 0.9926
- Macro F1: 0.9798
- Weighted F1: 0.9927
- Data training: 1620 baris
- Data uji: 540 baris

## Verifikasi Simulasi
- Verifikasi lulus: True
- Korelasi suhu air vs DO: -0.646
- Korelasi cahaya vs suhu udara: 0.75
- Timestamp per jam konsisten: True
- Rentang sensor realistis: True

## Catatan Referensi Domain
- OSU Extension: aquaponic pH 6.5-7.5 should be maintained; fish pH target near neutral. (https://extension.okstate.edu/fact-sheets/principles-of-small-scale-aquaponics)
- OSU Extension: tilapia optimal water temperature listed around 74-80 F. (https://extension.okstate.edu/fact-sheets/aquaponics)
- UF/IFAS: dissolved oxygen is a primary fish-production water quality parameter. (https://ask.ifas.ufl.edu/FA002)
- UF/IFAS: ammonia is a key intensive-system fish risk after oxygen; low oxygen can disrupt nitrification. (https://edis.ifas.ufl.edu/publication/FA031)
- FAO small-scale aquaponics: oxygen, pH, ammonia, nitrite, nitrate, and temperature are core monitoring variables. (https://www.fao.org/family-farming/detail/en/c/1743021/)
