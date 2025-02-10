# Laporan Proyek Machine Learning - Galih Anggoro Mukti

## Domain Proyek
Nilai tukar mata uang (exchange rate) memainkan peran krusial dalam ekonomi global. Ini adalah harga satu mata uang terhadap mata uang lain, dan fluktuasi nilai tukar dapat memengaruhi berbagai aspek. Proyek ini berfokus pada domain keuangan , khususnya dalam memprediksi nilai tukar mata uang USD/EUR  (Dolar AS terhadap Euro).

**Mengapa memilih domain proyek ini?**:
Nilai tukar mata uang adalah elemen penting dalam ekonomi global, tetapi fluktuasinya yang kompleks membuat prediksi manual menjadi sulit dan tidak efisien. Dengan menggunakan pendekatan machine learning, khususnya LSTM, kita dapat mengembangkan model prediktif yang akurat dan otomatis untuk memperkirakan nilai tukar USD/EUR.
  
## Business Understanding

Nilai tukar mata uang, seperti USD/EUR, memiliki dampak signifikan terhadap perdagangan internasional, investasi, dan stabilitas ekonomi. Namun, kompleksitas fluktuasi nilai tukar membuat prediksi yang akurat menjadi tantangan besar.

### Problem Statements
Berdasarkan kondisi yang telah diuraikan sebelumnya, saya mencoba mengembangkan sebuah sistem prediksi nilai tukar mata uang USD/EUR untuk menjawab permasalahan berikut:
- Bagaimana cara memprediksi nilai tukar USD/EUR dengan akurasi tinggi?
- Apakah model machine learning, terutama LSTM, dapat menangkap pola yang relevan dalam data time series untuk meningkatkan prediksi nilai tukar?

### Goals
Untuk menjawab pertanyaan di atas, saya akan membuat predictive modelling dengan tujuan sebagai berikut:
- Mengembangkan model machine learning yang dapat memprediksi nilai tukar USD/EUR di masa depan dengan akurasi tinggi.
- Membangun model yang mampu menangkap pola temporal dalam data dengan menggunakan data historis nilai tukar USD/EUR.
- Meningkatkan akurasi prediksi dibandingkan dengan metode manual atau tradisional, yang sering kali tidak efisien atau tidak konsisten.

Dengan mencapai tujuan-tujuan ini, proyek ini diharapkan dapat memberikan solusi praktis untuk mengatasi ketidakpastian dalam pergerakan nilai tukar.

**Solution Statement**:
Untuk mengatasi tantangan dalam memprediksi nilai tukar USD/EUR, kami mengusulkan solusi berbasis machine learning  menggunakan model LSTM (Long Short-Term Memory) . Berikut adalah langkah-langkah solusi yang diusulkan:
- Pengumpulan dan Pemrosesan Data : Menggunakan dataset harian nilai tukar USD/EUR dari tahun 2004 hingga saat ini.
- Pemodelan dengan LSTM : Memanfaatkan arsitektur LSTM, yang dirancang khusus untuk menangani data sekuensial seperti time series.
- Evaluasi Performa Model : Menggunakan metrik evaluasi seperti Mean Absolute Error (MAE) , Mean Squared Error (MSE) , dan Root Mean Squared Error (RMSE)  untuk mengukur akurasi prediksi.
- Memvisualisasikan hasil prediksi melalui grafik actual vs predicted dan analisis residuals untuk memastikan model bekerja dengan baik.

## Data Understanding
Dataset yang digunakan berasal dari Kaggle , yaitu dataset Daily Forex Exchange Rates yang mencakup nilai tukar harian lebih dari 160 mata uang terhadap Euro (EUR) sejak tahun 2004 hingga saat ini.
- Jumlah data asli sebanyak 401.606 record
- Jumlah kolom sebanyak 5 kolom.
- Terdapat missing value sebanyak 743 pada kolom `currency_name`.
- Tidak ditemukan adanya duplikasi data dalam dataset untuk tiap-tiap tanggal.

Setelah difilter untuk pasangan USD/EUR , dataset akhir hanya mencakup dua kolom:
- `date`: Tanggal nilai tukar.
- `USD_EUR`: Nilai tukar USD terhadap EUR.
- Tidak ada missing value pada data hasil filter.

**Link dataset:**
Dataset dapat diakses melalui link  [Forex Exchange Rates Since 2004](https://www.kaggle.com/datasets/asaniczka/forex-exchange-rate-since-2004-updated-daily/data).

**Fitur-fitur pada data:**
- `currency`: Mata uang target (contoh: USD, ZWL, GHS).
- `base_currency`: Mata uang dasar (selalu EUR dalam dataset ini).
- `currency_name`: Nama lengkap mata uang target.
- `exchange_rate`: Nilai tukar mata uang target terhadap EUR.
- `date`: Tanggal nilai tukar.


**Exploratory Data Analysis (EDA)**:
Melalui EDA, kami menemukan bahwa:
- Dataset tidak mengandung nilai yang hilang (missing values ) untuk pasangan USD/EUR.
- Tidak ada outlier signifikan berdasarkan metode IQR  dan Z-score , yang menunjukkan bahwa data relatif bersih dan stabil.

## Data Preparation
Langkah-langkah preprocessing yang dilakukan meliputi:
1. Filtering Data for USD/EUR.
2. Handling Missing Values.
3. Creating Lag Features.
4. Splitting the Data.
5. Reshaping for LSTM

**Penjelasan langkah-langkah pada preprocessing data**:
1. Dataset awal difilter untuk hanya menyertakan baris di mana kolom `currency` adalah USD . Hal ini untuk menghasilkan subset data yang berfokus pada nilai tukar USD/EUR.
2. Meskipun tidak ada nilai yang hilang dalam dataset USD/EUR, langkah ini tetap diperiksa untuk memastikan integritas data.
3. Untuk menangkap dependensi temporal dalam data, kami membuat lag features (`Lag_1`, `Lag_2`, `Lag_3`) yang mewakili nilai tukar dari hari-hari sebelumnya.
4. Dataset dibagi menjadi data latih (training)  dan data uji (testing)  dengan rasio 80:20. Dalam splitting ini kami memastikan urutan kronologis data dipertahankan dengan menonaktifkan pengocokan (shuffle=False).
5. Input data untuk model LSTM harus berupa tensor 3D dengan format (samples, timesteps, features). Oleh karena itu, data pelatihan dan pengujian diubah bentuknya.

## Modeling
**Model Selection**
Untuk memprediksi nilai tukar USD/EUR, kami memilih model LSTM (Long Short-Term Memory) , sebuah jenis arsitektur deep learning  yang dirancang khusus untuk menangani data sekuensial seperti time series. LSTM sangat cocok untuk proyek ini karena kemampuannya dalam menangkap pola jangka panjang dan dependensi temporal dalam data. 

**Model Architecture**
Model LSTM yang dibangun memiliki arsitektur sebagai berikut:
- Input Layer : Menerima input berupa lag features (`Lag_1`, `Lag_2`, `Lag_3`) yang telah diubah menjadi tensor 3D.
- LSTM Layer : Terdiri dari 50 unit LSTM, yang memungkinkan model untuk belajar pola kompleks dalam data.
- Output Layer : Lapisan Dense dengan satu neuron untuk memprediksi nilai tukar USD/EUR pada hari berikutnya.

**Training Process**
Model dilatih menggunakan parameter berikut:
- Epochs : 50 iterasi pelatihan.
- Batch Size : 32 sampel per pembaruan gradien.
- Loss Function : Mean Squared Error (MSE), yang umum digunakan untuk tugas regresi.
- Optimizer : Adam, yang secara otomatis menyesuaikan laju pembelajaran selama pelatihan.

Pelatihan dilakukan pada data latih (`X_train_lstm`, `y_train`), sementara data uji (`X_test_lstm`, `y_test`) digunakan untuk validasi. Proses pelatihan memungkinkan kami memantau performa model setelah setiap epoch.

## Evaluation
**Metrics**
Setelah pelatihan, model dievaluasi menggunakan metrik regresi berikut:
- Mean Absolute Error (MAE) : Mengukur rata-rata perbedaan absolut antara nilai aktual dan prediksi. Skor pada matrics MAE adalah: **0.009162290556496434**
- Mean Squared Error (MSE) : Mengukur rata-rata kuadrat perbedaan antara nilai aktual dan prediksi. Skor pada matrics MSE adalah: **0.0001474359784902076**
- Root Mean Squared Error (RMSE) : Akar kuadrat dari MSE, yang lebih mudah diinterpretasikan karena memiliki satuan yang sama dengan target. Skor pada matrics RMSE adalah: **0.01214232179157708**
![Screenshot 2025-02-10 at 21-19-54 Submission Predictive Analysis Dicoding ipynb - Colab](https://github.com/user-attachments/assets/3000c6f7-cee4-409f-a04e-cf45828ca3a0)

Dengan menggunakan model yang telah dikembangkan, maka kami dapat memprediksi nilai tukar USD/EUR dengan akurasi tinggi.
Model machine learning, terutama LSTM, terbukti dapat menangkap pola yang relevan dalam data time series untuk meningkatkan prediksi nilai tukar mata uang.

**Visualization**
Untuk memvisualisasikan performa model, kami membuat dua jenis plot:
1. Actual vs Predicted:
    - Plot ini membandingkan nilai tukar aktual dengan nilai tukar yang diprediksi oleh model.
    - Garis merah mewakili prediksi, sementara garis biru mewakili nilai aktual.
    - Visualisasi ini menunjukkan bahwa model secara umum dapat mengikuti tren nilai tukar dengan baik.
2. Residuals Analysis :
    - Residuals (selisih antara nilai aktual dan prediksi) diplot untuk melihat seberapa besar deviasi model.
    - Plot residuals membantu mengidentifikasi titik-titik di mana model mengalami kesulitan dalam memprediksi nilai tukar.

_Lampiran Gambar:_
- Plot nilai aktual vs prediksi
![ed27226e0706b790575abca05942b877](https://github.com/user-attachments/assets/a73e328e-bffc-41fc-8396-1e9123686e57)
- Tabel perbandingan nilai aktual vs prediksi.
![2f6b5ee3ecba84c43a39fd204caf0025](https://github.com/user-attachments/assets/b1b3778b-a8dc-44ec-9be0-2da24d685624)
- Visualisasi residual (selisih antara aktual vs prediksi).
![800bd5c3f17444eb5d128386e07a0fff](https://github.com/user-attachments/assets/f069a7a4-33af-48f3-8db7-ad3adef4c737)
