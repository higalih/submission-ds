# Laporan Proyek Machine Learning Recommender System - Galih Anggoro Mukti

## Project Overview
<a id="user-content-project-overview"></a>[](#project-overview)
Rekomendasi film merupakan salah satu aplikasi penting dari sistem rekomendasi dalam industri hiburan. Dengan semakin banyaknya konten film yang tersedia di platform streaming seperti Netflix, Amazon Prime, dan Disney+, pengguna sering kali mengalami kesulitan dalam memilih film yang sesuai dengan preferensi mereka. Sistem rekomendasi dapat membantu pengguna menemukan film yang relevan berdasarkan preferensi sebelumnya atau atribut film tertentu.  
**Mengapa proyek ini penting?**:
Proyek ini penting karena sistem rekomendasi yang efektif dapat meningkatkan pengalaman pengguna, meningkatkan engagement, dan membantu platform streaming mempertahankan pelanggan mereka. Selain itu, sistem rekomendasi juga dapat digunakan oleh industri perfilman untuk memahami tren penonton dan membuat keputusan bisnis yang lebih baik.
**Hasil riset sebagai referensi untuk penyelesaian proyek?**
1. [Collaborative Filtering for Implicit Feedback Datasets](https://ieeexplore.ieee.org/document/4781121?spm=5aebb161.76e4cdc2.0.0.4374c921IcX80i)
2. [Deep Cross-Domain Fashion Recommendation](https://doi.org/10.1145/3109859.3109861)

## Business Understanding
<a id="user-content-business-understanding"></a>[](#business-understanding)
Model ini bertujuan untuk meningkatkan pengalaman user dengan memberikan rekomendasi film secara personal berdasarkan preferensi individu dan juga fitur data dari dataset film.

### Problem Statements
<a id="user-content-problem-statements"></a>[](#problem-statements)
1. Bagaimana cara membantu pengguna menemukan film yang sesuai dengan preferensi mereka di tengah banyaknya pilihan film yang tersedia? 
2. Bagaimana cara meningkatkan relevansi rekomendasi film dengan mempertimbangkan fitur-fitur spesifik seperti genre, cast, director, dan overview?
3. Bagaimana cara memastikan bahwa dataset yang digunakan dalam sistem rekomendasi memiliki kualitas yang baik dan bebas dari noise atau missing values?

### Goals
<a id="user-content-goals"></a>[](#goals)
Secara spesifik, tujuan proyek ini meliputi:
1. Membangun sistem rekomendasi film yang dapat memberikan rekomendasi berdasarkan preferensi pengguna, seperti rating dan atribut film tertentu.
2. Mengintegrasikan fitur-fitur spesifik film seperti genre, cast, director, dan overview ke dalam model rekomendasi untuk meningkatkan relevansi hasil rekomendasi.
3. Membersihkan dan mempersiapkan data untuk memastikan kualitas dataset yang digunakan dalam sistem rekomendasi.

Dengan mencapai tujuan-tujuan ini, proyek ini diharapkan dapat memberikan solusi praktis untuk memberikan rekomendasi film kepada pengguna.

**Solution Statement**:
Untuk mengatasi tantangan dalam membuat model sistem rekomendasi film, kami melakukan pendekatan-pendekatan ini:
1. Weighted Rating Recommender : Menggunakan formula IMDB untuk merekomendasikan film berdasarkan vote count dan vote average.
2. Content-Based Recommender : Menggunakan TF-IDF dan cosine similarity untuk merekomendasikan film berdasarkan overview.
3. Advanced Content-Based Recommender : Menggabungkan fitur-fitur seperti genre, cast, director, dan keywords untuk memberikan rekomendasi yang lebih personal.

## Data Understanding
<a id="user-content-data-understanding"></a>[](#data-understanding)
Dataset yang digunakan dalam proyek ini adalah **TMDB Movie Metadata**, yang dapat diunduh dari [Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata?spm=5aebb161.76e4cdc2.0.0.4374c921IcX80i). Dataset ini terdiri dari dua file utama:
1. `tmdb_5000_movies.csv`: Berisi informasi tentang film seperti genre, keywords, production companies, dan overview.
2. `tmdb_5000_credits.csv`: Berisi informasi tambahan seperti cast, crew, dan karakter dalam film.

Dataset ini mencakup data film dengan berbagai fitur seperti judul, overview, genre, cast, director, dan rating pengguna.
- Jumlah data asli sebanyak 4.803 record
- Jumlah kolom sebanyak `4 kolom` dan `20 kolom`.
- Terdapat missing value sebanyak:
	- homepage = 3091 records
	- overview = 3 records
	- release_date = 1 record
	- runtime = 2 records
	- tagline = 844 records
- Tidak ditemukan adanya duplikasi data dalam dataset untuk fitur-fitur yang dibutuhkan.
- Mising value pada kolom yg kosong tidak berpengaruh signifikan dalam pembuatan model ditinjau dari fitur ataupun jumlah records pada fitur `overview`.

**Fitur-fitur pada data di antaranya:**
- id: ID unik untuk setiap film.  
- title: Judul film.
- overview: Deskripsi singkat tentang film.
- genres: Genre film dalam format JSON.
- keywords: Kata kunci yang menggambarkan elemen-elemen penting dalam film.
- cast: Informasi tentang aktor utama dalam film.
- crew: Informasi tentang tim produksi, termasuk sutradara.
- vote_average: Rata-rata rating pengguna untuk film tersebut.
- vote_count: Jumlah total voting yang diterima film tersebut.

**Exploratory Data Analysis (EDA)**:
Melalui EDA, kami menemukan insight awal dari dataset:
- Film dengan vote count tinggi cenderung memiliki vote average yang lebih stabil.
- Fitur seperti overview, genres, dan keywords sangat berguna untuk membangun sistem rekomendasi berbasis konten.  
- Missing values ditemukan dalam kolom `overview`, yang telah ditangani dengan mengisi nilai kosong menggunakan string kosong ('').

## Data Preparation
<a id="user-content-data-preparation"></a>[](#data-preparation)
Teknik data preparation yang dilakukan meliputi:
1. Merging Data : Menggabungkan dua dataset (`tmdb_5000_movies.csv` dan `tmdb_5000_credits.csv`) berdasarkan kolom `id`.
2. Handling Missing Values : Mengisi missing values dalam kolom `overview` dengan string kosong ('').
3. Parsing JSON Data : Mengubah kolom seperti `cast`, `crew`, `keywords`, dan `genres` dari format JSON menjadi list Python menggunakan `literal_eval`.
4. Feature Extraction : Mengekstrak fitur seperti nama sutradara, top 3 cast, dan genre untuk digunakan dalam sistem rekomendasi.
5. Text Cleaning : Membersihkan teks dengan menghapus spasi dan mengonversi semua teks menjadi huruf kecil untuk konsistensi.

**Penjelasan langkah-langkah pada preprocessing data**:
1. Proses data preparation bertujuan untuk memastikan bahwa dataset siap digunakan untuk membangun model rekomendasi. Langkah-langkah seperti parsing JSON dan cleaning text sangat penting untuk menghasilkan fitur yang dapat diproses oleh algoritma machine learning. 
2. Tahapan data preparation diperlukan untuk mengatasi masalah seperti missing values, format data yang tidak konsisten, dan noise dalam dataset. Hal ini memastikan bahwa model rekomendasi dapat bekerja secara efektif dan menghasilkan hasil yang akurat.

## Modeling
<a id="user-content-modeling"></a>[](#modeling)
**Model Selection**
Pada proyek ini, kami membangun dua jenis sistem rekomendasi untuk memberikan solusi terhadap permasalahan yang diidentifikasi:
1. Weighted Rating Recommender
	- Model ini menggunakan formula IMDB untuk menghitung skor berbobot berdasarkan `vote_average` dan `vote_count`. Film dengan vote count di atas persentil ke-90 dipilih untuk direkomendasikan.  
	- Output: Top 10 film berdasarkan skor berbobot. Contoh hasil rekomendasi:
![1](https://github.com/user-attachments/assets/43da95b8-a33a-401b-a2be-f30e6059fc64)
2. Content-Based Recommender
	- Model ini menggunakan TF-IDF Vectorizer dan cosine similarity untuk merekomendasikan film berdasarkan kemiripan teks pada kolom `overview`.    
	- Output: Top 10 film berdasarkan kemiripan overview dengan film input (misalnya, "Interstellar"). Contoh hasil rekomendasi:
![2](https://github.com/user-attachments/assets/d24245ff-634e-4375-8ae0-e95b8aa709ac)
3. Advanced Content-Based Recommender
	- Model ini menggabungkan fitur-fitur seperti `keywords`, `cast`, `director`, dan `genres` untuk membuat representasi film yang lebih kaya. Representasi ini kemudian digunakan untuk menghitung cosine similarity.    
	- Output: Top 10 film berdasarkan kombinasi fitur dengan film input (misalnya, "The Dark Knight Rises"). Contoh hasil rekomendasi:
![3](https://github.com/user-attachments/assets/6df96f64-2e53-4bc8-b3a5-aede18336d8d)

**Kami menyajikan dua solusi utama:**
1. **Weighted Rating Recommender** : Berfokus pada popularitas dan rating pengguna.
2. **Content-Based Recommender** : Berfokus pada kemiripan konten antara film.

**Kelebihan dan kekurangan masing-masing pendekatan:**
1. Weighted Rating Recommender    
	- Kelebihan: Mudah diimplementasikan dan memberikan rekomendasi berdasarkan preferensi umum pengguna.  
	- Kekurangan: Tidak mempertimbangkan preferensi individu atau fitur spesifik film.
2. Content-Based Recommender    
	- Kelebihan: Memberikan rekomendasi yang lebih personal berdasarkan fitur spesifik film.  
	- Kekurangan: Terbatas pada fitur yang tersedia dan tidak mempertimbangkan interaksi antar pengguna.
         
## Evaluation
<a id="user-content-evaluation"></a>[](#evaluation)
Untuk mengevaluasi kinerja sistem rekomendasi, kami menggunakan metrik Cosine Similarity  untuk mengukur kemiripan antara film. Cosine similarity menghitung sudut antara dua vektor representasi film, dengan nilai berkisar antara 0 hingga 1. Semakin tinggi nilai cosine similarity, semakin mirip dua film tersebut. 
Hasil evaluasi menunjukkan bahwa: 
1. Weighted Rating Recommender  berhasil memberikan rekomendasi film populer dengan akurasi tinggi berdasarkan vote count dan vote average.
2. Content-Based Recommender  berhasil memberikan rekomendasi film yang mirip dengan input berdasarkan overview.  
3. Advanced Content-Based Recommender  memberikan hasil yang lebih relevan dengan mempertimbangkan fitur tambahan seperti genre, cast, dan director.

**Penjelasan formula metrik dan bagaimana metrik tersebut bekerja.**
Cosine Similarity dengan formula:

`cos_sim(A, B) = (A · B) / (||A|| * ||B||)`

Di mana `A` dan `B` adalah vektor representasi film. Nilai cosine similarity mendekati 1 menunjukkan bahwa dua film sangat mirip. 
