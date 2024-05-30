# ML-Predictive-Analytics-Year-of-Experience-And-Employees-Salary
Project Machine Learning Terapan Predictive Analytics Year of Experience And Employees Salary

# Domain Proyek
Rekrutmen merupakan proses krusial dalam mendapatkan calon karyawan yang tepat untuk mengisi posisi atau jabatan tertentu. Proses ini tidak hanya mencari, tetapi juga menyeleksi pelamar yang memiliki potensi dan kualifikasi yang sesuai dengan kebutuhan perusahaan. Faktor yang mempengaruhi peningkatan produktivitas kerja karyawan antara lain pengalaman kerja dan gaji karyawan.  Oleh karena itu, dalam proyek ini, perusahaan akan mengembangkan beberapa model Machine Learning untuk memprediksi kisaran gaji berdasarkan pengalaman kerja calon pelamar, dengan harapan untuk mengevaluasi dan memilih model yang paling akurat dalam memberikan prediksi gaji yang sesuai.

Referensi : [Peningkatan Produktivitas Kerja Karyawan](https://jurnalpenyuluhan.ipb.ac.id/index.php/jabm/article/view/33128/21748)

# Business Understanding

## Problem Statements
1. Bagaimana algoritma yang tepat untuk memprediksi kisaran gaji karyawan?
2. Bagaimana cara menentukan hasil prediksi suatu Algoritma Machine Learning dapat dikatakan baik?

## Goals
1. Menentukan algoritma yang tepat untuk memprediksi kisaran gaji karyawan
2. Menentukan hasil prediksi suatu Algoritma Machine Learning dapat dikatakan baik

## Solution Statements
Solusi untuk masalah ini sebagai berikut.
1. Membuat 2 model Machine Learning yaitu dengan algoritma LinearRegression dan RandomForest.
   - Konsep dari algoritma LinearRegression adalah memprediksi nilai dari y dengan mengetahui nilai x dan menemukan nilai m dan b yang errornya paling minimal. Adapun kelebihan dari metode ini yakni metode ini mampu digunakan untuk memprediksi nilai yang ada pada masa depan jika hubungan antara variabel independen dan dependen memiliki hubungan linear. Kekurangan dari metode ini yaitu pada keadaan sesungguhnya jarang sekali variabel dependen dan independen menunjukkan hubungan yang jelas
   - Konsep dari algoritma RandomForest yaitu model prediksi yang terdiri dari beberapa model dan bekerja secara bersama-sama. Kelebihan dari metode ini yakni jika dataset berjumlah banyak maka RandomForest akan bekerja secara efisien.
2. Tujuan dari proyek ini adalah untuk membuat perkiraan gaji, yang merupakan variabel kontinu. Sebagai contoh, kita ingin memprediksi gaji berdasarkan pengalaman kerja seseorang. Dalam konteks ini, prediksi gaji merupakan masalah regresi karena variabel targetnya adalah nilai kontinu. Dalam masalah regresi seperti ini, salah satu metrik evaluasi yang umum digunakan adalah Mean Squared Error (MSE). Metrik ini mengukur seberapa jauh hasil prediksi dari nilai yang sebenarnya. Semakin kecil nilai MSE, semakin baik modelnya dalam memprediksi. Dengan demikian, dalam proyek ini, setiap model yang dikembangkan akan dievaluasi menggunakan MSE, dan algoritma yang menghasilkan nilai MSE terendah akan dipilih sebagai model yang terbaik untuk digunakan dalam memprediksi gaji berdasarkan pengalaman kerja calon pelamar.

# Data Understanding

## Sumber Dataset 
[Year of Experience and Employees Salary](https://www.kaggle.com/datasets/rubydoby/years-of-experience-and-employees-salary/data?select=employee_salaries.csv)

## Baris dan Kolom Dataset
Dataset terdapat 1500 baris dan 2 kolom.

## Variabel dataset
- "Years of Experience" yaitu total tahun pengalaman kerja.
- "Salary" yaitu total gaji karyawan per tahun dalam kurs dollar.

## Exploratory Data Analysis - Univariate Analysis
![unvariate](https://drive.google.com/uc?export=view&id=1wXTyVQh10Vh9CloSnSYoPSufizITg7Sp)

Dari hasil visualisasi di atas dapat disimpulkan bahwa:
- Sebagian besar sampel Years of experience berada di kisaran 8-14 tahun.
- Sebagian besar sampel Salary berada di kisaran 86000-90000.

## Exploratory Data Analysis - Multivariate Analysis
![multivariate](https://drive.google.com/uc?export=view&id=1QIODT48RKiquT16iArf6W8QAxPZiAEhm)

Dari hasil visualisasi data di atas dapat disimpulkan bahwa:
- Pola sebaran data pada grafik pairplot di atas memiliki korelasi posistif

![heatmap](https://drive.google.com/uc?export=view&id=1p6KPP5Om4ztWVwPVI6gRgCcHdZswlV-a)

Berdasarkan visualisasi heatmap di atas dapat disimpulkan bahwa:
- Variabel Years of experience berkorelasi positif dengan variabel Salary, skornya yaitu 0.8.

# Data Preparation
Berikut merupakan tahapan-tahapan dalam Data Preparation:
- Melakukan Split Data, dataset yang ada dibagi menjadi 2 bagian yaitu data latih dan data uji dengan rasio 90:10. Proses ini dilakukan dengan menggunakan modul train_test_split dari library scikit-learn.
- Melakukan standarisasi pada data latih dengan menggunakan StandardScaler dari library sckit-learn.

# Modeling
Setelah melakukan data preparation data yang sudah siap akan digunakan untuk membuat model, kali ini akan dibuat 2 model sebagai perbandingan.
- Membuat model dengan menggunakan algoritma LinearRegression, alasan menggunakan algoritma ini karena ini merupakan algortima yang umum untuk menyelesaikan permasalahan regresi, kelebihan dari algoritma ini yaitu mudah dipahami.
- Membuat model dengan menggunakan algoritma RandomForest, kelebihan dari menggunakan algoritma ini yaitu dapat mengatasi noise dan missing value serta dapat mengatasi data dalam jumlah yang besar, adapun kekurangan pada algoritma Random Forest yaitu interpretasi yang sulit dan membutuhkan tuning model yang tepat untuk data
  
# Evaluation
Proses evaluasi model pada proyek ini menggunakan metrik Mean Squared Error yang menghitung jumlah selisih kuadrat rata-rata nilai sebenarnya dengan nilai prediksi

## Model dengan Algoritma LinearRegression
![linearRegression](https://drive.google.com/uc?export=view&id=1Kfjht4qeni_KqgCMg4peWYbx4VMChseo)

Seperti terlihat pada gambar, model yang dibuat menggunakan algortima LinearRegression memiliki nilai MSE yang sangat tinggi hingga mencapai 113458.095232 pada saat training dan 127004.94334 pada saat test, hal ini menunjukkan algoritma ini kurang baik untuk melakukan prediksi.

![ujiLinearRegression](https://drive.google.com/uc?export=view&id=1gS_AyaN8WtI9LnLgM1EBS_56GJr7clTb)

Pada proses pengujian pun dapat terlihat hasil prediksi tidak akurat dengan nilai sebenarnya.

## Model dengan Algortima RandomForest
![randomForest](https://drive.google.com/uc?export=view&id=1nwErlaA6JiqtVprkYMdcFZLvo6_-jk7b)

Seperti terlihat pada gambar, saat dibandingkan dengan algoritma LinearRegression terlihat algortima RandomForest memiliki nilai MSE yang lebih rendah yaitu 13235.129443 pada saat training dan 15922.675464 pada saat test, hal ini menunjukkan algoritma RandomForest lebih baik untuk melakukan prediksi dibanding LinearRegression.

![ujiRandomForest](https://drive.google.com/uc?export=view&id=1Zuz8o0vin282SEFKZ6c1m4fhzePNTug7)

Pada proses pengujian dapat terlihat hasil prediksi dari model yang menggunakan RandomForest lebih akurat dengan nilai sebenarnya. Oleh karena itu algoritma ini yang akan dipilih sebagai model utama untuk memprediksi kisaran gaji karyawan.
