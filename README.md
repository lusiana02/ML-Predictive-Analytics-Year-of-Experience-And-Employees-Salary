# ML-Predictive-Analytics-Year-of-Experience-And-Employees-Salary
Project _Machine Learning_ Terapan _Predictive Analytics Year of Experience And Employees Salary_

# Domain Proyek
Rekrutmen merupakan proses krusial dalam mendapatkan calon karyawan yang tepat untuk mengisi posisi atau jabatan tertentu. Proses ini tidak hanya mencari, tetapi juga menyeleksi pelamar yang memiliki potensi dan kualifikasi yang sesuai dengan kebutuhan perusahaan. Faktor yang mempengaruhi peningkatan produktivitas kerja karyawan antara lain pengalaman kerja dan gaji karyawan [1].  Oleh karena itu, dalam proyek ini, perusahaan akan mengembangkan beberapa model _Machine Learning_ untuk memprediksi kisaran gaji berdasarkan pengalaman kerja calon pelamar, dengan harapan untuk mengevaluasi dan memilih model yang paling akurat dalam memberikan prediksi gaji yang sesuai.

# _Business Understanding_

## _Problem Statements_
1. Bagaimana algoritma yang tepat untuk memprediksi kisaran gaji karyawan?
2. Bagaimana cara menentukan hasil prediksi suatu Algoritma Machine Learning dapat dikatakan baik?

## _Goals_
1. Menentukan algoritma yang tepat untuk memprediksi kisaran gaji karyawan
2. Menentukan hasil prediksi suatu Algoritma Machine Learning dapat dikatakan baik

## _Solution Statements_
Solusi untuk masalah ini sebagai berikut.
1. Membuat 2 model _Machine Learning_ yaitu dengan algoritma _LinearRegression_ dan _RandomForest_.
   - Konsep dari algoritma _LinearRegression_ adalah memprediksi nilai dari y dengan mengetahui nilai x dan menemukan nilai m dan b yang errornya paling minimal. Adapun kelebihan dari metode ini yakni metode ini mampu digunakan untuk memprediksi nilai yang ada pada masa depan jika hubungan antara variabel independen dan dependen memiliki hubungan linear. Kekurangan dari metode ini yaitu pada keadaan sesungguhnya jarang sekali variabel dependen dan independen menunjukkan hubungan yang jelas
   - Konsep dari algoritma _RandomForest_ yaitu model prediksi yang terdiri dari beberapa model dan bekerja secara bersama-sama. Kelebihan dari metode ini yakni jika dataset berjumlah banyak maka RandomForest akan bekerja secara efisien.
2. Tujuan dari proyek ini adalah untuk membuat perkiraan gaji, yang merupakan variabel kontinu. Sebagai contoh, kita ingin memprediksi gaji berdasarkan pengalaman kerja seseorang. Dalam konteks ini, prediksi gaji merupakan masalah regresi karena variabel targetnya adalah nilai kontinu. Dalam masalah regresi seperti ini, salah satu metrik evaluasi yang umum digunakan adalah _Mean Squared Error_ (MSE). Metrik ini mengukur seberapa jauh hasil prediksi dari nilai yang sebenarnya. Semakin kecil nilai MSE, semakin baik modelnya dalam memprediksi. Dengan demikian, dalam proyek ini, setiap model yang dikembangkan akan dievaluasi menggunakan MSE, dan algoritma yang menghasilkan nilai MSE terendah akan dipilih sebagai model yang terbaik untuk digunakan dalam memprediksi gaji berdasarkan pengalaman kerja calon pelamar.

# _Data Understanding_
## Sumber Dataset 
[Year of Experience and Employees Salary](https://www.kaggle.com/datasets/rubydoby/years-of-experience-and-employees-salary/data?select=employee_salaries.csv)

## Baris dan Kolom Dataset
Dataset terdapat 1500 baris dan 2 kolom.

## Variabel dataset
- _"Years of Experience"_ yaitu total tahun pengalaman kerja.
- _"Salary"_ yaitu total gaji karyawan per tahun dalam kurs dollar.

## _Exploratory Data Analysis - Univariate Analysis_
![eda-unvariate](https://github.com/lusiana02/ML-Predictive-Analytics-Year-of-Experience-And-Employees-Salary/assets/123287899/914dddc5-cd76-4291-b66a-8ee3275ab40a)

Gambar 1 _EDA-Unvariate Analysis_

Dari hasil visualisasi di atas dapat disimpulkan bahwa:
- Sebagian besar sampel _Years of experience_ berada di kisaran 8-14 tahun.
- Sebagian besar sampel _Salary_ berada di kisaran 86000-90000.

## Exploratory Data Analysis - Multivariate Analysis
![eda-multivariate](https://github.com/lusiana02/ML-Predictive-Analytics-Year-of-Experience-And-Employees-Salary/assets/123287899/20d572dd-1ccb-4d0e-b08c-69b9b3250040)

Gambar 2 _EDA-Multivariate Analysis_

Dari hasil visualisasi data di atas dapat disimpulkan bahwa:
- Pola sebaran data pada grafik pairplot di atas memiliki korelasi posistif

![heatmap](https://github.com/lusiana02/ML-Predictive-Analytics-Year-of-Experience-And-Employees-Salary/assets/123287899/080eff85-4122-4fe5-904b-7d15205c39ae)

Gambar 3 Visualisasi _Heatmap_

Berdasarkan visualisasi _heatmap_ di atas dapat disimpulkan bahwa:
- Variabel _Years of experience_ berkorelasi positif dengan variabel _Salary_, skornya yaitu 0.8.

# _Data Preparation_
Berikut merupakan tahapan-tahapan dalam Data Preparation:
- Melakukan _Split Data_, dataset yang ada dibagi menjadi 2 bagian yaitu data latih dan data uji dengan rasio 90:10. Proses ini dilakukan dengan menggunakan modul train_test_split dari library scikit-learn. Dengan rasio 90:10, 90% data digunakan untuk pelatihan, memberikan model lebih banyak informasi untuk belajar pola dan variasi data. Meskipun data pelatihan lebih banyak, 10% data uji masih cukup untuk mengevaluasi performa model secara akurat. Ini memberikan gambaran yang realistis tentang bagaimana model akan bekerja pada data yang belum terlihat sebelumnya. Selain itu, Lebih banyak data pelatihan membantu mengurangi overfitting, di mana model terlalu sesuai dengan data pelatihan dan berkinerja buruk pada data baru.
- Melakukan standarisasi pada data latih dengan menggunakan _StandardScaler_ dari _library_ _sckit-learn_. Hal ini diperlukan untuk memastikan konsistensi skala fitur, mempercepat konvergensi dalam algoritma optimasi, dan menghindari dominasi oleh fitur dengan skala besar. Proses ini melibatkan menghitung mean dan standar deviasi dari data pelatihan dan menerapkan transformasi tersebut pada data pelatihan dan uji.

# _Modeling_
Setelah melakukan _data preparation_ data yang sudah siap akan digunakan untuk membuat model, kali ini akan dibuat 2 model sebagai perbandingan.
- Membuat model dengan menggunakan algoritma _LinearRegression_, alasan menggunakan algoritma ini karena ini merupakan algortima yang umum untuk menyelesaikan permasalahan regresi, kelebihan dari algoritma ini yaitu mudah dipahami. Proses dan tahapan yang dilakukan antara lain mempersiapkan _dataframe_ untuk analisis model, membuat model dengan algoritma _linear regression_, mengevaluasi model untuk memastikan model mampu membuat prediksi yang akurat, kemudian melakukan pengujian model. Dari hasil tersebut ternyata algoritma _LinearRegression_ masih kurang akurat, sehingga perlu menggunakan algoritma lain dalam project ini yaitu RandomForest.
- Membuat model dengan menggunakan algoritma _RandomForest_, kelebihan dari menggunakan algoritma ini yaitu dapat mengatasi _noise_ dan _missing value_ serta dapat mengatasi data dalam jumlah yang besar, adapun kekurangan pada algoritma _Random Forest_ yaitu interpretasi yang sulit dan membutuhkan tuning model yang tepat untuk data. Proses dan tahapan yang dilakukan antara lain membuat model dengan algoritma _RandomForest_, membandingkan Hasil Evaluasi Dari Kedua Model Menggunakan Metrik MSE, melakukan Pengujian Dari Kedua Model. Dari hasil pengujian algoritma RandomForest lebih baik dalam melakukan prediksi.
  
# _Evaluation_
Proses evaluasi model pada proyek ini menggunakan metrik _Mean Squared Error_ yang menghitung jumlah selisih kuadrat rata-rata nilai sebenarnya dengan nilai prediksi

## Model dengan Algoritma _LinearRegression_
![linear regression](https://github.com/lusiana02/ML-Predictive-Analytics-Year-of-Experience-And-Employees-Salary/assets/123287899/27e6cb4e-16e5-4ab8-a1a9-f3c178a757ab)

Gambar 4 Model _Linear Regression_

Seperti terlihat pada gambar, model yang dibuat menggunakan algortima _LinearRegression_ memiliki nilai MSE yang sangat tinggi hingga mencapai 113458.095232 pada saat _training_ dan 127004.94334 pada saat _test_, hal ini menunjukkan algoritma ini kurang baik untuk melakukan prediksi.

![uji linear regression](https://github.com/lusiana02/ML-Predictive-Analytics-Year-of-Experience-And-Employees-Salary/assets/123287899/95c232e2-eaf7-42d2-a1f4-46bbaf8ac956)

Gambar 5 Uji _Linear Regression_

Pada proses pengujian pun dapat terlihat hasil prediksi tidak akurat dengan nilai sebenarnya.

## Model dengan Algortima _RandomForest_
![random forest](https://github.com/lusiana02/ML-Predictive-Analytics-Year-of-Experience-And-Employees-Salary/assets/123287899/601279cb-6ab2-4057-a2d7-310617188668)

Gambar 6 Model _Random Forest_

Seperti terlihat pada gambar, saat dibandingkan dengan algoritma _LinearRegression_ terlihat algortima _RandomForest_ memiliki nilai MSE yang lebih rendah yaitu 13235.129443 pada saat _training_ dan 15922.675464 pada saat _test_, hal ini menunjukkan algoritma _RandomForest_ lebih baik untuk melakukan prediksi dibanding _LinearRegression._

![uji random forest](https://github.com/lusiana02/ML-Predictive-Analytics-Year-of-Experience-And-Employees-Salary/assets/123287899/0f958902-26be-4cd1-b68a-2233ae307311)

Gambar 7 Uji _Random Forest_

Pada proses pengujian dapat terlihat hasil prediksi dari model yang menggunakan _RandomForest_ lebih akurat dengan nilai sebenarnya. Oleh karena itu algoritma ini yang akan dipilih sebagai model utama untuk memprediksi kisaran gaji karyawan.

# Referensi
[1] Douw, N. I., Maarif, M. S., & Baga, L. M. "Peningkatan Produktivitas Kerja Karyawan Development Di Tambang Bawah Tanah Dmlz (Deep Mill Level Zone) Pt Freeport Indonesia," _Jurnal Aplikasi Bisnis dan Manajemen (JABM)_, 7(2), 316-316, 2021.
