
# Laporan Proyek Machine Learning - Abimanyu Sri Setyo

  

## Domain Proyek

Perpustakaan merupakan salah satu pusat penyedia layanan informasi yang bisa kita kunjungi. Namun. selama ini pengunjung perpustakaan kesulitan untuk mencari buku yang berkaitan dengan buku yang dibaca sebelumnya dan juga dalam menemukan alternatif buku lain ketika buku yang diinginkan tersebut telah dipinjam. Dengan adanya rekomendasi atau saran buku-buku lain yang berhubungan diharapkan membantu dalam mendapatkan buku yang sesuai dan diinginkan pengunjung perpustakaan [1]. Pada penelitian ini penerapan sistem rekomendasi menggunakan metode Content-Based Filtering dalam memberikan rekomendasi buku yang bekerja dengan melihat kemiripan item yang dianalisis dari fitur yang dikandungnya dengan *Weighted Tree Similarity*.
  

Pada proyek ini, dataset berasal dari situs web [Kaggle](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset), dan dataset ini menyediakan informasi buku yang mencakup dari 242135 judul buku.

  

## Business Understanding

### Problem Statement:

- Bagaimana cara membuat sistem yang dapat merekomendasikan buku dari buku yang pernah kita baca sebelumnya?
- Bagaimana cara mendapatkan rekomendasi buku dari penulis yang sama?

  

### Goals:

- Mengetahui cara membuat sistem yang dapat merekomendasikan buku dari buku yang pernah kita baca sebelumnya.
- Mengetahui cara mendapatkan buku yang ditulis oleh penulis yang sama dengan buku yang pernah kita baca sebelumnya.

  

### Solution Statements:

- Menggunakan metode *Content Based Filtering*, dimana metode ini merupakan metode sistem rekomendasi yang merekomendasikan item sesuai dengan item yang disukai oleh pengguna di masa lampau.
- Menggunakan metode *Collaborative Based Filtering*, dimana metode ini merupakan metode sistem rekomendasi yang merekomendasikan item berdasarkan pendapat suatu komunitas.

## Data Understanding

### **Analisis Data Eksplorasi:**

Dataset yang digunakan adalah ***Book Recommendation*** yang bersumber dari [Kaggle](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset). Dikutip dari [algorit.ma](https://algorit.ma/blog/exploratory-data-analysis-2022/), Analisis Data Eksplorasi mencakup proses kritis uji investigasi pendahuluan pada data untuk mengidentifikasi pola, menemukan anomali, menguji hipotesis, dan memeriksa asumsi melalui statistik ringkasan dan representasi visual.

- Dataset ini terdiri dari 271360 baris data dan memiliki 8 kolom data untuk ``books``, dan terdiri dari 1149780 baris data dan memiliki 3 kolom data untuk ``ratings``. 

- Dataset memiliki tidak memiliki nilai kosong atau NaN yang perlu dilakukan penanganan.
    - books
        | Kolom           	    | Jumlah NaN 	|
        |-----------------	    |------------	|
        | ISBN            	    | 0          	|
        | Book-Title            | 0          	|
        | Book-Author       	| 0          	|
        | Year-Of-Publication  	| 0         	|
        | Publisher         	| 0         	|
        | Image-URL-S 	        | 0          	|
        | Image-URL-M       	| 0          	|
        | Image-URL-L           | 0          	|
    - ratings
        | Kolom           	    | Jumlah NaN 	|
        |-----------------	    |------------	|
        | User-ID               | 0          	|
        | ISBN            	    | 0          	|
        | Book-Rating       	| 0          	|

  

### **Fitur pada Dataset:**

Dataset ini, ada 4 variabel dengan 4 fitur untuk books dan ada 3 variabel dengan 3 fitur untuk ratings.
- books
	| Fitur           	    | Penjelasan                                            	|
	|-----------------	    |---------------------------------------------------------	|
    | ISBN            	    | *International Standard Book Number* atau kode buku      	|
    | Book-Title            | Judul buku                                              	|
    | Book-Author       	| Nama penulis buku          	                            |
    | Year-Of-Publication  	| Tahun terbit buku                                     	|
    | Publisher         	| Penerbit buku                                          	|
- rating
	| Fitur           	    | Penjelasan                                            	|
	|-----------------	    |---------------------------------------------------------	|
    | User-ID               | Nomer pelanggan                                         	|
    | ISBN            	    | *International Standard Book Number* atau kode buku     	|
    | Book-Rating       	| *Rating* atau nilai buku                                	|


### **Korelasi pada Dataset:**

Untuk menjelaskan korelasi fitur – fitur pada dataset, diperlukan visualisasi data yang terdiri sebagai berikut. Namun, ada baiknya untuk melihat NaN *values* yang dimiliki oleh dataset dan dilakukan penanganan. Menurut [DQLab.id](https://www.dqlab.id/kursus-belajar-data-mengenal-apa-itu-missing-value), nilai NaN akan membuat data tidak dapat digunakan. Pada proyek ini, penanganan yang digunakan adalah *drop* atau pembersihan nilai NaN dari dataset dan juga dilakukan pembersihan dari duplikasi data.
  

## Data Preparation

  

Dalam *data preparation*, dilakukan beberapa hal berikut sebelum memasukkan data ke model latih:

  

- ***Drop data* untuk mengurangi jumlah data**<br>

	*Drop data* untuk mengurangi jumlah data. Hal ini akan membantu efektivitas dalam fitting model yang dilakukan. Pada dataset ini, dataframe rating dan buku cukup banyak, sehingga hanya perlu mengambil 10.000 baris dari dataset buku dan 5000 baris untuk dataset rating.

- ***Drop missing value***<br>

    *Drop missing value* bertujuan agar data bersih dari adanya data kosong yang dapat memengaruhi hasil akurasi dari model

- ***Drop duplicates values***<br>

    *Drop duplicates values* bertujuan agar data bersih dari adanya duplikasi data yang bernilai sama yang dapat memengaruhi hasil akurasi dari model

  

## Modeling

  

Model – model yang saya pakai dalam projek ini adalah:

-  ***Content Based Filtering***<br>

    *Content Based Filtering* adalah sistem rekomendasi yang merekomendasikan item sesuai dengan item yang disukai oleh pengguna di masa lampau. Pada proyek ini, digunakanlah ``TF-IDF Vectorizer`` untuk membangun sistem rekomendasi berdasarkan penulis buku, dimana ``TF-IDF`` berfungsi untuk mengukur seberapa pentingnya suatu kata terhadap kata-kata lain dalam dokumen dan metode ini sering digunakan dalam *Information Retrieval* dan *Text Mining*, dan untuk mengukur model digunakan adalah metrik ``accuracy``.

-  ***Collaborative Based Filtering***<br>

	*Collaborative Based Filtering* adalah sistem rekomendasi berdasarkan pendapat suatu komunitas. Pada proyek ini, digunakanlah ``RecommenderNet`` dimana proses *compile* pada model menggunakan *binary crossentropy* sebagai *loss function*, *adam* sebagai *optimizer*, dan ``RMSE`` sebagai metrik dari model. Kemudian, pada proses *training model*, nilai ``batch_size`` adalah 5 dengan ``epochs`` sebanyak 20.



## Evaluasi

Sebelum ke metrik evaluasi, ada istilah *confusion matrix* dimana di dalam *confusion matrix*, terdapat 4 kesimpulan yang dapat di ambil sebagai berikut.

| Confusion Matrix    	| Penjelasan                                                                 	|
|---------------------	|----------------------------------------------------------------------------	|
| *True Positive* (TP) 	| Jumlah prediksi positif yang benar terhadap jumlah positif yang sebenarnya 	|
| *False Positive* (FP) | Jumlah prediksi positif yang salah                                         	|
| *True Negative* (TN)  | Jumlah prediksi negatif yang benar terhadap jumlah negatif yang sebenarnya 	|
| *False Negative* (FN) | Jumlah prediksi negatif yang salah                                         	|
  

Dalam projek ini, metrik evaluasi yang digunakan adalah sebagai berikut.

-  ***Accuracy***<br>
	Akurasi adalah metrik yang paling umum dalam pemodelan klasifikasi. Ini adalah persentase jumlah data yang diprediksi dengan benar terhadap jumlah total data.

    ![1](https://raw.githubusercontent.com/bzizmza/Final-Project-Recommendation-System/main/img/1.png)

    Dengan hasil akurasi nya adalah:
    |               | *Content Based Filtering*     |
    |------------  	|---------------------------	|
    | *Accuracy*  	| 100%  	                    |

    Nilai akurasi didapatkan dengan membagi total buku hasil rekomendasi dengan nama dari penulis nya. Dimana buku yang di rekomendasi kan oleh sistem adalah sebagai berikut.

    ![3](https://raw.githubusercontent.com/bzizmza/Final-Project-Recommendation-System/main/img/3.png)

-  **RMSE**<br>
	*Root Mean Squared Error* (RMSE) adalah akar dari rata-rata kesalahan kuadrat diantara nilai aktual dan nilai prediksi. Metode *Root Mean Squared Error* secara umum digunakan untuk mengecek estimasi berapa nilai kesalahan pada prediksi.

    ![2](https://raw.githubusercontent.com/bzizmza/Final-Project-Recommendation-System/main/img/2.png)

    Pada plot dapat dilihat bahwa data sudah goodfit dan bisa menjelaskan data tanpa terpengaruh oleh data noise. Sebab, underfitting terjadi ketika model tidak bisa melihat logika dibelakang data, hingga tidak bisa melakukan prediksi dengan tepat, baik untuk dataset training maupun dataset lain yang serupa. Underfitting model akan memiliki high loss dan akurasi rendah. Sedangkan, Overfitting terjadi karena model yang dibuat terlalu fokus pada training dataset tertentu, hingga tidak bisa melakukan prediksi dengan tepat jika diberikan dataset lain yang serupa. Overfitting biasanya akan menangkap data noise yang seharusnya diabaikan. Overfitting model akan memiliki low loss dan akurasi rendah.
    Dari grafik di atas, dapat diketahui nilai RMSE sebagai berikut.
    | *Root Mean Squared Error* (RMSE)  | *Train*   | *Test*    |
    |--------------------------------	|--------	|--------	|
    | *Collaborative Based Filtering*  	| 0.2289  	| 0.3477    |

    Dimana buku yang di rekomendasi kan oleh sistem adalah sebagai berikut.

    ![4](https://raw.githubusercontent.com/bzizmza/Final-Project-Recommendation-System/main/img/4.png)
  
## Kesimpulan
Kesimpulan dari hasil proyek *recommendation system* ***Book Recomendation*** ini adalah sebagai berikut.
- Sistem rekomendasi dapat berjalan dengan baik, dimana sistem ini dapat memberikan rekomendasi buku yang sesuai dengan preferensi pembaca buku tersebut.
- Model terbaik, yaitu *Content Based Filtering* dapat memprediksi dengan akurasi hingga 100%.
- Melakukan evaluasi menggunakan nilai akurasi mudah untuk dipahami karena kesederhanaannya dan hanya berisi jumlah yang benar dibandingkan dengan keseluruhan jawaban. Namun, hal ini juga menjadi kekurangan dari evaluasi menggunakan nilai akurasi dimana tidak memperhitungkan aspek lainnya.
- Melakukan evaluasi menggunakan RMSE memiliki nilai yang lebih rendah dari MSE, nilai yang kecil ini membuat model memiliki nilai kesalahan yang kecil. Namun, terkadang juga perlu melihat peluang adanya *overfitting* atau *undefitting* dengan lebih teliti karena efek dari nilai yang kecil.
- Hasil prediksi mungkin akan lebih baik apabila data lebih banyak.

## Daftar Pustaka:

[1]	M. Alkaff, H. Khatimi, and A. Eriadi, “Sistem Rekomendasi Buku pada Perpustakaan Daerah Provinsi Kalimantan Selatan Menggunakan Metode Content-Based Filtering”, MATRIK : Jurnal Manajemen, Teknik Informatika dan Rekayasa Komputer, vol. 20, no. 1, pp. 193-202, Sep. 2020.