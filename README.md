# 🖋️ Word Detektor by Hakoji (Handwriting OCR) 🔍

Aplikasi pengenalan tulisan tangan menggunakan CNN dan OpenCV. 🧠💻

![Screenshot Aplikasi](github/assets/preview/preview.png)

# ![Maho](https://static.wikia.nocookie.net/princess-connect/images/a/a7/Maho_Box_Icon.png/revision/latest/scale-to-width-down/40?cb=20190925080932) Isi Project
1. 🐍 app_ocr.py sebagai main program yang akan dijalankan/dipakai
2. 🏗️ train_model.py sebagai arsitektur CNN serta menggunakan framework TensorFlow, untuk memperbagus akurasi lagi dapat meningkatkan melalui Hyperparameter Tuning, seperti menaikkan jumlah Epoch(bawaan 10) atau menyesuaikan learning rate pada Adam Optimizer
3. 📁 folder github adalah contoh bahan yang digunakan dan samples nya

# ![Shiori](https://static.wikia.nocookie.net/princess-connect/images/7/77/Shiori_Box_Icon.png/revision/latest/scale-to-width-down/40?cb=20190925113434) Cara Instalasi
1. Clone atau download repository ini.
2. Install library: `pip install -r requirements.txt`
3. Jalankan aplikasi: `python app_ocr.py` atau langsung mengklik file `app_ocr.py` didalam folder reposity

# ![Nozomi](https://static.wikia.nocookie.net/princess-connect/images/4/46/Nozomi_Box_Icon.png/revision/latest/scale-to-width-down/40?cb=20190925084658) Cara Menggunakan
1. Setelah aplikasi dijalankan pilih gambar yang ingin di deteksi (Harus 1 folder dengan project si gambar nya)
2. Model aplikasi akan menentukan gambar huruf/kata tersebut
3. Hasil akan terlihat seperti preview

# ![Tantei](https://static.wikia.nocookie.net/princess-connect/images/f/fb/Kasumi_Box_Icon.png/revision/latest/scale-to-width-down/40?cb=20190925082622) Fitur Utama
1. Upload Gambar: Mendukung format JPG, JPEG, dan PNG melalui fitur search image
2. Klasifikasi Teks: Mampu mengenali apakah input berupa Huruf Kapital/Lower atau Kalimat
3. Ekstraksi Kata: Mengenali karakter secara individu maupun merangkumnya menjadi kata/kalimat

# ![Saren](https://static.wikia.nocookie.net/princess-connect/images/b/b3/Saren_Box_Icon.png/revision/latest/scale-to-width-down/40?cb=20190925084518) Akurasi
1. Akurasi mencapai 95% Jika hanya 1 huruf
2. Akurasi mencapai 90% Jika lebih dari 3 huruf
3. Akurasi mencapai 80% Jika sebuah kata
4. Akurasi mencapai 70% Jika sebuah kalimat


# ![Kyouka](https://static.wikia.nocookie.net/princess-connect/images/3/39/Kyouka_Box_Icon.png/revision/latest/scale-to-width-down/40?cb=20190925113712) Author by Ranggi Febrian
