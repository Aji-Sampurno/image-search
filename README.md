# Batik Vision - High Accuracy Image Search

Sistem pencarian kemiripan motif batik tingkat lanjut menggunakan **DINOv2 (Vision Transformer)**, **FAISS**, dan **Hybrid Color Scoring**. Sistem ini dirancang untuk menangani variasi foto nyata (rotasi, pencahayaan, sudut pengambilan) dengan akurasi tinggi.

## Fitur Utama

- **DINOv2 Backbone**: Menggunakan model `dinov2_vits14` dari Meta AI untuk ekstraksi fitur motif yang sangat robust.
- **Multi-View Embedding**: Secara otomatis men-generate 6 view (original, rotasi 90/180/270, mirror, crop) untuk setiap gambar dan mengambil rata-ratanya. Ini membuat pencarian tahan terhadap perubahan sudut dan orientasi.
- **Hybrid Scoring (50/50)**: Menggabungkan kesamaan motif (Cosine Similarity) dan kesamaan warna (HSV Histogram) dengan bobot seimbang (50% Motif, 50% Warna) untuk memastikan varian warna yang benar muncul di atas.
- **Soft Color Scoring**: 
  - **Auto Center Crop**: Menganalisis warna hanya pada bagian tengah (50%) gambar query untuk mengabaikan background (seperti lantai/meja).
  - **Weighted Ranking**: Tidak ada pem filtering ketat (Strict Penalty) yang membuang hasil. Gambar dengan kemiripan warna rendah tetap muncul namun dengan skor akhir yang lebih kecil.
- **Incremental Indexing**: Mendukung penambahan gambar baru tanpa perlu memproses ulang seluruh database.
- **High Performance**: Menggunakan FAISS untuk pencarian vektor super cepat.

---

## Cara Menjalankan (Local Development)

### 1. Prasyarat
Pastikan Python 3.10+ terinstal.

```bash
pip install -r requirements.txt
```

### 2. Persiapan Data
Simpan semua gambar batik Anda di folder:
`static/images/`

### 3. Build & Update Index
Anda perlu membangun index sebelum menjalankan server. Script ini pintar: jika sudah ada index sebelumnya, dia hanya akan memproses **gambar baru** saja (Incremental).

```bash
python3 build_index.py
```
*Output akan disimpan di folder `data/` (`batik.faiss`, `batik_vectors.npy`, `batik_paths.pkl`).*

### 4. Jalankan Server
Jalankan aplikasi web menggunakan FastAPI.

```bash
python3 app.py
```
Server akan berjalan di `http://localhost:8080`.

---

## Penggunaan API

### Search Image
Endpoint untuk melakukan pencarian gambar.

**Request:**
- Method: `POST`
- URL: `http://localhost:8080/api/search`
- Body: `multipart/form-data` dengan key `file`.

**Contoh cURL:**
```bash
curl -X POST "http://localhost:8080/api/search" \
     -F "file=@/path/to/your/image.jpg"
```

**Response:**
Mengembalikan JSON berisi daftar kandidat yang relevan, diurutkan berdasarkan skor tertinggi.

```json
{
    "results": [
        {
            "filename": "BATIK_KHAKI.jpg",
            "score": 92.5,
            "motif_score": 88.2,
            "color_score": 96.8,
            "url": "/images/BATIK_KHAKI.jpg",
            "matches": 0
        },
        ...
    ]
}
```

---

## Struktur File Penting

- `app.py`: Main server entrypoint. Mengatur logika pencarian, scoring, dan API.
- `embedder.py`: Class `CNNEmbedder` yang menangani logika DINOv2 dan Multi-View Encoding.
- `build_index.py`: Script indexing. Mendukung **Incremental Indexing** (hanya proses file baru).
- `requirements.txt`: Daftar dependensi project.
- `data/`: Folder penyimpanan artifact index.
- `static/images/`: Folder dataset gambar.

---

## Troubleshooting

**Q: Server error saat startup?**
A: Pastikan Anda sudah menjalankan `python3 build_index.py` setidaknya sekali untuk membuat file index di folder `data/`.
