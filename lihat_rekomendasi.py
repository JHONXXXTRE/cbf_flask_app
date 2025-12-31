import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import os
from recommender_engine import dapatkan_rekomendasi # Impor fungsi terpusat

# --- Konfigurasi & Pemuatan Data ---
base_path = os.path.dirname(os.path.abspath(__file__))
# Langsung menunjuk ke file di direktori yang sama dengan skrip
latihan_path = os.path.join(base_path, 'data/data_latihan_processed.csv')
kuesioner_path = os.path.join(base_path, 'data/kuesioner_processed.csv')

try:
    df_latihan = pd.read_csv(latihan_path)
    df_kuesioner = pd.read_csv(kuesioner_path)
    print("File data berhasil dimuat.")
except FileNotFoundError as e:
    print(f"Error: File tidak ditemukan. Pastikan path sudah benar. Detail: {e}")
    exit()

# --- Pra-pemrosesan ---
df_latihan['fitur_gabungan_program'] = df_latihan['fitur_gabungan_program'].fillna('')
df_kuesioner['fitur_gabungan_pengguna'] = df_kuesioner['fitur_gabungan_pengguna'].fillna('')

# --- Pembangunan Model ---
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix_latihan = tfidf.fit_transform(df_latihan['fitur_gabungan_program'])
print(f"Dimensi Matriks TF-IDF Latihan: {tfidf_matrix_latihan.shape}")

# --- Fungsi Helper (disalin dari app.py untuk pengujian mandiri) ---
def tambah_kata_kunci_untuk_fitur(jenis_latihan_input_str):
    """
    Memproses string 'Jenis Latihan' untuk mengekstrak dan menstandarisasi kata kunci.
    """
    if not isinstance(jenis_latihan_input_str, str):
        return ''

    kata_kunci_standar = {
        'kekuatan': ['latihan fisik', 'angkat beban', 'push up', 'pull up', 'squat', 'kekuatan', 'bodyweight'],
        'kardio': ['kardio', 'lari', 'bersepeda', 'skipping', 'futsal', 'boxing', 'senam', 'jogging', 'running'],
        'hiit': ['hiit', 'fungsional']
    }
    input_lower = jenis_latihan_input_str.lower()
    jenis_latihan_parts = [part.strip() for part in jenis_latihan_input_str.split(',')]
    combined_keywords = {input_lower}
    for part in jenis_latihan_parts:
        combined_keywords.add(part)
        if '(' in part:
            main_type = part.split('(')[0].strip()
            if main_type: combined_keywords.add(main_type)
        for standar, sinonim_list in kata_kunci_standar.items():
            if any(sinonim in part for sinonim in sinonim_list):
                combined_keywords.add(standar)
    return ' '.join(sorted(list(filter(None, combined_keywords))))

def create_feature_string_for_new_user(user_data_dict):
    usia = str(user_data_dict.get('usia', ''))
    tujuan = str(user_data_dict.get('tujuan', ''))
    jenis_latihan_raw = str(user_data_dict.get('jenis_latihan', ''))
    hari_sibuk_user_str = str(user_data_dict.get('hari_sibuk', ''))
    all_days_form = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
    busy_days_list_form = [d.strip().capitalize() for d in hari_sibuk_user_str.split(',') if d.strip()]
    free_days_user_list = [d for d in all_days_form if d not in busy_days_list_form]
    free_days_user_feature_str = ' '.join(free_days_user_list)
    jenis_latihan_keywords = tambah_kata_kunci_untuk_fitur(jenis_latihan_raw)
    waktu_luang = str(user_data_dict.get('waktu_luang', ''))
    tempat = str(user_data_dict.get('tempat', ''))
    jenis_kelamin = str(user_data_dict.get('jenis_kelamin', '')).lower()
    pengalaman_user = str(user_data_dict.get('pengalaman', '')).lower()
    feature_string = (
        f"{usia} {tujuan} {jenis_latihan_keywords} {free_days_user_feature_str} "
        f"{waktu_luang} {tempat} {jenis_kelamin} {pengalaman_user}"
    )
    return ' '.join(feature_string.split())

# --- Jalankan dan Tampilkan Rekomendasi ---
if __name__ == "__main__":
    # --- PILIH PENGGUNA UNTUK DIUJI ---
    # Profil pengguna kustom sesuai permintaan Anda
    user_profile_dict = {
        'usia': '27',
        'tujuan': 'Menurunkan berat badan, Meningkatkan massa otot, Menjaga kesehatan',
        'jenis_latihan': '', # Dikosongkan karena tidak dispesifikkan
        'hari_sibuk': 'Senin,Selasa,Rabu,Kamis,Jumat,Sabtu', # Berdasarkan waktu luang di hari Minggu
        'waktu_luang': 'Sore (16:00-18:00)',
        'tempat': 'Rumah',
        'jenis_kelamin': 'Wanita', # dari 'perempuan'
        'pengalaman': 'Pemula' # dari 'pemula (jarang berolahraga)'
    }

    # Buat string fitur dari profil kustom
    user_profile_string = create_feature_string_for_new_user(user_profile_dict)

    print("\n=============================================")
    print("Profil Pengguna yang Digunakan untuk Rekomendasi:")
    for key, value in user_profile_dict.items():
        print(f"- {key.capitalize().replace('_', ' ')}: {value}")
    print("\nFeature String yang Dihasilkan:")
    print(f'"{user_profile_string}"')    
    print("=============================================\n")

    # Dapatkan 10 rekomendasi teratas
    rekomendasi_hasil = dapatkan_rekomendasi(
        profil_pengguna_string=user_profile_string,
        profil_pengguna_dict=user_profile_dict,
        df_latihan=df_latihan,
        tfidf_vectorizer=tfidf,
        tfidf_matrix_latihan=tfidf_matrix_latihan,
        final_top_n=10
    )

    # Tampilkan hasil dengan format yang rapi
    print("--- 10 Rekomendasi Latihan Teratas Untuk Anda ---")
    if not rekomendasi_hasil.empty:
        # Buat salinan untuk ditampilkan agar tidak mengubah data asli
        rekomendasi_tampil = rekomendasi_hasil.copy()
        # Bulatkan skor untuk keterbacaan yang lebih baik
        rekomendasi_tampil['original_similarity'] = rekomendasi_tampil['original_similarity'].round(4)
        rekomendasi_tampil['adjusted_similarity'] = rekomendasi_tampil['adjusted_similarity'].round(4)
        
        # Tampilkan kolom yang relevan
        print(rekomendasi_tampil[[
            'Nama Program Latihan', 
            'Tingkat Kebugaran Program', 
            'Tempat Program', 
            'original_similarity', 
            'adjusted_similarity'
        ]])
    else:
        print("Tidak ada rekomendasi yang cocok ditemukan untuk profil ini.")

    # --- Blok Evaluasi untuk Pengguna Tunggal ---
    print("\n--- Metrik Evaluasi untuk Pengguna Ini ---")

    # 1. Definisikan Ground Truth untuk pengguna ini
    user_kebugaran_pref = user_profile_dict.get('pengalaman', '').lower()
    user_tempat_pref = user_profile_dict.get('tempat', '').lower()
    user_gender_pref = user_profile_dict.get('jenis_kelamin', '').lower()

    # Normalisasi preferensi pengguna
    user_kebugaran_norm = ''
    if 'pemula' in user_kebugaran_pref: user_kebugaran_norm = 'pemula'
    elif 'menengah' in user_kebugaran_pref: user_kebugaran_norm = 'menengah'
    elif 'lanjut' in user_kebugaran_pref: user_kebugaran_norm = 'lanjut'

    user_tempat_norm = ''
    if 'rumah' in user_tempat_pref: user_tempat_norm = 'rumah'
    elif 'gym' in user_tempat_pref: user_tempat_norm = 'gym'
    elif 'outdoor' in user_tempat_pref: user_tempat_norm = 'outdoor'

    user_gender_norm = ''
    if 'pria' in user_gender_pref or 'laki-laki' in user_gender_pref: user_gender_norm = 'laki-laki'
    elif 'wanita' in user_gender_pref: user_gender_norm = 'wanita'

    # Dapatkan semua ID program yang relevan (ground truth)
    relevant_items = set()
    if user_kebugaran_norm and user_tempat_norm:
        ground_truth_filter = (
            (df_latihan['Tingkat Kebugaran Program'].str.lower() == user_kebugaran_norm) &
            (df_latihan['Tempat Program'].str.lower().str.contains(user_tempat_norm, na=False))
        )
        # Tambahkan filter gender jika ada
        if user_gender_norm:
            gender_filter = (df_latihan['Target Gender'].str.lower() == 'semua') | (df_latihan['Target Gender'].str.lower() == user_gender_norm)
            ground_truth_filter &= gender_filter

        relevant_items = set(df_latihan[ground_truth_filter]['ID Program'].astype(str))

    # 2. Dapatkan Rekomendasi dari Model
    recommended_items = set(rekomendasi_hasil['ID Program'].astype(str))

    # 3. Hitung Metrik
    if not relevant_items:
        print("Tidak dapat menghitung metrik: Tidak ada item 'ground truth' yang relevan ditemukan untuk profil pengguna ini.")
    else:
        true_positives = len(recommended_items.intersection(relevant_items))
        false_positives = len(recommended_items) - true_positives
        false_negatives = len(relevant_items) - true_positives
        total_items = len(df_latihan)
        true_negatives = total_items - true_positives - false_positives - false_negatives

        precision = true_positives / len(recommended_items) if recommended_items else 0
        recall = true_positives / len(relevant_items) if relevant_items else 0
        accuracy = (true_positives + true_negatives) / total_items if total_items > 0 else 0

        print(f"Total Item Relevan (Ground Truth): {len(relevant_items)}")
        print(f"Total Item Direkomendasikan:       {len(recommended_items)}")
        print(f"True Positives (Cocok):            {true_positives}")
        print("-" * 38)
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"Accuracy:  {accuracy:.4f}")
        print("--------------------------------------")