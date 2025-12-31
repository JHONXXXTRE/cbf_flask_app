import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import os
from recommender_engine import dapatkan_rekomendasi # Impor fungsi terpusat
import numpy as np

# --- Konfigurasi & Pemuatan Data ---
base_path = os.path.dirname(os.path.abspath(__file__))
# Langsung menunjuk ke file di direktori yang sama dengan skrip
latihan_path = os.path.join(base_path, 'data/data_latihan_processed.csv')
kuesioner_path = os.path.join(base_path, 'data/kuesioner_processed.csv')

try:
    df_latihan = pd.read_csv(latihan_path)
    df_kuesioner = pd.read_csv(kuesioner_path)
    print("File berhasil dimuat.")
except FileNotFoundError as e:
    print(f"Error: File tidak ditemukan. Pastikan path sudah benar. Detail: {e}")
    exit()

# --- Pembagian Data Uji dan Latih ---
# Bagi data kuesioner menjadi 80% untuk latih dan 20% untuk uji
train_df, test_df = train_test_split(df_kuesioner, test_size=0.2, random_state=42)
print(f"Data kuesioner dibagi: {len(train_df)} data latih, {len(test_df)} data uji.")

# --- Pra-pemrosesan ---
df_latihan['fitur_gabungan_program'] = df_latihan['fitur_gabungan_program'].fillna('')
df_kuesioner['fitur_gabungan_pengguna'] = df_kuesioner['fitur_gabungan_pengguna'].fillna('')

# --- Pembangunan Model ---
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix_latihan = tfidf.fit_transform(df_latihan['fitur_gabungan_program'])
print(f"Dimensi Matriks TF-IDF Latihan: {tfidf_matrix_latihan.shape}")

def _normalize_user_preferences(user_row: pd.Series) -> dict:
    """Mengekstrak dan menormalkan preferensi utama dari baris data pengguna."""
    prefs = {
        'kebugaran': user_row.get('4. Bagaimana tingkat kebugaran Anda saat ini?', '').lower(),
        'tempat': user_row.get('11. Apakah Anda lebih suka latihan di rumah atau di gym?', '').lower(),
        'gender': user_row.get('2. Jenis Kelamin', '').lower()
    }

    # Normalisasi Kebugaran
    if 'pemula' in prefs['kebugaran']: norm_kebugaran = 'pemula'
    elif 'menengah' in prefs['kebugaran']: norm_kebugaran = 'menengah'
    elif 'lanjut' in prefs['kebugaran']: norm_kebugaran = 'lanjut'
    else: norm_kebugaran = ''

    # Normalisasi Tempat
    if 'rumah' in prefs['tempat']: norm_tempat = 'rumah'
    elif 'gym' in prefs['tempat']: norm_tempat = 'gym'
    elif 'outdoor' in prefs['tempat']: norm_tempat = 'outdoor'
    else: norm_tempat = ''

    # Normalisasi Gender
    if 'pria' in prefs['gender'] or 'laki-laki' in prefs['gender']: norm_gender = 'laki-laki'
    elif 'wanita' in prefs['gender']: norm_gender = 'wanita'
    else: norm_gender = ''

    return {
        'kebugaran_norm': norm_kebugaran,
        'tempat_norm': norm_tempat,
        'gender_norm': norm_gender,
        'kebugaran_pref': prefs['kebugaran'], # Simpan juga preferensi asli untuk `dapatkan_rekomendasi`
        'tempat_pref': prefs['tempat'],
        'gender_pref': prefs['gender']
    }

def _get_ground_truth(normalized_prefs: dict, df_latihan: pd.DataFrame) -> set:
    """Mendapatkan set ID program yang relevan (ground truth) berdasarkan preferensi."""
    user_kebugaran_norm = normalized_prefs['kebugaran_norm']
    user_tempat_norm = normalized_prefs['tempat_norm']
    user_gender_norm = normalized_prefs['gender_norm']

    # Filter dasar berdasarkan level dan tempat
    if not user_kebugaran_norm or not user_tempat_norm:
        return set()

    ground_truth_filter = (
        (df_latihan['Tingkat Kebugaran Program'].str.lower() == user_kebugaran_norm) &
        (df_latihan['Tempat Program'].str.lower().str.contains(user_tempat_norm, na=False))
    )
    # Tambahkan filter gender jika ada
    if user_gender_norm:
        gender_filter = (df_latihan['Target Gender'].str.lower() == 'semua') | (df_latihan['Target Gender'].str.lower() == user_gender_norm)
        ground_truth_filter &= gender_filter

    return set(df_latihan[ground_truth_filter]['ID Program'].astype(str))

def evaluasi_model(data_uji, df_latihan, tfidf_vectorizer, tfidf_matrix_latihan, top_k=10):
    """
    Fungsi utama untuk mengevaluasi model rekomendasi pada data uji.
    """
    list_precision = []
    list_recall = []
    list_accuracy = []

    for _, user_row in data_uji.iterrows():
        # 1. Definisikan Ground Truth untuk pengguna ini
        norm_prefs = _normalize_user_preferences(user_row)
        relevant_items = _get_ground_truth(norm_prefs, df_latihan)
        
        if not relevant_items:
            continue # Lewati pengguna jika tidak ada item relevan di database

        # 2. Dapatkan Rekomendasi dari Model
        user_profile_string = user_row['fitur_gabungan_pengguna']
        user_profile_dict = {
            'tempat': norm_prefs['tempat_pref'],
            'pengalaman': norm_prefs['kebugaran_pref'],
            'jenis_kelamin': norm_prefs['gender_pref']
        } 
        
        rekomendasi_df = dapatkan_rekomendasi(
            profil_pengguna_string=user_profile_string,
            profil_pengguna_dict=user_profile_dict,
            df_latihan=df_latihan,
            tfidf_vectorizer=tfidf_vectorizer,
            tfidf_matrix_latihan=tfidf_matrix_latihan,
            final_top_n=top_k)      
        recommended_items = set(rekomendasi_df['ID Program'].astype(str))

        # 3. Hitung Metrik
        true_positives = len(recommended_items.intersection(relevant_items))
        
        false_positives = len(recommended_items) - true_positives
        false_negatives = len(relevant_items) - true_positives
        
        total_items = len(df_latihan)
        true_negatives = total_items - true_positives - false_positives - false_negatives

        precision = true_positives / len(recommended_items) if recommended_items else 0
        recall = true_positives / len(relevant_items) if relevant_items else 0
        accuracy = (true_positives + true_negatives) / total_items if total_items > 0 else 0

        list_precision.append(precision)
        list_recall.append(recall)
        list_accuracy.append(accuracy)

    # 4. Hitung Rata-rata Metrik
    avg_precision = np.mean(list_precision) if list_precision else 0
    avg_recall = np.mean(list_recall) if list_recall else 0
    avg_accuracy = np.mean(list_accuracy) if list_accuracy else 0

    return avg_precision, avg_recall, avg_accuracy, len(list_precision)

# --- Jalankan Evaluasi ---
if __name__ == "__main__":
    K = 10 # Jumlah rekomendasi yang akan dievaluasi (Precision@K, Recall@K)
    
    print(f"\nMemulai evaluasi model untuk Top-{K} rekomendasi...")
    
    precision, recall, accuracy, num_users_evaluated = evaluasi_model(test_df, df_latihan, tfidf, tfidf_matrix_latihan, top_k=K)
    
    print("\n--- Hasil Evaluasi Model ---")
    print(f"Jumlah Pengguna yang Dievaluasi: {num_users_evaluated} (dari {len(test_df)} data uji)")
    print(f"Rata-rata Precision@{K}: {precision:.4f}")
    print(f"Rata-rata Recall@{K}:    {recall:.4f}")
    print(f"Rata-rata Accuracy:      {accuracy:.4f}")
    print("----------------------------")
    print("\nPenjelasan:")
    print(f"Precision@{K}: Dari {K} item yang direkomendasikan, rata-rata {precision*100:.2f}% di antaranya benar-benar relevan (cocok level & tempat).")
    print(f"Recall@{K}:    Dari semua item yang relevan di database, sistem berhasil menemukan dan merekomendasikan rata-rata {recall*100:.2f}% di antaranya dalam {K} rekomendasi teratas.")
    print("Accuracy:      Persentase total prediksi yang benar (termasuk item yang benar-benar tidak direkomendasikan). Skor ini bisa tinggi meskipun Recall rendah.")
