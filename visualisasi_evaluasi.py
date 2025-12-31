import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import os
import numpy as np
import matplotlib.pyplot as plt

# --- Konfigurasi & Pemuatan Data ---
base_path = os.path.dirname(os.path.abspath(__file__))
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
train_df, test_df = train_test_split(df_kuesioner, test_size=0.2, random_state=42)
print(f"Data kuesioner dibagi: {len(train_df)} data latih, {len(test_df)} data uji.")

# --- Pra-pemrosesan ---
df_latihan['fitur_gabungan_program'] = df_latihan['fitur_gabungan_program'].fillna('')

# --- Pembangunan Model ---
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix_latihan = tfidf.fit_transform(df_latihan['fitur_gabungan_program'])
print(f"Dimensi Matriks TF-IDF Latihan: {tfidf_matrix_latihan.shape}")

def dapatkan_rekomendasi(
    profil_pengguna_string: str, 
    profil_pengguna_dict: dict, 
    candidate_pool_size: int = 200,
    final_top_n: int = 10) -> pd.DataFrame:
    """
    Fungsi rekomendasi yang disempurnakan untuk meningkatkan Recall.
    """
    tfidf_matrix_pengguna = tfidf.transform([profil_pengguna_string])
    cosine_similarities = cosine_similarity(tfidf_matrix_pengguna, tfidf_matrix_latihan).flatten()

    if len(cosine_similarities) < candidate_pool_size:
        candidate_pool_size = len(cosine_similarities)
    top_candidate_indices = cosine_similarities.argsort()[-candidate_pool_size:][::-1]
    
    df_latihan_temp = df_latihan.iloc[top_candidate_indices].copy()
    df_latihan_temp['original_similarity'] = cosine_similarities[top_candidate_indices]
    df_latihan_temp['adjusted_similarity'] = cosine_similarities[top_candidate_indices]

    tempat_preferensi = profil_pengguna_dict.get('tempat', '').lower()
    kebugaran_preferensi = profil_pengguna_dict.get('kebugaran', '').lower()

    PENALTI_TIDAK_COCOK = 0.25 
    BONUS_SANGAT_COCOK = 0.1

    user_kebugaran_keyword = ''
    if 'pemula' in kebugaran_preferensi: user_kebugaran_keyword = 'pemula'
    elif 'menengah' in kebugaran_preferensi: user_kebugaran_keyword = 'menengah'
    elif 'lanjut' in kebugaran_preferensi: user_kebugaran_keyword = 'lanjut'

    user_tempat_keyword = ''
    if 'rumah' in tempat_preferensi: user_tempat_keyword = 'rumah'
    elif 'gym' in tempat_preferensi: user_tempat_keyword = 'gym'
    elif 'outdoor' in tempat_preferensi: user_tempat_keyword = 'outdoor'

    for index, row in df_latihan_temp.iterrows():
        program_tempat = str(row.get('Tempat Program', '')).lower()
        program_kebugaran = str(row.get('Tingkat Kebugaran Program', '')).lower()

        if user_tempat_keyword:
            if user_tempat_keyword in program_tempat:
                df_latihan_temp.loc[index, 'adjusted_similarity'] += BONUS_SANGAT_COCOK
            else:
                df_latihan_temp.loc[index, 'adjusted_similarity'] -= PENALTI_TIDAK_COCOK

        if user_kebugaran_keyword:
            if user_kebugaran_keyword in program_kebugaran:
                df_latihan_temp.loc[index, 'adjusted_similarity'] += BONUS_SANGAT_COCOK
            elif any(level in program_kebugaran for level in ['pemula', 'menengah', 'lanjut']):
                 df_latihan_temp.loc[index, 'adjusted_similarity'] -= PENALTI_TIDAK_COCOK

    sorted_df = df_latihan_temp.sort_values(by='adjusted_similarity', ascending=False)
    return sorted_df.head(final_top_n)

def evaluasi_untuk_kurva(data_uji, k_values):
    """
    Menjalankan evaluasi untuk serangkaian nilai K dan mengembalikan skor untuk plot.
    """
    avg_precisions_per_k = []
    avg_recalls_per_k = []

    for k in k_values:
        print(f"Mengevaluasi untuk K={k}...")
        list_precision_for_this_k = []
        list_recall_for_this_k = []
        
        for _, user_row in data_uji.iterrows():
            user_kebugaran_pref = user_row.get('4. Bagaimana tingkat kebugaran Anda saat ini?', '').lower()
            user_tempat_pref = user_row.get('11. Apakah Anda lebih suka latihan di rumah atau di gym?', '').lower()

            user_kebugaran_norm = ''
            if 'pemula' in user_kebugaran_pref: user_kebugaran_norm = 'pemula'
            elif 'menengah' in user_kebugaran_pref: user_kebugaran_norm = 'menengah'
            elif 'lanjut' in user_kebugaran_pref: user_kebugaran_norm = 'lanjut'

            user_tempat_norm = ''
            if 'rumah' in user_tempat_pref: user_tempat_norm = 'rumah'
            elif 'gym' in user_tempat_pref: user_tempat_norm = 'gym'
            elif 'outdoor' in user_tempat_pref: user_tempat_norm = 'outdoor'

            relevant_items = set(df_latihan[
                (df_latihan['Tingkat Kebugaran Program'].str.lower() == user_kebugaran_norm) &
                (df_latihan['Tempat Program'].str.lower().str.contains(user_tempat_norm, na=False))
            ]['ID Program'].astype(str))

            if not relevant_items:
                continue

            user_profile_string = user_row['fitur_gabungan_pengguna']
            user_profile_dict = {'tempat': user_tempat_pref, 'kebugaran': user_kebugaran_pref}
            
            rekomendasi_df = dapatkan_rekomendasi(
                profil_pengguna_string=user_profile_string, 
                profil_pengguna_dict=user_profile_dict, 
                candidate_pool_size=200,
                final_top_n=k)
            recommended_items = set(rekomendasi_df['ID Program'].astype(str))

            true_positives = len(recommended_items.intersection(relevant_items))
            precision = true_positives / len(recommended_items) if recommended_items else 0
            recall = true_positives / len(relevant_items) if relevant_items else 0
            
            list_precision_for_this_k.append(precision)
            list_recall_for_this_k.append(recall)
            
        avg_prec = np.mean(list_precision_for_this_k) if list_precision_for_this_k else 0
        avg_rec = np.mean(list_recall_for_this_k) if list_recall_for_this_k else 0
        
        avg_precisions_per_k.append(avg_prec)
        avg_recalls_per_k.append(avg_rec)
        
    return avg_precisions_per_k, avg_recalls_per_k

# --- Jalankan Evaluasi dan Visualisasi ---
if __name__ == "__main__":
    # Tentukan rentang K yang akan diuji
    K_VALUES = range(1, 21) # Evaluasi dari K=1 hingga K=20
    
    print("\nMemulai evaluasi untuk membuat kurva Precision-Recall...")
    
    precisions, recalls = evaluasi_untuk_kurva(test_df, K_VALUES)
    
    # Tampilkan hasil dalam bentuk tabel
    print("\n--- Hasil Evaluasi per K ---")
    print(" K | Precision | Recall")
    print("---|-----------|---------")
    for i, k in enumerate(K_VALUES):
        print(f"{k:2d} |  {precisions[i]:.4f}   | {recalls[i]:.4f}")
    print("----------------------------")

    # Buat plot
    plt.figure(figsize=(10, 7))
    plt.plot(recalls, precisions, marker='o', linestyle='-', color='b')
    
    # Anotasi titik-titik dengan nilai K
    for i, k in enumerate(K_VALUES):
        if k % 2 == 0 or k == 1 or k == len(K_VALUES): # Beri label pada beberapa titik agar tidak terlalu ramai
            plt.annotate(f'K={k}', (recalls[i], precisions[i]), textcoords="offset points", xytext=(0,10), ha='center')

    plt.title('Kurva Precision-Recall untuk Berbagai Nilai K')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([0.0, max(recalls) * 1.1 if recalls else 1.0])
    plt.ylim([min(precisions) * 0.9 if precisions else 0.0, 1.05])
    plt.grid(True)
    plt.show()