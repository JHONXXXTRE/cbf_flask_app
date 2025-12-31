import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import re

# --- KONFIGURASI MODEL REKOMENDASI ---
# Pindahkan parameter ke sini agar mudah disesuaikan
CANDIDATE_POOL_SIZE = 450  # Jumlah kandidat awal yang akan dipertimbangkan
# Penalti & Bonus
PENALTI_TIDAK_COCOK = 0.25 # Penalti jika tempat tidak cocok
BONUS_SANGAT_COCOK = 0.1   # Bonus jika kriteria (tempat, level, dll) sangat cocok
# Penalti & Bonus Spesifik untuk meningkatkan akurasi
BONUS_GENDER_COCOK = 0.15      # Bonus jika gender cocok
PENALTI_GENDER_TIDAK_COCOK = 0.30 # Penalti jika gender tidak cocok
BONUS_TUJUAN_COCOK = 0.20         # Bonus signifikan jika tujuan utama cocok
BONUS_JENIS_LATIHAN_COCOK = 0.15  # Bonus jika jenis latihan yang disukai cocok
BONUS_USIA_COCOK = 0.10           # Bonus jika usia pengguna masuk dalam rentang
PENALTI_USIA_TIDAK_COCOK = 0.20   # Penalti jika usia pengguna di luar rentang

def _parse_age_range(age_range_str: str):
    """
    Mem-parsing string rentang usia seperti '18-25 Tahun' menjadi (min_age, max_age).
    Mengembalikan (None, None) jika parsing gagal.
    """
    if not age_range_str or not isinstance(age_range_str, str):
        return None, None
    # Temukan semua angka dalam string
    numbers = re.findall(r'\d+', age_range_str)
    if len(numbers) >= 2:
        try:
            return int(numbers[0]), int(numbers[1])
        except (ValueError, IndexError):
            return None, None
    elif len(numbers) == 1:
        try:
            return int(numbers[0]), int(numbers[0])
        except ValueError:
            return None, None
    return None, None
def dapatkan_rekomendasi(
    profil_pengguna_string: str,
    profil_pengguna_dict: dict,
    df_latihan: pd.DataFrame,
    tfidf_vectorizer,
    tfidf_matrix_latihan,
    final_top_n: int = 10) -> pd.DataFrame:
    """
    Fungsi rekomendasi terpusat yang disempurnakan.
    Menerima data dan model sebagai argumen untuk fleksibilitas.
    Menggunakan hard filter untuk tingkat kebugaran.
    """
    tfidf_matrix_pengguna = tfidf_vectorizer.transform([profil_pengguna_string])
    cosine_similarities = cosine_similarity(tfidf_matrix_pengguna, tfidf_matrix_latihan).flatten()

    # --- Pass 1: Candidate Generation (Menjaring Kandidat) ---
    pool_size = CANDIDATE_POOL_SIZE
    if len(cosine_similarities) < pool_size:
        pool_size = len(cosine_similarities)
    top_candidate_indices = cosine_similarities.argsort()[-pool_size:][::-1]

    df_latihan_temp = df_latihan.iloc[top_candidate_indices].copy()
    df_latihan_temp['original_similarity'] = cosine_similarities[top_candidate_indices]
    df_latihan_temp['adjusted_similarity'] = cosine_similarities[top_candidate_indices]

    # --- Pass 2: Re-ranking (Bonus & Penalti) ---
    tempat_preferensi = profil_pengguna_dict.get('tempat', '').lower()
    kebugaran_preferensi = profil_pengguna_dict.get('pengalaman', '').lower() # FIX: Menggunakan kunci 'pengalaman' yang benar dari form
    gender_preferensi = profil_pengguna_dict.get('jenis_kelamin', '').lower()
    tujuan_preferensi = profil_pengguna_dict.get('tujuan', '').lower()
    jenis_latihan_preferensi = profil_pengguna_dict.get('jenis_latihan', '').lower()
    # Ambil usia pengguna dan konversi ke integer
    try:
        usia_pengguna = int(profil_pengguna_dict.get('usia'))
    except (ValueError, TypeError):
        usia_pengguna = None

    # Normalisasi preferensi pengguna sekali saja untuk efisiensi
    user_kebugaran_keyword = ''
    if 'pemula' in kebugaran_preferensi: user_kebugaran_keyword = 'pemula'
    elif 'menengah' in kebugaran_preferensi: user_kebugaran_keyword = 'menengah'
    elif 'lanjut' in kebugaran_preferensi or 'mahir' in kebugaran_preferensi: user_kebugaran_keyword = 'lanjut' # FIX: Menambahkan 'mahir'

    user_tempat_keyword = ''
    if 'rumah' in tempat_preferensi: user_tempat_keyword = 'rumah'
    elif 'gym' in tempat_preferensi: user_tempat_keyword = 'gym'
    elif 'outdoor' in tempat_preferensi: user_tempat_keyword = 'outdoor'

    user_gender_keyword = ''
    if 'pria' in gender_preferensi or 'laki-laki' in gender_preferensi:
        user_gender_keyword = 'laki-laki'
    elif 'wanita' in gender_preferensi:
        user_gender_keyword = 'wanita'

    # Normalisasi tujuan dan jenis latihan dari input pengguna
    user_tujuan_keyword = ''
    if 'menurunkan berat badan' in tujuan_preferensi: user_tujuan_keyword = 'menurunkan berat badan'
    elif 'meningkatkan massa otot' in tujuan_preferensi: user_tujuan_keyword = 'meningkatkan massa otot'
    elif 'menjaga kesehatan' in tujuan_preferensi: user_tujuan_keyword = 'menjaga kesehatan'

    user_jenis_latihan_keyword = ''
    # Mencocokkan dengan kata kunci yang lebih umum
    if 'kardio' in jenis_latihan_preferensi: user_jenis_latihan_keyword = 'kardio'
    elif 'latihan fisik' in jenis_latihan_preferensi or 'angkat beban' in jenis_latihan_preferensi: user_jenis_latihan_keyword = 'kekuatan'
    elif 'hiit' in jenis_latihan_preferensi: user_jenis_latihan_keyword = 'hiit'

    for index, row in df_latihan_temp.iterrows():
        program_tempat = str(row.get('Tempat Program', '')).lower()
        program_kebugaran = str(row.get('Tingkat Kebugaran Program', '')).lower()
        program_gender = str(row.get('Target Gender', '')).lower()
        program_tujuan = str(row.get('Tujuan Latihan', '')).lower()
        program_jenis_latihan = str(row.get('Jenis Latihan Program', '')).lower()
        program_rentang_usia = str(row.get('Rentang Usia', ''))

        # Bonus jika tingkat kebugaran cocok persis (penalti dihapus, akan di-handle oleh hard filter)
        if user_kebugaran_keyword and user_kebugaran_keyword in program_kebugaran:
            df_latihan_temp.loc[index, 'adjusted_similarity'] += BONUS_SANGAT_COCOK

        # Penalti & Bonus Gender
        if user_gender_keyword and program_gender and 'semua' not in program_gender:
            if user_gender_keyword in program_gender:
                df_latihan_temp.loc[index, 'adjusted_similarity'] += BONUS_GENDER_COCOK
            else:
                df_latihan_temp.loc[index, 'adjusted_similarity'] -= PENALTI_GENDER_TIDAK_COCOK

        # Bonus Tujuan
        if user_tujuan_keyword and user_tujuan_keyword in program_tujuan:
            df_latihan_temp.loc[index, 'adjusted_similarity'] += BONUS_TUJUAN_COCOK

        # Bonus Jenis Latihan
        if user_jenis_latihan_keyword and user_jenis_latihan_keyword in program_jenis_latihan:
            df_latihan_temp.loc[index, 'adjusted_similarity'] += BONUS_JENIS_LATIHAN_COCOK

        # Penalti & Bonus Usia (BARU)
        if usia_pengguna is not None and program_rentang_usia:
            min_usia, max_usia = _parse_age_range(program_rentang_usia)
            if min_usia is not None and max_usia is not None:
                # FIX: Jika rentang usia program adalah 18-35, anggap cocok untuk semua (beri bonus)
                if min_usia == 18 and max_usia == 35:
                    df_latihan_temp.loc[index, 'adjusted_similarity'] += BONUS_USIA_COCOK
                # Logika standar untuk rentang usia lainnya
                elif min_usia <= usia_pengguna <= max_usia:
                    df_latihan_temp.loc[index, 'adjusted_similarity'] += BONUS_USIA_COCOK
                else:
                    df_latihan_temp.loc[index, 'adjusted_similarity'] -= PENALTI_USIA_TIDAK_COCOK

    # --- Pass 2.5: Hard Filtering for Age Range (Custom Logic) ---
    if usia_pengguna is not None:
        if 18 <= usia_pengguna <= 25:
            def usia_match(rentang):
                min_usia, max_usia = _parse_age_range(str(rentang))
                return (
                    (min_usia == 18 and max_usia == 25) or
                    (min_usia == 26 and max_usia == 35) or
                    (min_usia == 18 and max_usia == 35)
                )
            df_latihan_temp = df_latihan_temp[df_latihan_temp['Rentang Usia'].apply(usia_match)]
        elif 26 <= usia_pengguna <= 35:
            def usia_match(rentang):
                min_usia, max_usia = _parse_age_range(str(rentang))
                # Ambil program untuk 26-35, 18-35, dan juga 18-25 (jika ingin lebih inklusif)
                return (
                    (min_usia == 26 and max_usia == 35) or
                    (min_usia == 18 and max_usia == 35) or
                    (min_usia == 18 and max_usia == 25)
                )
            df_latihan_temp = df_latihan_temp[df_latihan_temp['Rentang Usia'].apply(usia_match)]
        else:
            def usia_match(rentang):
                min_usia, max_usia = _parse_age_range(str(rentang))
                return min_usia is not None and max_usia is not None and min_usia <= usia_pengguna <= max_usia
            df_latihan_temp = df_latihan_temp[df_latihan_temp['Rentang Usia'].apply(usia_match)]

    # --- Pass 3: Hard Filtering (Filter Wajib untuk Tingkat Kebugaran) ---
    # Langkah ini memastikan preferensi level pengguna dipatuhi secara mutlak.
    # Filter ketat: hanya tampilkan program dengan tingkat kebugaran yang sama persis dengan pilihan pengguna.
    if user_kebugaran_keyword:
        df_latihan_temp = df_latihan_temp[df_latihan_temp['Tingkat Kebugaran Program'].str.lower() == user_kebugaran_keyword]

    # --- Pass 4: Hard Filtering for Location (Filter Wajib untuk Tempat) ---
    # Ini memastikan bahwa hanya program yang cocok dengan lokasi pilihan pengguna yang akan dipertimbangkan.
    if user_tempat_keyword:
        df_latihan_temp = df_latihan_temp[df_latihan_temp['Tempat Program'].str.lower().str.contains(user_tempat_keyword, na=False)]

    # --- Pass 5: Hard Filtering for Exercise Type (Filter Wajib untuk Jenis Latihan) ---
    if user_jenis_latihan_keyword:
        df_latihan_temp = df_latihan_temp[df_latihan_temp['Jenis Latihan Program'].str.lower().str.contains(user_jenis_latihan_keyword)]

    sorted_df = df_latihan_temp.sort_values(by='adjusted_similarity', ascending=False)
    return sorted_df.head(final_top_n)