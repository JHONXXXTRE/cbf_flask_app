import pandas as pd
from pymongo import MongoClient
import os

# Konfigurasi MongoDB
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = "cbf_program_db"
PROGRAM_COLLECTION_NAME = "programs"

# Path ke dataset program latihan
PROGRAM_CSV_PATH = 'data/data_latihan.csv' # Menggunakan data_latihan.csv sebagai sumber utama

  # Fungsi ini disalin/disesuaikan dari app.py atau olah_data.py untuk konsistensi
def tambah_kata_kunci_untuk_fitur(jenis_latihan_input_str):
    if not isinstance(jenis_latihan_input_str, str):
        return ''
    jenis_latihan_parts = [part.strip() for part in jenis_latihan_input_str.split(',')]
    combined_keywords = [jenis_latihan_input_str.lower()]
    for jenis_str_single in jenis_latihan_parts:
        jenis = jenis_str_single.lower()
        if 'latihan fisik' in jenis or 'angkat beban' in jenis or 'push up' in jenis or 'squat' in jenis or 'kekuatan' in jenis : combined_keywords.append('kekuatan') # Memastikan 'kekuatan' ada jika salah satu keyword ini muncul
        if 'kardio' in jenis: combined_keywords.append('kardio')
        # if 'yoga' in jenis or 'pilates' in jenis or 'fleksibilitas' in jenis: combined_keywords.append('fleksibilitas') # Dihapus
        if 'hiit' in jenis: combined_keywords.append('hiit')
        # if 'zumba' in jenis or 'aerobik' in jenis: combined_keywords.append('aerobik') # Dihapus
        if '(' in jenis:
            main_type = jenis.split('(')[0].strip()
            if main_type: combined_keywords.append(main_type)
        else:
            if jenis: combined_keywords.append(jenis)
    return ' '.join(list(set(filter(None, combined_keywords))))

def main():
    print(f"Mencoba memuat data program dari: {PROGRAM_CSV_PATH}")
    if not os.path.exists(PROGRAM_CSV_PATH):
        print(f"ERROR: File dataset program '{PROGRAM_CSV_PATH}' tidak ditemukan.")
        return

    try:
        df_prog = pd.read_csv(PROGRAM_CSV_PATH)
    except Exception as e:
        print(f"Error membaca file CSV program: {e}")
        return

    print(f"Dataset program berhasil dimuat. Jumlah baris: {len(df_prog)}")

    # Mengisi nilai NaN dengan string kosong sebelum membuat fitur gabungan
    df_prog.fillna('', inplace=True)

    print("Membuat 'fitur_gabungan_program' agar konsisten dengan cbf_rekomendasi.py...")
    # Gunakan fungsi tambah_kata_kunci_untuk_fitur untuk memperkaya Jenis Latihan Program
    df_prog['Jenis Latihan Program Keywords'] = df_prog['Jenis Latihan Program'].apply(tambah_kata_kunci_untuk_fitur)
    df_prog['fitur_gabungan_program'] = (
        df_prog['Nama Program Latihan'].astype(str).str.lower() + ' ' +         # Sesuai urutan CSV (Kolom 2)
        df_prog['Deskripsi Program'].astype(str).str.lower() + ' ' +            # Sesuai urutan CSV (Kolom 3)
        df_prog['Jenis Latihan Program Keywords'] + ' ' +                   # Diolah dari 'Jenis Latihan Program' (Kolom 4 CSV), sudah lowercase
        df_prog['Tujuan Latihan'].astype(str).str.lower() + ' ' +               # Sesuai urutan CSV (Kolom 5)
        df_prog['Durasi Program (menit)'].astype(str).str.lower() + ' ' +       # Kolom 6
        df_prog['Tempat Program'].astype(str).str.lower() + ' ' +               # Kolom 7
        df_prog['Peralatan Program (Ya/Tidak)'].astype(str).str.lower() + ' ' + # Kolom 8
        df_prog['Tingkat Kebugaran Program'].astype(str).str.lower() + ' ' +     # Kolom 9
        df_prog['Waktu Ideal Program'].astype(str).str.lower() + ' ' +          # Kolom 10
        df_prog['Target Gender'].astype(str).str.lower() + ' ' +                # Kolom 12
        df_prog['Rentang Usia'].astype(str).str.lower()                         # Kolom 13 (Baru)
    ).str.strip() # Hapus spasi berlebih di awal/akhir
    print("'fitur_gabungan_program' telah dibuat.")
    print("Contoh fitur_gabungan_program:")
    print(df_prog['fitur_gabungan_program'].head())


    # Menghubungkan ke MongoDB
    try:
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        programs_collection = db[PROGRAM_COLLECTION_NAME]
    except Exception as e:
        print(f"Error menghubungkan ke MongoDB: {e}")
        return

    # Menghapus data lama (jika ada) dan memasukkan data baru
    try:
        programs_collection.delete_many({})
        print("Data lama di koleksi program berhasil dihapus.")
        
        # Mengubah DataFrame ke format dictionary untuk dimasukkan ke MongoDB
        programs_dict = df_prog.to_dict(orient='records')
        if programs_dict:
            programs_collection.insert_many(programs_dict)
            print(f"{len(programs_dict)} data program berhasil dimasukkan ke MongoDB.")
        else:
            print("Tidak ada data program untuk dimasukkan ke MongoDB.")

    except Exception as e:
        print(f"Error saat operasi database: {e}")
    finally:
        client.close()
        print("Koneksi MongoDB ditutup.")

if __name__ == '__main__':
    main()