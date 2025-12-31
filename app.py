# Standard Library Imports
import ast
import csv
import json
import os
import re
from contextlib import contextmanager
import datetime

# Third-party Library Imports
import pandas as pd
from bson import ObjectId
from bson.errors import InvalidId
from dotenv import load_dotenv
from flask import (Flask, flash, make_response, redirect, render_template, request,
                   session, url_for)
from flask_login import (  # type: ignore
    LoginManager,
    UserMixin,
    current_user,
    login_required,
    login_user,
    logout_user,
)
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename
from recommender_engine import dapatkan_rekomendasi as get_recommendations_from_engine # Impor engine

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# Muat environment variables dari .env
load_dotenv()

app = Flask(__name__, template_folder=os.path.join(APP_ROOT, 'templates'))
app.secret_key = os.getenv("FLASK_SECRET_KEY", "supersecretkey_dev_default_flask")
app.config['UPLOAD_FOLDER'] = os.path.join(APP_ROOT, 'static', 'profile_pics')
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True) # Pastikan folder ada

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# --- Konfigurasi MongoDB ---
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = "cbf_program_db"
PROGRAM_COLLECTION_NAME = "programs" # Menggunakan data_latihan.csv sebagai sumber utama
USER_COLLECTION_NAME = "users" # Definisikan nama koleksi pengguna

# Path ke file kuesioner
KUESIONER_CSV_PATH = 'data/kuesioner_bersih.csv'
EXERCISES_CSV_PATH = 'data/exercises.csv' # Path untuk data latihan

# Konstanta untuk nama kolom kuesioner (diurutkan berdasarkan nomor)
COL_KUESIONER_USIA = '1. Usia'
COL_KUESIONER_JENIS_KELAMIN = '2. Jenis Kelamin'
COL_KUESIONER_PENGALAMAN = '4. Bagaimana tingkat kebugaran Anda saat ini?'
COL_KUESIONER_TUJUAN = '5. Apa tujuan utama Anda dalam berolahraga? (Bisa pilih lebih dari satu)'
COL_KUESIONER_JENIS_LATIHAN_PRIMARY = '6. Jenis latihan apa yang paling Anda sukai? (Bisa pilih lebih dari satu)'
COL_KUESIONER_JENIS_LATIHAN_FALLBACK = '6. Jenis latihan apa yang paling Anda sukai? (Bisa pilih lebih dari satu' # Nama kolom dengan potensi typo
COL_KUESIONER_HARI_SIBUK = '8. Hari apa saja Anda merasa sangat sibuk? (Bisa pilih lebih dari satu)'
COL_KUESIONER_JAM_LUANG = '9. Pada jam berapa Anda biasanya memiliki waktu luang untuk berolahraga?'
COL_KUESIONER_TEMPAT = '11. Apakah Anda lebih suka latihan di rumah atau di gym?'

# Tambahkan fungsi helper ini
@contextmanager
def mongo_db_connection():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    try:
        yield db
    finally:
        client.close()

IMAGE_SUBDIR = 'images'

# --- Variabel Global untuk Data dan Model ---
df_prog = None
tfidf_vectorizer = None
df_exercises = None # Tambahkan variabel global untuk data latihan
tfidf_matrix_prog = None

class User(UserMixin):
    def __init__(self, user_doc):
        self.id = str(user_doc['_id']) if '_id' in user_doc else user_doc.get('id', None)
        self.username = user_doc['username']
        self.nama = user_doc.get('nama', '')
        self.berat = user_doc.get('berat', None)
        self.tinggi = user_doc.get('tinggi', None)
        self.gol_darah = user_doc.get('gol_darah', None)
        self.foto = user_doc.get('foto', None)
        self.usia = user_doc.get('usia', None)
        self.jenis_kelamin = user_doc.get('jenis_kelamin', None)
        self.pengalaman = user_doc.get('pengalaman', None) # Akan diisi dari form preferensi

@login_manager.user_loader
def load_user(user_id):
    with mongo_db_connection() as db:
        user_col = db[USER_COLLECTION_NAME]
        user_doc = None
        try:
            user_doc = user_col.find_one({'_id': ObjectId(user_id)})
        except (InvalidId, TypeError):
            user_doc = user_col.find_one({'id': user_id}) # Fallback untuk ID lama
        if user_doc:
            return User(user_doc)
    return None

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# --- Fungsi Helper ---
def tambah_kata_kunci_untuk_fitur(jenis_latihan_input_str):
    """
    Memproses string 'Jenis Latihan' untuk mengekstrak dan menstandarisasi kata kunci.
    Fungsi ini menangani berbagai format input, memetakan sinonim ke kategori standar
    (misalnya, 'push up' -> 'kekuatan'), dan mengekstrak tipe utama dari format seperti 'Kardio (...)'.
    """
    if not isinstance(jenis_latihan_input_str, str):
        return ''

    # Kata kunci standar yang akan kita gunakan
    kata_kunci_standar = {
        'kekuatan': ['latihan fisik', 'angkat beban', 'push up', 'pull up', 'squat', 'kekuatan', 'bodyweight'],
        'kardio': ['kardio', 'lari', 'bersepeda', 'skipping', 'futsal', 'boxing', 'senam', 'jogging', 'running'],
        'hiit': ['hiit', 'fungsional']
    }

    # Normalisasi input awal
    input_lower = jenis_latihan_input_str.lower()
    
    # Pisahkan input jika ada beberapa jenis latihan (dipisahkan koma)
    jenis_latihan_parts = [part.strip() for part in jenis_latihan_input_str.split(',')]
    
    combined_keywords = set()

    # Tambahkan seluruh string input mentah yang sudah di-lowercase
    combined_keywords.add(input_lower)

    for part in jenis_latihan_parts:
        # Tambahkan bagian individu
        combined_keywords.add(part)

        # Ekstrak tipe utama jika ada format 'tipe (detail)'
        if '(' in part:
            main_type = part.split('(')[0].strip()
            if main_type:
                combined_keywords.add(main_type)

        # Cek dan tambahkan kata kunci standar berdasarkan sinonim
        for standar, sinonim_list in kata_kunci_standar.items():
            if any(sinonim in part for sinonim in sinonim_list):
                combined_keywords.add(standar)
    
    # Hapus string kosong dan gabungkan menjadi satu string, diurutkan untuk konsistensi
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
    pengalaman_user = str(user_data_dict.get('pengalaman', '')).lower() # Tambahkan pengalaman
    feature_string = (
        f"{usia} {tujuan} {jenis_latihan_keywords} {free_days_user_feature_str} "
        f"{waktu_luang} {tempat} {jenis_kelamin} {pengalaman_user}" # Tambahkan pengalaman ke string fitur
    )
    return ' '.join(feature_string.split())

def create_feature_string_for_historical_user(row):
    try:
        usia = str(row.get(COL_KUESIONER_USIA, ''))
        tujuan = str(row.get(COL_KUESIONER_TUJUAN, ''))

        jenis_latihan_raw_val = row.get(COL_KUESIONER_JENIS_LATIHAN_PRIMARY)
        if jenis_latihan_raw_val is None or pd.isna(jenis_latihan_raw_val) or str(jenis_latihan_raw_val).strip() == '':
            jenis_latihan_raw_val = row.get(COL_KUESIONER_JENIS_LATIHAN_FALLBACK, '')
        jenis_latihan_raw = str(jenis_latihan_raw_val)
        jenis_latihan_keywords = tambah_kata_kunci_untuk_fitur(jenis_latihan_raw)
        hari_sibuk_raw = str(row.get(COL_KUESIONER_HARI_SIBUK, ''))
        jam_luang_raw = str(row.get(COL_KUESIONER_JAM_LUANG, ''))
        tempat_raw = str(row.get(COL_KUESIONER_TEMPAT, ''))
        gender_raw = str(row.get(COL_KUESIONER_JENIS_KELAMIN, '')).lower()
        # Ambil pengalaman dari kuesioner historis
        pengalaman_raw = str(row.get(COL_KUESIONER_PENGALAMAN, '')).lower()

        all_days = ['Senin', 'Selasa', 'Rabu', 'Kamis', "Jum'at", 'Sabtu', 'Minggu']
        busy_days_list = [d.strip().capitalize() for d in hari_sibuk_raw.split(',') if d.strip()]
        free_days_str = ' '.join([d for d in all_days if d not in busy_days_list and d])
        feature_string = (
            f"{usia} {tujuan} {jenis_latihan_keywords} {free_days_str} "
            f"{jam_luang_raw} {tempat_raw} {gender_raw} {pengalaman_raw}" # Tambahkan pengalaman
        )
        return ' '.join(feature_string.split())
    except Exception as e:
        return ""

def load_and_preprocess_data_from_db():
    global df_prog, tfidf_vectorizer, tfidf_matrix_prog
    try:
        with mongo_db_connection() as db:
            programs_collection = db[PROGRAM_COLLECTION_NAME]
            programs_cursor = programs_collection.find({})
            df_prog_list = list(programs_cursor)
            if not df_prog_list:
                print("PERINGATAN: Tidak ada data program ditemukan di MongoDB.")
                df_prog = pd.DataFrame()
                return False

        df_prog = pd.DataFrame(df_prog_list)
        if '_id' in df_prog.columns: df_prog = df_prog.drop('_id', axis=1) # Gunakan df_prog = df_prog.drop() untuk menghindari SettingWithCopyWarning
        df_prog = df_prog.astype(str).fillna('')
        if 'fitur_gabungan_program' not in df_prog.columns or df_prog['fitur_gabungan_program'].isnull().all():
            print("PERINGATAN: Kolom 'fitur_gabungan_program' tidak ada atau kosong.")
            return False
        program_features_list = df_prog['fitur_gabungan_program'].tolist()
        historical_user_features_list = []
        if os.path.exists(KUESIONER_CSV_PATH):
            try:
                df_user_historical = pd.read_csv(KUESIONER_CSV_PATH)
                df_user_historical.fillna('', inplace=True)
                historical_user_features_list = df_user_historical.apply(create_feature_string_for_historical_user, axis=1).tolist()
                print(f"Data kuesioner historis dimuat ({len(df_user_historical)} rekaman).")
            except Exception as e:
                print(f"Error memproses '{KUESIONER_CSV_PATH}': {e}")
        else:
            print(f"PERINGATAN: File '{KUESIONER_CSV_PATH}' tidak ditemukan.")
        all_text_features_for_fitting = program_features_list[:]
        if historical_user_features_list:
            all_text_features_for_fitting.extend(filter(None, historical_user_features_list))
        if not all_text_features_for_fitting:
            print("ERROR: Tidak ada fitur teks untuk melatih TF-IDF.")
            return False
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_vectorizer.fit(all_text_features_for_fitting)
        print(f"TF-IDF Vectorizer dilatih pada {len(all_text_features_for_fitting)} dokumen.")
        tfidf_matrix_prog = tfidf_vectorizer.transform(program_features_list)
        print(f"Matriks TF-IDF program: {tfidf_matrix_prog.shape}")
        return True
    except Exception as e:
        print(f"Error signifikan saat load/preprocess data: {e}")
        return False

# Fungsi untuk memuat data latihan dari CSV
def load_exercises_data():
    global df_exercises
    if df_exercises is None: # Hanya muat jika belum ada
        try:
            if os.path.exists(EXERCISES_CSV_PATH):
                df_exercises = pd.read_csv(EXERCISES_CSV_PATH)
                df_exercises.fillna('', inplace=True) # Ganti NaN dengan string kosong
                print(f"Data latihan '{EXERCISES_CSV_PATH}' berhasil dimuat ({len(df_exercises)} rekaman).")
                return True
            else:
                print(f"PERINGATAN: File latihan '{EXERCISES_CSV_PATH}' tidak ditemukan.")
                df_exercises = pd.DataFrame() # Inisialisasi sebagai DataFrame kosong jika file tidak ada
                return False
        except Exception as e:
            print(f"Error saat memuat data latihan: {e}")
            df_exercises = pd.DataFrame()
            return False
    return True

def get_recommendations_from_model(user_input_data, top_n=10):
    global df_prog, tfidf_vectorizer, tfidf_matrix_prog
    if df_prog is None or tfidf_vectorizer is None or tfidf_matrix_prog is None or df_prog.empty:
        if not load_and_preprocess_data_from_db():
            flash("Sistem sedang mempersiapkan data, mohon coba lagi.", "warning")
            return []

    if df_prog.empty:
        flash("Tidak ada data program untuk rekomendasi.", "error")
        return []

    user_feature_string = create_feature_string_for_new_user(user_input_data)
    
    # Panggil engine rekomendasi yang sudah terpusat
    recommendations_df = get_recommendations_from_engine(
        profil_pengguna_string=user_feature_string,
        profil_pengguna_dict=user_input_data,
        df_latihan=df_prog,
        tfidf_vectorizer=tfidf_vectorizer,
        tfidf_matrix_latihan=tfidf_matrix_prog,
        final_top_n=top_n
    )

    # Ubah DataFrame hasil menjadi list of dictionaries untuk template
    recommendations_list = recommendations_df.to_dict(orient='records')
    return recommendations_list

def get_hari_luang(hari_sibuk_str, waktu_luang_user=None):
    semua_hari = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
    hari_sibuk_input = [h.strip().capitalize() for h in hari_sibuk_str.split(',') if h.strip()]
    hari_luang = [h for h in semua_hari if h not in hari_sibuk_input]
    if not hari_luang:
        return f"Tidak ada hari luang, manfaatkan waktu luang di hari sibuk pada jam: {waktu_luang_user}" if waktu_luang_user else "Tidak ada hari luang"
    return ' '.join(hari_luang)

def get_program_details_by_id(program_id):
    with mongo_db_connection() as db:
        prog_col = db[PROGRAM_COLLECTION_NAME]
        # Coba cari dengan ID sebagai string (default)
        prog = prog_col.find_one({'ID Program': program_id})
        if prog:
            return prog

        # Jika tidak ketemu dan ID adalah numerik, coba cari sebagai integer
        # Ini untuk kompatibilitas jika DB memiliki ID numerik
        if program_id.isdigit():
            try:
                prog = prog_col.find_one({'ID Program': int(program_id)})
                if prog:
                    return prog
            except (ValueError, TypeError):
                pass # Abaikan jika konversi gagal

    return None # Kembalikan None jika tidak ditemukan sama sekali

def parse_age_range(age_range_str):
    """
    Parses an age range string like '18-25 Tahun' or '26-35' into (min_age, max_age).
    Returns (None, None) if parsing fails.
    """
    if not age_range_str or not isinstance(age_range_str, str):
        return None, None
    
    # Find all numbers in the string
    numbers = re.findall(r'\d+', age_range_str)
    
    if len(numbers) >= 2: # Use >= 2 to be safe
        try:
            # Assume first two numbers are min and max
            return int(numbers[0]), int(numbers[1])
        except (ValueError, IndexError):
            return None, None
    elif len(numbers) == 1:
        try:
            return int(numbers[0]), int(numbers[0])
        except ValueError:
            return None, None
    return None, None

def format_description_for_html(description_str):
    return description_str.replace('\n', '<br>') if isinstance(description_str, str) else "Deskripsi tidak tersedia."

def extract_youtube_id(url):
    if not url or not isinstance(url, str): return None
    if 'youtu.be' in url: return url.split('/')[-1]
    if 'youtube.com' in url:
        import urllib.parse
        query = urllib.parse.urlparse(url).query
        params = urllib.parse.parse_qs(query)
        return params.get('v', [None])[0]
    return None

# --- Rute Flask ---
@app.route('/')
def index():
    return redirect(url_for('dashboard'))

@app.route('/recommend', methods=['POST'])
@login_required # Pastikan pengguna login untuk mendapatkan rekomendasi
def recommend_route():
    raw_selected_jenis_latihan_from_form = request.form.getlist('jenis_latihan') # Ambil list mentah
    if request.method == 'POST':
        user_input_from_form = {
            'usia': request.form.get('usia'),
            'jenis_kelamin': request.form.get('jenis_kelamin'),
            'tujuan': request.form.get('tujuan'), # Diubah ke .get() untuk dropdown
            'jenis_latihan': request.form.get('jenis_latihan'), # Diubah ke .get() untuk dropdown
            'hari_sibuk': ', '.join(request.form.getlist('hari_sibuk')),
            'waktu_luang': request.form.get('waktu_luang'),
            'tempat': request.form.get('tempat'),
            'pengalaman': request.form.get('pengalaman')
        }
        try:
            usia_val = int(user_input_from_form['usia'])
            if not (18 <= usia_val <= 100): # Batas usia lebih realistis
                flash("Input usia harus antara 18 dan 100 tahun.", "error")
                return redirect(url_for('form_rekomendasi'))
        except (ValueError, TypeError):
            flash("Format usia tidak valid.", "error")
            return redirect(url_for('form_rekomendasi'))

        valid_waktu_luang = ["Pagi (06:00-09:00)", "Siang (12:00-14:00)", "Sore (16:00-18:00)"]
        if user_input_from_form['waktu_luang'] not in valid_waktu_luang:
            flash("Pilihan waktu luang tidak valid.", "error")
            return redirect(url_for('form_rekomendasi'))

        required_fields = ['jenis_kelamin', 'tujuan', 'jenis_latihan', 'waktu_luang', 'tempat', 'pengalaman']
        for field in required_fields:
            if not user_input_from_form.get(field):
                flash(f"Mohon isi semua field wajib, termasuk {field.replace('_', ' ').capitalize()}.", "error")
                return redirect(url_for('form_rekomendasi'))
        
        # Simpan preferensi pengguna saat ini ke session untuk pre-fill form jika user ingin mengubahnya
        session['last_preferences'] = user_input_from_form

        # Tentukan berapa banyak rekomendasi yang akan ditampilkan
        try: # Jika 'num_to_show_next' datang dari tombol "More"
            # Jika 'num_to_show_next' datang dari tombol "More"
            num_to_display = int(request.form.get('num_to_show_next', 3)) # Default 4 untuk tampilan awal
        except ValueError:
            num_to_display = 4 # Fallback jika ada error konversi

        # Simpan 'pengalaman' ke profil pengguna
        with mongo_db_connection() as db:
            user_col = db[USER_COLLECTION_NAME]
            user_col.update_one({'_id': ObjectId(current_user.id)}, {'$set': {'pengalaman': user_input_from_form['pengalaman']}})

        # Perbarui current_user object agar perubahan tercermin segera
        current_user.pengalaman = user_input_from_form['pengalaman']


        user_input_from_form['hari_luang_user'] = get_hari_luang(
            user_input_from_form['hari_sibuk'], user_input_from_form['waktu_luang']
        )

        recommendations = get_recommendations_from_model(user_input_from_form, top_n=num_to_display)
        
        actual_num_shown = len(recommendations)
        # Tampilkan tombol "More" jika jumlah rekomendasi yang didapat sama dengan yang diminta,
        # menandakan kemungkinan masih ada lebih banyak.
        show_more_button_flag = actual_num_shown == num_to_display 
        next_num_to_show_for_button_val = actual_num_shown + 10 # Increment untuk klik "More" berikutnya menjadi 10

        for rec in recommendations:
            rec['yt_id'] = extract_youtube_id(rec.get('video_url', ''))
            rec['is_favorited'] = is_program_favorited(current_user.id, rec.get('ID Program'))

        # --- Logika untuk saran aktivitas di hari luang ---
        activity_schedule_suggestion_html = None
        # --- Logika BARU untuk saran aktivitas di hari luang berdasarkan skor tertinggi ---
        if recommendations: # Hanya buat saran jika ada rekomendasi
            all_days_ordered = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
            hari_sibuk_input_str = user_input_from_form.get('hari_sibuk', '')
            hari_sibuk_input_list = [h.strip().capitalize() for h in hari_sibuk_input_str.split(',') if h.strip()]
            actual_free_days_list = [day for day in all_days_ordered if day not in hari_sibuk_input_list]

            if actual_free_days_list:
                day_suggestions = []
                num_recommendations = len(recommendations)
                waktu_luang_user = user_input_from_form.get('waktu_luang', 'Kapan saja') # Ambil preferensi waktu
                
                for i, day in enumerate(actual_free_days_list):
                    # Ambil program dari daftar rekomendasi secara berurutan (cycling)
                    program_to_suggest = recommendations[i % num_recommendations]
                    program_name = program_to_suggest.get('Nama Program Latihan', 'Program Pilihan')
                    program_id = program_to_suggest.get('ID Program')
                    
                    # Buat link ke halaman detail program
                    program_link = url_for('program_detail_route', program_id=program_id)
                    
                    # Buat teks saran dengan link
                    suggestion_text = f'&bull; Hari {day} (Waktu: {waktu_luang_user}): Coba lakukan <a href="{program_link}" class="text-primary fw-bold"><strong>{program_name}</strong></a>.'
                    day_suggestions.append(suggestion_text)

                activity_schedule_suggestion_html = (
                    "<strong>Saran Jadwal Latihan di Hari Luang Anda:</strong><br>" +
                    "<br>".join(day_suggestions)
                )
        # --- Akhir logika saran aktivitas ---
        if not recommendations:
            flash("Maaf, belum ada rekomendasi yang cocok. Coba ubah pilihan Anda.", "info")
        return render_template('recommendations.html',
                               user_input=user_input_from_form,
                               recommendations=recommendations,
                               activity_schedule_suggestion=activity_schedule_suggestion_html,
                               show_more_button_flag=show_more_button_flag,
                               next_num_to_show_for_button=next_num_to_show_for_button_val)
    return redirect(url_for('form_rekomendasi'))

@app.route('/program/<program_id>')
def program_detail_route(program_id):
    program_details = get_program_details_by_id(program_id)
    if program_details:
        program_details['Deskripsi Program HTML'] = format_description_for_html(program_details.get('Deskripsi Program'))
        program_details['yt_id'] = extract_youtube_id(program_details.get('video_url'))
        program_details['is_favorited'] = is_program_favorited(current_user.id, program_id) if current_user.is_authenticated else False
        return render_template('program_detail.html', program=program_details)
    else:
        flash(f"Detail program ID '{program_id}' tidak ditemukan.", "error")
        return redirect(url_for('dashboard')) # Arahkan ke dashboard jika program tidak ada

def is_program_favorited(user_id_str, program_id_str):
    if not user_id_str or not program_id_str: return False
    with mongo_db_connection() as db:
        user_col = db[USER_COLLECTION_NAME]
        user_doc = user_col.find_one({'_id': ObjectId(user_id_str)})

    return bool(user_doc and program_id_str in user_doc.get('favorite_program_ids', []))

@app.route('/dashboard')
@login_required
def dashboard():
    stats = {}
    try:
        df_kuesioner = pd.read_csv(KUESIONER_CSV_PATH)
        stats["total_historical_users"] = len(df_kuesioner)
        # Ambil total program dari df_prog yang sudah dimuat
        if df_prog is not None and not df_prog.empty:
            stats["total_programs"] = len(df_prog)
        else:
            # Coba muat jika belum ada
            if load_and_preprocess_data_from_db() and df_prog is not None:
                stats["total_programs"] = len(df_prog)
            else:
                stats["total_programs"] = 0
    except Exception as e:
        print(f"Error saat mengambil statistik program: {e}") # Logging error
        flash(f"Gagal mengambil data statistik program: {e}", "error")
        stats["total_programs"] = 0
    
    # Statistik untuk jumlah latihan dari exercises.csv
    if df_exercises is not None and not df_exercises.empty:
        stats["total_exercises"] = len(df_exercises)
    else:
        stats["total_exercises"] = 0

    # Ambil contoh latihan
    sample_exercises_list = []
    if df_exercises is not None and not df_exercises.empty:
        # Ambil 3 contoh acak
        sample_df = df_exercises.sample(n=min(3, len(df_exercises)))
        for _, row in sample_df.iterrows():
            exercise_item = row.to_dict()
            # Proses gambar untuk exercise
            image_filenames_str = exercise_item.get('images', '')
            if isinstance(image_filenames_str, str) and image_filenames_str:
                first_image_filename = image_filenames_str.split(',')[0].strip()
                exercise_id_folder = exercise_item.get('id', '').strip()
                if first_image_filename and exercise_id_folder:
                    # Asumsi struktur folder: static/images/<exercise_id>/<image_filename>
                    # Perlu disesuaikan jika struktur folder gambar latihan berbeda
                    # Untuk saat ini, kita asumsikan gambar ada di static/images/nama_file_gambar.jpg
                    # Jika gambar ada di subfolder per exercise ID:
                    # exercise_item['image_url'] = url_for('static', filename=f'images/{exercise_id_folder}/{first_image_filename}')
                    # Jika gambar langsung di static/images:
                    exercise_item['image_url'] = url_for('static', filename=f'images/{first_image_filename}')
                else:
                    exercise_item['image_url'] = None
            else:
                exercise_item['image_url'] = None
            sample_exercises_list.append(exercise_item)

    favorited_programs_details = []
    scheduled_programs_details = []
    if current_user.is_authenticated:
        with mongo_db_connection() as db:
            user_col = db[USER_COLLECTION_NAME]
            user_doc = user_col.find_one({'_id': ObjectId(current_user.id)})
            if user_doc:
                if 'favorite_program_ids' in user_doc:
                    for prog_id_str in user_doc['favorite_program_ids']:
                        if not isinstance(prog_id_str, str): prog_id_str = str(prog_id_str)
                        if not prog_id_str: continue
                        program_detail = get_program_details_by_id(prog_id_str) # Ini akan membuka koneksi lagi, idealnya passing db
                        if program_detail: favorited_programs_details.append(program_detail)
                if 'saved_programs' in user_doc:
                    for scheduled_item in user_doc['saved_programs']:
                        if isinstance(scheduled_item, dict):
                            program_detail = get_program_details_by_id(scheduled_item.get('program_id')) # Sama seperti di atas
                            if program_detail:
                                program_detail['jadwal_hari'] = scheduled_item.get('jadwal_hari')
                                program_detail['jadwal_jam'] = scheduled_item.get('jadwal_jam')
                                scheduled_programs_details.append(program_detail)


    bmi = None
    if current_user.is_authenticated and current_user.berat and current_user.tinggi:
        try:
            berat_kg = float(current_user.berat)
            tinggi_m = float(current_user.tinggi) / 100
            if tinggi_m > 0: bmi = round(berat_kg / (tinggi_m ** 2), 1)
        except (ValueError, TypeError, AttributeError): bmi = None

    current_year = datetime.datetime.now().year # Dapatkan tahun saat ini
    return render_template(
        'dashboard.html', data=stats,
        sample_exercises=sample_exercises_list, # Kirim contoh latihan ke template
        favorited_programs_list=favorited_programs_details,
        scheduled_programs_list=scheduled_programs_details, bmi=bmi,
        current_year=current_year # Kirim tahun saat ini ke template
    )

@app.route('/form', methods=['GET'])
@login_required
def form_rekomendasi():
    salam_pembuka = "Silakan isi preferensi Anda untuk mendapatkan rekomendasi program latihan."
    form_fields_desc = {
        "usia": "1. Usia Anda (dalam tahun)",
        "jenis_kelamin": "2. Jenis Kelamin Anda",
        "tujuan": "3. Apa tujuan utama Anda berolahraga? (Bisa pilih lebih dari satu)",
        "jenis_latihan": "4. Jenis latihan apa yang paling Anda sukai? (Bisa pilih lebih dari satu)",
        "hari_sibuk": "5. Hari apa saja Anda biasanya sibuk? (Pilih satu atau lebih)",
        "waktu_luang": "6. Pada jam berapa Anda biasanya punya waktu luang untuk olahraga? (06:00 - 18:00)",
        "tempat": "8. Di mana Anda biasanya berolahraga?",
        "pengalaman": "9. Bagaimana tingkat pengalaman Anda dalam berolahraga?"
    }
    options_for_form = {
        "jenis_kelamin": ["Pria", "Wanita"],
        "tujuan": ["Menurunkan berat badan", "Meningkatkan massa otot", "Menjaga kesehatan"],
        "jenis_latihan": ["Kardio (Lari, Sepeda, Renang)", "Latihan Fisik (Angkat Beban, Push up, Squat)", "HIIT"],
        "hari_sibuk": ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"],
        "waktu_luang": ["Pagi (06:00-09:00)", "Siang (12:00-14:00)", "Sore (16:00-18:00)"],
        "tempat": ["Rumah", "Gym/Fitness Center", "Outdoor (Taman, Lapangan)"],
        "pengalaman": ["Pemula (Baru memulai atau jarang)", "Menengah (Cukup rutin, paham dasar)", "Mahir (Sangat rutin, teknik bagus)"]
    }

    # Coba ambil preferensi terakhir dari session, jika ada
    last_preferences = session.pop('last_preferences', None) # Gunakan pop untuk menghapus dari session setelah diambil
    user_data_for_form = {}

    if last_preferences:
        user_data_for_form = last_preferences
        # Konversi string koma-separated kembali ke list untuk checkbox
        # Hanya 'hari_sibuk' yang masih menggunakan checkbox
        if 'hari_sibuk' in user_data_for_form and isinstance(user_data_for_form['hari_sibuk'], str):
            user_data_for_form['hari_sibuk'] = [s.strip() for s in user_data_for_form['hari_sibuk'].split(',') if s.strip()]
    elif current_user.is_authenticated:
        # Fallback ke data user dari database jika tidak ada preferensi terakhir di session
        user_data_for_form['usia'] = current_user.usia
        user_data_for_form['jenis_kelamin'] = current_user.jenis_kelamin
        user_data_for_form['pengalaman'] = current_user.pengalaman
    
    global df_prog
    if df_prog is None or df_prog.empty:
        if not load_and_preprocess_data_from_db():
            flash("Gagal memuat data program yang diperlukan.", "warning")
    current_year = datetime.datetime.now().year # Dapatkan tahun saat ini
    return render_template('index.html', fields=form_fields_desc, options=options_for_form,
                           salam_pembuka=salam_pembuka, user_data=user_data_for_form,
                           current_year=current_year,
                           options_json=json.dumps(options_for_form)) # Kirim opsi sebagai JSON ke template

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated: return redirect(url_for('dashboard'))
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        with mongo_db_connection() as db:
            user_col = db[USER_COLLECTION_NAME]
            user_doc = user_col.find_one({'username': username})
            if user_doc and check_password_hash(user_doc['password'], password):
                login_user(User(user_doc))
                return redirect(url_for('dashboard'))
        flash('Username atau password salah', 'error') # Pindahkan flash di luar with jika user_doc None
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated: return redirect(url_for('dashboard'))
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        nama = request.form['nama']
        berat = request.form.get('berat')
        tinggi = request.form.get('tinggi')
        gol_darah = request.form.get('gol_darah')
        usia = request.form.get('usia')
        jenis_kelamin = request.form.get('jenis_kelamin')

        with mongo_db_connection() as db:
            user_col = db[USER_COLLECTION_NAME]
            if user_col.find_one({'username': username}):
                flash('Username sudah ada.', 'error')
                return redirect(url_for('register'))

            required_fields = {'username': username, 'password': password, 'nama': nama, 'usia': usia, 'jenis_kelamin': jenis_kelamin}
            missing_fields = [key for key, value in required_fields.items() if not value]
            if missing_fields:
                flash(f"Field berikut wajib diisi: {', '.join(missing_fields).replace('_', ' ').title()}.", 'error')
                return redirect(url_for('register'))
            try:
                if not (18 <= int(usia) <= 100):
                    flash("Usia harus antara 18 dan 100 tahun.", "error")
                    return redirect(url_for('register'))
            except ValueError:
                flash("Format usia tidak valid.", "error")
                return redirect(url_for('register'))

            user_doc_data = {
                'username': username, 'password': generate_password_hash(password), 'nama': nama,
                'berat': int(berat) if berat and berat.isdigit() else None,
                'tinggi': int(tinggi) if tinggi and tinggi.isdigit() else None,
                'gol_darah': gol_darah if gol_darah else None, 'foto': None,
                'usia': int(usia), 'jenis_kelamin': jenis_kelamin, 'pengalaman': None # Pengalaman awal None
            }
            user_col.insert_one(user_doc_data)
            flash('Registrasi berhasil! Silakan login.', 'success')
            return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    if request.method == 'POST':
        # Ambil data dari form
        nama = request.form.get('nama')
        berat = request.form.get('berat')
        tinggi = request.form.get('tinggi')
        gol_darah = request.form.get('gol_darah')
        
        update_data = {
            'nama': nama,
            'berat': int(berat) if berat and berat.isdigit() else current_user.berat,
            'tinggi': int(tinggi) if tinggi and tinggi.isdigit() else current_user.tinggi,
            'gol_darah': gol_darah if gol_darah else current_user.gol_darah
        }

        # Handle upload foto
        if 'foto' in request.files:
            file = request.files['foto']
            if file and file.filename != '' and allowed_file(file.filename):
                # Buat nama file yang aman dan unik
                filename = secure_filename(file.filename)
                unique_filename = f"{current_user.id}_{filename}"
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                
                # Hapus foto lama jika ada
                if current_user.foto and os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], os.path.basename(current_user.foto))):
                    try:
                        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], os.path.basename(current_user.foto)))
                    except OSError as e:
                        print(f"Error menghapus file lama: {e}")

                file.save(file_path)
                # Simpan path relatif ke DB
                update_data['foto'] = url_for('static', filename=f'profile_pics/{unique_filename}')

        with mongo_db_connection() as db:
            user_col = db[USER_COLLECTION_NAME]
            user_col.update_one({'_id': ObjectId(current_user.id)}, {'$set': update_data})
        
        flash('Profil berhasil diperbarui!', 'success')
        return redirect(url_for('dashboard'))

    # Untuk GET request, tampilkan form dengan data saat ini
    return render_template('profile.html')

@app.route('/history')
@login_required
def history():
    # TODO: Ganti data dummy ini dengan query ke database untuk riwayat latihan pengguna yang sebenarnya.
    # Anda perlu mengambil data latihan yang telah diselesaikan oleh current_user.
    dummy_history = [
        {'date': '2023-10-26', 'program': 'Full Body Strength', 'duration': '45 menit', 'status': 'Selesai'},
        {'date': '2023-10-24', 'program': 'Cardio Blast', 'duration': '30 menit', 'status': 'Selesai'},
        {'date': '2023-10-22', 'program': 'Yoga Pagi', 'duration': '20 menit', 'status': 'Selesai'},
    ]
    # Jika tidak ada riwayat, kirim list kosong agar template bisa menampilkan pesan "Riwayat Latihan Kosong"
    # history_data = [] 
    return render_template('history.html', history=dummy_history)


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash("Anda telah berhasil logout.", "success")
    return redirect(url_for('login'))

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/simpan_program', methods=['POST'])
@login_required
def simpan_program():
    program_id = request.form.get('program_id')
    jadwal_hari = request.form.get('jadwal_hari')
    jadwal_jam = request.form.get('jadwal_jam')
    if not program_id or not jadwal_hari or not jadwal_jam:
        flash("Lengkapi semua field jadwal.", "error")
        return redirect(url_for('dashboard'))
    with mongo_db_connection() as db:
        user_col = db[USER_COLLECTION_NAME]
        user_doc = user_col.find_one({'_id': ObjectId(current_user.id)})
        if not user_doc:
            flash("User tidak ditemukan.", "error")
            return redirect(url_for('dashboard'))
        saved_programs = user_doc.get('saved_programs', [])
        already_saved = any(isinstance(p, dict) and p.get('program_id') == program_id and p.get('jadwal_hari') == jadwal_hari and p.get('jadwal_jam') == jadwal_jam for p in saved_programs)
        if not already_saved:
            saved_programs.append({"program_id": program_id, "jadwal_hari": jadwal_hari, "jadwal_jam": jadwal_jam})
            user_col.update_one({'_id': ObjectId(current_user.id)}, {'$set': {'saved_programs': saved_programs}})
            flash("Program berhasil disimpan ke jadwal!", "success")
        else:
            flash("Program dengan jadwal ini sudah ada.", "info")
    return redirect(url_for('dashboard'))

@app.route('/toggle_favorite_program', methods=['POST'])
@login_required
def toggle_favorite_program():
    program_id = request.form.get('program_id')
    action = request.form.get('action') # 'favorite' or 'unfavorite'
    if not program_id:
        flash("ID program tidak valid.", "error")
        return redirect(request.referrer or url_for('dashboard'))
    with mongo_db_connection() as db:
        user_col = db[USER_COLLECTION_NAME]
        user_doc = user_col.find_one({'_id': ObjectId(current_user.id)})
        if not user_doc:
            flash("User tidak ditemukan.", "error")
            return redirect(request.referrer or url_for('dashboard'))
        fav_list = user_doc.get('favorite_program_ids', [])
        if action == "favorite" and program_id not in fav_list:
            fav_list.append(program_id)
            flash("Program ditambahkan ke favorit.", "success")
        elif action == "unfavorite" and program_id in fav_list:
            fav_list.remove(program_id)
            flash("Program dihapus dari favorit.", "success")
        user_col.update_one({'_id': ObjectId(current_user.id)}, {'$set': {'favorite_program_ids': fav_list}})
    return redirect(request.referrer or url_for('dashboard'))

@app.route('/delete_scheduled_program', methods=['POST'])
@login_required
def delete_scheduled_program():
    program_id_to_delete = request.form.get('program_id')
    jadwal_hari_to_delete = request.form.get('jadwal_hari')
    jadwal_jam_to_delete = request.form.get('jadwal_jam')
    if not program_id_to_delete or not jadwal_hari_to_delete or not jadwal_jam_to_delete:
        flash("Informasi program/jadwal tidak lengkap.", "error")
        return redirect(url_for('dashboard'))
    with mongo_db_connection() as db:
        user_col = db[USER_COLLECTION_NAME]
        user_doc = user_col.find_one({'_id': ObjectId(current_user.id)})
        if not user_doc:
            flash("User tidak ditemukan.", "error")
            return redirect(url_for('dashboard'))
        saved_programs = user_doc.get('saved_programs', [])
        updated_saved_programs = [p for p in saved_programs if not (isinstance(p, dict) and p.get('program_id') == program_id_to_delete and p.get('jadwal_hari') == jadwal_hari_to_delete and p.get('jadwal_jam') == jadwal_jam_to_delete)]
        if len(updated_saved_programs) < len(saved_programs):
            user_col.update_one({'_id': ObjectId(current_user.id)}, {'$set': {'saved_programs': updated_saved_programs}})
            flash("Jadwal program berhasil dihapus.", "success")
        else:
            flash("Jadwal program tidak ditemukan.", "info")
    return redirect(url_for('dashboard'))

@app.route('/exercise', methods=['GET', 'POST'])
@login_required
def exercise():
    selectedPrimaryMuscle = request.cookies.get('selectedPrimaryMuscle', '') # Ambil dari cookie
    selectedEquipment = '' # Untuk beginner, equipment dipilih di halaman yang sama
    
    if request.method == 'POST':
        # Ini adalah POST dari form pemilihan equipment di halaman beginner
        # selectedPrimaryMuscle sudah ada dari cookie
        selectedEquipment = request.form.get('equipment', '')
        # Simpan equipment ke cookie jika perlu, atau langsung gunakan untuk rekomendasi
        # Untuk saat ini, kita akan langsung membuat user_input untuk rekomendasi
        
        user_input_for_exercise = {
            "primaryMuscles": [selectedPrimaryMuscle] if selectedPrimaryMuscle else [],
            "equipment": [selectedEquipment] if selectedEquipment else []
        }
        # Redirect ke halaman rekomendasi dengan data ini
        # Kita perlu cara untuk mengirim user_input_for_exercise ke exercises_recommendations
        # Salah satu cara adalah menyimpannya di session atau mengirim sebagai parameter query (kurang ideal untuk JSON besar)
        # Untuk konsistensi, kita akan gunakan form POST ke /exercises_recommendations
        
        # Buat form tersembunyi dan submit via JS, atau langsung render template rekomendasi
        # Untuk kesederhanaan, kita akan langsung memanggil fungsi dan render template rekomendasi
        recommendations = get_exercise_recommendations_for_user(user_input_for_exercise, is_advanced_filter=False)
        
        resp = make_response(render_template(
            'exercises _recommendations.html',
            recommendations=recommendations,
            user_input=user_input_for_exercise # Kirim user_input agar bisa digunakan tombol "More"
        ))
        # Hapus cookie setelah digunakan agar tidak mengganggu sesi berikutnya
        resp.set_cookie('selectedPrimaryMuscle', '', expires=0, samesite='Lax')
        if selectedEquipment: # Simpan equipment jika ada, untuk tombol "More"
             resp.set_cookie('selectedEquipment', selectedEquipment, samesite='Lax')
        else:
             resp.set_cookie('selectedEquipment', '', expires=0, samesite='Lax')
        return resp

    # Untuk GET request atau jika bukan POST dari form equipment
    # Disesuaikan untuk beginner.jpg (tampilan depan)
    primary_muscles_list = [
        "Chest", # Kiri atas
        "Biceps",     # Kiri tengah-atas
        "Abs",    # Kiri tengah-bawah
        "Legs",       # Tengah
        "Back",      # Kanan atas (mewakili Quads) / Kaki umum
        "Glutes",   # Kanan tengah (meskipun lebih ke belakang, sering dilatih bersamaan)
        "Hamstring",  
        "Calves"  
    ]
    
    resp = make_response(render_template(
        'exercise.html',
        primary_muscles=primary_muscles_list,
        selectedPrimaryMuscle=selectedPrimaryMuscle
    ))
    # Set cookie untuk primary muscle jika ada perubahan dari JS (walaupun JS di exercise.html sudah handle ini)
    # Ini lebih sebagai fallback jika JS tidak berjalan atau untuk memastikan state
    if request.form.get('selectedPrimaryMuscle'): # Jika ada POST dari pemilihan otot
         resp.set_cookie('selectedPrimaryMuscle', request.form.get('selectedPrimaryMuscle'), samesite='Lax')
    return resp


@app.route('/exercises_recommendations', methods=['POST'])
@login_required
def exercises_recommendations():
    user_input_json = request.form.get('user_input')
    if not user_input_json:
        flash("Tidak ada input preferensi latihan.", "warning")
        return redirect(url_for('exercise')) # Arahkan kembali ke form beginner
    
    try:
        user_input = json.loads(user_input_json)
    except json.JSONDecodeError:
        flash("Format input preferensi tidak valid.", "error")
        return redirect(url_for('exercise'))

    # Ambil dari cookie jika ada untuk tombol "More"
    if not user_input.get("primaryMuscles"):
        selectedPrimaryMuscle_cookie = request.cookies.get('selectedPrimaryMuscle')
        if selectedPrimaryMuscle_cookie:
            user_input["primaryMuscles"] = [selectedPrimaryMuscle_cookie]
    
    if not user_input.get("equipment"):
        selectedEquipment_cookie = request.cookies.get('selectedEquipment')
        if selectedEquipment_cookie:
            user_input["equipment"] = [selectedEquipment_cookie]

    recommendations = get_exercise_recommendations_for_user(user_input, is_advanced_filter=False)
    
    # Simpan kembali ke cookie untuk tombol "More" berikutnya
    resp = make_response(render_template(
        'exercises _recommendations.html',
        recommendations=recommendations,
        user_input=user_input
    ))
    if user_input.get("primaryMuscles"):
        resp.set_cookie('selectedPrimaryMuscle', user_input["primaryMuscles"][0] if user_input["primaryMuscles"] else '', samesite='Lax')
    if user_input.get("equipment"):
        resp.set_cookie('selectedEquipment', user_input["equipment"][0] if user_input["equipment"] else '', samesite='Lax')
    return resp

def _check_advanced_filter_condition(user_pref_value, exercise_attr_value, is_list_preference=False):
    """
    Memeriksa apakah atribut latihan cocok dengan preferensi pengguna untuk filter lanjutan.
    user_pref_value: Preferensi pengguna (str atau list str).
    exercise_attr_value: Nilai atribut latihan (str).
    is_list_preference: True jika user_pref_value adalah list dan salah satu item cocok sudah cukup.
    """
    if not user_pref_value:  # Tidak ada preferensi, jadi lolos filter
        return True
    
    exercise_attr_value_lower = (exercise_attr_value or '').lower().strip()
    if not exercise_attr_value_lower: # Latihan tidak punya info untuk atribut ini, tapi pengguna punya preferensi
        return False

    if is_list_preference: # Misal: secondaryMuscles
        user_pref_list = [item.lower().strip() for item in user_pref_value if item]
        return any(pref_item in exercise_attr_value_lower for pref_item in user_pref_list)
    else: # Misal: level, force, mechanic, category (perlu kecocokan persis)
        user_pref_lower = user_pref_value.lower().strip()
        return user_pref_lower == exercise_attr_value_lower

def get_exercise_recommendations_for_user(user_input_dict, is_advanced_filter=False):
    results = []
    MUSCLE_GROUP_MAPPING = {
        "neck": ["neck"],
        "shoulders": ["shoulder", "shoulders", "deltoid", "delts", "rotator cuff", "traps"],
        "back": ["back", "lats", "latissimus", "traps", "trapezius", "rhomboids", "erector spinae", "teres major", "teres minor"],
        "chest": ["chest", "pectoral", "pectorals"],
        "legs": ["leg", "legs", "quadriceps", "quads", "hamstrings", "hams", "glutes", "calves", "thigh", "adductor", "abductor"],
        "arms": ["arm", "arms", "biceps", "triceps", "forearm", "brachialis", "brachioradialis"],
        "abs": ["abs", "abdominals", "abdominal", "core", "obliques", "rectus abdominis", "transverse abdominis"],
        "glutes": ["glute", "glutes", "buttock", "buttocks", "hip"],
        "calves": ["calf", "calves", "gastrocnemius", "soleus"],
        "traps": ["traps", "trapezius"],
        "forearms": ["forearm", "forearms", "brachioradialis"],
        
    }
    try:
        with open(EXERCISES_CSV_PATH, mode='r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                passes_primary_muscle_filter = True
                if 'primaryMuscles' in user_input_dict and user_input_dict['primaryMuscles'] and user_input_dict['primaryMuscles'][0]:
                    user_selected_groups = [g.lower().strip() for g in user_input_dict['primaryMuscles']]
                    target_keywords = set()
                    for group_name in user_selected_groups:
                        target_keywords.update(MUSCLE_GROUP_MAPPING.get(group_name, [group_name]))
                    csv_primary_muscles_str = row.get('primaryMuscles', '').lower()
                    match_found = any(keyword in csv_primary_muscles_str for keyword in target_keywords) if csv_primary_muscles_str else False
                    passes_primary_muscle_filter = match_found
                if not passes_primary_muscle_filter: continue

                passes_equipment_filter = True
                if 'equipment' in user_input_dict and user_input_dict['equipment'] and user_input_dict['equipment'][0]:
                    user_selected_equipment_list = [eq.lower().strip() for eq in user_input_dict['equipment']]
                    csv_equipment_str = row.get('equipment', '').lower().strip()
                    equipment_match = False
                    if "body only" in user_selected_equipment_list:
                        if not csv_equipment_str or "body weight" in csv_equipment_str or "body only" in csv_equipment_str or "no equipment" in csv_equipment_str:
                            equipment_match = True
                    elif csv_equipment_str and any(user_eq_keyword in csv_equipment_str for user_eq_keyword in user_selected_equipment_list):
                        equipment_match = True
                    passes_equipment_filter = equipment_match
                if not passes_equipment_filter: continue
                
                if is_advanced_filter:
                    if not _check_advanced_filter_condition(user_input_dict.get('secondaryMuscles'), row.get('secondaryMuscles'), is_list_preference=True):
                        continue
                    if not _check_advanced_filter_condition(user_input_dict.get('level'), row.get('level')):
                        continue
                    if not _check_advanced_filter_condition(user_input_dict.get('force'), row.get('force')):
                        continue
                    if not _check_advanced_filter_condition(user_input_dict.get('mechanic'), row.get('mechanic')):
                        continue
                    if not _check_advanced_filter_condition(user_input_dict.get('category'), row.get('category')):
                        continue

                exercise_name = row.get('name', 'Unnamed Exercise')
                exercise_id = row.get('id', f'exid_{exercise_name.lower().replace(" ", "_")[:20]}')
                images_raw = row.get('images', '')
                images = [img.strip() for img in images_raw.split(',') if img.strip()]
                instructions_str = row.get('instructions', '').strip()
                parsed_instructions = []
                if instructions_str:
                    if instructions_str.startswith('[') and instructions_str.endswith(']'):
                        try:
                            evaluated = ast.literal_eval(instructions_str)
                            parsed_instructions = [str(item).strip() for item in evaluated if str(item).strip()] if isinstance(evaluated, list) else ([str(evaluated).strip()] if evaluated else [])
                        except (ValueError, SyntaxError):
                            parsed_instructions = [s.strip() for s in instructions_str.split(',') if s.strip()]
                    else:
                        parsed_instructions = [s.strip() for s in instructions_str.split(',') if s.strip()]
                results.append({'name': exercise_name, 'images': images, 'instructions': '<br>'.join(parsed_instructions), 'id': exercise_id})
    except FileNotFoundError:
        print(f"Error: '{EXERCISES_CSV_PATH}' not found.")
        return []
    except Exception as e:
        print(f"Error processing {EXERCISES_CSV_PATH}: {e}")
        return []
    return results

@app.route('/advanced', methods=['GET', 'POST'])
@login_required
def advanced_exercise_form():
    selectedPrimaryMuscle = request.form.get('selectedPrimaryMuscle', request.cookies.get('selectedPrimaryMuscle', ''))
    primary_muscles_list = ["Chest", "Back", "Shoulders", "Legs", "Biceps", "Triceps", "Abs", "Glutes", "Calves", "Traps", "Forearms", "Neck"] # Modifikasi di sini
    
    resp = make_response(render_template(
        'advanced.html', primary_muscles=primary_muscles_list, selectedPrimaryMuscle=selectedPrimaryMuscle
    ))
    if selectedPrimaryMuscle: resp.set_cookie('selectedPrimaryMuscle', selectedPrimaryMuscle, samesite='Lax')
    return resp

@app.route('/advanced_recommendations', methods=['POST'])
@login_required
def advanced_exercises_recommendations():
    user_input = {
        'primaryMuscles': [request.form.get('selectedPrimaryMuscleHidden')] if request.form.get('selectedPrimaryMuscleHidden') else [],
        'secondaryMuscles': request.form.getlist('secondaryMuscles[]'),
        'level': request.form.get('level'),
        'equipment': [request.form.get('equipment')] if request.form.get('equipment') else [],
        'force': request.form.get('force'),
        'mechanic': request.form.get('mechanic'),
        'category': request.form.get('category')
    }
    recommendations = get_exercise_recommendations_for_user(user_input, is_advanced_filter=True)
    
    # Simpan primary muscle dan equipment ke cookie untuk tombol "More"
    resp = make_response(render_template('exercises _recommendations.html', recommendations=recommendations, user_input=user_input))
    if user_input.get("primaryMuscles"):
        resp.set_cookie('selectedPrimaryMuscle', user_input["primaryMuscles"][0] if user_input["primaryMuscles"] else '', samesite='Lax')
    if user_input.get("equipment"):
        resp.set_cookie('selectedEquipment', user_input["equipment"][0] if user_input["equipment"] else '', samesite='Lax')
    # Simpan filter lanjutan lainnya di session jika ingin "More" mempertahankan filter tersebut
    session['advanced_filters_for_more'] = {
        'secondaryMuscles': user_input['secondaryMuscles'],
        'level': user_input['level'],
        'force': user_input['force'],
        'mechanic': user_input['mechanic'],
        'category': user_input['category']
    }
    return resp

# --- Inisialisasi Aplikasi ---
if __name__ == '__main__':
    print("Memulai aplikasi CBF Rekomendasi...")
    if not load_and_preprocess_data_from_db():
        print("PERHATIAN: Gagal memuat data program saat startup.")
    if not load_exercises_data(): # Muat data latihan saat startup
        print("PERHATIAN: Gagal memuat data latihan (exercises.csv) saat startup.")

    app.run(host='0.0.0.0', port=5000, debug=True)