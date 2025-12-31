from pymongo import MongoClient
import re
import os

client = MongoClient("mongodb://localhost:27017/")
db = client["cbf_program_db"]
col = db["programs"]

for prog in col.find():
    nama = prog.get("Nama Program Latihan")
    gambar_field_value = prog.get("gambar") # Field ini bisa berisi nama file atau path yang sudah diproses sebelumnya

    if not nama or not gambar_field_value:
        continue

    # Ekstrak nama file dasar, menangani kemungkinan path yang ada atau beberapa file yang dipisahkan ';'
    actual_filename = os.path.basename(gambar_field_value.split(";")[0].strip().replace("\\", "/"))

    # Sanitasi nama program untuk digunakan dalam path: lowercase, ganti non-alphanumeric dengan underscore
    program_name_slug = re.sub(r'\W+', '_', nama.lower()).strip('_') or "default_program"

    # Path relatif terhadap folder 'static'. Contoh: "images/knee_push_up/knee_push_up.jpg"
    new_gambar_path = f"images/{program_name_slug}/{actual_filename}"
    col.update_one(
        {"_id": prog["_id"]},
        {"$set": {"gambar": new_gambar_path}}  # Simpan path relatif yang baru
    )
