import pandas as pd
from deep_translator import GoogleTranslator

df = pd.read_csv('data/exercises.csv')

def translate_instruction(text):
    if pd.isna(text):
        return ""
    try:
        # Memecah instruksi menjadi kalimat-kalimat berdasarkan koma,
        # kemudian menerjemahkan setiap kalimat.
        sentences = [s.strip() for s in text.split(',')]
        translated_sentences = []
        for s in sentences:
            if s: # Hanya terjemahkan jika kalimat tidak kosong
                translated_sentences.append(GoogleTranslator(source='auto', target='id').translate(s))
            else:
                translated_sentences.append("") # Tambahkan string kosong jika kalimat asli kosong
        
        return ', '.join(filter(None, translated_sentences)) # Gabungkan kembali, abaikan string kosong
    except Exception as e:
        print(f"Error saat menerjemahkan: '{text}'. Error: {e}")
        return text # Kembalikan teks asli jika ada error

df['instructions'] = df['instructions'].apply(translate_instruction)
df.to_csv('data/exercises_id.csv', index=False)
print("Selesai! File hasil: data/exercises_id.csv")