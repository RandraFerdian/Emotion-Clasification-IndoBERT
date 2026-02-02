import streamlit as st
import torch
import torch.nn.functional as F
import pandas as pd
import altair as alt
from transformers import BertTokenizer, BertForSequenceClassification

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="IndoBERT Emotion Analysis",
    page_icon="🧠",
    layout="centered"
)

# Judul dan Deskripsi
st.title("🧠 IndoBERT Emotion Detector")
st.markdown("Aplikasi deteksi emosi teks Bahasa Indonesia menggunakan model **IndoBERT**.")
st.write("---")

# ==========================================
# 2. LOAD MODEL (CACHED)
# ==========================================
# @st.cache_resource agar model tidak di-load berulang kali setiap klik tombol
@st.cache_resource
def load_model():
    model_path = './models' # Pastikan nama folder sesuai!
    
    try:
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model = BertForSequenceClassification.from_pretrained(model_path)
        return tokenizer, model
    except Exception as e:
        st.error(f"Gagal memuat model. Pastikan folder '{model_path}' ada.")
        st.error(str(e))
        return None, None

# Load model saat aplikasi mulai
tokenizer, model = load_model()

# ==========================================
# 3. FUNGSI PREDIKSI
# ==========================================
def predict_emotion(text, model, tokenizer):
    # Mapping Label (Sesuaikan urutan saat training!)
    # Urutan: 0:Anger, 1:Joy, 2:Sadness, 3:Fear, 4:Love, 5:Neutral
    label_map = {
        0: 'Anger 😡',
        1: 'Fear 😱',
        2: 'Joy 😂',
        3: 'Love ❤️',
        4: 'Neutral 😐',
        5: 'Sadness 😭'
    }

    # 1. Tokenisasi
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    
    # 2. Prediksi ke Model
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 3. Hitung Probabilitas (Softmax)
    probs = F.softmax(outputs.logits, dim=1)
    probs = probs.detach().numpy()[0] # Ambil array probabilitas
    
    # 4. Ambil index dengan nilai tertinggi
    pred_idx = torch.argmax(outputs.logits).item()
    pred_label = label_map[pred_idx]
    confidence = probs[pred_idx]
    
    return pred_label, confidence, probs, label_map

# ==========================================
# 4. USER INTERFACE (UI)
# ==========================================

# Input Teks dari User
input_text = st.text_area("Masukkan kalimat yang ingin dianalisis:", height=100, placeholder="Contoh: Aku sangat bahagia hari ini karena lulus ujian!")

if st.button("🔍 Analisis Emosi"):
    if input_text.strip() == "":
        st.warning("Mohon masukkan teks terlebih dahulu!")
    else:
        if model is not None:
            # Lakukan Prediksi
            with st.spinner('Sedang menganalisis otak IndoBERT...'):
                label, conf, probs, label_map = predict_emotion(input_text, model, tokenizer)
            
            # Tampilkan Hasil Utama
            st.success("Analisis Selesai!")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.metric("Emosi Terdeteksi", label)
                st.metric("Keyakinan (Confidence)", f"{conf*100:.2f}%")
            
            with col2:
                # Visualisasi Grafik Batang
                st.subheader("Detail Probabilitas")
                
                # Buat DataFrame untuk grafik
                df_probs = pd.DataFrame({
                    'Emosi': [label_map[i] for i in range(len(probs))],
                    'Probabilitas': probs
                })
                
                # Bikin grafik cantik pakai Altair
                chart = alt.Chart(df_probs).mark_bar().encode(
                    x=alt.X('Probabilitas', axis=alt.Axis(format='%')),
                    y=alt.Y('Emosi', sort='-x'),
                    color=alt.Color('Probabilitas', scale={'scheme': 'blues'}),
                    tooltip=['Emosi', alt.Tooltip('Probabilitas', format='.2%')]
                ).properties(height=300)
                
                st.altair_chart(chart, use_container_width=True)

# Footer
st.write("---")
st.caption("Dibuat dengan ❤️ menggunakan IndoBERT & Streamlit")