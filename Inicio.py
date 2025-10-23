# -*- coding: utf-8 -*-
# TF-IDF QA — Estaciones & Clima Edition
# Requisitos: streamlit, scikit-learn, pandas, nltk
# (NLTK: usa SnowballStemmer inglés, no precisa descargar corpora)

import re
import io
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import SnowballStemmer

# ------------------------ Config básica ------------------------
st.set_page_config(page_title="TF-IDF QA — Estaciones", page_icon="🌦️", layout="centered")

# Paletas por estación
SEASONS = {
    "Primavera 🌸": {
        "bg1": "#fff7fb", "bg2": "#eafff5",
        "card": "rgba(255,255,255,0.92)", "accent": "#ff7ac8", "accent2": "#b8f2cf",
        "text": "#111", "chip": "#ffe6f4",
        "svg": """<svg viewBox="0 0 220 90" xmlns="http://www.w3.org/2000/svg">
          <rect rx="16" width="220" height="90" fill="#ffe6f4"/>
          <g transform="translate(20,20)">
            <circle cx="20" cy="20" r="10" fill="#ff8dcf"/>
            <circle cx="40" cy="16" r="7" fill="#ffd1e6"/>
            <circle cx="30" cy="34" r="8" fill="#ff9ecf"/>
            <rect x="90" y="30" width="80" height="8" rx="4" fill="#b8f2cf"/>
            <rect x="90" y="46" width="60" height="8" rx="4" fill="#ff8dcf"/>
          </g></svg>"""
    },
    "Verano ☀️": {
        "bg1": "#fffbe6", "bg2": "#ffe7b7",
        "card": "rgba(255,255,255,0.92)", "accent": "#ff9f1a", "accent2": "#ffd166",
        "text": "#111", "chip": "#fff0c2",
        "svg": """<svg viewBox="0 0 220 90" xmlns="http://www.w3.org/2000/svg">
          <rect rx="16" width="220" height="90" fill="#fff0c2"/>
          <g transform="translate(30,20)">
            <circle cx="25" cy="25" r="18" fill="#ffd166"/>
            <rect x="90" y="30" width="80" height="8" rx="4" fill="#ff9f1a"/>
            <rect x="90" y="46" width="60" height="8" rx="4" fill="#ffd166"/>
          </g></svg>"""
    },
    "Otoño 🍂": {
        "bg1": "#fff3e6", "bg2": "#ffe0c7",
        "card": "rgba(255,255,255,0.92)", "accent": "#c26d3f", "accent2": "#f5c49b",
        "text": "#111", "chip": "#ffe9d7",
        "svg": """<svg viewBox="0 0 220 90" xmlns="http://www.w3.org/2000/svg">
          <rect rx="16" width="220" height="90" fill="#ffe9d7"/>
          <g transform="translate(20,18)">
            <path d="M20 20 C30 0, 40 0, 48 20 C40 28, 30 28, 20 20" fill="#c26d3f"/>
            <path d="M42 38 C52 18, 62 18, 70 38 C62 46, 52 46, 42 38" fill="#f5c49b"/>
            <rect x="90" y="30" width="80" height="8" rx="4" fill="#c26d3f"/>
            <rect x="90" y="46" width="60" height="8" rx="4" fill="#f5c49b"/>
          </g></svg>"""
    },
    "Invierno ❄️": {
        "bg1": "#eef7ff", "bg2": "#e8eaff",
        "card": "rgba(255,255,255,0.92)", "accent": "#5aa9ff", "accent2": "#cfe3ff",
        "text": "#111", "chip": "#e6f1ff",
        "svg": """<svg viewBox="0 0 220 90" xmlns="http://www.w3.org/2000/svg">
          <rect rx="16" width="220" height="90" fill="#e6f1ff"/>
          <g transform="translate(22,18)" fill="#5aa9ff">
            <circle cx="18" cy="18" r="6"/><circle cx="32" cy="12" r="4"/><circle cx="28" cy="28" r="5"/>
            <rect x="90" y="30" width="80" height="8" rx="4" fill="#5aa9ff"/>
            <rect x="90" y="46" width="60" height="8" rx="4" fill="#cfe3ff"/>
          </g></svg>"""
    },
}

with st.sidebar:
    st.markdown("## 🎨 Estación")
    season = st.selectbox("Elige el tema", list(SEASONS.keys()), index=0)
S = SEASONS[season]

# Estilos que sí aplican a Streamlit
st.markdown(f"""
<style>
[data-testid="stAppViewContainer"] {{
  background: linear-gradient(180deg, {S['bg1']} 0%, {S['bg2']} 100%) !important;
}}
[data-testid="stSidebarContent"] {{
  background: {S['card']}; border: 2px solid {S['accent2']};
  border-radius: 18px; padding: .8rem; box-shadow: 0 10px 28px rgba(0,0,0,.06);
}}
h1,h2,h3,label,p,span,div {{ color: {S['text']} !important; }}
.card {{
  background: {S['card']}; border: 2px solid {S['accent2']};
  border-radius: 22px; padding: 1rem 1.1rem; backdrop-filter: blur(6px);
  box-shadow: 0 12px 30px rgba(0,0,0,.08);
}}
.chip {{
  display:inline-flex; align-items:center; gap:.4rem; padding:.35rem .7rem; border-radius:999px;
  background:{S['chip']}; border:1.5px solid {S['accent2']}; font-weight:700; font-size:.8rem;
  margin-right:.25rem;
}}
div.stButton>button {{
  background:{S['accent']}; color:#fff; border:none; border-radius:16px; padding:.6rem 1rem; font-weight:800;
  box-shadow:0 8px 16px rgba(0,0,0,.08); transition:transform .06s ease, filter .2s ease;
}}
div.stButton>button:hover {{ transform: translateY(-1px); filter: brightness(1.06); }}
.stTextArea textarea, .stTextInput input, .stSelectbox [data-baseweb="select"]>div {{
  border-radius:14px !important; border:2px solid {S['accent2']} !important;
}}
.progress {{
  width:100%; height:12px; border-radius:999px; background:#eee; overflow:hidden; border:2px solid {S['accent2']};
}}
.fill {{ height:100%; background:{S['accent']}; transition:width .4s ease; }}
</style>
""", unsafe_allow_html=True)

# ------------------------ Cabecera ------------------------
st.markdown(f"""
<div class="card" style="display:flex;gap:16px;align-items:center">
  <div>{S['svg']}</div>
  <div>
    <h1 style="margin:0">Demo de TF-IDF con Preguntas y Respuestas</h1>
    <div class="chip">🔎 Normalización + Stemming</div>
    <div class="chip">📚 Multi-documentos</div>
    <div class="chip">☁️ Estación: <b>{season}</b></div>
  </div>
</div>
""", unsafe_allow_html=True)

st.write("""
Cada línea se trata como un **documento** (frase, párrafo o texto largo).  
⚠️ **Trabajamos en inglés** (el *stemming* y las stopwords están configuradas para ese idioma).
""")

# ------------------------ Controles extra ------------------------
with st.expander("⚙️ Opciones avanzadas"):
    use_stem = st.toggle("Usar stemming Snowball (recommended)", value=True)
    ngram_max = st.selectbox("n-gramas (máx)", [1, 2], index=0)
    norm_lower = st.toggle("Normalizar a minúsculas", value=True)
    show_top_terms = st.toggle("Mostrar Top TF-IDF del documento más relevante", value=True)

# Sugerencias estacionales de pregunta
SUGGESTIONS = {
    "Primavera 🌸": ["What blooms?", "Who is planting?", "Which park has flowers?"],
    "Verano ☀️": ["Who is swimming?", "Where is the beach party?", "Who plays outside?"],
    "Otoño 🍂": ["Who is raking leaves?", "Which tree changes color?", "What falls in autumn?"],
    "Invierno ❄️": ["Who is skiing?", "Where does it snow?", "Who is playing with snow?"]
}

# ------------------------ Entrada ------------------------
default_docs = "The dog barks loudly.\nThe cat meows at night.\nThe dog and the cat play together."
text_input = st.text_area("Escribe tus documentos (uno por línea, en inglés):", default_docs, height=140)

question = st.text_input("Escribe una pregunta (en inglés):", SUGGESTIONS[season][0])
if st.button("Usar otra sugerencia"):
    # simplemente rota por la lista
    q_list = SUGGESTIONS[season]
    try:
        i = (q_list.index(question) + 1) % len(q_list)
    except ValueError:
        i = 0
    question = q_list[i]
    st.session_state["__question__"] = question
    st.experimental_rerun()

# ------------------------ Prepro + Tokenizer ------------------------
stemmer = SnowballStemmer("english")

def tokenize_and_stem(text: str):
    if norm_lower:
        text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [t for t in text.split() if len(t) > 1]
    if use_stem:
        tokens = [stemmer.stem(t) for t in tokens]
    return tokens

# ------------------------ Acción principal ------------------------
if st.button("Calcular TF-IDF y buscar respuesta"):
    documents = [d.strip() for d in text_input.split("\n") if d.strip()]
    if len(documents) < 1:
        st.warning("⚠️ Ingresa al menos un documento.")
    else:
        vectorizer = TfidfVectorizer(
            tokenizer=tokenize_and_stem,
            stop_words="english",
            token_pattern=None,
            ngram_range=(1, ngram_max)
        )
        X = vectorizer.fit_transform(documents)

        df_tfidf = pd.DataFrame(
            X.toarray(),
            columns=vectorizer.get_feature_names_out(),
            index=[f"Doc {i+1}" for i in range(len(documents))]
        )
        st.markdown("### 📊 Matriz TF-IDF")
        st.dataframe(df_tfidf.round(3), use_container_width=True)

        # Vector de la pregunta + similitud
        question_vec = vectorizer.transform([question])
        similarities = cosine_similarity(question_vec, X).flatten()
        best_idx = similarities.argmax()
        best_doc = documents[best_idx]
        best_score = float(similarities[best_idx])

        st.markdown("### 🤖 Pregunta y respuesta")
        st.write(f"**Tu pregunta:** {question}")
        st.write(f"**Documento más relevante (Doc {best_idx+1}):** {best_doc}")

        # Barra de similitud con color de estación
        pct = max(0.0, min(1.0, best_score))
        st.markdown(f'<div class="progress"><div class="fill" style="width:{pct*100:.1f}%"></div></div>', unsafe_allow_html=True)
        st.write(f"**Puntaje de similitud:** {best_score:.3f}")

        # Ranking de similitudes
        sim_df = pd.DataFrame({
            "Documento": [f"Doc {i+1}" for i in range(len(documents))],
            "Texto": documents,
            "Similitud": similarities
        }).sort_values("Similitud", ascending=False)
        st.markdown("### 🧭 Puntajes de similitud (ordenados)")
        st.dataframe(sim_df, use_container_width=True)

        # Stems de la pregunta presentes
        vocab = set(vectorizer.get_feature_names_out())
        q_stems = tokenize_and_stem(question)
        matched = [s for s in q_stems if s in vocab and df_tfidf.iloc[best_idx].get(s, 0) > 0]
        st.markdown("### 🔎 Stems/n-grams de la pregunta presentes en el documento elegido")
        if matched:
            st.write(", ".join(f"`{m}`" for m in matched))
        else:
            st.write("_No se hallaron coincidencias directas en el vocabulario._")

        # Top términos del doc ganador
        if show_top_terms:
            row = df_tfidf.iloc[best_idx]
            top_terms = row.sort_values(ascending=False).head(10)
            st.markdown("### 🏅 Top términos TF-IDF del documento ganador")
            st.dataframe(top_terms.reset_index().rename(columns={"index": "Término", best_idx: "TF-IDF"}).round(3))

        # Descarga CSV
        csv = df_tfidf.round(6).to_csv().encode("utf-8")
        st.download_button("⬇️ Descargar matriz TF-IDF (CSV)", data=csv, file_name="tfidf_matrix.csv", mime="text/csv")

        # Efectos según estación + score
        if season == "Invierno ❄️" and best_score >= 0.5:
            st.snow()
        if season == "Verano ☀️" and best_score >= 0.7:



