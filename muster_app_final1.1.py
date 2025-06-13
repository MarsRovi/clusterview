# Mustererkennungs-Web-App (Streamlit) mit Upload, Clustering, Zeitreihen- & Netzwerkmodul

import streamlit as st
import subprocess
import importlib.util
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import io
from datetime import datetime

# Sicheres Laden des deutschen spaCy-Modells
def load_spacy_model(model_name):
    if importlib.util.find_spec(model_name) is None:
        subprocess.run(["python", "-m", "spacy", "download", model_name])
    return spacy.load(model_name)

nlp = load_spacy_model("de_core_news_sm")
model = SentenceTransformer("distiluse-base-multilingual-cased")

# Titel
st.title("📊 Neutrale Mustererkennung in Texten und Daten")
st.write("Upload von Textdateien oder CSV-Zeitreihen zur strukturierten Mustererkennung ohne Bewertung.")

# Upload-Funktion
uploaded_file = st.file_uploader("Wähle eine Textdatei (.txt) oder CSV-Zeitreihe:", type=["txt", "csv"])

if uploaded_file:
    file_type = uploaded_file.name.split(".")[-1].lower()

    if file_type == "txt":
        text = uploaded_file.read().decode("utf-8")
        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 20]
        embeddings = model.encode(sentences)

        # Clustering
        n_clusters = st.slider("Anzahl Themencluster", 2, 10, 3)
        clustering = KMeans(n_clusters=n_clusters, random_state=0)
        labels = clustering.fit_predict(embeddings)

        # Cluster-Anzeige
        for i in range(n_clusters):
            st.subheader(f"🧩 Cluster {i + 1}")
            for j, s in enumerate(sentences):
                if labels[j] == i:
                    st.write("-", s)

        # Netzwerk-Visualisierung
        st.subheader("🔗 Semantisches Netzwerk (Co-Cluster)")
        G = nx.Graph()
        for i in range(len(sentences)):
            for j in range(i+1, len(sentences)):
                if labels[i] == labels[j]:
                    G.add_edge(f"Satz {i+1}", f"Satz {j+1}")
        fig, ax = plt.subplots(figsize=(8, 6))
        nx.draw_networkx(G, ax=ax, with_labels=True, node_size=300, font_size=8)
        st.pyplot(fig)

    elif file_type == "csv":
        df = pd.read_csv(uploaded_file)
        st.write("Vorschau der Daten:", df.head())

        # Annahme: erste Spalte = Datum, zweite Spalte = Wert
        df.columns = ["Datum", "Wert"]
        df["Datum"] = pd.to_datetime(df["Datum"], errors="coerce")
        df = df.dropna()
        df = df.sort_values("Datum")

        # Zeitreihen-Plot
        st.subheader("📈 Zeitverlauf der Werte")
        fig, ax = plt.subplots()
        ax.plot(df["Datum"], df["Wert"], marker="o")
        ax.set_xlabel("Datum")
        ax.set_ylabel("Wert")
        ax.set_title("Zeitreihe")
        st.pyplot(fig)

        # Einfache Mustererkennung: Peaks & Anomalien
        st.subheader("🔍 Einfache Muster (Anstieg, Rückgang)")
        df["Differenz"] = df["Wert"].diff()
        df["Muster"] = df["Differenz"].apply(lambda x: "📈 Anstieg" if x > 0 else ("📉 Rückgang" if x < 0 else "➖ Gleich"))
        st.write(df[["Datum", "Wert", "Muster"]].tail(10))

else:
    st.info("Bitte lade eine .txt-Datei (Text) oder .csv-Datei (Zeitreihe) hoch, um zu starten.")
