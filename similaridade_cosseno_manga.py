import sys
import subprocess

def instalar(pacote):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pacote])

try:
    import nltk
except ImportError:
    print("Pacote 'requests' não encontrado. Instalando automaticamente...")
    instalar("nltk")
    import nltk

import pandas as pd
import numpy as np
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from math import acos, degrees

nltk.download('punkt')
nltk.download('stopwords')

def tratamento_dados(texto):
    if not isinstance(texto, str):
        return ''
    texto = texto.lower()
    texto = texto.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(texto)
    stop_words = set(stopwords.words('english'))
    tokens = [palavra for palavra in tokens if palavra not in stop_words]
    return ' '.join(tokens)

def cosine_sim_sparse(v1, v2):
    numerador = v1.multiply(v2).sum()
    denominador = np.sqrt(v1.multiply(v1).sum()) * np.sqrt(v2.multiply(v2).sum())
    if denominador == 0:
        return 0.0
    return numerador / denominador

df = pd.read_csv("data.csv")

df['description'] = df['description'].fillna('')
df['tags'] = df['tags'].fillna('')

df['desc_tratada'] = df['description'].apply(tratamento_dados)
df['tags_tratadas'] = df['tags'].apply(tratamento_dados)

vectorizer_desc = TfidfVectorizer(max_features=10000)
vectorizer_tags = TfidfVectorizer(max_features=5000)

tfidf_desc = vectorizer_desc.fit_transform(df['desc_tratada'])
tfidf_tags = vectorizer_tags.fit_transform(df['tags_tratadas'])

def recomenda_top3_dividido(nome_obra, df, tfidf_desc, tfidf_tags):
    if nome_obra not in df['title'].values:
        return "Obra não encontrada."

    idx_entrada = df[df['title'] == nome_obra].index[0]
    vec_desc_input = tfidf_desc[idx_entrada]
    vec_tags_input = tfidf_tags[idx_entrada]

    similaridades = []

    for i in range(len(df)):
        if i == idx_entrada:
            continue
        sim_desc = cosine_sim_sparse(vec_desc_input, tfidf_desc[i])
        sim_tags = cosine_sim_sparse(vec_tags_input, tfidf_tags[i])
        sim_total = 0.7 * sim_desc + 0.3 * sim_tags
        angulo = degrees(acos(min(max(sim_total, -1.0), 1.0)))
        similaridades.append((i, sim_total, angulo))

    top3 = sorted(similaridades, key=lambda x: x[1], reverse=True)[:3]

    resultados = []
    for idx, sim, angulo in top3:
        resultados.append({
            "title": df.at[idx, 'title'],
            "similaridade": round(sim, 4),
            "angulo": round(angulo, 2)
        })

    return resultados

entrada = input("Digite o nome de uma obra: ")
recomendacoes = recomenda_top3_dividido(entrada, df, tfidf_desc, tfidf_tags)

if isinstance(recomendacoes, str):
    print(recomendacoes)
else:
    print(f"\n  Recomendações similares a: {entrada}\n")
    for i, res in enumerate(recomendacoes, 1):
        print(f"{i}. Título - {res['title']}")
        print(f"     Ângulo: {res['angulo']}° | Similaridade: {res['similaridade']}\n")
