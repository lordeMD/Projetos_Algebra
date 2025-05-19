import pandas as pd
import numpy as np
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from math import acos, degrees
from scipy.sparse import csr_matrix

# Downloads necessários
nltk.download('punkt')
nltk.download('stopwords')

# Função de pré-processamento de texto
def tratamento_dados(texto):
    if not isinstance(texto, str):
        return ''
    texto = texto.lower()
    texto = texto.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(texto)
    stop_words = set(stopwords.words('english'))
    tokens = [palavra for palavra in tokens if palavra not in stop_words]
    return ' '.join(tokens)

# Função de similaridade do cosseno para matrizes esparsas
def cosine_sim_sparse(v1, v2):
    numerador = v1.multiply(v2).sum()
    denominador = np.sqrt(v1.multiply(v1).sum()) * np.sqrt(v2.multiply(v2).sum())
    if denominador == 0:
        return 0.0
    return numerador / denominador

# Leitura do dataset
arquivo = r"C:\Users\Antônio\Downloads\data\data.csv"
df = pd.read_csv(arquivo)

# Preenchendo valores nulos
df['description'] = df['description'].fillna('')
df['tags'] = df['tags'].fillna('')

# Tratamento separado
df['desc_tratada'] = df['description'].apply(tratamento_dados)
df['tags_tratadas'] = df['tags'].apply(tratamento_dados)

# Vetorização com limitação de vocabulário
vectorizer_desc = TfidfVectorizer(max_features=10000)
vectorizer_tags = TfidfVectorizer(max_features=5000)

tfidf_desc = vectorizer_desc.fit_transform(df['desc_tratada'])
tfidf_tags = vectorizer_tags.fit_transform(df['tags_tratadas'])

# Função de recomendação com cosseno manual separado
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
        sim_total = 0.7*sim_desc + 0.3*sim_tags
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

# Exemplo de uso
#entrada = "Moriarty the Patriot"
entrada = input("Digite o nome de um manga: ")
recomendacoes = recomenda_top3_dividido(entrada, df, tfidf_desc, tfidf_tags)

print(f"\n  Recomendações similares a: {entrada}\n")
for i, res in enumerate(recomendacoes, 1):
    print(f"{i}. Titulo - {res['title']}")
    print(f"     Ângulo: {res['angulo']}° | Similaridade: {res['similaridade']}\n")
