import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import plotly.graph_objects as go
import numpy as np
import re



# Cargar los datos
df = pd.read_csv("tabla_unida.csv")


df_violin = df.dropna(subset=[df.columns[3], df.columns[4]])
df_violin['Horas'] = pd.to_numeric(df_violin.iloc[:, 4].str.extract('(\d+)')[0], errors='coerce')
df_violin = df_violin.assign(Género=df_violin.iloc[:, 3].str.split(',')).explode('Género')
df_violin['Género'] = df_violin['Género'].str.strip()



# --- 5. GRÁFICO RADIAL: POPULARIDAD DE GÉNEROS ---
df_radial = df.dropna(subset=[df.columns[3]])
df_radial = df_radial.assign(Género=df_radial.iloc[:, 3].str.split(',')).explode('Género')
df_radial['Género'] = df_radial['Género'].str.strip()

conteo_gen = df_radial['Género'].value_counts()
categorias = conteo_gen.index
valores = conteo_gen.values

theta = np.linspace(0, 2 * np.pi, len(valores), endpoint=False)
valores = np.concatenate((valores, [valores[0]]))
theta = np.concatenate((theta, [theta[0]]))

plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)
ax.plot(theta, valores, 'o-', linewidth=2)
ax.fill(theta, valores, alpha=0.25)
ax.set_thetagrids(theta[:-1] * 180/np.pi, categorias)
plt.title("Popularidad de géneros (gráfico radial)")
plt.show()


from itertools import combinations

df_gen = df.dropna(subset=[df.columns[3]])
df_gen['Géneros'] = df_gen.iloc[:, 3].str.split(',').apply(lambda x: [i.strip() for i in x])
combs = []

for genres in df_gen['Géneros']:
    combs.extend(combinations(sorted(set(genres)), 2))

cooc = Counter(combs)
all_genres = sorted(set([g for pair in cooc for g in pair]))

matrix = pd.DataFrame(0, index=all_genres, columns=all_genres)
for (g1, g2), count in cooc.items():
    matrix.loc[g1, g2] = count
    matrix.loc[g2, g1] = count  # simétrico

df_corr = df.dropna(subset=[df.columns[1], df.columns[3]])
df_corr = df_corr.assign(
    Dispositivo=df_corr.iloc[:, 1].str.split(','),
    Genero=df_corr.iloc[:, 3].str.split(',')
)
df_corr = df_corr.explode('Dispositivo').explode('Genero')
df_corr['Dispositivo'] = df_corr['Dispositivo'].str.strip()
df_corr['Genero'] = df_corr['Genero'].str.strip()

tabla = pd.crosstab(df_corr['Dispositivo'], df_corr['Genero'])

# Mapa de calor
plt.figure(figsize=(12, 8))
sns.heatmap(tabla, cmap="YlGnBu", annot=False, cbar_kws={'label': 'Frecuencia'})
plt.title("Mapa de calor: Dispositivo vs Género")
plt.xlabel("Género")
plt.ylabel("Dispositivo")
plt.tight_layout()
plt.show()


