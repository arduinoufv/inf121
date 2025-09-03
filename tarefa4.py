#@title Exploração de Dados - Predição de Cultura
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configurações visuais
sns.set(style="whitegrid", palette="muted", color_codes=True)

# Ler CSV
#df = pd.read_csv("predicao_cultura.csv")
# Harp
df = pd.read_csv("/home/inf121/pratica/predicao_cultura.csv")


# Informações básicas
print("📌 Número de amostras:", df.shape[0])
print("📌 Número de atributos:", df.shape[1])
print("\n🔹 Primeiras linhas do dataset:")
display(df.head())

# Identificar colunas numéricas e categóricas
num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
cat_cols = df.select_dtypes(include=['object']).columns.tolist()

print("\n📊 Colunas Numéricas:", num_cols)
print("📊 Colunas Categóricas:", cat_cols)

# -----------------------------
# Distribuição da coluna target
# -----------------------------
plt.figure(figsize=(8,5))
target_counts = df['target'].value_counts(normalize=True) * 100
sns.barplot(x=target_counts.index, y=target_counts.values, palette="viridis")
plt.ylabel("Percentual (%)")
plt.title("Distribuição da coluna Target (%)")
plt.xticks(rotation=45)
plt.show()

# -----------------------------
# Boxplots das colunas numéricas
# -----------------------------
for col in num_cols:
    plt.figure(figsize=(8,5))
    sns.boxplot(x="target", y=col, data=df, palette="Set2")
    plt.title(f"Boxplot de {col} por classe target")
    plt.xticks(rotation=45)
    plt.show()

# -----------------------------
# Histogramas das colunas numéricas
# -----------------------------
df[num_cols].hist(bins=20, figsize=(15,10), layout=(len(num_cols)//3+1, 3))
plt.suptitle("Distribuição das variáveis numéricas", fontsize=16)
plt.show()

# -----------------------------
# Gráficos de barras para colunas categóricas (exceto target)
# -----------------------------
for col in [c for c in cat_cols if c != 'target']:
    plt.figure(figsize=(8,5))
    sns.countplot(x=col, data=df, palette="coolwarm")
    plt.title(f"Distribuição da coluna categórica: {col}")
    plt.xticks(rotation=45)
    plt.show()
