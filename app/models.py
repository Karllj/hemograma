# treino.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Obtém o diretório do script
script_dir = os.path.dirname(__file__)

# Carrega o conjunto de dados
dados_path = r"C:\Users\carlos.pereira\PycharmProjects\hemogram\modelo\dados.csv"
df = pd.read_csv(dados_path)

# Correção: substituir vírgulas por pontos e converter para float
df.replace(',', '.', regex=True, inplace=True)
df = df.apply(pd.to_numeric, errors='ignore')

# Dividir o conjunto de dados em features (X) e rótulos (y)
X = df.drop('Diagnóstico', axis=1)
y = df['Diagnóstico']

# Codificar rótulos categóricos
le = LabelEncoder()
y = le.fit_transform(y)

# Correção: lidar com variáveis categóricas usando pd.get_dummies()
X = pd.get_dummies(X)

# Dividir o conjunto de dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir colunas categóricas e numéricas
numeric_features = X.select_dtypes(include=['float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Criar transformers para aplicar imputação em valores numéricos e categóricos separadamente
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
])

# Combinar transformers em um pré-processador
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
    ])

# Adicionar o pré-processador ao pipeline do modelo
modelo = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', RandomForestClassifier())])

# Treinar o modelo
modelo.fit(X_train, y_train)

# Avaliar o desempenho do modelo nos dados de teste, se desejar
# (adicione código de avaliação aqui, se necessário)

# Cria o diretório 'modelo' se não existir
modelo_dir = os.path.join(script_dir, 'modelo')
if not os.path.exists(modelo_dir):
    os.makedirs(modelo_dir)

# Salvar o LabelEncoder treinado em um arquivo pkl
le_path = os.path.join(modelo_dir, 'label_encoder.pkl')
joblib.dump(le, le_path)

# Salvar o modelo treinado em um arquivo pkl
modelo_path = os.path.join(modelo_dir, 'modelo_treinado.pkl')
joblib.dump(modelo, modelo_path)

# Verificar se o modelo foi salvo corretamente
try:
    # Carregar o modelo do arquivo pkl
    modelo_carregado = joblib.load(modelo_path)

    # Imprimir algumas informações sobre o modelo
    print("Modelo carregado com sucesso!")
    print("Parâmetros do modelo:")
    print(modelo_carregado.named_steps['classifier'].get_params())
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")
