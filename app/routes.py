from flask import request, jsonify, Blueprint
from app import app
import joblib
import pandas as pd

# Criar o Blueprint
app = Blueprint('app', __name__)

# Carregar o modelo treinado
modelo_path = "C:\\Users\\carlos.pereira\\PycharmProjects\\hemogram\\modelo\\modelo\\modelo_treinado.pkl"
modelo = joblib.load(modelo_path)

# Carregar o LabelEncoder usado no treinamento
le_path = "C:\\Users\\carlos.pereira\\PycharmProjects\\hemogram\\modelo\\modelo\\label_encoder.pkl"
le = joblib.load(le_path)

@app.route('/prever_diagnostico', methods=['POST'])
def prever_diagnostico():
    try:
        # Obter dados da solicitação POST em formato JSON
        dados_json = request.get_json()

        # Extrair dados do JSON, assumindo que as chaves correspondem aos nomes das colunas
        eritrocitos = dados_json['Eritrócitos (10/µL)']
        hemoglobina = dados_json['Hemoglobina (g/dL)']
        hematocrito = dados_json['Hematócrito (%)']
        hcm = dados_json['HCM']
        vgm = dados_json['VGM (fL)']
        chgm = dados_json['CHGM(%)']
        metarrubricitos = dados_json['Metarrubrícitos']
        proteina_plasmatica = dados_json['Proteína Plasmática']
        leucocitos = dados_json['Leucócitos (/µL)']
        leucograma = dados_json['Leucograma']
        segmentados = dados_json['Segmentados (/µL)']
        bastonetes = dados_json['Bastonetes (/µL)']
        blastos = dados_json['Blastos']
        metamielocitos = dados_json['Metamielócitos (/µL)']
        mielocitos = dados_json['Mielócitos (/µL)']
        linfocitos = dados_json['Linfócitos (/µL)']
        monocitos = dados_json['Monócitos (/µL)']
        eosinofilos = dados_json['Eosinófilos (/µL)']
        basofilos = dados_json['Basófilos (/µL)']
        plaquetas = dados_json['Plaquetas (/µL)']

        # Criar um DataFrame com os dados
        dados = pd.DataFrame({
            'Eritrócitos (10/µL)': [eritrocitos],
            'Hemoglobina (g/dL)': [hemoglobina],
            'Hematócrito (%)': [hematocrito],
            'HCM': [hcm],
            'VGM (fL)': [vgm],
            'CHGM(%)': [chgm],
            'Metarrubrícitos': [metarrubricitos],
            'Proteína Plasmática': [proteina_plasmatica],
            'Leucócitos (/µL)': [leucocitos],
            'Leucograma': [leucograma],
            'Segmentados (/µL)': [segmentados],
            'Bastonetes (/µL)': [bastonetes],
            'Blastos': [blastos],
            'Metamielócitos (/µL)': [metamielocitos],
            'Mielócitos (/µL)': [mielocitos],
            'Linfócitos (/µL)': [linfocitos],
            'Monócitos (/µL)': [monocitos],
            'Eosinófilos (/µL)': [eosinofilos],
            'Basófilos (/µL)': [basofilos],
            'Plaquetas (/µL)': [plaquetas],
        })

        # Fazer previsão usando o modelo
        previsao = modelo.predict(dados)

        # Obter a classe prevista (se necessário)
        classe_prevista = le.inverse_transform(previsao)[0]

        # Retornar o resultado
        return jsonify({"diagnostico_previsto": classe_prevista})

    except Exception as e:
        return jsonify({"erro": str(e)})
