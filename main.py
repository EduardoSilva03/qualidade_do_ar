import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

warnings.filterwarnings("ignore")

ARQUIVO_CSV = 'AirQualityUCI.csv'

ALVOS_PARA_PREVER = [
    'CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)', 'PT08.S2(NMHC)', 
    'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 
    'T', 'RH', 'AH'
]

DICIONARIO_UNIDADES = {
    'T': 'Temperatura (°C)',
    'RH': 'Umidade Relativa (%)',
    'AH': 'Umidade Absoluta',
    'CO(GT)': 'Concentração de CO (mg/m³)',
    'NO2(GT)': 'Concentração de NO2 (microg/m³)',
    'C6H6(GT)': 'Concentração de Benzeno (microg/m³)',
    'NMHC(GT)': 'Hidrocarbonetos (microg/m³)',
    'NOx(GT)': 'Óxidos de Nitrogênio (ppb)',
    'PT08.S1(CO)': 'Resposta do Sensor (CO)',
    'PT08.S2(NMHC)': 'Resposta do Sensor (NMHC)',
    'PT08.S3(NOx)': 'Resposta do Sensor (NOx)',
    'PT08.S4(NO2)': 'Resposta do Sensor (NO2)',
    'PT08.S5(O3)': 'Resposta do Sensor (O3)'
}

def carregar_dados_completo(caminho):
    print(f"--- Carregando Dataset: {caminho} ---")
    df = pd.read_csv(caminho, sep=';', decimal=',')
    
    df.dropna(how='all', axis=1, inplace=True)
    df.dropna(how='any', inplace=True)

    df['Timestamp'] = pd.to_datetime(
        df['Date'] + ' ' + df['Time'].astype(str).str.replace('.', ':', regex=False),
        format='%d/%m/%Y %H:%M:%S'
    )
    df.set_index('Timestamp', inplace=True)
    df.drop(['Date', 'Time'], axis=1, inplace=True)

    df.replace(-200, np.nan, inplace=True)
    df.interpolate(method='time', inplace=True)
    df.fillna(method='bfill', inplace=True)
    
    df = df.sort_index()
    
    print(f"    Período: {df.index.min()} até {df.index.max()}")
    return df

def engenharia_features(df):
    df_feat = df.copy()
    df_feat['Hora'] = df_feat.index.hour
    df_feat['DiaSemana'] = df_feat.index.dayofweek
    df_feat['Mes'] = df_feat.index.month
    return df_feat

def processar_periodo_completo(df_base, alvo):
    print(f"\n>>> Analisando: {alvo}")
    
    df_atual = df_base.copy()
    
    if alvo not in df_atual.columns:
        print(f"    AVISO: Coluna '{alvo}' não existe.")
        return None, None

    df_atual['Lag_1h'] = df_atual[alvo].shift(1)
    df_atual['Lag_24h'] = df_atual[alvo].shift(24)
    df_atual.dropna(inplace=True)
    
    if df_atual.empty:
        print(f"    ERRO CRÍTICO: Sem dados válidos para '{alvo}'.")
        return None, None
    
    features = [c for c in df_atual.columns if c != alvo]
    X = df_atual[features]
    y = df_atual[alvo]
    
    model = RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=42)
    model.fit(X, y)
    
    previsoes = model.predict(X)
    mae = mean_absolute_error(y, previsoes)
    print(f"    Erro Médio (MAE): {mae:.2f}")
    
    return y, previsoes

def plotar_grafico_full(y_real, y_pred, alvo):
    plt.close('all')
    
    plt.figure(figsize=(16, 6))
    
    plt.plot(y_real.index, y_real.values, label='Real (Histórico)', color='gray', alpha=0.5, linewidth=0.8)
    plt.plot(y_real.index, y_pred, label='Modelo Ajustado', color='#ff4500', alpha=0.7, linewidth=0.8)
    
    data_ini = y_real.index.min().strftime('%d/%m/%Y')
    data_fim = y_real.index.max().strftime('%d/%m/%Y')
    
    plt.title(f'Análise Completa: {alvo} | De {data_ini} até {data_fim}')
    plt.xlabel('Linha do Tempo (Data)')
    
    label_y = DICIONARIO_UNIDADES.get(alvo, 'Valor / Intensidade')
    plt.ylabel(label_y)
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    nome_limpo = alvo.replace('(', '').replace(')', '').replace('.', '')
    nome_arquivo = f"FullRange_{nome_limpo}.png"
    
    plt.savefig(nome_arquivo, dpi=150)
    print(f"    Gráfico salvo: {nome_arquivo}")
    
    plt.close()

if __name__ == "__main__":
    try:
        df_full = carregar_dados_completo(ARQUIVO_CSV)
        df_proc = engenharia_features(df_full)
        
        for alvo in ALVOS_PARA_PREVER:
            y_real, y_pred = processar_periodo_completo(df_proc, alvo)
            
            if y_real is not None:
                plotar_grafico_full(y_real, y_pred, alvo)
                
        print(f"\n--- SUCESSO! Gráficos gerados com as novas legendas. ---")
        
    except FileNotFoundError:
        print(f"ERRO: O arquivo '{ARQUIVO_CSV}' não foi encontrado.")
    except Exception as e:
        print(f"ERRO GERAL: {e}")