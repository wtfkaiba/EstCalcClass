# calculator.py
import pandas as pd
import numpy as np

def calcular_estatisticas(lim_inf_primeira, amplitude, qtd_classes, frequencias):
    # ... (todo o código da função como está no app.py) ...
    if not frequencias:
        # Retorna um DataFrame vazio e estatísticas NaN se frequências estiverem vazias
        # para manter a consistência do tipo de retorno para os testes.
        cols = ['Limite Inferior (LI)', 'Limite Superior (LS)', 'Ponto Médio (xi)', 
                'Frequência Absoluta (fi)', 'Frequência Acumulada (Fi)']
        df_empty = pd.DataFrame(columns=cols)
        estatisticas_nan = {
            "Média": np.nan, "Variância": np.nan, "Desvio Padrão": np.nan,
            "Coeficiente de Variação (%)": np.nan, "Mediana": np.nan,
            "Moda Bruta (Ponto Médio da Classe Modal)": np.nan, "Moda de Czuber": np.nan
        }
        return df_empty, estatisticas_nan # Ou pode retornar (None, "mensagem de erro")
                                         # mas para testes, ter uma estrutura consistente é melhor

    if not all(isinstance(f, (int, float)) for f in frequencias):
        # Similar ao de cima, ou levantar um ValueError
        cols = ['Limite Inferior (LI)', 'Limite Superior (LS)', 'Ponto Médio (xi)', 
                'Frequência Absoluta (fi)', 'Frequência Acumulada (Fi)']
        df_empty = pd.DataFrame(columns=cols)
        estatisticas_nan = {
            "Média": np.nan, "Variância": np.nan, "Desvio Padrão": np.nan,
            "Coeficiente de Variação (%)": np.nan, "Mediana": np.nan,
            "Moda Bruta (Ponto Médio da Classe Modal)": np.nan, "Moda de Czuber": np.nan
        }
        return df_empty, estatisticas_nan # ou (None, "mensagem de erro")


    classes_lim_inf = []
    classes_lim_sup = []
    classes_ponto_medio = []
    for i in range(qtd_classes):
        li = lim_inf_primeira + (i * amplitude)
        ls = li + amplitude
        pm = (li + ls) / 2
        classes_lim_inf.append(li)
        classes_lim_sup.append(ls)
        classes_ponto_medio.append(pm)

    df = pd.DataFrame({
        'Limite Inferior (LI)': classes_lim_inf,
        'Limite Superior (LS)': classes_lim_sup,
        'Ponto Médio (xi)': classes_ponto_medio,
        'Frequência Absoluta (fi)': frequencias
    })
    df['Frequência Acumulada (Fi)'] = df['Frequência Absoluta (fi)'].cumsum()

    N = df['Frequência Absoluta (fi)'].sum()
    estatisticas_base = {
        "Média": np.nan, "Variância": np.nan, "Desvio Padrão": np.nan,
        "Coeficiente de Variação (%)": np.nan, "Mediana": np.nan,
        "Moda Bruta (Ponto Médio da Classe Modal)": np.nan, "Moda de Czuber": np.nan
    }
    if N == 0:
        return df, estatisticas_base # df ainda é útil para mostrar a estrutura de classes

    soma_xi_fi = (df['Ponto Médio (xi)'] * df['Frequência Absoluta (fi)']).sum()
    media = soma_xi_fi / N
    soma_desvios_quad_ponderados = (((df['Ponto Médio (xi)'] - media)**2) * df['Frequência Absoluta (fi)']).sum()
    variancia = soma_desvios_quad_ponderados / (N - 1) if N > 1 else 0.0
    desvio_padrao = np.sqrt(variancia)
    cv = (desvio_padrao / abs(media)) * 100 if media != 0 else np.nan

    pos_mediana = N / 2
    classe_mediana_idx_series = df[df['Frequência Acumulada (Fi)'] >= pos_mediana].index
    if classe_mediana_idx_series.empty: 
        mediana = np.nan # Caso N > 0 mas nenhuma classe atinge pos_mediana (improvável com cumsum)
    else:
        classe_mediana_idx = classe_mediana_idx_series[0]
        L_md = df.loc[classe_mediana_idx, 'Limite Inferior (LI)']
        f_md = df.loc[classe_mediana_idx, 'Frequência Absoluta (fi)']
        Fi_ant_md = df.loc[classe_mediana_idx - 1, 'Frequência Acumulada (Fi)'] if classe_mediana_idx > 0 else 0
        h = amplitude
        mediana = L_md + ((pos_mediana - Fi_ant_md) / f_md) * h if f_md != 0 else L_md

    idx_moda_bruta = df['Frequência Absoluta (fi)'].idxmax() # idxmax pode dar erro se todas as freqs forem iguais e mínimas
                                                           # ou se df estiver vazio, mas N=0 já trata df vazio.
                                                           # Se todas as freqs forem iguais, pega o primeiro.
    moda_bruta = df.loc[idx_moda_bruta, 'Ponto Médio (xi)']
    classe_modal_idx = idx_moda_bruta

    L_mo = df.loc[classe_modal_idx, 'Limite Inferior (LI)']
    f_mo = df.loc[classe_modal_idx, 'Frequência Absoluta (fi)']
    f_ant = df.loc[classe_modal_idx - 1, 'Frequência Absoluta (fi)'] if classe_modal_idx > 0 else 0
    f_pos = df.loc[classe_modal_idx + 1, 'Frequência Absoluta (fi)'] if classe_modal_idx < (qtd_classes - 1) else 0
    
    delta1 = f_mo - f_ant
    delta2 = f_mo - f_pos
    
    if (delta1 + delta2) == 0:
        # Se f_mo for 0 e f_ant/f_pos também, ainda será 0.
        # Ou se f_mo > 0 mas f_ant = f_mo e f_pos = f_mo (platô).
        moda_czuber = L_mo + (amplitude / 2) 
    else:
        moda_czuber = L_mo + (delta1 / (delta1 + delta2)) * amplitude
        moda_czuber = max(L_mo, min(moda_czuber, L_mo + amplitude))

    estatisticas_calculadas = {
        "Média": media, "Variância": variancia, "Desvio Padrão": desvio_padrao,
        "Coeficiente de Variação (%)": cv, "Mediana": mediana,
        "Moda Bruta (Ponto Médio da Classe Modal)": moda_bruta, "Moda de Czuber": moda_czuber
    }
    return df, estatisticas_calculadas