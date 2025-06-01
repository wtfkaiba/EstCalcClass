import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- Color Palette for Dark Theme ---
DARK_BACKGROUND_COLOR = '#1e2A38'
PLOT_BACKGROUND_COLOR = '#283442'
TEXT_COLOR_DARK_THEME = '#EAEAEA'
GRID_COLOR_DARK_THEME = 'rgba(180, 180, 180, 0.15)'
BAR_COLOR_HISTOGRAM_DEFAULT = '#3498db'
BAR_COLOR_HISTOGRAM_MODAL = '#e74c3c'
BAR_BORDER_COLOR_MODAL = '#c0392b'
HOVER_BG_COLOR_HISTOGRAM = '#15202b'

# --- Funções de Cálculo ---
def calcular_estatisticas(lim_inf_primeira, amplitude, qtd_classes, frequencias):
    if not frequencias:
        return None, "A lista de frequências está vazia. Por favor, insira valores."
    if not all(isinstance(f, (int, float)) and f >= 0 for f in frequencias): # Frequências devem ser numéricas e não negativas
        return None, "Todas as frequências devem ser números positivos ou zero."
    if len(frequencias) != qtd_classes:
        return None, f"Esperava {qtd_classes} frequências, mas recebeu {len(frequencias)}."


    classes_lim_inf, classes_lim_sup, classes_ponto_medio = [], [], []
    for i in range(qtd_classes):
        li = float(lim_inf_primeira + (i * amplitude)) # Garantir float para consistência
        ls = float(li + amplitude)
        pm = float((li + ls) / 2)
        classes_lim_inf.append(li)
        classes_lim_sup.append(ls)
        classes_ponto_medio.append(pm)
    
    df = pd.DataFrame({
        'Limite Inferior (LI)': classes_lim_inf, 
        'Limite Superior (LS)': classes_lim_sup,
        'Ponto Médio (xi)': classes_ponto_medio, 
        'Frequência Absoluta (fi)': frequencias
    })
    df['Frequência Acumulada (Fi)'] = df['Frequência Absoluta (fi)'].astype(float).cumsum() # Assegurar float para cumsum

    N = df['Frequência Absoluta (fi)'].sum()

    # Inicializar todas as chaves que a interface espera
    estatisticas_resultados = {
        "Média": np.nan, "Variância": np.nan, "Desvio Padrão": np.nan,
        "Coeficiente de Variação (%)": np.nan, "Mediana": np.nan,
        "Classificação Modal": "Amodal", # Default se N=0 ou não houver moda clara
        "Modas Brutas": [],
        "Modas de Czuber": [],
        "Índices das Classes Modais": []
    }

    if N == 0:
        return df, estatisticas_resultados # Retorna df para tabela, mas estatísticas NaN/Amodal

    # Média, Variância, DP, CV, Mediana
    soma_xi_fi = (df['Ponto Médio (xi)'] * df['Frequência Absoluta (fi)']).sum()
    media = soma_xi_fi / N
    estatisticas_resultados["Média"] = float(media)

    soma_desvios_quad_ponderados = (((df['Ponto Médio (xi)'] - media)**2) * df['Frequência Absoluta (fi)']).sum()
    variancia = soma_desvios_quad_ponderados / (N - 1) if N > 1 else 0.0
    estatisticas_resultados["Variância"] = float(variancia)

    desvio_padrao = np.sqrt(variancia) if variancia >= 0 else np.nan # Evitar erro com variância negativa (improvável aqui)
    estatisticas_resultados["Desvio Padrão"] = float(desvio_padrao) if not np.isnan(desvio_padrao) else np.nan

    cv = (desvio_padrao / abs(media)) * 100 if media != 0 and not np.isnan(desvio_padrao) else np.nan
    estatisticas_resultados["Coeficiente de Variação (%)"] = float(cv) if not np.isnan(cv) else np.nan
    
    pos_mediana = N / 2.0
    classe_mediana_idx_series = df[df['Frequência Acumulada (Fi)'] >= pos_mediana].index
    
    if not classe_mediana_idx_series.empty:
        classe_mediana_idx = classe_mediana_idx_series[0]
        L_md = df.loc[classe_mediana_idx, 'Limite Inferior (LI)']
        f_md = df.loc[classe_mediana_idx, 'Frequência Absoluta (fi)']
        Fi_ant_md = df.loc[classe_mediana_idx - 1, 'Frequência Acumulada (Fi)'] if classe_mediana_idx > 0 else 0.0
        h = float(amplitude)
        mediana_calc = L_md + ((pos_mediana - Fi_ant_md) / f_md) * h if f_md != 0 else L_md
        estatisticas_resultados["Mediana"] = float(mediana_calc)
    else:
        estatisticas_resultados["Mediana"] = np.nan


    # Lógica Modal Aprimorada
    frequencias_abs_series = df['Frequência Absoluta (fi)']
    max_freq = frequencias_abs_series.max()

    # Considerar amodal se todas as frequências forem iguais (e > 0)
    # ou se a frequência máxima for 0 (já coberto por N=0, mas é uma boa verificação)
    if max_freq == 0 or (len(frequencias_abs_series.unique()) == 1 and N > 0) :
        estatisticas_resultados["Classificação Modal"] = "Amodal"
    else:
        indices_modais = frequencias_abs_series[frequencias_abs_series == max_freq].index.tolist()
        estatisticas_resultados["Índices das Classes Modais"] = indices_modais
        
        num_modas = len(indices_modais)
        if num_modas == 1: estatisticas_resultados["Classificação Modal"] = "Unimodal"
        elif num_modas == 2: estatisticas_resultados["Classificação Modal"] = "Bimodal"
        elif num_modas == 3: estatisticas_resultados["Classificação Modal"] = "Trimodal"
        elif num_modas == 4: estatisticas_resultados["Classificação Modal"] = "Tetramodal"
        elif num_modas == 5: estatisticas_resultados["Classificação Modal"] = "Pentamodal"
        else: estatisticas_resultados["Classificação Modal"] = "Hexamodal"

        modas_brutas_list = []
        modas_czuber_list = []

        for idx_modal_atual in indices_modais:
            modas_brutas_list.append(float(df.loc[idx_modal_atual, 'Ponto Médio (xi)']))

            L_mo_atual = df.loc[idx_modal_atual, 'Limite Inferior (LI)']
            f_mo_atual = df.loc[idx_modal_atual, 'Frequência Absoluta (fi)'] # É max_freq
            
            f_ant_atual = df.loc[idx_modal_atual - 1, 'Frequência Absoluta (fi)'] if idx_modal_atual > 0 else 0.0
            f_pos_atual = df.loc[idx_modal_atual + 1, 'Frequência Absoluta (fi)'] if idx_modal_atual < (qtd_classes - 1) else 0.0
            
            delta1_atual = f_mo_atual - f_ant_atual
            delta2_atual = f_mo_atual - f_pos_atual
            
            cz_mode_atual_calc = np.nan
            if (delta1_atual + delta2_atual) == 0:
                # Se f_mo_atual é 0, então é L_mo + amp/2, mas f_mo_atual não deveria ser 0 aqui
                # devido à condição max_freq > 0.
                # Isso acontece se for um platô perfeito ou isolada com f_ant=f_pos=f_mo_atual.
                # Ou se f_ant = f_mo e f_pos = f_mo
                cz_mode_atual_calc = L_mo_atual + (float(amplitude) / 2.0)
            else:
                # Garante que o denominador não é zero antes da divisão
                cz_mode_calc_temp = L_mo_atual + (delta1_atual / (delta1_atual + delta2_atual)) * float(amplitude)
                cz_mode_atual_calc = max(L_mo_atual, min(cz_mode_calc_temp, L_mo_atual + float(amplitude)))
            modas_czuber_list.append(float(cz_mode_atual_calc))
            
        estatisticas_resultados["Modas Brutas"] = modas_brutas_list
        estatisticas_resultados["Modas de Czuber"] = modas_czuber_list
        
    return df, estatisticas_resultados


# --- Interface Streamlit ---
st.set_page_config(layout="wide", page_title="Estatística Aplicada - Dados Agrupados")
st.markdown(f"""
<style>
    .reportview-container {{ background-color: {DARK_BACKGROUND_COLOR}; color: {TEXT_COLOR_DARK_THEME}; }} /* Deprecated */
    .main .block-container {{ padding-top: 2rem; padding-bottom: 2rem; }} 
    .sidebar .sidebar-content {{ background-color: {DARK_BACKGROUND_COLOR}; }}
    h1, h2, h3, h4, h5, h6 {{ color: {TEXT_COLOR_DARK_THEME}; }}
    .stMetric {{ background-color: {PLOT_BACKGROUND_COLOR} !important; border-radius: 7px; padding: 10px; }}
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {{ color: {TEXT_COLOR_DARK_THEME}; }}
    .stTabs [data-baseweb="tab-list"] button {{ background-color: transparent; }}
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {{ background-color: {PLOT_BACKGROUND_COLOR}; border-bottom: 3px solid {BAR_COLOR_HISTOGRAM_DEFAULT}; }}
    .modal-block {{ background-color: {PLOT_BACKGROUND_COLOR}; border-radius: 7px; padding: 15px; margin-bottom: 10px; border: 1px solid rgba(255,255,255,0.1); }}
    .modal-block .modal-title {{ font-size: 0.875rem; color: rgba(234, 234, 234, 0.7); margin-bottom: 5px; display: block; }}
    .modal-block .modal-value {{ font-size: 1.5rem; color: {TEXT_COLOR_DARK_THEME}; font-weight: 500; margin-bottom: 8px; display: block; }}
    .modal-block .modal-list-label {{ font-weight: bold; color: {TEXT_COLOR_DARK_THEME}; margin-top: 8px; }}
    .modal-block .modal-list-value {{ color: {TEXT_COLOR_DARK_THEME}; font-size: 0.95rem; }}
</style>
""", unsafe_allow_html=True)

st.title("📊 Calculadora de Estatística para Dados Agrupados em Classes")
st.markdown("---")

st.sidebar.header("Insira os Dados da Distribuição:")
lim_inf_primeira = st.sidebar.number_input("Limite Inferior da 1ª Classe:", value=5.0, step=1.0, format="%.2f") # Permitir decimais
amplitude = st.sidebar.number_input("Amplitude de Classe (h):", value=3.0, min_value=1.0, step=1.0, format="%.2f") # Permitir decimais
qtd_classes_opcoes = [3, 5, 7]
qtd_classes = st.sidebar.selectbox("Quantidade de Classes (ímpar):", options=qtd_classes_opcoes, index=1)

st.sidebar.markdown("---")
st.sidebar.subheader("Frequências Absolutas (fi) de cada Classe:")

# Ajustar default_frequencies com base no qtd_classes selecionado
if qtd_classes == 5: default_frequencies = [10, 10, 10, 10, 10] 
elif qtd_classes == 3: default_frequencies = [10, 10, 10]
else: default_frequencies = [10, 10, 10, 10, 10, 10, 10] 

frequencias_input = []
for i in range(qtd_classes):
    # Garantir que default_val seja pego corretamente
    default_val = default_frequencies[i] if i < len(default_frequencies) else 10 
    freq = st.sidebar.number_input(f"Frequência da Classe {i+1}:", min_value=0, value=int(default_val), step=1, format="%d")
    frequencias_input.append(freq)

if st.sidebar.button("Calcular Estatísticas", type="primary", use_container_width=True):
    df_resultados, estatisticas_data = calcular_estatisticas(float(lim_inf_primeira), float(amplitude), qtd_classes, frequencias_input)

    if df_resultados is None: 
        st.error(estatisticas_data) # estatisticas_data aqui é a mensagem de erro
    else:
        soma_frequencias = df_resultados['Frequência Absoluta (fi)'].sum()
        
        classificacao_modal = estatisticas_data.get("Classificação Modal", "N/A")
        modas_brutas = estatisticas_data.get("Modas Brutas", [])
        modas_czuber = estatisticas_data.get("Modas de Czuber", [])
        indices_classes_modais = estatisticas_data.get("Índices das Classes Modais", [])

        tab_tabela_metricas, tab_graficos = st.tabs(["📋 Tabela e Métricas", "📈 Gráficos"])

        with tab_tabela_metricas:
            st.subheader("Tabela de Distribuição de Frequência:")
            st.dataframe(df_resultados.style.format("{:.2f}", subset=pd.IndexSlice[:, ['Limite Inferior (LI)', 'Limite Superior (LS)', 'Ponto Médio (xi)']]))
            
            st.subheader("Resultados Estatísticos Calculados:")
            if soma_frequencias == 0: st.warning("⚠️ A soma das frequências é zero. As estatísticas podem ser N/A.")

            metricas_principais = {
                "Média (x̅)": estatisticas_data.get("Média"),
                "Variância (s²)": estatisticas_data.get("Variância"),
                "Desvio Padrão (s)": estatisticas_data.get("Desvio Padrão"),
                "Coeficiente de Variação (%)": estatisticas_data.get("Coeficiente de Variação (%)"),
                "Mediana (Me)": estatisticas_data.get("Mediana"),
            }
            cols_principais = st.columns(len(metricas_principais) if len(metricas_principais) <= 3 else 3)
            idx_metric = 0
            for nome, valor in metricas_principais.items():
                with cols_principais[idx_metric % len(cols_principais)]:
                    # Tratar NaN explicitamente para st.metric
                    valor_display = f"{valor:.2f}" if isinstance(valor, (float, int)) and not np.isnan(valor) else "N/A"
                    st.metric(label=nome, value=valor_display)
                idx_metric +=1
            
            st.markdown("---")
            st.subheader("Análise Modal:")

            modal_block_html = "<div class='modal-block'>"
            modal_block_html += f"<span class='modal-title'>Classificação Modal da Amostra</span>"
            modal_block_html += f"<span class='modal-value'>{classificacao_modal}</span>"

            if classificacao_modal == "Amodal":
                modal_block_html += f"<div class='modal-list-label'>Moda(s) Bruta(s):</div> <div class='modal-list-value'>∄</div>"
                modal_block_html += f"<div class='modal-list-label'>Moda(s) de Czuber:</div> <div class='modal-list-value'>∄</div>"
            else:
                modas_brutas_str = ", ".join([f"{mb:.2f}" for mb in modas_brutas if not np.isnan(mb)]) if modas_brutas else "N/A"
                modal_block_html += f"<div class='modal-list-label'>Moda(s) Bruta(s) (PM):</div> <div class='modal-list-value'>{modas_brutas_str if modas_brutas_str else 'N/A'}</div>"
                
                modas_czuber_str = ", ".join([f"{mc:.2f}" for mc in modas_czuber if not np.isnan(mc)]) if modas_czuber else "N/A"
                modal_block_html += f"<div class='modal-list-label'>Moda(s) de Czuber:</div> <div class='modal-list-value'>{modas_czuber_str if modas_czuber_str else 'N/A'}</div>"
            
            modal_block_html += "</div>"
            st.markdown(modal_block_html, unsafe_allow_html=True)

        with tab_graficos:
            st.markdown(f"<h2 style='text-align: center; color: {TEXT_COLOR_DARK_THEME}; font-family: sans-serif; margin-bottom: 10px; margin-top: 0px;'>Gráficos</h2>", unsafe_allow_html=True)
            if soma_frequencias == 0: 
                st.info("Gráficos não podem ser gerados pois a soma total das frequências é zero.")
            else:
                # --- HISTOGRAMA COM DESTAQUE DAS CLASSES MODAIS ---
                class_labels_xaxis = []
                for _, row in df_resultados.iterrows():
                    li_val = float(row['Limite Inferior (LI)'])
                    ls_val = float(row['Limite Superior (LS)'])
                    li_str = str(int(li_val)) if li_val.is_integer() else f"{li_val:.2f}"
                    ls_str = str(int(ls_val)) if ls_val.is_integer() else f"{ls_val:.2f}"
                    class_labels_xaxis.append(f"{li_str} - {ls_str}")

                hover_texts_hist = [f"<b>Classe:</b> {class_labels_xaxis[i]}<br><b>Frequência (fi):</b> {int(df_resultados['Frequência Absoluta (fi)'].iloc[i])}" for i in range(len(class_labels_xaxis))]
                
                bar_colors = [BAR_COLOR_HISTOGRAM_DEFAULT] * qtd_classes
                bar_line_colors = ['rgba(0,0,0,0)'] * qtd_classes
                bar_line_widths = [0] * qtd_classes

                if indices_classes_modais: 
                    for idx_modal_atual_raw in indices_classes_modais:
                        if not np.isnan(idx_modal_atual_raw): # Checar se o índice é válido
                            idx_modal_atual = int(idx_modal_atual_raw) 
                            if 0 <= idx_modal_atual < qtd_classes:
                                bar_colors[idx_modal_atual] = BAR_COLOR_HISTOGRAM_MODAL
                                bar_line_colors[idx_modal_atual] = BAR_BORDER_COLOR_MODAL
                                bar_line_widths[idx_modal_atual] = 2
                
                y_max_freq_hist = df_resultados['Frequência Absoluta (fi)'].max()
                y_range_max_hist = y_max_freq_hist * 1.15 if soma_frequencias > 0 and y_max_freq_hist > 0 else 10 # Evitar range negativo ou zero

                fig_hist = go.Figure()
                fig_hist.add_trace(go.Bar(
                    x=df_resultados['Ponto Médio (xi)'],
                    y=df_resultados['Frequência Absoluta (fi)'],
                    marker=dict(
                        color=bar_colors, 
                        line=dict(color=bar_line_colors, width=bar_line_widths)
                    ),
                    text=hover_texts_hist, hoverinfo='text',
                    name='Frequência (fi)',
                    width=float(amplitude)
                ))
                fig_hist.update_layout(
                    title=dict(text='Histograma de Frequências', x=0.5, font=dict(color=TEXT_COLOR_DARK_THEME, size=20, family='sans-serif')),
                    plot_bgcolor=PLOT_BACKGROUND_COLOR, paper_bgcolor=PLOT_BACKGROUND_COLOR,
                    font=dict(color=TEXT_COLOR_DARK_THEME, family='sans-serif'),
                    xaxis=dict(tickvals=df_resultados['Ponto Médio (xi)'].tolist(), ticktext=class_labels_xaxis, showgrid=False, showline=False, zeroline=False, linecolor=GRID_COLOR_DARK_THEME, tickcolor=TEXT_COLOR_DARK_THEME),
                    yaxis=dict(title_text='', showgrid=True, gridcolor=GRID_COLOR_DARK_THEME, griddash='dash', showline=False, zeroline=False, range=[0, y_range_max_hist], dtick=max(1, int(np.ceil(y_range_max_hist / 8))), linecolor=GRID_COLOR_DARK_THEME, tickcolor=TEXT_COLOR_DARK_THEME),
                    bargap=0.15, 
                    showlegend=False, 
                    hoverlabel=dict(bgcolor=HOVER_BG_COLOR_HISTOGRAM, font_size=14, font_family="sans-serif", font_color=TEXT_COLOR_DARK_THEME, bordercolor=HOVER_BG_COLOR_HISTOGRAM),
                    margin=dict(l=50, r=30, t=80, b=80) 
                )
                st.plotly_chart(fig_hist, use_container_width=True)
                st.markdown("---")

                # --- OGIVA DE FREQUÊNCIAS ACUMULADAS ---
                st.markdown(f"<h4 style='text-align: center; color: {TEXT_COLOR_DARK_THEME}; font-family: sans-serif;'>Ogiva de Frequências Acumuladas</h4>", unsafe_allow_html=True)
                x_ogiva = [float(df_resultados['Limite Inferior (LI)'].iloc[0])] + df_resultados['Limite Superior (LS)'].astype(float).tolist()
                y_ogiva = [0.0] + df_resultados['Frequência Acumulada (Fi)'].astype(float).tolist()
                hover_texts_ogiva = [f"<b>Limite:</b> {x_ogiva[0]:.2f}<br><b>Freq. Acum.:</b> {int(y_ogiva[0])}"] + \
                                    [f"<b>Limite Sup.:</b> {x_ogiva[i]:.2f}<br><b>Freq. Acum.:</b> {int(y_ogiva[i])}" for i in range(1, len(x_ogiva))]
                fig_ogiva = go.Figure(data=[go.Scatter(
                    x=x_ogiva, y=y_ogiva, mode='lines+markers',
                    marker=dict(color=BAR_COLOR_HISTOGRAM_DEFAULT, size=9, line=dict(color=TEXT_COLOR_DARK_THEME, width=1)),
                    line=dict(color=BAR_COLOR_HISTOGRAM_DEFAULT, width=3),
                    text=hover_texts_ogiva, hoverinfo='text'
                )])
                fig_ogiva.update_layout(
                    plot_bgcolor=PLOT_BACKGROUND_COLOR, paper_bgcolor=PLOT_BACKGROUND_COLOR,
                    font=dict(color=TEXT_COLOR_DARK_THEME, family='sans-serif'),
                    xaxis_title='Limites das Classes', yaxis_title='Frequência Acumulada Absoluta (Fac)',
                    xaxis=dict(tickvals=x_ogiva, ticktext=[f"{val:.2f}" for val in x_ogiva], showgrid=True, gridcolor=GRID_COLOR_DARK_THEME, zeroline=False, linecolor=GRID_COLOR_DARK_THEME, tickcolor=TEXT_COLOR_DARK_THEME),
                    yaxis=dict(showgrid=True, gridcolor=GRID_COLOR_DARK_THEME, zeroline=True, zerolinecolor=GRID_COLOR_DARK_THEME, dtick=max(1, int(np.ceil(soma_frequencias / 10.0 if soma_frequencias > 0 else 1.0))), linecolor=GRID_COLOR_DARK_THEME, tickcolor=TEXT_COLOR_DARK_THEME),
                    hoverlabel=dict(bgcolor=HOVER_BG_COLOR_HISTOGRAM, font_size=14, font_family="sans-serif", font_color=TEXT_COLOR_DARK_THEME, bordercolor=HOVER_BG_COLOR_HISTOGRAM),
                    showlegend=False, margin=dict(l=50, r=20, t=40, b=50)
                )
                st.plotly_chart(fig_ogiva, use_container_width=True)
else:
    st.info("Aguardando a inserção dos dados e o clique no botão 'Calcular Estatísticas' na barra lateral.")
st.markdown("---")
st.caption(f"<p style='color:{TEXT_COLOR_DARK_THEME};'>© 2025 | Desenvolvido com Python, Pandas, Numpy, Streamlit e Plotly.</p>", unsafe_allow_html=True)
st.caption(f"<p style='color:{TEXT_COLOR_DARK_THEME};'>Desenvolvido por: Caio Vinícius Passos; Gabriel Henrique Godoy Ribeiro; Gabrielly Lima Paranhos; Geovanna Gabrieli Evangelista e Haisa Rodrigues Brandão.</p>", unsafe_allow_html=True)