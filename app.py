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
    if not frequencias: return None, "A lista de frequências está vazia. Por favor, insira valores."
    if not all(isinstance(f, (int, float)) and f >= 0 for f in frequencias):
        return None, "Todas as frequências devem ser números positivos ou zero."
    if len(frequencias) != qtd_classes:
        return None, f"Esperava {qtd_classes} frequências, mas recebeu {len(frequencias)}."

    frequencias_int = [int(round(f)) for f in frequencias]
    classes_lim_inf, classes_lim_sup, classes_ponto_medio = [], [], []
    for i in range(qtd_classes):
        li = float(lim_inf_primeira + (i * amplitude))
        ls = float(li + amplitude)
        pm = float((li + ls) / 2)
        classes_lim_inf.append(li); classes_lim_sup.append(ls); classes_ponto_medio.append(pm)
    
    df = pd.DataFrame({
        'Limite Inferior (LI)': classes_lim_inf, 
        'Limite Superior (LS)': classes_lim_sup,
        'Ponto Médio (xi)': classes_ponto_medio, 
        'Frequência Absoluta (fi)': frequencias_int
    })
    df['Frequência Acumulada (Fac)'] = df['Frequência Absoluta (fi)'].cumsum().astype(int)
    df['Frequência Absoluta (fi)'] = df['Frequência Absoluta (fi)'].astype(int)
    N = df['Frequência Absoluta (fi)'].sum()
    estatisticas_resultados = {
        "Média": np.nan, "Variância": np.nan, "Desvio Padrão": np.nan,
        "Coeficiente de Variação (%)": np.nan, "Mediana": np.nan,
        "Classificação Modal": "Amodal", "Modas Brutas": [],
        "Modas de Czuber": [], "Índices das Classes Modais": []}
    if N == 0: return df, estatisticas_resultados
    soma_xi_fi = (df['Ponto Médio (xi)'] * df['Frequência Absoluta (fi)']).sum(); media = soma_xi_fi / N
    estatisticas_resultados["Média"] = float(media)
    soma_desvios_quad_ponderados = (((df['Ponto Médio (xi)'] - media)**2) * df['Frequência Absoluta (fi)']).sum()
    variancia = soma_desvios_quad_ponderados / (N - 1) if N > 1 else 0.0
    estatisticas_resultados["Variância"] = float(variancia)
    desvio_padrao = np.sqrt(variancia) if variancia >= 0 else np.nan
    estatisticas_resultados["Desvio Padrão"] = float(desvio_padrao) if not np.isnan(desvio_padrao) else np.nan
    cv = (desvio_padrao / abs(media)) * 100 if media != 0 and not np.isnan(desvio_padrao) else np.nan
    estatisticas_resultados["Coeficiente de Variação (%)"] = float(cv) if not np.isnan(cv) else np.nan
    pos_mediana = N / 2.0
    classe_mediana_idx_series = df[df['Frequência Acumulada (Fac)'] >= pos_mediana].index
    if not classe_mediana_idx_series.empty:
        classe_mediana_idx = classe_mediana_idx_series[0]
        L_md = df.loc[classe_mediana_idx, 'Limite Inferior (LI)']
        f_md = df.loc[classe_mediana_idx, 'Frequência Absoluta (fi)']
        Fi_ant_md = df.loc[classe_mediana_idx - 1, 'Frequência Acumulada (Fac)'] if classe_mediana_idx > 0 else 0.0
        h = float(amplitude); mediana_calc = L_md + ((pos_mediana - Fi_ant_md) / f_md) * h if f_md != 0 else L_md
        estatisticas_resultados["Mediana"] = float(mediana_calc)
    else: estatisticas_resultados["Mediana"] = np.nan
    frequencias_abs_series = df['Frequência Absoluta (fi)']
    max_freq = frequencias_abs_series.max()
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
        modas_brutas_list, modas_czuber_list = [], []
        for idx_modal_atual in indices_modais:
            modas_brutas_list.append(float(df.loc[idx_modal_atual, 'Ponto Médio (xi)']))
            L_mo_atual = df.loc[idx_modal_atual, 'Limite Inferior (LI)']
            f_mo_atual = df.loc[idx_modal_atual, 'Frequência Absoluta (fi)']
            f_ant_atual = df.loc[idx_modal_atual - 1, 'Frequência Absoluta (fi)'] if idx_modal_atual > 0 else 0.0
            f_pos_atual = df.loc[idx_modal_atual + 1, 'Frequência Absoluta (fi)'] if idx_modal_atual < (qtd_classes - 1) else 0.0
            delta1_atual = f_mo_atual - f_ant_atual; delta2_atual = f_mo_atual - f_pos_atual
            cz_mode_atual_calc = np.nan
            if (delta1_atual + delta2_atual) == 0:
                cz_mode_atual_calc = L_mo_atual + (float(amplitude) / 2.0)
            else:
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
    .main .block-container {{ padding-top: 1rem; padding-bottom: 1rem; }} 
    body {{ background-color: {DARK_BACKGROUND_COLOR}; color: {TEXT_COLOR_DARK_THEME}; }} /* Aplica a todo o corpo */
    .stApp {{ background-color: {DARK_BACKGROUND_COLOR}; }} /* Garante que o app principal tenha o fundo */
    .sidebar .sidebar-content {{ background-color: {DARK_BACKGROUND_COLOR}; }}
    h1, h2, h3, h4, h5, h6, .stMarkdown p {{ color: {TEXT_COLOR_DARK_THEME}; }} /* Aplica cor a parágrafos markdown também */
    .stButton>button {{ 
        border: 1px solid {BAR_COLOR_HISTOGRAM_DEFAULT}; border-radius: 5px;
        padding: 0.4rem 0.75rem; margin: 0.2rem;
        background-color: transparent; color: {BAR_COLOR_HISTOGRAM_DEFAULT};
    }}
    .stButton>button:hover {{ background-color: {BAR_COLOR_HISTOGRAM_DEFAULT}; color: {PLOT_BACKGROUND_COLOR}; }}
    .stButton>button:focus {{ /* Estilo para botão selecionado (se o browser suportar :focus bem para isso) */
        box-shadow: 0 0 0 0.2rem {BAR_COLOR_HISTOGRAM_MODAL}70; /* Leve brilho ao focar/clicar */
    }}
    .metric-block {{ 
        background-color: {PLOT_BACKGROUND_COLOR}; border-radius: 7px; padding: 15px;
        margin-top: 10px; border: 1px solid rgba(255,255,255,0.1);
    }}
    .metric-block .metric-label {{ font-size: 1rem; color: rgba(234, 234, 234, 0.8); display: block; margin-bottom: 5px;}}
    .metric-block .metric-value {{ font-size: 1.8rem; color: {TEXT_COLOR_DARK_THEME}; font-weight: bold;}}
    .modal-block {{ background-color: {PLOT_BACKGROUND_COLOR}; border-radius: 7px; padding: 15px; margin-bottom: 10px; border: 1px solid rgba(255,255,255,0.1); }}
    .modal-block .modal-title {{ font-size: 0.875rem; color: rgba(234, 234, 234, 0.7); margin-bottom: 5px; display: block; }}
    .modal-block .modal-value {{ font-size: 1.5rem; color: {TEXT_COLOR_DARK_THEME}; font-weight: 500; margin-bottom: 8px; display: block; }}
    .modal-block .modal-list-label {{ font-weight: bold; color: {TEXT_COLOR_DARK_THEME}; margin-top: 8px; }}
    .modal-block .modal-list-value {{ color: {TEXT_COLOR_DARK_THEME}; font-size: 0.95rem; }}
</style>
""", unsafe_allow_html=True)

st.title("📊 Calculadora de Estatística para Dados Agrupados em Classes")
st.subheader("Estatística Aplicada - Prof. Eng. MSc. João Carlos dos Santos")
#st.markdown("---")

if 'resultados_calculados' not in st.session_state: st.session_state.resultados_calculados = None
if 'df_resultados' not in st.session_state: st.session_state.df_resultados = None
if 'soma_frequencias' not in st.session_state: st.session_state.soma_frequencias = 0
if 'mostrar_calculo' not in st.session_state: st.session_state.mostrar_calculo = "nenhum"

st.sidebar.header("Insira os Dados da Distribuição:")
lim_inf_primeira = st.sidebar.number_input("Limite Inferior da 1ª Classe:", value=5.0, step=1.0, format="%.2f", key="lim_inf")
amplitude = st.sidebar.number_input("Amplitude de Classe (h):", value=3.0, min_value=1.0, step=1.0, format="%.2f", key="amp")
qtd_classes = st.sidebar.selectbox("Quantidade de Classes:", options=[3, 5, 7], index=1, key="qtd_cls")
#st.sidebar.markdown("---")
st.sidebar.subheader("Frequências Absolutas (fi) de cada Classe:")
if qtd_classes == 5: default_frequencies = [10, 10, 10, 10, 10] 
elif qtd_classes == 3: default_frequencies = [10, 10, 10]
else: default_frequencies = [10, 10, 10, 10, 10, 10, 10] 
frequencias_input = [st.sidebar.number_input(f"Frequência da Classe {i+1}:", min_value=0, value=int(default_frequencies[i] if i < len(default_frequencies) else 10), step=1, format="%d", key=f"freq_{i}") for i in range(qtd_classes)]

if st.sidebar.button("Calcular Estatísticas", type="primary", use_container_width=True, key="btn_calcular_main"):
    df_res, stats_res = calcular_estatisticas(float(lim_inf_primeira), float(amplitude), qtd_classes, frequencias_input)
    if df_res is None:
        st.session_state.resultados_calculados = None; st.session_state.df_resultados = None
        st.session_state.soma_frequencias = 0; st.error(stats_res) 
    else:
        st.session_state.resultados_calculados = stats_res; st.session_state.df_resultados = df_res
        st.session_state.soma_frequencias = df_res['Frequência Absoluta (fi)'].sum()
        st.session_state.mostrar_calculo = "nenhum" 
    st.rerun() # Forçar rerun para atualizar a interface principal após o cálculo

# --- Exibição dos Botões de Cálculo e Resultados ---
mostrar_atual = st.session_state.mostrar_calculo # Pega o estado atual

if st.session_state.resultados_calculados is not None and st.session_state.df_resultados is not None:
    st.markdown("---"); st.subheader("Selecione o Cálculo para Visualizar:")
    botoes_calculo = {
        "Média (x̅)": "media", "Variância (s²)": "variancia", "Desvio Padrão (s)": "desvio_padrao",
        "Coef. Variação": "cv", "Mediana (Me)": "mediana", "Análise Modal": "modal", 
        "Tabela de Freq.": "tabela", "Gráficos": "graficos", "Mostrar Tudo": "todos"}
    cols_botoes = st.columns(len(botoes_calculo) if len(botoes_calculo) <=5 else 5) 
    for i, (nome_botao, chave_calculo) in enumerate(botoes_calculo.items()):
        with cols_botoes[i % len(cols_botoes)]:
            if st.button(nome_botao, key=f"btn_show_{chave_calculo}", use_container_width=True):
                st.session_state.mostrar_calculo = chave_calculo
                st.rerun() 
    st.markdown("---") 
    
    estatisticas_data = st.session_state.resultados_calculados
    df_resultados = st.session_state.df_resultados
    soma_frequencias = st.session_state.soma_frequencias

    def exibir_metrica_individual(label, valor_chave):
        valor = estatisticas_data.get(valor_chave)
        valor_display = f"{valor:.2f}" if isinstance(valor, (float, int)) and not np.isnan(valor) else "N/A"
        st.markdown(f"<div class='metric-block'><span class='metric-label'>{label}</span><span class='metric-value'>{valor_display}</span></div>", unsafe_allow_html=True)

    def exibir_analise_modal_completa():
        classificacao = estatisticas_data.get("Classificação Modal", "N/A")
        modas_b = estatisticas_data.get("Modas Brutas", []); modas_c = estatisticas_data.get("Modas de Czuber", [])
        modal_html = (f"<div class='modal-block'><span class='modal-title'>Classificação Modal</span><span class='modal-value'>{classificacao}</span>")
        if classificacao == "Amodal":
            modal_html += "<div class='modal-list-label'>Moda(s) Bruta(s):</div><div class='modal-list-value'>∄</div>"
            modal_html += "<div class='modal-list-label'>Moda(s) de Czuber:</div><div class='modal-list-value'>∄</div>"
        else:
            mb_str = ", ".join([f"{mb:.2f}" for mb in modas_b if not np.isnan(mb)]) if modas_b else "N/A"
            mc_str = ", ".join([f"{mc:.2f}" for mc in modas_c if not np.isnan(mc)]) if modas_c else "N/A"
            modal_html += f"<div class='modal-list-label'>Moda(s) Bruta(s) (PM):</div><div class='modal-list-value'>{mb_str if mb_str else 'N/A'}</div>"
            modal_html += f"<div class='modal-list-label'>Moda(s) de Czuber:</div><div class='modal-list-value'>{mc_str if mc_str else 'N/A'}</div>"
        modal_html += "</div>"; st.markdown(modal_html, unsafe_allow_html=True)

    # --- Funções para Gerar Gráficos ---
    def gerar_histograma(df_res, stats_data, amp, q_classes, s_freq):
        indices_modais = stats_data.get("Índices das Classes Modais", [])
        labels_x = [f"{int(r['Limite Inferior (LI)']) if float(r['Limite Inferior (LI)']).is_integer() else r['Limite Inferior (LI)']:.2f} - {int(r['Limite Superior (LS)']) if float(r['Limite Superior (LS)']).is_integer() else r['Limite Superior (LS)']:.2f}" for _, r in df_res.iterrows()]
        hover_txt = [f"<b>Classe:</b> {labels_x[j]}<br><b>Frequência (fi):</b> {int(df_res['Frequência Absoluta (fi)'].iloc[j])}" for j in range(len(labels_x))]
        bar_cols, line_cols, line_w = [BAR_COLOR_HISTOGRAM_DEFAULT]*q_classes, ['rgba(0,0,0,0)']*q_classes, [0]*q_classes
        if indices_modais:
            for idx_r in indices_modais:
                if not np.isnan(idx_r):
                    idx_m = int(idx_r)
                    if 0 <= idx_m < q_classes: bar_cols[idx_m], line_cols[idx_m], line_w[idx_m] = BAR_COLOR_HISTOGRAM_MODAL, BAR_BORDER_COLOR_MODAL, 2
        y_max = df_res['Frequência Absoluta (fi)'].max(); y_range = y_max * 1.15 if s_freq > 0 and y_max > 0 else 10
        fig = go.Figure(data=[go.Bar(x=df_res['Ponto Médio (xi)'], y=df_res['Frequência Absoluta (fi)'], marker=dict(color=bar_cols, line=dict(color=line_cols, width=line_w)), text=hover_txt, hoverinfo='text', name='Freq. (fi)', width=float(amp))])
        fig.update_layout(plot_bgcolor=PLOT_BACKGROUND_COLOR, paper_bgcolor=PLOT_BACKGROUND_COLOR, font=dict(color=TEXT_COLOR_DARK_THEME), xaxis=dict(tickvals=df_res['Ponto Médio (xi)'].tolist(), ticktext=labels_x, showgrid=False,showline=False,zeroline=False), yaxis=dict(title_text='',showgrid=True,gridcolor=GRID_COLOR_DARK_THEME,griddash='dash',showline=False,zeroline=False,range=[0,y_range],dtick=max(1,int(np.ceil(y_range/8)))), bargap=0.15,showlegend=False,hoverlabel=dict(bgcolor=HOVER_BG_COLOR_HISTOGRAM,font_color=TEXT_COLOR_DARK_THEME), margin=dict(l=40,r=20,t=30,b=50))
        return fig

    def gerar_ogiva(df_res, s_freq):
        x_og = [float(df_res['Limite Inferior (LI)'].iloc[0])] + df_res['Limite Superior (LS)'].astype(float).tolist()
        y_og = [0.0] + df_res['Frequência Acumulada (Fac)'].tolist()
        hover_txt_og = [f"<b>Limite:</b> {x_og[0]:.2f}<br><b>Freq. Acum.:</b> {int(y_og[0])}"] + [f"<b>Limite Sup.:</b> {x_og[k]:.2f}<br><b>Freq. Acum.:</b> {int(y_og[k])}" for k in range(1, len(x_og))]
        fig = go.Figure(data=[go.Scatter(x=x_og, y=y_og, mode='lines+markers', marker=dict(color=BAR_COLOR_HISTOGRAM_DEFAULT,size=9,line=dict(color=TEXT_COLOR_DARK_THEME,width=1)), line=dict(color=BAR_COLOR_HISTOGRAM_DEFAULT,width=3), text=hover_txt_og,hoverinfo='text')])
        fig.update_layout(plot_bgcolor=PLOT_BACKGROUND_COLOR, paper_bgcolor=PLOT_BACKGROUND_COLOR, font=dict(color=TEXT_COLOR_DARK_THEME), xaxis_title='Limites das Classes', yaxis_title='Freq. Acumulada (Fac)', xaxis=dict(tickvals=x_og,ticktext=[f"{v:.2f}" for v in x_og],showgrid=True,gridcolor=GRID_COLOR_DARK_THEME,zeroline=False), yaxis=dict(showgrid=True,gridcolor=GRID_COLOR_DARK_THEME,zeroline=True,zerolinecolor=GRID_COLOR_DARK_THEME,dtick=max(1,int(np.ceil(s_freq/10.0 if s_freq>0 else 1.0)))), hoverlabel=dict(bgcolor=HOVER_BG_COLOR_HISTOGRAM,font_color=TEXT_COLOR_DARK_THEME), showlegend=False,margin=dict(l=50,r=20,t=30,b=50))
        return fig
    
    # Lógica de Exibição Principal
    if soma_frequencias == 0 and mostrar_atual not in ["nenhum", "tabela"]:
        st.warning("⚠️ Soma das frequências é zero. Cálculos podem ser N/A.")
    
    if mostrar_atual == "media": exibir_metrica_individual("Média (x̅)", "Média")
    elif mostrar_atual == "variancia": exibir_metrica_individual("Variância (s²)", "Variância")
    elif mostrar_atual == "desvio_padrao": exibir_metrica_individual("Desvio Padrão (s)", "Desvio Padrão")
    elif mostrar_atual == "cv": exibir_metrica_individual("Coeficiente de Variação (%)", "Coeficiente de Variação (%)")
    elif mostrar_atual == "mediana": exibir_metrica_individual("Mediana (Me)", "Mediana")
    elif mostrar_atual == "modal": exibir_analise_modal_completa()
    elif mostrar_atual == "tabela":
        st.subheader("Tabela de Distribuição de Frequência:")
        st.dataframe(df_resultados.style.format({'Limite Inferior (LI)': "{:.2f}", 'Limite Superior (LS)': "{:.2f}", 'Ponto Médio (xi)': "{:.2f}", 'Frequência Absoluta (fi)': "{:d}", 'Frequência Acumulada (Fac)': "{:d}"}))
    elif mostrar_atual == "graficos":
        if soma_frequencias == 0: st.info("Gráficos não gerados (soma de frequências é zero).")
        else:
            st.markdown(f"<h4 style='text-align: center; color: {TEXT_COLOR_DARK_THEME};'>Histograma de Frequências</h4>", unsafe_allow_html=True)
            st.plotly_chart(gerar_histograma(df_resultados, estatisticas_data, amplitude, qtd_classes, soma_frequencias), use_container_width=True)
            st.markdown("---")
            st.markdown(f"<h4 style='text-align: center; color: {TEXT_COLOR_DARK_THEME};'>Ogiva de Frequências Acumuladas</h4>", unsafe_allow_html=True)
            st.plotly_chart(gerar_ogiva(df_resultados, soma_frequencias), use_container_width=True)
    elif mostrar_atual == "todos":
        st.subheader("Tabela de Distribuição de Frequência:")
        st.dataframe(df_resultados.style.format({'Limite Inferior (LI)': "{:.2f}", 'Limite Superior (LS)': "{:.2f}", 'Ponto Médio (xi)': "{:.2f}", 'Frequência Absoluta (fi)': "{:d}", 'Frequência Acumulada (Fac)': "{:d}"}))
        st.markdown("---"); st.subheader("Resultados Estatísticos:")
        if soma_frequencias == 0: st.warning("⚠️ Soma das frequências é zero.")
        metricas_para_exibir_todos = {"Média (x̅)": estatisticas_data.get("Média"), "Variância (s²)": estatisticas_data.get("Variância"), "Desvio Padrão (s)": estatisticas_data.get("Desvio Padrão"), "Coef. Variação (%)": estatisticas_data.get("Coeficiente de Variação (%)"), "Mediana (Me)": estatisticas_data.get("Mediana")}
        cols_todos_metricas = st.columns(len(metricas_para_exibir_todos) if len(metricas_para_exibir_todos) <=3 else 3)
        for i_td, (n_td, v_td) in enumerate(metricas_para_exibir_todos.items()):
            with cols_todos_metricas[i_td % len(cols_todos_metricas)]:
                st.metric(label=n_td, value=(f"{v_td:.2f}" if isinstance(v_td, (float,int)) and not np.isnan(v_td) else "N/A"))
        exibir_analise_modal_completa(); st.markdown("---")
        if soma_frequencias == 0: st.info("Gráficos não gerados.")
        else:
            st.subheader("Gráficos:")
            st.markdown(f"<h4 style='text-align: center; color: {TEXT_COLOR_DARK_THEME};'>Histograma de Frequências</h4>", unsafe_allow_html=True)
            st.plotly_chart(gerar_histograma(df_resultados, estatisticas_data, amplitude, qtd_classes, soma_frequencias), use_container_width=True)
            st.markdown("---")
            st.markdown(f"<h4 style='text-align: center; color: {TEXT_COLOR_DARK_THEME};'>Ogiva de Frequências Acumuladas</h4>", unsafe_allow_html=True)
            st.plotly_chart(gerar_ogiva(df_resultados, soma_frequencias), use_container_width=True)

elif st.session_state.mostrar_calculo == "nenhum" and (st.session_state.resultados_calculados is None or st.session_state.df_resultados is None):
    st.info("Aguardando a inserção dos dados e o clique no botão 'Calcular Estatísticas' na barra lateral.")

st.markdown("---")
st.caption(f"<p style='color:{TEXT_COLOR_DARK_THEME}; text-align: center;'>Desenvolvido por: Caio Vinícius Passos; Gabriel Henrique Godoy Ribeiro; Gabrielly Lima Paranhos; Geovanna Gabrieli Evangelista e Haisa Rodrigues Brandão.</p>", unsafe_allow_html=True)
st.caption(f"<p style='color:{TEXT_COLOR_DARK_THEME}; text-align: center;'>© 2025 | Desenvolvido com Python, Pandas, Numpy, Streamlit e Plotly.</p>", unsafe_allow_html=True)