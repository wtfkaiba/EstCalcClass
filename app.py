import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- Color Palette for Dark Theme ---
DARK_BACKGROUND_COLOR = '#1e2A38' # Slightly different from pure #222730 for variety
PLOT_BACKGROUND_COLOR = '#283442' # A bit lighter for the plot area itself
TEXT_COLOR_DARK_THEME = '#EAEAEA'
GRID_COLOR_DARK_THEME = 'rgba(180, 180, 180, 0.15)'
BAR_COLOR_HISTOGRAM = '#3498db' # A nice blue
HOVER_BG_COLOR_HISTOGRAM = '#15202b'
SHAPE_COLOR_LIGHT = 'rgba(200, 200, 220, 0.07)' # Very subtle
SHAPE_COLOR_DARKER = 'rgba(180, 180, 200, 0.12)' # Slightly more visible for the middle band

# --- Funções de Cálculo (permanecem as mesmas) ---
def calcular_estatisticas(lim_inf_primeira, amplitude, qtd_classes, frequencias):
    if not frequencias: return None, "A lista de frequências está vazia."
    if not all(isinstance(f, (int, float)) for f in frequencias): return None, "Todas as frequências devem ser números."
    classes_lim_inf, classes_lim_sup, classes_ponto_medio = [], [], []
    for i in range(qtd_classes):
        li = lim_inf_primeira + (i * amplitude); ls = li + amplitude; pm = (li + ls) / 2
        classes_lim_inf.append(li); classes_lim_sup.append(ls); classes_ponto_medio.append(pm)
    df = pd.DataFrame({'Limite Inferior (LI)': classes_lim_inf, 'Limite Superior (LS)': classes_lim_sup,
                       'Ponto Médio de Classe (xi)': classes_ponto_medio, 'Frequência Absoluta (fi)': frequencias})
    df['Frequência Acumulada (Fac)'] = df['Frequência Absoluta (fi)'].cumsum()
    N = df['Frequência Absoluta (fi)'].sum()
    estatisticas_base = {"Média": np.nan, "Variância": np.nan, "Desvio Padrão": np.nan,
                         "Coeficiente de Variação (%)": np.nan, "Mediana": np.nan,
                         "Moda Bruta (Ponto Médio da Classe Modal)": np.nan, "Moda de Czuber": np.nan}
    if N == 0: return df, estatisticas_base
    soma_xi_fi = (df['Ponto Médio de Classe (xi)'] * df['Frequência Absoluta (fi)']).sum(); media = soma_xi_fi / N
    soma_desvios_quad_ponderados = (((df['Ponto Médio de Classe (xi)'] - media)**2) * df['Frequência Absoluta (fi)']).sum()
    variancia = soma_desvios_quad_ponderados / (N - 1) if N > 1 else 0.0; desvio_padrao = np.sqrt(variancia)
    cv = (desvio_padrao / abs(media)) * 100 if media != 0 else np.nan
    pos_mediana = N / 2
    classe_mediana_idx_series = df[df['Frequência Acumulada (Fac)'] >= pos_mediana].index
    if classe_mediana_idx_series.empty: mediana = np.nan
    else:
        classe_mediana_idx = classe_mediana_idx_series[0]
        L_md = df.loc[classe_mediana_idx, 'Limite Inferior (LI)']
        f_md = df.loc[classe_mediana_idx, 'Frequência Absoluta (fi)']
        Fi_ant_md = df.loc[classe_mediana_idx - 1, 'Frequência Acumulada (Fac)'] if classe_mediana_idx > 0 else 0
        h = amplitude; mediana = L_md + ((pos_mediana - Fi_ant_md) / f_md) * h if f_md != 0 else L_md
    idx_moda_bruta = df['Frequência Absoluta (fi)'].idxmax(); moda_bruta = df.loc[idx_moda_bruta, 'Ponto Médio de Classe (xi)']
    classe_modal_idx = idx_moda_bruta
    L_mo = df.loc[classe_modal_idx, 'Limite Inferior (LI)']
    f_mo = df.loc[classe_modal_idx, 'Frequência Absoluta (fi)']
    f_ant = df.loc[classe_modal_idx - 1, 'Frequência Absoluta (fi)'] if classe_modal_idx > 0 else 0
    f_pos = df.loc[classe_modal_idx + 1, 'Frequência Absoluta (fi)'] if classe_modal_idx < (qtd_classes - 1) else 0
    delta1 = f_mo - f_ant; delta2 = f_mo - f_pos
    if (delta1 + delta2) == 0: moda_czuber = L_mo + (amplitude / 2)
    else:
        moda_czuber = L_mo + (delta1 / (delta1 + delta2)) * amplitude
        moda_czuber = max(L_mo, min(moda_czuber, L_mo + amplitude))
    return df, {"Média": media, "Variância": variancia, "Desvio Padrão": desvio_padrao,
                 "Coeficiente de Variação (%)": cv, "Mediana": mediana,
                 "Moda Bruta (Ponto Médio da Classe Modal)": moda_bruta, "Moda de Czuber": moda_czuber}

# --- Interface Streamlit ---
st.set_page_config(layout="wide", page_title="Estatística Aplicada - Dados Agrupados")
st.markdown(f"""
<style>
    .reportview-container {{
        background-color: {DARK_BACKGROUND_COLOR};
        color: {TEXT_COLOR_DARK_THEME};
    }}
    .sidebar .sidebar-content {{
        background-color: {DARK_BACKGROUND_COLOR};
    }}
    h1, h2, h3, h4, h5, h6 {{
        color: {TEXT_COLOR_DARK_THEME};
    }}
    .stMetric {{
        background-color: {PLOT_BACKGROUND_COLOR} !important; /* Corrigido com !important */
        border-radius: 7px;
        padding: 10px;
    }}
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {{
        color: {TEXT_COLOR_DARK_THEME};
    }}
    .stTabs [data-baseweb="tab-list"] button {{
        background-color: transparent; /* Para melhor visual */
    }}
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {{
        background-color: {PLOT_BACKGROUND_COLOR};
         border-bottom: 2px solid {BAR_COLOR_HISTOGRAM};
    }}
</style>
""", unsafe_allow_html=True)

st.title("📊 Calculadora de Estatística para Dados Agrupados em Classes")
st.markdown("---")

st.sidebar.header("Insira os Dados da Distribuição:")
lim_inf_primeira = st.sidebar.number_input("Limite Inferior da 1ª Classe:", value=1, step=1)
amplitude = st.sidebar.number_input("Amplitude de Classe (h):", value=3, min_value=1, step=1)
qtd_classes = st.sidebar.selectbox("Quantidade de Classes (ímpar):", options=[3, 5, 7], index=1) # Padrão 5

st.sidebar.markdown("---")
st.sidebar.subheader("Frequências Absolutas (fi) de cada Classe:")
default_frequencies = [10, 10, 10, 10, 10] if qtd_classes == 5 else ([10, 10, 10] if qtd_classes == 3 else [10, 10, 10, 10, 10, 10, 10])
frequencias_input = [st.sidebar.number_input(f"Frequência da Classe {i+1}:", min_value=0, value=(default_frequencies[i] if i < len(default_frequencies) else 10), step=1) for i in range(qtd_classes)]

if st.sidebar.button("Calcular Estatísticas", type="primary", use_container_width=True):
    df_resultados, estatisticas_data = calcular_estatisticas(lim_inf_primeira, amplitude, qtd_classes, frequencias_input)

    if df_resultados is None: st.error(estatisticas_data)
    else:
        soma_frequencias = df_resultados['Frequência Absoluta (fi)'].sum()
        tab_tabela_metricas, tab_graficos = st.tabs(["📋 Tabela e Métricas", "📈 Gráficos"])

        with tab_tabela_metricas:
            st.subheader("Tabela de Distribuição de Frequência:")
            st.dataframe(df_resultados.style.format("{:.1f}", subset=pd.IndexSlice[:, ['Limite Inferior (LI)', 'Limite Superior (LS)', 'Ponto Médio de Classe (xi)']]))
            st.subheader("Resultados Estatísticos Calculados:")
            if soma_frequencias == 0: st.warning("⚠️ A soma das frequências é zero. As estatísticas podem não ser significativas ou foram definidas como 'N/A'.")
            cols = st.columns(3)
            for i, (nome, valor) in enumerate(estatisticas_data.items()):
                with cols[i % 3]:
                    st.metric(label=nome, value=f"{valor:.1f}" if isinstance(valor, (int, float)) and not np.isnan(valor) else "N/A")
        
        with tab_graficos:
            st.markdown(f"<h2 style='text-align: center; color: {TEXT_COLOR_DARK_THEME}; font-family: sans-serif; margin-bottom: 0px;'>Gráficos</h2>", unsafe_allow_html=True)
            if soma_frequencias == 0: st.info("Gráficos não podem ser gerados pois a soma total das frequências é zero.")
            else:
                # --- HISTOGRAMA ---
                class_labels_xaxis = [f"{int(row['Limite Inferior (LI)']) if row['Limite Inferior (LI)'].is_integer() else row['Limite Inferior (LI)']:.1f} - {int(row['Limite Superior (LS)']) if row['Limite Superior (LS)'].is_integer() else row['Limite Superior (LS)']:.1f}" for _, row in df_resultados.iterrows()]
                hover_texts_hist = [f"<b>Classe:</b> {class_labels_xaxis[i]}<br><b>Frequência (fi):</b> {df_resultados['Frequência Absoluta (fi)'].iloc[i]}" for i in range(len(class_labels_xaxis))]
                
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Bar(
                    x=df_resultados['Ponto Médio de Classe (xi)'],
                    y=df_resultados['Frequência Absoluta (fi)'],
                    marker_color=BAR_COLOR_HISTOGRAM,
                    text=hover_texts_hist, hoverinfo='text',
                    name='Frequência (fi)', # Para a legenda
                    width=amplitude * 0.65 # Barras um pouco mais finas para mais espaçamento
                ))

                # Adicionar as formas (bandas verticais)
                shapes = []
                y_range_max = max(df_resultados['Frequência Absoluta (fi)']) * 1.15
                
                if qtd_classes == 5: # Para replicar o visual da imagem com 5 classes
                    # Banda 1 (leve)
                    shapes.append(go.layout.Shape(type="rect", xref="x", yref="paper", x0=df_resultados['Limite Inferior (LI)'].iloc[0], x1=df_resultados['Limite Superior (LS)'].iloc[0], y0=0, y1=1, fillcolor=SHAPE_COLOR_LIGHT, layer="below", line_width=0))
                    # Banda 2 (mais escura, cobre 2 classes)
                    shapes.append(go.layout.Shape(type="rect", xref="x", yref="paper", x0=df_resultados['Limite Inferior (LI)'].iloc[1], x1=df_resultados['Limite Superior (LS)'].iloc[2], y0=0, y1=1, fillcolor=SHAPE_COLOR_DARKER, layer="below", line_width=0))
                    # Banda 3 (leve, cobre 2 classes)
                    shapes.append(go.layout.Shape(type="rect", xref="x", yref="paper", x0=df_resultados['Limite Inferior (LI)'].iloc[3], x1=df_resultados['Limite Superior (LS)'].iloc[4], y0=0, y1=1, fillcolor=SHAPE_COLOR_LIGHT, layer="below", line_width=0))
                elif qtd_classes == 3:
                     shapes.append(go.layout.Shape(type="rect", xref="x", yref="paper", x0=df_resultados['Limite Inferior (LI)'].iloc[0], x1=df_resultados['Limite Superior (LS)'].iloc[0], y0=0, y1=1, fillcolor=SHAPE_COLOR_LIGHT, layer="below", line_width=0))
                     shapes.append(go.layout.Shape(type="rect", xref="x", yref="paper", x0=df_resultados['Limite Inferior (LI)'].iloc[1], x1=df_resultados['Limite Superior (LS)'].iloc[1], y0=0, y1=1, fillcolor=SHAPE_COLOR_DARKER, layer="below", line_width=0))
                     shapes.append(go.layout.Shape(type="rect", xref="x", yref="paper", x0=df_resultados['Limite Inferior (LI)'].iloc[2], x1=df_resultados['Limite Superior (LS)'].iloc[2], y0=0, y1=1, fillcolor=SHAPE_COLOR_LIGHT, layer="below", line_width=0))
                # Adicionar mais lógica para qtd_classes == 7 se necessário para as bandas

                fig_hist.update_layout(
                    title=dict(text='Histograma de Frequências', x=0.5, font=dict(color=TEXT_COLOR_DARK_THEME, size=20, family='sans-serif')),
                    plot_bgcolor=PLOT_BACKGROUND_COLOR, paper_bgcolor=PLOT_BACKGROUND_COLOR,
                    font=dict(color=TEXT_COLOR_DARK_THEME, family='sans-serif'),
                    xaxis=dict(tickvals=df_resultados['Ponto Médio de Classe (xi)'], ticktext=class_labels_xaxis, showgrid=False, showline=False, zeroline=False, linecolor=GRID_COLOR_DARK_THEME, tickcolor=TEXT_COLOR_DARK_THEME),
                    yaxis=dict(title_text='', showgrid=True, gridcolor=GRID_COLOR_DARK_THEME, griddash='dash', showline=False, zeroline=False, range=[0, y_range_max], dtick=4, linecolor=GRID_COLOR_DARK_THEME, tickcolor=TEXT_COLOR_DARK_THEME),
                    bargap=0.3, # Aumenta o espaço entre as barras
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, bgcolor='rgba(0,0,0,0)', font=dict(color=TEXT_COLOR_DARK_THEME)), # Legenda no estilo da imagem (movida para o topo para não sobrepor X-axis)
                    hoverlabel=dict(bgcolor=HOVER_BG_COLOR_HISTOGRAM, font_size=14, font_family="sans-serif", font_color=TEXT_COLOR_DARK_THEME, bordercolor=HOVER_BG_COLOR_HISTOGRAM),
                    shapes=shapes,
                    margin=dict(l=50, r=30, t=100, b=80) # Ajuste de margens
                )
                st.plotly_chart(fig_hist, use_container_width=True)
                st.markdown("---")

                # --- OGIVA DE FREQUÊNCIAS ACUMULADAS (DARK MODE) ---
                st.markdown(f"<h4 style='text-align: center; color: {TEXT_COLOR_DARK_THEME}; font-family: sans-serif;'>Ogiva de Frequências Acumuladas</h4>", unsafe_allow_html=True)
                x_ogiva = [df_resultados['Limite Inferior (LI)'].iloc[0]] + df_resultados['Limite Superior (LS)'].tolist()
                y_ogiva = [0] + df_resultados['Frequência Acumulada (Fac)'].tolist()
                hover_texts_ogiva = [f"<b>Limite:</b> {x_ogiva[0]:.1f}<br><b>Freq. Acum.:</b> 0"] + \
                                    [f"<b>Limite Sup.:</b> {x_ogiva[i]:.1f}<br><b>Freq. Acum.:</b> {y_ogiva[i]}" for i in range(1, len(x_ogiva))]
                fig_ogiva = go.Figure(data=[go.Scatter(
                    x=x_ogiva, y=y_ogiva, mode='lines+markers',
                    marker=dict(color=BAR_COLOR_HISTOGRAM, size=9, line=dict(color=TEXT_COLOR_DARK_THEME, width=1)), # Usar cor da barra do histograma para consistência
                    line=dict(color=BAR_COLOR_HISTOGRAM, width=3),
                    text=hover_texts_ogiva, hoverinfo='text'
                )])
                fig_ogiva.update_layout(
                    plot_bgcolor=PLOT_BACKGROUND_COLOR, paper_bgcolor=PLOT_BACKGROUND_COLOR,
                    font=dict(color=TEXT_COLOR_DARK_THEME, family='sans-serif'),
                    xaxis_title='Limites das Classes', yaxis_title='Frequência Acumulada (Fac)',
                    xaxis=dict(tickvals=x_ogiva, ticktext=[f"{val:.1f}" for val in x_ogiva], showgrid=True, gridcolor=GRID_COLOR_DARK_THEME, zeroline=False, linecolor=GRID_COLOR_DARK_THEME, tickcolor=TEXT_COLOR_DARK_THEME),
                    yaxis=dict(showgrid=True, gridcolor=GRID_COLOR_DARK_THEME, zeroline=True, zerolinecolor=GRID_COLOR_DARK_THEME, dtick=max(1, int(np.ceil(soma_frequencias / 10))), linecolor=GRID_COLOR_DARK_THEME, tickcolor=TEXT_COLOR_DARK_THEME),
                    hoverlabel=dict(bgcolor=HOVER_BG_COLOR_HISTOGRAM, font_size=14, font_family="sans-serif", font_color=TEXT_COLOR_DARK_THEME, bordercolor=HOVER_BG_COLOR_HISTOGRAM),
                    showlegend=False, margin=dict(l=50, r=20, t=40, b=50)
                )
                st.plotly_chart(fig_ogiva, use_container_width=True)
else:
    st.info("Aguardando a inserção dos dados e o clique no botão 'Calcular Estatísticas' na barra lateral.")
st.markdown("---")
st.caption(f"<p style='color:{TEXT_COLOR_DARK_THEME};'>© 2025 | Desenvolvido com Python, Pandas, Numpy, Streamlit e Plotly.</p>", unsafe_allow_html=True)
