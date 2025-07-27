"""
Dashboard de Análise de E-commerce

Este script cria um dashboard interativo para análise de dados de e-commerce,
com visualizações estatísticas e filtros por grupo de consumidores.

Tecnologias utilizadas:
- Plotly Express para visualizações
- Dash para construção do dashboard
- Scikit-learn para análise de regressão linear
- Pandas para manipulação de dados
"""

# Importações de bibliotecas
import plotly.express as px  # Para criação de gráficos interativos
import pandas as pd  # Para manipulação e análise de dados
from sklearn.linear_model import LinearRegression  # Para análise de regressão
from dash import Dash, html, dcc, Input, Output  # Para construção do dashboard web

# Carrega os dados do arquivo CSV
df = pd.read_csv('ecommerce_estatistica.csv')

# Processamento inicial dos dados:
# Cria uma coluna 'Grupo_Consolidado' baseada no gênero
df['Grupo_Consolidado'] = df['Gênero'].apply(
    lambda x: 'Infantil' if ('menino' in str(x).lower() or 'menina' in str(x).lower())
    else 'Homens' if ('masculino' in str(x).lower())
    else 'Mulheres' if ('feminino' in str(x).lower())
    else 'Outros')

# Filtra apenas os grupos principais para análise
df_filtrado = df[df['Grupo_Consolidado'].isin(['Homens', 'Mulheres', 'Infantil'])]
contagem = df_filtrado['Grupo_Consolidado'].value_counts()


def padronizar_grafico(fig, titulo=None, altura=500):
    """
    Padroniza o layout dos gráficos para manter consistência visual.

    Args:
        fig: Figura Plotly a ser formatada
        titulo: Título do gráfico (opcional)
        altura: Altura do gráfico em pixels (padrão: 500)

    Returns:
        Figura Plotly formatada
    """
    fig.update_layout(
        title_text=titulo,
        title_x=0.5,  # Centraliza o título
        height=altura,
        plot_bgcolor='rgba(240,240,240,0.9)',  # Cor de fundo do gráfico
        paper_bgcolor='white',  # Cor de fundo da área externa
        hoverlabel=dict(bgcolor="white"),  # Estilo do tooltip
        margin=dict(l=20, r=20, t=60, b=20),  # Margens
        font=dict(family="Arial")  # Fonte padrão
    )
    return fig


def calcular_coeficientes(df):
    """
    Calcula coeficientes de regressão linear para cada categoria.

    Args:
        df: DataFrame com os dados de entrada

    Returns:
        DataFrame com os coeficientes para cada categoria
    """
    categories = df['Grupo_Consolidado'].unique()
    coeff_data = []

    for cat in categories:
        # Filtra dados para a categoria atual
        temp_df = df[df['Grupo_Consolidado'] == cat]

        # Variáveis independentes (features)
        X = temp_df[['Preço_MinMax', 'Desconto_MinMax', 'N_Avaliações_MinMax']]
        # Variável dependente (target)
        y = temp_df['Qtd_Vendidos_Cod']

        # Ajusta modelo de regressão linear
        model = LinearRegression().fit(X, y)

        # Armazena os coeficientes encontrados
        coeff_data.append({
            'Categoria': cat,
            'Preço': model.coef_[0],  # Coeficiente para Preço
            'Desconto': model.coef_[1],  # Coeficiente para Desconto
            'Avaliações': model.coef_[2]  # Coeficiente para Número de Avaliações
        })

    return pd.DataFrame(coeff_data).set_index('Categoria')


def cria_graficos(df):
    """
    Cria todos os gráficos para o dashboard.

    Args:
        df: DataFrame com os dados processados

    Returns:
        Tupla com todas as figuras Plotly criadas
    """
    # Gráfico 1: Histograma de distribuição de notas
    fig1 = px.histogram(df, x='Nota', nbins=30, color='Grupo_Consolidado',
                        hover_name='Grupo_Consolidado')
    fig1 = padronizar_grafico(fig1, titulo='Distribuição de Notas')
    fig1.update_traces(hovertemplate="<br>Notas: %{x}<br>")
    fig1.update_layout(
        xaxis_title='Notas',
        yaxis_title='Quantidade de Notas',
        legend_title='Notas por Gênero'
    )

    # Gráfico 2: Dispersão Preço vs Quantidade Vendida
    fig2 = px.scatter(df, x='Preço', y='Qtd_Vendidos_Cod',
                      color='Grupo_Consolidado', hover_name='Grupo_Consolidado')
    fig2.update_traces(
        hovertemplate="<b>%{hovertext}</b><br>Preço Unitário: %{x}<br>Quantidade Vendida: %{y}<extra></extra>"
    )
    fig2 = padronizar_grafico(fig2, titulo='Preço vs Quantidade Vendida')
    fig2.update_layout(
        yaxis_title='Quantidade de Vendas',
        legend_title='Vendas por Gênero'
    )

    # Gráfico 3: Mapa de calor de correlação entre variáveis
    df_corr = df[['Nota', 'N_Avaliações', 'Preço', 'Qtd_Vendidos_Cod', 'Desconto']].corr()
    fig3 = px.imshow(df_corr, text_auto=True, aspect='auto')
    fig3 = padronizar_grafico(fig3, titulo='Mapa de Calor de Correlação', altura=550)

    # Gráfico 4: Top 10 marcas por vendas (gráfico de barras)
    top_marcas = df.groupby('Marca')['Qtd_Vendidos_Cod'].sum().nlargest(10).reset_index()
    fig4 = px.bar(top_marcas, x='Marca', y='Qtd_Vendidos_Cod', color='Marca',
                  hover_name='Marca', labels={'Qtd_Vendidos_Cod': 'Quantidade Vendida '})
    fig4 = padronizar_grafico(fig4, titulo='Top 10 Marcas por Vendas')
    fig4.update_traces(hovertemplate="<b>%{label}</b><br>Quantidade: %{value}")
    fig4.update_layout(
        xaxis_title='Marca',
        yaxis_title='Quantidade de Vendas',
        legend_title='Top 10 Marcas por Vendas'
    )

    # Gráfico 5: Distribuição por grupo (gráfico de pizza)
    contagem_df = df['Grupo_Consolidado'].value_counts().reset_index()
    contagem_df.columns = ['Grupo', 'Quantidade']

    # Cores personalizadas para cada grupo
    cores = {
        'Homens': '#3498db',  # Azul
        'Mulheres': '#e74c3c',  # Vermelho
        'Infantil': '#f39c12',  # Laranja
        'Outros': '#2ecc71'  # Verde
    }

    fig5 = px.pie(contagem_df, names='Grupo', values='Quantidade',
                  color='Grupo', color_discrete_map=cores)
    fig5 = padronizar_grafico(fig5, titulo='Distribuição por Grupo', altura=450)
    fig5.update_traces(
        textinfo='percent+label',  # Mostra percentual e label
        hovertemplate="<b>%{label}</b><br>Quantidade: %{value}<br>Percentual: %{percent}",
        textfont_size=12,
        marker=dict(line=dict(color='white', width=1))  # Borda branca para separação
    )
    fig5.update_layout(
        legend_title='Distribuição de Produtos por Gênero',
        legend=dict(
            orientation="h",  # Legenda horizontal
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        uniformtext_minsize=10,
        uniformtext_mode='hide'
    )

    # Gráfico 6: Densidade (KDE) Preço vs Vendas
    fig6 = px.density_contour(df, x='Preço', y='Qtd_Vendidos_Cod')
    fig6 = padronizar_grafico(fig6, titulo='Densidade: Preço vs Vendas')
    fig6.update_traces(
        contours_coloring="fill",
        contours_showlabels=True,
        hovertemplate="<b>Preço</b>: %{x}<br><b>Vendas</b>: %{y}<br><b>Densidade</b>: %{z}<extra></extra>"
    )
    fig6.update_layout(yaxis_title='Quantidade de Vendas')

    # Gráfico 7: Dispersão multivariada com facetas por grupo
    fig7 = px.scatter(
        df,
        x='Preço_MinMax',
        y='Qtd_Vendidos_Cod',
        color='Desconto_MinMax',  # Cor representa desconto
        size='N_Avaliações_MinMax',  # Tamanho representa avaliações
        facet_col='Grupo_Consolidado',  # Separa por grupo
        trendline="ols",  # Linha de tendência
        title='Relação Preço-Vendas com Contexto',
        labels={
            'Preço_MinMax': 'Preço (Normalizado)',
            'Qtd_Vendidos_Cod': 'Vendas (Normalizado)',
            'Desconto_MinMax': 'Desconto'
        },
        height=500
    )
    fig7.update_layout(
        title_x=0.5,  # Centraliza o título
        margin=dict(l=50, r=50, b=50),
        coloraxis_colorbar=dict(title="Desconto"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3)
    )

    # Gráfico 8: Heatmap de sensibilidade (coeficientes de regressão)
    coeff_df = calcular_coeficientes(df)
    fig8 = px.imshow(
        coeff_df.T,
        text_auto=".2f",  # Mostra valores com 2 casas decimais
        color_continuous_scale='RdBu',  # Escala de cores divergente
        aspect="auto",
        title='Sensibilidade por Categoria (Coeficientes Padronizados)',
        labels=dict(x="Categoria", y="Fator", color="Impacto"),
        zmin=-1,  # Mínimo da escala
        zmax=1  # Máximo da escala
    )
    fig8.update_layout(
        height=400,
        margin=dict(l=100, r=50, t=80, b=50),
        title_x=0.5,
        title_font=dict(size=16),
        coloraxis_colorbar=dict(
            title="Impacto",
            tickvals=[-1, -0.5, 0, 0.5, 1],
            ticktext=["Negativo Forte", "Negativo", "Neutro", "Positivo", "Positivo Forte"]
        )
    )
    fig8.update_xaxes(tickangle=45)  # Inclina labels do eixo X

    return fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8


def cria_app(df):
    """
    Cria e configura a aplicação Dash.

    Args:
        df: DataFrame com os dados processados

    Returns:
        Aplicação Dash configurada
    """
    app = Dash(__name__)

    # Cria todos os gráficos iniciais
    fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8 = cria_graficos(df)

    # Define o layout do dashboard
    app.layout = html.Div([
        # Cabeçalho com título e filtro
        html.Link(
            href="https://fonts.googleapis.com/css2?family=Montserrat:wght@600&display=swap",
            rel="stylesheet"),
        html.Div([
            html.H1("Dashboard: E-commerce Análise Estatística", style={
                'textAlign': 'center',
                'color': '#2c3e50',
                'marginBottom': '30px',
                'fontFamily': 'Montserrat, sans-serif',
                'fontWeight': '600',
                'fontSize': '2.5rem',
                'letterSpacing': '1px',
                'textTransform': 'uppercase'
            }),

            # Filtro por gênero
            html.Div([
                dcc.Dropdown(
                    id='filtro-genero',
                    options=[{'label': g, 'value': g} for g in ['Todos'] + sorted(df['Grupo_Consolidado'].unique())],
                    value='Todos',
                    placeholder="Filtrar por gênero",
                    style={'width': '300px', 'margin': '0 auto'}
                )
            ], style={'marginBottom': '30px'})
        ], style={'backgroundColor': '#f8f9fa', 'padding': '10px'}),

        # Container principal com os gráficos
        html.Div([
            # Linha 1: Histograma e Dispersão
            html.Div([
                dcc.Graph(figure=fig1, style=graph_style, id='graph1'),
                dcc.Graph(figure=fig2, style=graph_style, id='graph2')
            ], style=row_style),

            # Linha 2: Mapa de Calor e Top Marcas
            html.Div([
                dcc.Graph(figure=fig3, style=graph_style, id='graph3'),
                dcc.Graph(figure=fig4, style={**graph_style, 'height': '70vh'}, id='graph4')
            ], style=row_style),

            # Linha 3: Pizza e Densidade
            html.Div([
                dcc.Graph(figure=fig5, style=graph_style, id='graph5'),
                dcc.Graph(figure=fig6, style=graph_style, id='graph6')
            ], style=row_style),

            # Gráfico 7: Dispersão multivariada (largura completa)
            dcc.Graph(figure=fig7, style={**graph_style, 'width': '96%'}, id='graph7'),

            # Gráfico 8: Heatmap de sensibilidade (centralizado)
            dcc.Graph(
                figure=fig8,
                style={
                    'width': '60%',
                    'height': '50vh',
                    'margin': '20px auto',
                    'backgroundColor': 'white'
                },
                id='graph8',
                config={'responsive': True}
            )
        ], style={'maxWidth': '1800px', 'margin': '0 auto'})
    ])

    # Callback para atualizar os gráficos quando o filtro muda
    @app.callback(
        [Output(f'graph{i}', 'figure') for i in range(1, 9)],
        [Input('filtro-genero', 'value')]
    )
    def update_graphs(genero_selecionado):
        """
        Atualiza todos os gráficos com base no filtro selecionado.

        Args:
            genero_selecionado: Valor do dropdown ('Todos' ou um grupo específico)

        Returns:
            Lista com todas as figuras atualizadas
        """
        df_filtrado = df if genero_selecionado == 'Todos' else df[df['Grupo_Consolidado'] == genero_selecionado]
        return cria_graficos(df_filtrado)

    return app


# Estilos CSS reutilizáveis
graph_style = {
    'width': '48%',  # Largura do gráfico
    'height': '60vh',  # Altura do gráfico
    'display': 'inline-block',  # Layout inline
    'margin': '10px',  # Margem externa
    'backgroundColor': 'white'  # Cor de fundo
}

row_style = {
    'textAlign': 'center',  # Alinhamento do conteúdo
    'marginBottom': '20px'  # Margem inferior
}

# Ponto de entrada da aplicação
if __name__ == '__main__':
    # Cria e executa a aplicação
    app = cria_app(df)
    app.run(debug=True, port=8050)  # Executa em modo debug na porta 8050