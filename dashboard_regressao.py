# app.py
import argparse
import math
import warnings
from io import StringIO
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import curve_fit
import scipy.cluster.hierarchy as sch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import statsmodels.api as sm
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Supervisado imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
                             classification_report, roc_auc_score, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

warnings.filterwarnings('ignore')

# ---------------------------- UTILITÁRIOS ----------------------------
def carregar_dados(caminho='tabelinha2.xlsx', colunas_necessarias=None):
    if colunas_necessarias is None:
        colunas_necessarias = ["Ano Modelo", "Tamanho do motor", "Preço médio brl"]
    df = pd.read_excel(caminho)
    df.columns = [c.strip() for c in df.columns]
    existentes = [c for c in colunas_necessarias if c in df.columns]
    if not existentes:
        existentes = df.columns.tolist()
    df_limpo = df.copy()
    for c in existentes:
        try:
            df_limpo[c] = pd.to_numeric(df_limpo[c], errors='coerce')
        except Exception:
            pass
    df_limpo = df_limpo.dropna(subset=existentes).reset_index(drop=True)
    return df_limpo, existentes

def resumo_estatistico(df, colunas):
    resultados = {}
    for col in colunas:
        s = df[col]
        resultados[col] = {
            'media': float(s.mean()) if s.size>0 else np.nan,
            'mediana': float(s.median()) if s.size>0 else np.nan,
            'moda': float(s.mode().iat[0]) if not s.mode().empty else np.nan,
            'variancia': float(s.var()) if s.size>0 else np.nan,
            'desvio_padrao': float(s.std()) if s.size>0 else np.nan,
            'assimetria': float(s.skew()) if s.size>0 else np.nan,
            'curtose': float(s.kurtosis()) if s.size>0 else np.nan,
            'min': float(s.min()) if s.size>0 else np.nan,
            'max': float(s.max()) if s.size>0 else np.nan,
            'n': int(s.count())
        }
    return resultados

# ---------------------------- FEATURE ENGINEERING ----------------------------
def criar_features_temporais(df, col_ano='Ano Modelo'):
    """Cria features de idade e depreciação"""
    df_eng = df.copy()
    ano_atual = 2025  # Pode ser parametrizado
    if col_ano in df_eng.columns:
        df_eng['Idade_Veiculo'] = ano_atual - df_eng[col_ano]
        df_eng['Anos_desde_2000'] = df_eng[col_ano] - 2000
    return df_eng

# ---------------------------- MODELOS REGRESSÃO (mantidos) ----------------------------
def ajusta_regressao_linear_simples(x, y):
    x_resh = x.reshape(-1, 1)
    model = LinearRegression()
    model.fit(x_resh, y)
    y_pred = model.predict(x_resh)
    r2 = r2_score(y, y_pred)
    rmse = math.sqrt(mean_squared_error(y, y_pred))
    return model, y_pred, r2, rmse

def ajusta_regressao_linear_multivariada(X, y):
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = math.sqrt(mean_squared_error(y, y_pred))
    return model, y_pred, r2, rmse

def ajusta_regressao_polinomial(x, y, grau=2):
    model = make_pipeline(PolynomialFeatures(degree=grau, include_bias=False), LinearRegression())
    x_resh = x.reshape(-1, 1)
    model.fit(x_resh, y)
    y_pred = model.predict(x_resh)
    r2 = r2_score(y, y_pred)
    rmse = math.sqrt(mean_squared_error(y, y_pred))
    return model, y_pred, r2, rmse

def modelo_exponencial(x, a, b):
    return a * np.exp(b * x)

def ajusta_exponencial(x, y, p0=None):
    if p0 is None:
        p0 = [y.max() if hasattr(y, 'max') and y.max() > 0 else 1.0, 0.01]
    try:
        popt, pcov = curve_fit(modelo_exponencial, x, y, p0=p0, maxfev=10000)
        y_pred = modelo_exponencial(x, *popt)
        r2 = r2_score(y, y_pred)
        rmse = math.sqrt(mean_squared_error(y, y_pred))
        return popt, y_pred, r2, rmse
    except Exception:
        return None, None, None, None

def modelo_logistico(x, a, b, c):
    return a / (1 + np.exp(-b * (x - c)))

def ajusta_logistico(x, y, p0=None):
    if p0 is None:
        try:
            p0 = [max(y), 1, np.median(x)]
        except Exception:
            p0 = [1,1,np.median(x)]
    try:
        popt, pcov = curve_fit(modelo_logistico, x, y, p0=p0, maxfev=10000)
        y_pred = modelo_logistico(x, *popt)
        r2 = r2_score(y, y_pred)
        rmse = math.sqrt(mean_squared_error(y, y_pred))
        return popt, y_pred, r2, rmse
    except Exception:
        return None, None, None, None

def modelo_potencia(x, a, b):
    return a * np.power(x, b)

def ajusta_potencia(x, y, p0=None):
    if p0 is None:
        p0 = [1, 1]
    try:
        popt, pcov = curve_fit(modelo_potencia, x, y, p0=p0, maxfev=10000)
        y_pred = modelo_potencia(x, *popt)
        r2 = r2_score(y, y_pred)
        rmse = math.sqrt(mean_squared_error(y, y_pred))
        return popt, y_pred, r2, rmse
    except Exception:
        return None, None, None, None

# ---------------------------- ANÁLISE DE RESÍDUOS ----------------------------
def analisar_residuos(y_true, y_pred):
    """Calcula resíduos e testes estatísticos"""
    residuos = y_true - y_pred
    resultado = {
        'residuos': residuos,
        'media_residuos': float(np.mean(residuos)),
        'std_residuos': float(np.std(residuos)),
        'shapiro_stat': None,
        'shapiro_p': None,
        'normalidade': None
    }
    
    # Teste de normalidade (Shapiro-Wilk)
    if len(residuos) >= 3:
        try:
            stat, p = stats.shapiro(residuos)
            resultado['shapiro_stat'] = float(stat)
            resultado['shapiro_p'] = float(p)
            resultado['normalidade'] = 'Normal' if p > 0.05 else 'Não Normal'
        except Exception:
            pass
    
    return resultado

# ---------------------------- CLUSTERIZAÇÃO ----------------------------
def aplicar_kmeans(X, n_clusters=3, random_state=0):
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(X)
    return labels, km

def aplicar_agglomerative(X, n_clusters=3, linkage='ward'):
    ag = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = ag.fit_predict(X)
    return labels, ag

def aplicar_gmm(X, n_components=3, covariance_type='full', random_state=0):
    gm = GaussianMixture(n_components=n_components, covariance_type=covariance_type, random_state=random_state)
    labels = gm.fit_predict(X)
    return labels, gm

def avaliar_clusters(X, labels):
    result = {}
    if len(np.unique(labels)) <= 1 or len(labels) < 2:
        result['silhouette'] = None
        result['calinski_harabasz'] = None
        result['davies_bouldin'] = None
        return result
    try:
        result['silhouette'] = float(silhouette_score(X, labels))
    except Exception:
        result['silhouette'] = None
    try:
        result['calinski_harabasz'] = float(calinski_harabasz_score(X, labels))
    except Exception:
        result['calinski_harabasz'] = None
    try:
        result['davies_bouldin'] = float(davies_bouldin_score(X, labels))
    except Exception:
        result['davies_bouldin'] = None
    return result

# ---------------------------- SUPERVISIONADOS ----------------------------
def is_classification_target(y):
    try:
        if y.dtype == object:
            return True
    except Exception:
        pass
    uniques = np.unique(y[~pd.isna(y)])
    if len(uniques) <= 10 and np.all(np.mod(uniques, 1) == 0):
        return True
    return False

def treinar_e_avaliar_supervisionado(X, y, test_size=0.25, random_state=42, task=None):
    """Treina e avalia modelos supervisionados de forma robusta (sem travar)."""
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
    from sklearn.neural_network import MLPRegressor, MLPClassifier
    from sklearn.metrics import (
        mean_squared_error, r2_score,
        accuracy_score, precision_score, recall_score, f1_score,
        classification_report, confusion_matrix, roc_auc_score, roc_curve
    )
    import math

    # Garante que X e y sejam arrays numéricos
    X_df = pd.DataFrame(X).apply(pd.to_numeric, errors='coerce')
    y_series = pd.Series(y).dropna()
    X_df = X_df.loc[y_series.index].dropna()

    if X_df.shape[0] < 3:
        raise ValueError("Poucos dados válidos para treinar modelos supervisionados.")

    y_clean = y_series.values
    X_clean = X_df.values

    # --- detecção robusta da tarefa ---
    if task is None:
        y_unique = np.unique(y_clean)
        if y_series.dtype == object:
            task = 'classification'
        elif np.all(np.mod(y_unique, 1) == 0) and len(y_unique) <= 10:
            task = 'classification'
        else:
            task = 'regression'

    stratify = y_clean if task == 'classification' and len(np.unique(y_clean)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean, test_size=test_size, random_state=random_state, stratify=stratify
    )

    results, models = {}, {}

    if task == 'regression':
        model_defs = {
            'DecisionTree': DecisionTreeRegressor(random_state=random_state),
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=random_state),
            'KNN': KNeighborsRegressor(n_neighbors=5),
            'MLP': MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=1000, random_state=random_state)
        }
    else:
        model_defs = {
            'DecisionTree': DecisionTreeClassifier(random_state=random_state),
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=random_state),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'MLP': MLPClassifier(hidden_layer_sizes=(50, 50), max_iter=1000, random_state=random_state)
        }

    for name, model in model_defs.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            res = {'model': model, 'y_pred': y_pred}

            if task == 'regression':
                r2 = r2_score(y_test, y_pred)
                rmse = math.sqrt(mean_squared_error(y_test, y_pred))
                try:
                    cv = cross_val_score(model, X_clean, y_clean, cv=5, scoring='r2')
                    res['cv_r2_mean'], res['cv_r2_std'] = float(np.mean(cv)), float(np.std(cv))
                except Exception:
                    res['cv_r2_mean'] = res['cv_r2_std'] = None
                res.update({'R2': float(r2), 'RMSE': float(rmse)})

            else:
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                res.update({'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1})
                res['classification_report'] = classification_report(y_test, y_pred, zero_division=0)
                res['confusion_matrix'] = confusion_matrix(y_test, y_pred)
                if len(np.unique(y_test)) == 2 and hasattr(model, "predict_proba"):
                    try:
                        y_proba = model.predict_proba(X_test)[:, 1]
                        auc = roc_auc_score(y_test, y_proba)
                        fpr, tpr, _ = roc_curve(y_test, y_proba)
                        res.update({'roc_auc': auc, 'roc_curve': (fpr, tpr)})
                    except Exception:
                        pass

            results[name] = res
            models[name] = model
        except Exception as e:
            results[name] = {'error': str(e)}

    # Escolher melhor modelo
    best = None
    if task == 'regression':
        valids = [(r['R2'], n) for n, r in results.items() if 'R2' in r]
        if valids:
            best = sorted(valids, reverse=True)[0][1]
    else:
        valids = [(r['f1'], n) for n, r in results.items() if 'f1' in r]
        if valids:
            best = sorted(valids, reverse=True)[0][1]

    return {
        'task': task, 'results': results, 'models': models, 'best': best,
        'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test
    }

# Plot helpers (matplotlib)
def plot_confusion_matrix(cm, labels=None, figsize=(6,5)):
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title("Matriz de Confusão")
    plt.colorbar(im, ax=ax)
    if labels is not None:
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45)
        ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(int(cm[i, j]), 'd'), ha="center", va="center", color="white" if cm[i,j] > cm.max()/2 else "black")
    ax.set_ylabel('Verdadeiro')
    ax.set_xlabel('Previsto')
    plt.tight_layout()
    return fig

def plot_regression_pred_vs_obs(y_true, y_pred, titulo="Predito vs Observado"):
    fig, ax = plt.subplots(figsize=(6,5))
    ax.scatter(y_true, y_pred, alpha=0.6)
    minv = min(np.min(y_true), np.min(y_pred))
    maxv = max(np.max(y_true), np.max(y_pred))
    ax.plot([minv, maxv], [minv, maxv], '--', linewidth=1, label='y = Predito', color='red')
    ax.set_xlabel('Observado')
    ax.set_ylabel('Predito')
    ax.set_title(titulo)
    ax.legend()
    plt.tight_layout()
    return fig

def plot_residuos(y_true, y_pred):
    """Gera plots de análise de resíduos"""
    residuos = y_true - y_pred
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot 1: Resíduos vs Preditos
    axes[0].scatter(y_pred, residuos, alpha=0.6)
    axes[0].axhline(y=0, color='r', linestyle='--')
    axes[0].set_xlabel('Valores Preditos')
    axes[0].set_ylabel('Resíduos')
    axes[0].set_title('Resíduos vs Preditos')
    
    # Plot 2: Histograma dos Resíduos
    axes[1].hist(residuos, bins=20, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Resíduos')
    axes[1].set_ylabel('Frequência')
    axes[1].set_title('Distribuição dos Resíduos')
    
    # Plot 3: Q-Q Plot
    stats.probplot(residuos, dist="norm", plot=axes[2])
    axes[2].set_title('Q-Q Plot')
    
    plt.tight_layout()
    return fig

# ---------------------------- RELATÓRIO ----------------------------
def resumo_modelo_linear_statsmodels(X, y):
    Xc = sm.add_constant(X)
    model = sm.OLS(y, Xc).fit()
    return model

def gerar_relatorio_texto(df, colunas, resultados_stats, modelos_resumo, clusters_info=None, supervised_summary=None):
    buf = StringIO()
    print("RELATÓRIO - ANÁLISE E MODELOS", file=buf)
    print("="*80, file=buf)
    print(f"Registros: {len(df)}\n", file=buf)
    print("== Estatísticas descritivas ==\n", file=buf)
    for col in colunas:
        r = resultados_stats[col]
        print(
            f"{col} - n={r['n']}, média={r['media']:.3f}, mediana={r['mediana']:.3f}, "
            f"moda={r['moda']:.3f}, variância={r['variancia']:.3f}, dp={r['desvio_padrao']:.3f}, "
            f"assimetria={r['assimetria']:.3f}, curtose={r['curtose']:.3f}, min={r['min']:.3f}, max={r['max']:.3f}",
            file=buf
        )
    print("\n== Modelos ajustados (Regressões) ==\n", file=buf)
    for nome, info in modelos_resumo.items():
        print(f"Modelo: {nome}", file=buf)
        for k, v in info.items():
            print(f" {k}: {v}", file=buf)
        print("\n", file=buf)
    if clusters_info:
        print("== Clusterização ==\n", file=buf)
        for nome, info in clusters_info.items():
            print(f"{nome}:", file=buf)
            for k, v in info.items():
                print(f"  {k}: {v}", file=buf)
            print("", file=buf)
    if supervised_summary:
        print("== Modelos Supervisionados ==\n", file=buf)
        best = supervised_summary.get('best')
        for name, info in supervised_summary.get('results', {}).items():
            print(f"Modelo {name}:", file=buf)
            for k, v in info.items():
                if k in ['model', 'y_pred', 'confusion_matrix', 'roc_curve']:
                    continue
                print(f"  {k}: {v}", file=buf)
            print("", file=buf)
        print(f"Melhor método sugerido (supervisionado): {best}\n", file=buf)
        print("== Recomendação / Prescrição ==\n", file=buf)
        if supervised_summary.get('task') == 'regression':
            print(f"Recomendamos utilizar o modelo {best} para previsões (priorizando R² e RMSE).", file=buf)
            print("Ações prescritivas sugeridas: validar com cross-validation, monitorar resíduos e ajustar features.", file=buf)
        else:
            print(f"Recomendamos utilizar o modelo {best} (priorizando F1/accuracy).", file=buf)
            print("Ações prescritivas sugeridas: calibrar thresholds, balancear classes se necessário, pipeline de deploy e monitoramento de drift.", file=buf)
    print("\n== Matriz de Covariância ==\n", file=buf)
    try:
        print(df[colunas].cov(), file=buf)
    except Exception:
        print("não foi possível calcular matriz de covariância.", file=buf)
    print("\n== Matriz de Correlação ==\n", file=buf)
    try:
        print(df[colunas].corr(), file=buf)
    except Exception:
        print("não foi possível calcular matriz de correlação.", file=buf)
    return buf.getvalue()

def gerar_report_cli(df, colunas, output_path='relatorio_regressao.txt'):
    stats = resumo_estatistico(df, colunas)
    modelos_resumo = {}
    supervised_summary = None
    try:
        if len(colunas) >= 2:
            target = colunas[-1]
            features = colunas[:-1]
            X = df[features].values
            y = df[target].values
            supervised_summary = treinar_e_avaliar_supervisionado(X, y)
            if supervised_summary['task'] == 'regression':
                for nome, info in supervised_summary['results'].items():
                    if 'R2' in info and 'RMSE' in info:
                        modelos_resumo[nome] = {'R2': info['R2'], 'RMSE': info['RMSE']}
    except Exception as e:
        print("Aviso: não foi possível treinar modelos supervisionados no modo CLI:", e)
    texto = gerar_relatorio_texto(df, colunas, stats, modelos_resumo, clusters_info=None, supervised_summary=supervised_summary)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(texto)
    print(f"Relatório salvo em {output_path}")

# ---------------------------- DASHBOARD STREAMLIT ----------------------------
def streamlit_app(df, colunas):
    st.set_page_config(page_title='Dashboard de Previsão de Automóveis', layout='wide')
    st.title('🚗 Dashboard Completo: Previsão de Valores de Automóveis')
    
    # Criar features temporais
    df = criar_features_temporais(df)
    colunas_eng = [c for c in df.columns if c in colunas or c in ['Idade_Veiculo', 'Anos_desde_2000']]
    
    st.sidebar.header('Configurações')
    target = st.sidebar.selectbox('Escolha a variável alvo (y) para regressões', colunas, index=len(colunas)-1)
    features = st.sidebar.multiselect('Escolha variáveis independentes (X) para regressões', [c for c in colunas_eng if c != target], default=[c for c in colunas if c != target])
    grau_poly = st.sidebar.slider('Grau do polinômio (para regressão polinomial univariada)', 2, 5, 2)

    # Estatísticas básicas (rápido)
    stats = resumo_estatistico(df, colunas)

    # INICIALIZA VARIÁVEIS (cálculo sob demanda)
    r2_lin = rmse_lin = r2_poly = rmse_poly = r2_exp = rmse_exp = r2_mv = rmse_mv = r2_log = rmse_log = r2_pot = rmse_pot = None
    model_lin = model_poly = model_mv = None
    y_pred_lin = y_pred_poly = y_pred_exp = y_pred_log = y_pred_pot = y_pred_mv = None

    # ABAS
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
        "📊 Estatísticas Descritivas",
        "📉 Gráficos por Variável",
        "🔗 Correlações & Modelos",
        "📋 Resumo Final",
        "🖼️ Infográfico",
        "🔍 Clusterização",
        "🧠 Supervisionado",
        "🔮 Previsão Futura",
        "📈 Análise Temporal",
        "🏆 Comparação Global"
    ])

    # TAB 1 - Estatísticas (mantida)
    with tab1:
        st.subheader("📊 Estatísticas Descritivas e Gráficos")
        df_stats = pd.DataFrame.from_dict(stats, orient='index')
        st.write(df_stats)
        st.markdown("---")
        st.subheader("📈 Gráficos Estatísticos Individuais")
        fig_media = px.bar(df_stats.reset_index(), x='index', y='media', title="📊 Média por Variável", color='media', color_continuous_scale='Blues')
        st.plotly_chart(fig_media, use_container_width=True)
        fig_mediana = px.bar(df_stats.reset_index(), x='index', y='mediana', title="📊 Mediana por Variável", color='mediana', color_continuous_scale='Greens')
        st.plotly_chart(fig_mediana, use_container_width=True)
        fig_moda = px.bar(df_stats.reset_index(), x='index', y='moda', title="📊 Moda por Variável", color='moda', color_continuous_scale='Purples')
        st.plotly_chart(fig_moda, use_container_width=True)
        st.markdown("---")
        st.subheader("📉 Variância e Desvio Padrão")
        fig_var = px.bar(df_stats.reset_index(), x='index', y='variancia', title="📉 Variância por Variável", color='variancia', color_continuous_scale='Viridis')
        st.plotly_chart(fig_var, use_container_width=True)
        fig_std = px.bar(df_stats.reset_index(), x='index', y='desvio_padrao', title="📉 Desvio Padrão por Variável", color='desvio_padrao', color_continuous_scale='Cividis')
        st.plotly_chart(fig_std, use_container_width=True)
        st.markdown("---")
        st.subheader("🧮 Matriz de Covariância")
        cov_matrix = df[colunas].cov().round(3)
        fig_cov = px.imshow(cov_matrix, text_auto=True, title="🧮 Matriz de Covariância entre Variáveis", color_continuous_scale='RdBu', zmin=-abs(cov_matrix.values).max(), zmax=abs(cov_matrix.values).max())
        st.plotly_chart(fig_cov, use_container_width=True)

    # TAB 2 - Gráficos por Variável (mantida)
    with tab2:
        st.subheader("📉 Distribuições e Relações")
        col1, col2 = st.columns(2)
        with col1:
            var_hist = st.selectbox("Selecione variável para histograma", colunas, key="hist_var")
            fig_hist = px.histogram(df, x=var_hist, nbins=20, title=f"Distribuição de {var_hist}")
            st.plotly_chart(fig_hist, use_container_width=True)
        with col2:
            xvar = st.selectbox("Eixo X (relação)", colunas, key="xvar_grafico")
            yvar = st.selectbox("Eixo Y (relação)", [c for c in colunas if c != xvar], key="yvar_grafico")
            fig_scatter = px.scatter(df, x=xvar, y=yvar, trendline="ols", title=f"Relação entre {xvar} e {yvar}")
            st.plotly_chart(fig_scatter, use_container_width=True)

    # TAB 3 - Modelos (CÁLCULO SOB DEMANDA)
    with tab3:
        st.subheader("📈 Ajustes de Regressão")
        
        # Botão para calcular modelos
        calcular = st.button("🔄 Calcular Modelos de Regressão", type="primary")
        
        if calcular or 'modelos_calculados' in st.session_state:
            if calcular:
                with st.spinner("Calculando modelos... aguarde"):
                    ycol = target
                    xcol = features[0] if features else None
                    y = df[ycol].values if ycol in df.columns else None
                    
                    if xcol and len(features) >= 1:
                        x = df[xcol].values
                        try:
                            model_lin, y_pred_lin, r2_lin, rmse_lin = ajusta_regressao_linear_simples(x, y)
                        except Exception:
                            model_lin = y_pred_lin = r2_lin = rmse_lin = None
                        try:
                            model_poly, y_pred_poly, r2_poly, rmse_poly = ajusta_regressao_polinomial(x, y, grau=grau_poly)
                        except Exception:
                            model_poly = y_pred_poly = r2_poly = rmse_poly = None
                        try:
                            if np.all(y > 0):
                                _, y_pred_exp, r2_exp, rmse_exp = ajusta_exponencial(x, y)
                            else:
                                y_pred_exp = r2_exp = rmse_exp = None
                        except Exception:
                            y_pred_exp = r2_exp = rmse_exp = None
                        try:
                            if np.all(x > 0):
                                _, y_pred_pot, r2_pot, rmse_pot = ajusta_potencia(x, y)
                            else:
                                y_pred_pot = r2_pot = rmse_pot = None
                        except Exception:
                            y_pred_pot = r2_pot = rmse_pot = None
                        try:
                            _, y_pred_log, r2_log, rmse_log = ajusta_logistico(x, y)
                        except Exception:
                            y_pred_log = r2_log = rmse_log = None
                        if len(features) >= 2:
                            X = df[features].values
                            try:
                                model_mv, y_pred_mv, r2_mv, rmse_mv = ajusta_regressao_linear_multivariada(X, y)
                            except Exception:
                                model_mv = y_pred_mv = r2_mv = rmse_mv = None
                        
                        # Salvar em session_state
                        st.session_state['modelos_calculados'] = {
                            'x': x, 'y': y, 'xcol': xcol, 'ycol': ycol,
                            'r2_lin': r2_lin, 'rmse_lin': rmse_lin, 'y_pred_lin': y_pred_lin,
                            'r2_poly': r2_poly, 'rmse_poly': rmse_poly, 'y_pred_poly': y_pred_poly,
                            'r2_exp': r2_exp, 'rmse_exp': rmse_exp, 'y_pred_exp': y_pred_exp,
                            'r2_log': r2_log, 'rmse_log': rmse_log, 'y_pred_log': y_pred_log,
                            'r2_pot': r2_pot, 'rmse_pot': rmse_pot, 'y_pred_pot': y_pred_pot,
                            'r2_mv': r2_mv, 'rmse_mv': rmse_mv, 'y_pred_mv': y_pred_mv,
                            'grau_poly': grau_poly
                        }
                    else:
                        st.warning("Selecione pelo menos 1 feature no painel lateral")
            
            # Recuperar dados
            if 'modelos_calculados' in st.session_state:
                dados = st.session_state['modelos_calculados']
                x, y, xcol, ycol = dados['x'], dados['y'], dados['xcol'], dados['ycol']
                r2_lin, rmse_lin, y_pred_lin = dados['r2_lin'], dados['rmse_lin'], dados['y_pred_lin']
                r2_poly, rmse_poly, y_pred_poly = dados['r2_poly'], dados['rmse_poly'], dados['y_pred_poly']
                r2_exp, rmse_exp, y_pred_exp = dados['r2_exp'], dados['rmse_exp'], dados['y_pred_exp']
                r2_log, rmse_log, y_pred_log = dados['r2_log'], dados['rmse_log'], dados['y_pred_log']
                r2_pot, rmse_pot, y_pred_pot = dados['r2_pot'], dados['rmse_pot'], dados['y_pred_pot']
                r2_mv, rmse_mv, y_pred_mv = dados['r2_mv'], dados['rmse_mv'], dados['y_pred_mv']
                grau_poly = dados['grau_poly']
                
                df_plot = pd.DataFrame({xcol: x, ycol: y})
                if y_pred_lin is not None:
                    df_plot['lin_pred'] = y_pred_lin
                if y_pred_poly is not None:
                    df_plot['poly_pred'] = y_pred_poly
                if y_pred_exp is not None:
                    df_plot['exp_pred'] = y_pred_exp
                if y_pred_log is not None:
                    df_plot['log_pred'] = y_pred_log
                if y_pred_pot is not None:
                    df_plot['pot_pred'] = y_pred_pot
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_plot[xcol], y=df_plot[ycol], mode='markers', name='Observado'))
                if 'lin_pred' in df_plot:
                    fig.add_trace(go.Scatter(x=df_plot[xcol], y=df_plot['lin_pred'], mode='lines', name=f'Linear (R²={r2_lin:.3f})' if r2_lin else 'Linear'))
                if 'poly_pred' in df_plot:
                    fig.add_trace(go.Scatter(x=df_plot[xcol], y=df_plot['poly_pred'], mode='lines', name=f'Polinomial {grau_poly} (R²={r2_poly:.3f})' if r2_poly else f'Polinomial {grau_poly}'))
                if 'exp_pred' in df_plot:
                    fig.add_trace(go.Scatter(x=df_plot[xcol], y=df_plot['exp_pred'], mode='lines', name=f'Exponencial (R²={r2_exp:.3f})' if r2_exp else 'Exponencial'))
                if 'log_pred' in df_plot:
                    fig.add_trace(go.Scatter(x=df_plot[xcol], y=df_plot['log_pred'], mode='lines', name=f'Logístico (R²={r2_log:.3f})' if r2_log else 'Logístico'))
                if 'pot_pred' in df_plot:
                    fig.add_trace(go.Scatter(x=df_plot[xcol], y=df_plot['pot_pred'], mode='lines', name=f'Potência (R²={r2_pot:.3f})' if r2_pot else 'Potência'))
                fig.update_layout(title=f"Ajustes de Regressão: {ycol} vs {xcol}", xaxis_title=xcol, yaxis_title=ycol)
                st.plotly_chart(fig, use_container_width=True)

                met_df = pd.DataFrame({
                    'Modelo': ['Linear', f'Polinomial grau {grau_poly}', 'Exponencial', 'Logístico', 'Potência'],
                    'R²': [r2_lin, r2_poly, r2_exp, r2_log, r2_pot],
                    'RMSE': [rmse_lin, rmse_poly, rmse_exp, rmse_log, rmse_pot]
                })
                st.table(met_df)
                
                # ANÁLISE DE RESÍDUOS
                st.markdown("---")
                st.subheader("🔬 Análise de Resíduos")
                modelo_residuo = st.selectbox("Selecione modelo para análise de resíduos", ['Linear', 'Polinomial', 'Exponencial', 'Logístico', 'Potência'])
                y_pred_sel = None
                if modelo_residuo == 'Linear' and y_pred_lin is not None:
                    y_pred_sel = y_pred_lin
                elif modelo_residuo == 'Polinomial' and y_pred_poly is not None:
                    y_pred_sel = y_pred_poly
                elif modelo_residuo == 'Exponencial' and y_pred_exp is not None:
                    y_pred_sel = y_pred_exp
                elif modelo_residuo == 'Logístico' and y_pred_log is not None:
                    y_pred_sel = y_pred_log
                elif modelo_residuo == 'Potência' and y_pred_pot is not None:
                    y_pred_sel = y_pred_pot
                
                if y_pred_sel is not None:
                    res_info = analisar_residuos(y, y_pred_sel)
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Média dos Resíduos", f"{res_info['media_residuos']:.4f}")
                    col2.metric("Desvio Padrão", f"{res_info['std_residuos']:.4f}")
                    if res_info['normalidade']:
                        col3.metric("Normalidade", res_info['normalidade'], f"p={res_info['shapiro_p']:.4f}")
                    
                    fig_res = plot_residuos(y, y_pred_sel)
                    st.pyplot(fig_res)
        else:
            st.info("👆 Clique no botão acima para calcular os modelos de regressão")

    # TAB 4 - Resumo Final (sob demanda)
    with tab4:
        st.subheader("📋 Resumo e Relatório Final")
        
        if st.button("📝 Gerar Relatório Completo"):
            with st.spinner("Gerando relatório..."):
                modelos_resumo = {}
                
                # Pegar dados dos modelos se existirem
                if 'modelos_calculados' in st.session_state:
                    dados = st.session_state['modelos_calculados']
                    for nome, r2, rmse in [
                        ('Linear', dados.get('r2_lin'), dados.get('rmse_lin')),
                        (f"Polinomial {dados.get('grau_poly', 2)}", dados.get('r2_poly'), dados.get('rmse_poly')),
                        ('Exponencial', dados.get('r2_exp'), dados.get('rmse_exp')),
                        ('Logístico', dados.get('r2_log'), dados.get('rmse_log')),
                        ('Potência', dados.get('r2_pot'), dados.get('rmse_pot'))
                    ]:
                        if r2 is not None:
                            modelos_resumo[nome] = {'R2': r2, 'RMSE': rmse}
                
                supervised_summary = st.session_state.get('modelo_treinado')
                texto = gerar_relatorio_texto(df, colunas, stats, modelos_resumo, clusters_info=None, supervised_summary=supervised_summary)
                st.text_area("Relatório (texto)", texto, height=400)
                st.download_button("📥 Download relatório .txt", texto, file_name="relatorio_regressao.txt")
        else:
            st.info("👆 Clique no botão acima para gerar o relatório completo")

    # TAB 5 - Infográfico (mantida)
    with tab5:
        st.subheader("🖼️ Infográfico das Teorias Estatísticas")
        st.markdown("""
        Este infográfico resume as principais teorias estatísticas abordadas:
        - Teorema Central do Limite
        - Correlação
        - Amostragem e Distribuição Normal (Curva de Gauss)
        - Teste T-Student
        - Teste Qui-Quadrado
        """)
        try:
            st.image("InfoGráfico.png", use_container_width=True, caption="Infográfico - Fundamentos Estatísticos")
        except Exception:
            st.info("Coloque um arquivo 'InfoGráfico.png' na pasta do app para visualizar o infográfico.")

    # TAB 6 - CLUSTERIZAÇÃO (mantida)
    with tab6:
        st.subheader("🔍 Clusterização Não Supervisionada")
        st.markdown("Escolha variáveis (2D) para visualizar clusters e compare KMeans, Agglomerative e EM (GMM).")
        cols_for_cluster = st.multiselect("Selecione 2 variáveis para cluster (X, Y)", colunas, default=colunas[:2], max_selections=2)
        n_clusters = st.slider("Número de clusters", 2, 10, 3)
        scale_data = st.checkbox("Padronizar variáveis antes do cluster", value=True)
        linkage = st.selectbox("Linkage (Agglomerative)", ['ward', 'complete', 'average', 'single'])
        covariance_type = st.selectbox("Covariance type (GMM)", ['full', 'tied', 'diag', 'spherical'])

        if len(cols_for_cluster) != 2:
            st.warning("Selecione exatamente 2 variáveis para visualização (X, Y).")
        else:
            X_df = df[cols_for_cluster].dropna().reset_index(drop=True)
            X = X_df.values.astype(float)
            if scale_data:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
            else:
                X_scaled = X

            km_labels, km_model = aplicar_kmeans(X_scaled, n_clusters=n_clusters, random_state=42)
            km_metrics = avaliar_clusters(X_scaled, km_labels)
            ag_labels, ag_model = aplicar_agglomerative(X_scaled, n_clusters=n_clusters, linkage=linkage)
            ag_metrics = avaliar_clusters(X_scaled, ag_labels)
            gmm_labels, gmm_model = aplicar_gmm(X_scaled, n_components=n_clusters, covariance_type=covariance_type, random_state=42)
            gmm_metrics = avaliar_clusters(X_scaled, gmm_labels)

            X_plot = X_df.copy()
            X_plot['KMeans'] = km_labels.astype(int)
            X_plot['Agglomerative'] = ag_labels.astype(int)
            X_plot['GMM'] = gmm_labels.astype(int)

            fig_km = px.scatter(X_plot, x=cols_for_cluster[0], y=cols_for_cluster[1], color='KMeans', title='KMeans Clustering', symbol='KMeans')
            fig_ag = px.scatter(X_plot, x=cols_for_cluster[0], y=cols_for_cluster[1], color='Agglomerative', title='Agglomerative Clustering', symbol='Agglomerative')
            fig_gmm = px.scatter(X_plot, x=cols_for_cluster[0], y=cols_for_cluster[1], color='GMM', title='GMM (EM) Clustering', symbol='GMM')

            c1, c2, c3 = st.columns(3)
            with c1:
                st.plotly_chart(fig_km, use_container_width=True)
            with c2:
                st.plotly_chart(fig_ag, use_container_width=True)
            with c3:
                st.plotly_chart(fig_gmm, use_container_width=True)

            with st.expander("Mostrar dendrograma (Agglomerative)"):
                try:
                    linkage_matrix = sch.linkage(X_scaled, method=linkage if linkage != 'ward' else 'ward')
                    lm_df = pd.DataFrame(linkage_matrix, columns=['idx1','idx2','dist','sample_count'])
                    st.write("Matriz de linkage (primeiras linhas):")
                    st.dataframe(lm_df.head())
                except Exception as e:
                    st.write("Não foi possível gerar dendrograma:", e)

            metrics_table = pd.DataFrame({
                'Método': ['KMeans', 'Agglomerative', 'GMM (EM)'],
                'Silhouette': [km_metrics['silhouette'], ag_metrics['silhouette'], gmm_metrics['silhouette']],
                'Calinski-Harabasz': [km_metrics['calinski_harabasz'], ag_metrics['calinski_harabasz'], gmm_metrics['calinski_harabasz']],
                'Davies-Bouldin': [km_metrics['davies_bouldin'], ag_metrics['davies_bouldin'], gmm_metrics['davies_bouldin']]
            })
            st.subheader("📋 Métricas de Avaliação dos Clusters")
            st.table(metrics_table)

            def pontuar(row):
                score = 0.0
                if not pd.isna(row['Silhouette']):
                    score += row['Silhouette'] * 3
                if not pd.isna(row['Calinski-Harabasz']):
                    score += (row['Calinski-Harabasz'] / (1 + row['Calinski-Harabasz'])) * 1.5
                if not pd.isna(row['Davies-Bouldin']):
                    score += (1 / (1 + row['Davies-Bouldin'])) * 1.0
                return score

            metrics_table['score'] = metrics_table.apply(pontuar, axis=1)
            melhor = metrics_table.sort_values('score', ascending=False).iloc[0]
            st.success(f"Melhor método sugerido: {melhor['Método']} (score={melhor['score']:.3f})")

            if st.button("Adicionar rótulos ao DataFrame original e permitir download"):
                temp = df[cols_for_cluster].reset_index().dropna().reset_index(drop=True)
                temp_out = temp.copy()
                temp_out['KMeans'] = km_labels
                temp_out['Agglomerative'] = ag_labels
                temp_out['GMM'] = gmm_labels
                csv = temp_out.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Baixar CSV com labels", csv, file_name="clusters_labels.csv", mime="text/csv")

    # TAB 7 - SUPERVISIONADO (corrigido e otimizado)
    with tab7:
        st.subheader("🧠 Modelos Supervisionados: DecisionTree, RandomForest, KNN, MLP")
        st.markdown("Selecione target e features para treinar e comparar modelos supervisionados.")

        # Inicializa session_state seguro
        st.session_state.setdefault('modelo_treinado', None)
        st.session_state.setdefault('features_usadas', [])
        st.session_state.setdefault('scaler_usado', None)

        target_sup = st.selectbox("Variável alvo (y)", colunas, index=len(colunas)-1, key="target_sup")
        features_sup = st.multiselect("Variáveis (X)", [c for c in colunas_eng if c != target_sup],
                                      default=[c for c in colunas if c != target_sup], key="features_sup")
        test_size = st.slider("Proporção de teste (%)", 10, 50, 25, key="test_size_sup") / 100.0
        scale_sup = st.checkbox("Padronizar features (recomendado para KNN/MLP)", value=True)

        @st.cache_data(show_spinner=False)
        def treinar_modelos_cache(X, y, test_size, random_state):
            return treinar_e_avaliar_supervisionado(X, y, test_size=test_size, random_state=random_state)

        if st.button("Treinar modelos supervisionados"):
            if len(features_sup) < 1:
                st.warning("Selecione pelo menos 1 feature.")
            else:
                X = df[features_sup].apply(pd.to_numeric, errors='coerce').dropna().values
                y = df[target_sup].dropna().values

                if scale_sup:
                    scaler = StandardScaler()
                    X = scaler.fit_transform(X)
                else:
                    scaler = None

                try:
                    with st.spinner("Treinando modelos..."):
                        out = treinar_modelos_cache(X, y, test_size, 42)
                    st.write("Tarefa detectada:", out['task'])
                    st.write("Melhor método sugerido:", out['best'])

                    rows = []
                    for nome, r in out['results'].items():
                        if 'error' in r:
                            rows.append({'Método': nome, 'Obs': 'Erro: ' + r['error']})
                        else:
                            if out['task'] == 'regression':
                                rows.append({'Método': nome, 'R²': r.get('R2'), 'RMSE': r.get('RMSE'),
                                             'CV R² (média)': r.get('cv_r2_mean'), 'CV R² (std)': r.get('cv_r2_std')})
                            else:
                                rows.append({'Método': nome, 'Accuracy': r.get('accuracy'),
                                             'Precision': r.get('precision'), 'Recall': r.get('recall'),
                                             'F1': r.get('f1')})
                    st.table(pd.DataFrame(rows))

                    best = out['best']
                    if best and 'y_pred' in out['results'][best]:
                        st.markdown(f"### Resultados detalhados - Melhor: {best}")
                        if out['task'] == 'regression':
                            st.pyplot(plot_regression_pred_vs_obs(out['y_test'], out['results'][best]['y_pred']))
                            st.pyplot(plot_residuos(out['y_test'], out['results'][best]['y_pred']))
                        else:
                            st.text(out['results'][best].get('classification_report', ''))
                            cm = out['results'][best].get('confusion_matrix')
                            if cm is not None:
                                st.pyplot(plot_confusion_matrix(cm))
                            if 'roc_curve' in out['results'][best]:
                                fpr, tpr = out['results'][best]['roc_curve']
                                fig, ax = plt.subplots()
                                ax.plot(fpr, tpr, label=f"AUC={out['results'][best].get('roc_auc'):.3f}")
                                ax.plot([0, 1], [0, 1], '--')
                                ax.set_xlabel('FPR')
                                ax.set_ylabel('TPR')
                                ax.legend()
                                ax.set_title(f"ROC - {best}")
                                st.pyplot(fig)

                    # Salva o modelo e o scaler
                    st.session_state['modelo_treinado'] = out
                    st.session_state['features_usadas'] = features_sup
                    st.session_state['scaler_usado'] = scaler
                    st.success("✅ Modelo salvo! Vá para a aba 'Previsão Futura' para usar o modelo.")
                except Exception as e:
                    st.error(f"Erro ao treinar modelos supervisionados: {e}")


    # TAB 8 - PREVISÃO FUTURA (corrigido)
    with tab8:
        st.subheader("🔮 Previsão de Valor Futuro de Automóveis")
        st.markdown("Use o modelo treinado na aba anterior para prever o valor de um veículo.")

        # Garante persistência
        st.session_state.setdefault('modelo_treinado', None)

        if not st.session_state['modelo_treinado']:
            st.warning("⚠️ Treine um modelo primeiro na aba 'Supervisionado'.")
        else:
            out = st.session_state['modelo_treinado']
            best = out.get('best')
            if not best:
                st.error("❌ Nenhum modelo ativo encontrado. Reentre na aba 'Supervisionado' e treine novamente.")
            else:
                features_usadas = st.session_state.get('features_usadas', [])
                scaler_usado = st.session_state.get('scaler_usado', None)

                st.info(f"Modelo ativo: **{best}** | Tarefa: **{out['task']}**")

                st.markdown("### Insira as características do veículo:")
                input_values = {}
                cols_input = st.columns(3)
                for i, feat in enumerate(features_usadas):
                    with cols_input[i % 3]:
                        min_val = float(df[feat].min())
                        max_val = float(df[feat].max())
                        mean_val = float(df[feat].mean())
                        input_values[feat] = st.number_input(
                            f"{feat}",
                            min_value=min_val,
                            max_value=max_val * 1.5,
                            value=mean_val,
                            key=f"input_{feat}"
                        )

                if st.button("🚀 Fazer Previsão", type="primary"):
                    try:
                        X_novo = np.array([[input_values[f] for f in features_usadas]])
                        if scaler_usado:
                            X_novo = scaler_usado.transform(X_novo)
                        modelo = out['models'][best]
                        predicao = modelo.predict(X_novo)[0]

                        if out['task'] == 'regression':
                            st.success(f"💰 Previsão: R$ {predicao:,.2f}")
                            y_train = out['y_train']
                            y_pred_train = modelo.predict(out['X_train'])
                            std_residuos = np.std(y_train - y_pred_train)
                            ic_lower = predicao - 1.96 * std_residuos
                            ic_upper = predicao + 1.96 * std_residuos
                            st.info(f"📊 Intervalo de Confiança (95%): R$ {ic_lower:,.2f} - R$ {ic_upper:,.2f}")
                        else:
                            st.success(f"🎯 Classe prevista: {predicao}")

                        # Histórico de previsões
                        hist = st.session_state.setdefault('historico_previsoes', [])
                        hist.append({'timestamp': pd.Timestamp.now(), 'modelo': best,
                                     'predicao': predicao, **input_values})
                        st.session_state['historico_previsoes'] = hist

                        st.markdown("### 📝 Histórico de Previsões")
                        df_hist = pd.DataFrame(hist)
                        st.dataframe(df_hist)
                        csv_hist = df_hist.to_csv(index=False).encode('utf-8')
                        st.download_button("📥 Baixar Histórico CSV", csv_hist, "historico_previsoes.csv", "text/csv")
                    except Exception as e:
                        st.error(f"Erro ao fazer previsão: {e}")

    # TAB 9 - ANÁLISE TEMPORAL (NOVO!)
    with tab9:
        st.subheader("📈 Análise Temporal e Tendências")
        st.markdown("Análise da evolução dos preços ao longo do tempo")
        
        if 'Ano Modelo' in df.columns and target in df.columns:
            # Evolução temporal
            df_temp = df.groupby('Ano Modelo')[target].agg(['mean', 'median', 'std', 'count']).reset_index()
            
            fig_evolucao = go.Figure()
            fig_evolucao.add_trace(go.Scatter(
                x=df_temp['Ano Modelo'],
                y=df_temp['mean'],
                mode='lines+markers',
                name='Média',
                line=dict(color='blue', width=3)
            ))
            fig_evolucao.add_trace(go.Scatter(
                x=df_temp['Ano Modelo'],
                y=df_temp['median'],
                mode='lines+markers',
                name='Mediana',
                line=dict(color='green', width=2, dash='dash')
            ))
            fig_evolucao.update_layout(
                title=f"Evolução do {target} por Ano",
                xaxis_title="Ano Modelo",
                yaxis_title=target,
                hovermode='x unified'
            )
            st.plotly_chart(fig_evolucao, use_container_width=True)
            
            # Taxa de variação anual
            df_temp['variacao_pct'] = df_temp['mean'].pct_change() * 100
            fig_var = px.bar(
                df_temp,
                x='Ano Modelo',
                y='variacao_pct',
                title="Taxa de Variação Anual (%)",
                color='variacao_pct',
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig_var, use_container_width=True)
            
            # Estatísticas
            col1, col2, col3 = st.columns(3)
            taxa_media = df_temp['variacao_pct'].mean()
            taxa_recente = df_temp['variacao_pct'].iloc[-1] if len(df_temp) > 1 else 0
            col1.metric("Taxa de Crescimento Média Anual", f"{taxa_media:.2f}%")
            col2.metric("Taxa Mais Recente", f"{taxa_recente:.2f}%")
            col3.metric("Amplitude de Preços", f"R$ {df[target].max() - df[target].min():,.2f}")
            
            # Depreciação
            if 'Idade_Veiculo' in df.columns:
                st.markdown("### 📉 Análise de Depreciação")
                fig_depr = px.scatter(
                    df,
                    x='Idade_Veiculo',
                    y=target,
                    trendline="lowess",
                    title="Depreciação por Idade do Veículo",
                    labels={'Idade_Veiculo': 'Idade (anos)', target: 'Preço'}
                )
                st.plotly_chart(fig_depr, use_container_width=True)
        else:
            st.info("Coluna 'Ano Modelo' não encontrada para análise temporal")

    # TAB 10 - COMPARAÇÃO GLOBAL (otimizado)
    with tab10:
        st.subheader("🏆 Comparação Global de Todos os Métodos")
        st.markdown("Visão consolidada de performance de TODOS os modelos testados")
        
        if st.button("🔄 Gerar Comparação Completa"):
            with st.spinner("Compilando resultados..."):
                # Coletar todos os resultados
                todos_resultados = []
                
                # Regressões clássicas (se existirem)
                if 'modelos_calculados' in st.session_state:
                    dados = st.session_state['modelos_calculados']
                    for nome, r2, rmse in [
                        ('Regressão Linear', dados.get('r2_lin'), dados.get('rmse_lin')),
                        (f"Regressão Polinomial (grau {dados.get('grau_poly', 2)})", dados.get('r2_poly'), dados.get('rmse_poly')),
                        ('Regressão Exponencial', dados.get('r2_exp'), dados.get('rmse_exp')),
                        ('Regressão Logística', dados.get('r2_log'), dados.get('rmse_log')),
                        ('Regressão Potência', dados.get('r2_pot'), dados.get('rmse_pot')),
                        ('Regressão Multivariada', dados.get('r2_mv'), dados.get('rmse_mv'))
                    ]:
                        if r2 is not None:
                            todos_resultados.append({
                                'Categoria': 'Regressão Clássica',
                                'Método': nome,
                                'R²': r2,
                                'RMSE': rmse,
                                'Score': r2 * 100
                            })
                
                # Modelos Supervisionados (se existirem)
                if 'modelo_treinado' in st.session_state:
                    out_sup = st.session_state['modelo_treinado']
                    for nome, info in out_sup['results'].items():
                        if 'R2' in info and 'RMSE' in info:
                            todos_resultados.append({
                                'Categoria': 'Aprendizado Supervisionado',
                                'Método': nome,
                                'R²': info['R2'],
                                'RMSE': info['RMSE'],
                                'Score': info['R2'] * 100
                            })
                        elif 'f1' in info:
                            todos_resultados.append({
                                'Categoria': 'Classificação Supervisionada',
                                'Método': nome,
                                'F1-Score': info['f1'],
                                'Accuracy': info['accuracy'],
                                'Score': info['f1'] * 100
                            })
                
                if len(todos_resultados) > 0:
                    df_comparacao = pd.DataFrame(todos_resultados)
                    
                    # Tabela completa
                    st.subheader("📋 Tabela Completa de Resultados")
                    st.dataframe(df_comparacao.style.highlight_max(subset=['Score'], color='lightgreen'))
                    
                    # Gráfico de barras comparativo
                    st.subheader("📊 Comparação Visual - R² Score")
                    df_r2 = df_comparacao[df_comparacao['R²'].notna()].sort_values('R²', ascending=True)
                    
                    fig_comp = px.bar(
                        df_r2,
                        y='Método',
                        x='R²',
                        color='Categoria',
                        orientation='h',
                        title='Comparação de R² entre todos os métodos',
                        text='R²',
                        color_discrete_sequence=px.colors.qualitative.Set2
                    )
                    fig_comp.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                    fig_comp.update_layout(height=max(400, len(df_r2) * 40))
                    st.plotly_chart(fig_comp, use_container_width=True)
                    
                    # Gráfico RMSE
                    st.subheader("📉 Comparação Visual - RMSE")
                    df_rmse = df_comparacao[df_comparacao['RMSE'].notna()].sort_values('RMSE', ascending=False)
                    
                    fig_rmse = px.bar(
                        df_rmse,
                        y='Método',
                        x='RMSE',
                        color='Categoria',
                        orientation='h',
                        title='Comparação de RMSE entre todos os métodos (menor é melhor)',
                        text='RMSE',
                        color_discrete_sequence=px.colors.qualitative.Pastel
                    )
                    fig_rmse.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                    fig_rmse.update_layout(height=max(400, len(df_rmse) * 40))
                    st.plotly_chart(fig_rmse, use_container_width=True)
                    
                    # Ranking final
                    st.markdown("---")
                    st.subheader("🏆 Ranking Final dos Métodos")
                    df_ranking = df_comparacao.sort_values('Score', ascending=False).reset_index(drop=True)
                    df_ranking.index = df_ranking.index + 1
                    df_ranking.index.name = 'Posição'
                    
                    # Destacar top 3
                    def highlight_top3(row):
                        if row.name <= 3:
                            return ['background-color: gold' if row.name == 1 
                                   else 'background-color: silver' if row.name == 2 
                                   else 'background-color: #cd7f32'] * len(row)
                        return [''] * len(row)
                    
                    st.dataframe(df_ranking.style.apply(highlight_top3, axis=1))
                    
                    # Melhor método geral
                    melhor_geral = df_ranking.iloc[0]
                    st.success(f"""
                    ### 🎯 RECOMENDAÇÃO FINAL
                    
                    **Melhor Método:** {melhor_geral['Método']}  
                    **Categoria:** {melhor_geral['Categoria']}  
                    **Score:** {melhor_geral['Score']:.2f}  
                    
                    Este método apresentou o melhor desempenho considerando as métricas de avaliação.
                    Para previsão de valores futuros de automóveis, recomendamos utilizar este modelo.
                    """)
                    
                    # Conclusão e prescrição
                    st.markdown("---")
                    st.subheader("📝 Conclusão e Ações Prescritivas")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("""
                        **Conclusões:**
                        - Testamos múltiplos métodos de regressão e aprendizado de máquina
                        - Comparamos performance usando R², RMSE e outras métricas
                        - Identificamos o método mais adequado para este conjunto de dados
                        """)
                    
                    with col2:
                        st.markdown(f"""
                        **Ações Recomendadas:**
                        1. ✅ Implementar o modelo **{melhor_geral['Método']}** em produção
                        2. 📊 Monitorar métricas continuamente (drift detection)
                        3. 🔄 Re-treinar periodicamente com novos dados
                        4. 🧪 Validar com dados externos (cross-validation temporal)
                        5. 📈 Criar pipeline de deploy automatizado
                        """)
                    
                    # Download do relatório completo
                    st.markdown("---")
                    csv_comparacao = df_comparacao.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Baixar Relatório Completo (CSV)",
                        csv_comparacao,
                        "comparacao_metodos_completa.csv",
                        "text/csv",
                        key='download_comparacao'
                    )
                    
                    # Gráfico radar de comparação (top 5)
                    if len(df_ranking) >= 3:
                        st.markdown("---")
                        st.subheader("🎯 Análise Radar - Top 5 Métodos")
                        
                        top5 = df_ranking.head(5)
                        if 'R²' in top5.columns and 'RMSE' in top5.columns:
                            # Normalizar métricas
                            top5_norm = top5.copy()
                            top5_norm['R²_norm'] = (top5_norm['R²'] / top5_norm['R²'].max()) * 100
                            top5_norm['RMSE_norm'] = (1 - (top5_norm['RMSE'] / top5_norm['RMSE'].max())) * 100
                            
                            fig_radar = go.Figure()
                            
                            for idx, row in top5_norm.iterrows():
                                fig_radar.add_trace(go.Scatterpolar(
                                    r=[row['R²_norm'], row['RMSE_norm'], row['Score']],
                                    theta=['R² (norm)', 'RMSE Inverso (norm)', 'Score'],
                                    fill='toself',
                                    name=row['Método']
                                ))
                            
                            fig_radar.update_layout(
                                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                                showlegend=True,
                                title="Comparação Multidimensional dos Top 5 Métodos"
                            )
                            st.plotly_chart(fig_radar, use_container_width=True)
                    
                else:
                    st.warning("⚠️ Nenhum modelo foi calculado ainda.")
        else:
            st.info("""
            👆 **Clique no botão acima para gerar a comparação completa**
            
            **Para gerar a comparação:**
            1. Vá para a aba 'Correlações & Modelos' e calcule os modelos de regressão
            2. Vá para a aba 'Supervisionado' e treine os modelos de ML
            3. Retorne aqui e clique em 'Gerar Comparação Completa'
            """)

# ---------------------------- MAIN ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', action='store_true', help='Gerar relatório estático e sair')
    parser.add_argument('--file', type=str, default='tabelinha2.xlsx', help='Arquivo Excel com dados')
    args = parser.parse_args()
    df, colunas = carregar_dados(args.file)
    if args.report:
        gerar_report_cli(df, colunas)
    else:
        streamlit_app(df, colunas)

if __name__ == '__main__':
    main()

#poderia separar em subarquivos para melhor organizacao