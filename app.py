"""
Streamlit app: Fama-French factor analysis of S&P 500 stocks.
Loads pre-computed results from the Colab notebook (streamlit_data/ folder)
and provides interactive exploration across the homework questions.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title='FF5 Factor Explorer',
    page_icon='📈',
    layout='wide',
    initial_sidebar_state='expanded'
)

DATA_DIR = Path('data')

# ---------------------------------------------------------------------------
# Cached data loaders — Streamlit re-runs the script on every interaction,
# so without caching we'd re-read parquet files every time.
# ---------------------------------------------------------------------------
@st.cache_data
def load_results():
    """Load all four results matrices + metadata."""
    out = {
        'daily': pd.read_parquet(DATA_DIR / 'results_daily.parquet'),
        'monthly': pd.read_parquet(DATA_DIR / 'results_monthly.parquet'),
        'lead': pd.read_parquet(DATA_DIR / 'results_lead.parquet'),
        'ff6': pd.read_parquet(DATA_DIR / 'results_ff6.parquet'),
        'ivol': pd.read_parquet(DATA_DIR / 'results_ivol.parquet'),
        'meta': pd.read_parquet(DATA_DIR / 'meta.parquet'),
    }
    # Attach sector + company name to each results frame for convenience
    for k in ['daily', 'monthly', 'lead', 'ff6', 'ivol']:
        new_cols = out['meta'].columns.difference(out[k].columns)
        out[k] = out[k].join(out['meta'][new_cols], how='left')
    return out

@st.cache_data
def load_optional(name):
    """Optional files — return None if user didn't save them."""
    path = DATA_DIR / f'{name}.parquet'
    return pd.read_parquet(path) if path.exists() else None

# ---------------------------------------------------------------------------
# Sidebar — global filters
# ---------------------------------------------------------------------------
st.sidebar.title('Filters')

data = load_results()

freq = st.sidebar.radio(
    'Frequency',
    ['Daily', 'Monthly', 'Daily (predictive, t+1)', 'Daily FF6 (with momentum)', 'Daily FF5 + IVOL'],
    help='Switches the underlying results matrix used by every tab.'
)

freq_map = {
    'Daily': 'daily',
    'Monthly': 'monthly',
    'Daily (predictive, t+1)': 'lead',
    'Daily FF6 (with momentum)': 'ff6',
    'Daily FF5 + IVOL': 'ivol',
}
df = data[freq_map[freq]].copy()

# Sector filter
all_sectors = sorted(df['Sector'].dropna().unique())
selected_sectors = st.sidebar.multiselect(
    'GICS Sectors',
    all_sectors,
    default=all_sectors,
    help='Filter the cross-section by sector.'
)
df = df[df['Sector'].isin(selected_sectors)]

# R² floor
r2_min = st.sidebar.slider(
    'Minimum R²', 0.0, 1.0, 0.0, 0.05,
    help='Hide stocks the model fits poorly.'
)
df = df[df['R2'] >= r2_min]

# Derive column lists dynamically so FF6 (b_MOM) and IVOL (b_IVOL) are included
COEF_COLS  = ['alpha'] + [c for c in df.columns if c.startswith('b_')]
TSTAT_COLS = [c for c in df.columns if c.startswith('t_')]

st.sidebar.markdown('---')
st.sidebar.metric('Stocks in current view', f'{len(df):,}')
st.sidebar.metric('Mean R²', f'{df["R2"].mean():.3f}')

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title('Fama-French Factor Analysis')
st.markdown(
    'Interactive exploration of FF5 regressions across S&P 500 stocks. '
    'Switch frequency and filter by sector in the sidebar; results update across all tabs.'
)

# ---------------------------------------------------------------------------
# Tabs — one per major analysis question
# ---------------------------------------------------------------------------
tabs = st.tabs([
    'Overview', 'Sectors', 'Clusters',
    'Top/Bottom α', 'Stock Detail',
    'Model Comparison', 'IVOL Factor',
])

# ===========================================================================
# TAB 1 — OVERVIEW (Q2 + Q3)
# ===========================================================================
with tabs[0]:
    st.header('Cross-sectional descriptive statistics')

    desc = df[COEF_COLS + TSTAT_COLS + ['R2']].describe().T
    desc['skew'] = df[COEF_COLS + TSTAT_COLS + ['R2']].skew()
    desc['kurtosis'] = df[COEF_COLS + TSTAT_COLS + ['R2']].kurt()
    st.dataframe(desc.round(4), use_container_width=True)

    st.markdown('### Kernel density plots of t-statistics')
    st.caption('Red dashed lines mark the conventional ±1.96 (5%) significance threshold.')

    # Which t-stats exist depends on the chosen specification (FF5 has 6, FF6 has 7)
    available_tstats = [c for c in df.columns if c.startswith('t_')]

    cols = st.columns(3)
    for i, col in enumerate(available_tstats):
        with cols[i % 3]:
            fig = go.Figure()
            # Plotly doesn't have a native KDE, so we use violin to get a smoothed shape
            fig.add_trace(go.Violin(
                x=df[col], side='positive', line_color='steelblue',
                fillcolor='rgba(70,130,180,0.4)', meanline_visible=True,
                points=False, name=col
            ))
            for x in [-1.96, 1.96]:
                fig.add_vline(x=x, line_dash='dash', line_color='red')
            fig.update_layout(
                title=col, height=280, showlegend=False,
                margin=dict(l=10, r=10, t=40, b=10)
            )
            fig.update_yaxes(showticklabels=False)
            st.plotly_chart(fig, use_container_width=True)

    # Significance shares
    st.markdown('### Share of stocks with significant loadings (|t| > 1.96)')
    sig_shares = pd.Series(
        {c: (df[c].abs() > 1.96).mean() * 100 for c in available_tstats},
        name='Pct significant'
    ).round(1)
    st.bar_chart(sig_shares)

# ===========================================================================
# TAB 2 — SECTORS (Q4)
# ===========================================================================
with tabs[1]:
    st.header('Coefficients by GICS sector')

    sector_means = df.groupby('Sector')[COEF_COLS].mean().round(3)

    # Heatmap — diverging colour scale centred at 0
    st.markdown('### Mean coefficient heatmap')
    fig = px.imshow(
        sector_means,
        color_continuous_scale='RdBu_r', color_continuous_midpoint=0,
        aspect='auto', text_auto='.2f',
    )
    fig.update_layout(height=500, xaxis_title='', yaxis_title='')
    st.plotly_chart(fig, use_container_width=True)

    # Per-coefficient boxplot — pick which one to show
    st.markdown('### Distribution within sector')
    chosen = st.selectbox('Coefficient', COEF_COLS, index=1)
    fig = px.box(
        df.dropna(subset=['Sector']),
        x='Sector', y=chosen, color='Sector',
        points='outliers'
    )
    fig.add_hline(y=0, line_dash='dash', line_color='black')
    fig.update_layout(height=500, showlegend=False, xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

# ===========================================================================
# TAB 3 — CLUSTERS (Q5)
# ===========================================================================
with tabs[2]:
    st.header('Unsupervised clustering of factor loadings')
    st.caption(
        'K-means is recomputed on the fly across the filtered cross-section. '
        'Use the slider to change the number of clusters.'
    )

    from sklearn.cluster import KMeans

    k = st.slider('Number of clusters', 2, 8, 4)

    feat = df[COEF_COLS].dropna()
    if len(feat) < k:
        st.warning('Not enough stocks in the current filter for clustering.')
    else:
        scaler = StandardScaler()
        X_std = scaler.fit_transform(feat)

        km = KMeans(n_clusters=k, n_init=20, random_state=42).fit(X_std)
        labels = pd.Series(km.labels_, index=feat.index, name='Cluster')

        # 2-D PCA projection for plotting
        pca = PCA(n_components=2)
        coords = pca.fit_transform(X_std)
        plot_df = pd.DataFrame(coords, columns=['PC1', 'PC2'], index=feat.index)
        plot_df['Cluster'] = labels.astype(str)
        plot_df['Sector']  = df.loc[feat.index, 'Sector']
        plot_df['Name']    = df.loc[feat.index, 'Name']
        plot_df['Ticker']  = feat.index

        fig = px.scatter(
            plot_df, x='PC1', y='PC2', color='Cluster',
            hover_data=['Ticker', 'Name', 'Sector'],
            title=f'PCA projection (explained variance: '
                  f'{pca.explained_variance_ratio_.sum()*100:.1f}%)'
        )
        fig.update_traces(marker=dict(size=8, opacity=0.7))
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

        # Centroids back in original units for interpretation
        centroids = pd.DataFrame(
            scaler.inverse_transform(km.cluster_centers_),
            columns=COEF_COLS,
            index=[f'Cluster {i}' for i in range(k)]
        ).round(3)
        st.markdown('### Cluster centroids (mean coefficient per cluster)')
        st.dataframe(centroids, use_container_width=True)

        # Sector composition of each cluster
        st.markdown('### Sector composition (%)')
        comp = (
            pd.crosstab(labels, df.loc[feat.index, 'Sector'], normalize='index') * 100
        ).round(1)
        comp.index = [f'Cluster {i}' for i in comp.index]
        fig = px.imshow(comp, color_continuous_scale='Blues',
                        aspect='auto', text_auto='.0f')
        fig.update_layout(height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

# ===========================================================================
# TAB 4 — TOP / BOTTOM ALPHAS (Q6)
# ===========================================================================
with tabs[3]:
    st.header('Highest and lowest estimated alphas')

    n = st.slider('How many to show on each side?', 3, 25, 5)
    sorted_df = df.sort_values('alpha', ascending=False)

    top = sorted_df.head(n)[['Name', 'Sector', 'alpha', 't_alpha', 'R2']]
    bot = sorted_df.tail(n)[['Name', 'Sector', 'alpha', 't_alpha', 'R2']]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f'### Top {n} α (best risk-adjusted)')
        st.dataframe(top.round(4), use_container_width=True)
    with c2:
        st.markdown(f'### Bottom {n} α (worst risk-adjusted)')
        st.dataframe(bot.round(4), use_container_width=True)

    # Combined horizontal bar chart
    combined = pd.concat([top, bot]).sort_values('alpha')
    fig = px.bar(
        combined, x='alpha', y=combined.index, orientation='h',
        color='alpha', color_continuous_scale='RdYlGn',
        hover_data=['Name', 'Sector', 't_alpha'],
    )
    fig.add_vline(x=0, line_color='black')
    fig.update_layout(height=max(400, 25 * len(combined)),
                      yaxis_title='', xaxis_title='Alpha')
    st.plotly_chart(fig, use_container_width=True)

# ===========================================================================
# TAB 5 — STOCK DETAIL
# ===========================================================================
with tabs[4]:
    st.header('Per-stock factor breakdown')

    if len(df) == 0:
        st.warning('No stocks match the current filters.')
    else:
        # Build a "Ticker — Name" label so the user can search either way
        labels = df.apply(lambda r: f'{r.name} — {r["Name"]}', axis=1)
        chosen_label = st.selectbox('Pick a stock', labels.tolist())
        ticker = chosen_label.split(' — ')[0]
        row = df.loc[ticker]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric('Sector', row['Sector'])
        c2.metric('R²',     f'{row["R2"]:.3f}')
        c3.metric('α',      f'{row["alpha"]:.4f}')
        c4.metric('t(α)',   f'{row["t_alpha"]:.2f}')

        # Bar chart of factor coefficients with t-stats annotated
        beta_cols  = [c for c in df.columns if c.startswith('b_')]
        tstat_betas = [c.replace('b_', 't_') for c in beta_cols]

        plot_df = pd.DataFrame({
            'Factor': [c.replace('b_', '') for c in beta_cols],
            'Coefficient': row[beta_cols].values,
            't-statistic': row[tstat_betas].values,
        })
        plot_df['Significant'] = plot_df['t-statistic'].abs() > 1.96

        fig = px.bar(
            plot_df, x='Factor', y='Coefficient', color='Significant',
            color_discrete_map={True: 'steelblue', False: 'lightgray'},
            text=plot_df['t-statistic'].round(2).astype(str).radd('t='),
        )
        fig.add_hline(y=0, line_color='black')
        fig.update_layout(height=450, title=f'{ticker} — factor loadings')
        st.plotly_chart(fig, use_container_width=True)

        # Show this stock's position in the cross-sectional R² distribution
        st.markdown('### Where does this stock sit in the cross-section?')
        fig = px.histogram(df, x='R2', nbins=40)
        fig.add_vline(x=row['R2'], line_color='red', line_width=3,
                      annotation_text=f'{ticker}', annotation_position='top')
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

# ===========================================================================
# TAB 6 — ML MODEL COMPARISON (Q9)
# ===========================================================================
with tabs[5]:
    st.header('Cross-validated R² across models')

    cv = load_optional('cv_results')
    if cv is None:
        st.info(
            'No `cv_results.parquet` found in `streamlit_data/`. '
            'Re-run Q9 in the notebook with the save cell to enable this tab.'
        )
    else:
        st.caption(f'5-fold CV R² for {len(cv)} stocks across {cv.shape[1]} model families.')

        # Box plot — one per model
        long = cv.reset_index().melt(id_vars=cv.index.name or 'index',
                                     var_name='Model', value_name='CV R²')
        fig = px.box(long, x='Model', y='CV R²', color='Model', points='outliers')
        fig.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # Mean / median table
        summary = pd.DataFrame({
            'Mean CV R²':   cv.mean(),
            'Median CV R²': cv.median(),
            'Std':          cv.std(),
        }).round(4)
        st.dataframe(summary, use_container_width=True)

# ===========================================================================
# TAB 7 — IVOL FACTOR (Q10)
# ===========================================================================
with tabs[6]:
    st.header('The Idiosyncratic Volatility Factor')

    st.markdown(
        'Long low-IVOL, short high-IVOL portfolio. Following '
        '[Ang, Hodrick, Xing & Zhang (2006)](https://doi.org/10.1111/j.1540-6261.2006.00836.x).'
    )

    with st.expander('How was IVOL built?', expanded=False):
        st.markdown('**Step 1: Measure how volatile each stock is**')
        st.markdown(
            'Every month, for each stock, we run a regression of its daily returns against '
            'the three Fama-French factors (market, size, value). The part the model *cannot* '
            'explain is the residual. IVOL is simply the standard deviation of those residuals: '
            'a measure of how much each stock moves for reasons unrelated to broad market forces.'
        )
        st.latex(r'r_t^i = \alpha^i + \beta^i_{MKT}\,MKT_t + \beta^i_{SMB}\,SMB_t + \beta^i_{HML}\,HML_t + \varepsilon^i_t')
        st.latex(r'IVOL_i = \sqrt{\,\widehat{\mathrm{Var}}(\varepsilon^i_t)\,}')
        st.markdown(
            'We use FF3 (not FF5) to match the paper, keeping our numbers comparable to the literature.'
        )

        st.markdown('**Step 2: Build a long-short portfolio from those IVOL numbers**')
        st.markdown(
            'At the end of each month we rank all stocks by IVOL and form a zero-cost portfolio:'
        )
        st.markdown('- **Long** the 20% with the *lowest* IVOL (calm, predictable stocks)\n'
                    '- **Short** the 20% with the *highest* IVOL (erratic, hard-to-predict stocks)')
        st.markdown('The daily return of this portfolio is the IVOL factor time series.')

        st.markdown('**Step 3: Add the IVOL factor to the FF5 regression**')
        st.markdown(
            'We add that factor as a sixth regressor. The `b_IVOL` coefficient tells you '
            'how much a stock co-moves with high-volatility names.'
        )
        st.latex(r'r_t^i = \alpha^i + \beta^i_{MKT}\,MKT_t + \beta^i_{SMB}\,SMB_t + \beta^i_{HML}\,HML_t + \beta^i_{RMW}\,RMW_t + \beta^i_{CMA}\,CMA_t + \beta^i_{IVOL}\,IVOL_t + \varepsilon^i_t')

        st.markdown('**Why does this matter?**')
        st.markdown(
            'Ang et al. (2006) found that the most volatile stocks earn *much lower* returns '
            'than calm stocks, which is counter-intuitive since more risk usually means more reward. '
            'Their long-short strategy earned roughly +12%/year over 1963–2000. '
            'We test whether that finding holds in our sample.'
        )


    ivol_ts = load_optional('ivol_factor')
    if ivol_ts is None:
        st.info('Save `ivol_factor.parquet` from the notebook to enable this tab.')
    else:
        ivol_ts = ivol_ts.iloc[:, 0]   # parquet round-trips Series as 1-col DataFrame

        c1, c2, c3 = st.columns(3)
        c1.metric('Mean daily return', f'{ivol_ts.mean():.4f}%')
        c2.metric('Daily volatility',   f'{ivol_ts.std():.4f}%')
        sharpe = (ivol_ts.mean() * 252) / (ivol_ts.std() * np.sqrt(252))
        c3.metric('Annualised Sharpe',  f'{sharpe:.2f}')

        # Cumulative return — the canonical low-vol-anomaly chart
        cum = ivol_ts.cumsum()
        fig = px.line(cum, title='Cumulative IVOL factor return (low minus high)')
        fig.update_layout(height=450, yaxis_title='Cumulative log return (%)',
                          showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # --- Interpretation ---
        st.markdown('### Interpretation of our results')

        cum_total   = ivol_ts.cumsum().iloc[-1]
        pct_pos     = (ivol_ts > 0).mean() * 100
        ann_ret     = ivol_ts.mean() * 252

        sign_word   = 'lost' if cum_total < 0 else 'gained'

        r2_ff6  = data['ff6']['R2'].mean()
        r2_ivol = data['ivol']['R2'].mean()
        r2_gain = (r2_ivol - r2_ff6) * 100

        st.markdown(f"""
**Does our strategy make money?**

Over the full sample the strategy accumulated **{cum_total:.1f}%** in total:
it **{sign_word}** money. The annualised return is **{ann_ret:.2f}%**, and only
**{pct_pos:.0f}%** of daily returns were positive. For reference, Ang et al.
reported roughly +12%/year going the same direction over 1963–2000.

**Our results are the opposite of the paper. Why?**

The original paper found that high-IVOL stocks earn *very low* returns, so
buying low-IVOL stocks and shorting high-IVOL ones was profitable. In our sample, the
strategy loses money. Three possible reasons explain the gap:

- **We only use S&P 500 stocks.** The IVOL anomaly is strongest among small,
  less known companies where it is hard to trade. Large-cap stocks in the S&P 500
  are heavily covered and efficiently priced, so the gap between low-IVOL and high-IVOL
  stocks is much smaller.

- **The anomaly weakened after it was published.** Once the paper came out,
  hedge funds started trading it. That extra buying pressure on low-IVOL stocks and
  selling pressure on high-IVOL ones gradually closed the gap: a classic case of
  arbitrage correcting an anomaly.

- **Tech stocks dominated our sample period (2010–2026).** Companies like
  Apple, Nvidia, and Tesla are extremely volatile (high IVOL) yet delivered
  exceptional returns. Our strategy was *short* those names, which hurt
  performance badly.

**Does IVOL improve the model fit?**

Yes. Adding IVOL raises the mean R² from **{r2_ff6:.3f}** (FF6 with momentum)
to **{r2_ivol:.3f}** (+{r2_gain:.1f} pp). The long-short portfolio may not be
profitable, but IVOL still captures genuine co-movement in stock returns that
the other five factors miss.

**Takeaway**

The IVOL anomaly does not appear exploitable in large-cap US stocks over
the past 15 years. However, the improved R² confirms that `b_IVOL` is a
meaningful exposure measure: it tells you how much each stock co-moves with
high-volatility names, which is useful for risk management even when the
trading signal itself is weak.
""")
