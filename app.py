import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Global Stock Market ML", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0a0e1a; color: #e8eaf6; }
.main-header { background: linear-gradient(135deg, #0d1b2a 0%, #1a2744 50%, #0d1b2a 100%); border: 1px solid #1e3a5f; border-radius: 16px; padding: 2rem 2.5rem; margin-bottom: 2rem; }
.main-header h1 { font-family: 'Space Mono', monospace; font-size: 2rem; font-weight: 700; background: linear-gradient(90deg, #00d4ff, #7c3aed, #00d4ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0 0 0.4rem 0; }
.main-header p { color: #8892a4; font-size: 0.95rem; margin: 0; }
.metric-card { background: linear-gradient(135deg, #111827, #1a2234); border: 1px solid #1e3a5f; border-radius: 12px; padding: 1.2rem 1.4rem; text-align: center; }
.metric-card .metric-value { font-family: 'Space Mono', monospace; font-size: 1.7rem; font-weight: 700; color: #00d4ff; }
.metric-card .metric-label { color: #8892a4; font-size: 0.8rem; margin-top: 0.2rem; }
.section-title { font-family: 'Space Mono', monospace; font-size: 1.1rem; font-weight: 700; color: #00d4ff; border-left: 3px solid #7c3aed; padding-left: 0.8rem; margin: 1.5rem 0 1rem 0; }
.signal-buy { background: rgba(16,185,129,0.15); border: 1px solid #10b981; color: #10b981; padding: 0.5rem 1.5rem; border-radius: 50px; font-family: 'Space Mono', monospace; font-weight: 700; font-size: 1.2rem; display: inline-block; }
.signal-sell { background: rgba(239,68,68,0.15); border: 1px solid #ef4444; color: #ef4444; padding: 0.5rem 1.5rem; border-radius: 50px; font-family: 'Space Mono', monospace; font-weight: 700; font-size: 1.2rem; display: inline-block; }
div[data-testid="stSidebar"] { background: #0d1421 !important; border-right: 1px solid #1e3a5f; }
.stButton > button { background: linear-gradient(135deg, #7c3aed, #00d4ff) !important; color: white !important; border: none !important; border-radius: 8px !important; font-weight: 600 !important; font-family: 'Space Mono', monospace !important; }
</style>
""", unsafe_allow_html=True)

PLOTLY_THEME = "plotly_dark"

@st.cache_data
def load_data():
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    excel_path = os.path.join(base_dir, 'Global_Stock_Market_Master_Dataset.xlsx')
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"File not found at: {excel_path}")
    df = pd.read_excel(excel_path, header=2)
    df.columns = ['Date','Country','Company','Sector','Sub_Sector','Open','High','Low','Close','Volume','BUY','SELL','Daily_Return','War_Period']
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[(df['Close']>0)&(df['Volume']>0)].reset_index(drop=True)
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Price_Range'] = df['High'] - df['Low']
    df['BuySell_Ratio'] = df['BUY'] / (df['SELL']+1)
    return df

df = None
try:
    df = load_data()
except FileNotFoundError as e:
    st.error(f"Dataset file not found: {e}")
if df is None:
    st.stop()

with st.sidebar:
    st.markdown("<div style='text-align:center;padding:1rem 0;'><div style='font-family:Space Mono,monospace;font-size:1.1rem;color:#00d4ff;font-weight:700;'>📈 STOCK ML</div><div style='color:#8892a4;font-size:0.75rem;'>Global Market Intelligence</div></div>", unsafe_allow_html=True)
    st.divider()
    page = st.radio("Navigate", [
        "🏠 Overview","📊 PS1 — Price Prediction","🎯 PS2 — Buy/Sell Signal",
        "🌍 PS3 — Market Analysis","💭 PS4 — Sentiment Analysis","💼 PS5 — Portfolio Optimizer",
        "📉 PS6 — Volatility Forecast","🚨 PS7 — Anomaly Detection",
        "📈 PS8 — Trend Classification","⚔️ PS9 — War Period Impact","🔄 PS10 — Sector Rotation"
    ])
    st.divider()
    st.markdown("<div style='color:#8892a4;font-size:0.75rem;'>Dataset: 90,040 records<br>200 companies · 10 countries<br>Jan 2023 – Mar 2026</div>", unsafe_allow_html=True)

# ── OVERVIEW ──────────────────────────────────────────────────────────────────
if page == "🏠 Overview":
    st.markdown("<div class='main-header'><h1>🌐 Global Stock Market ML</h1><p>Machine Learning insights across 200 companies · 10 countries · 10 problem statements</p></div>", unsafe_allow_html=True)
    c1,c2,c3,c4,c5 = st.columns(5)
    for col,val,label in zip([c1,c2,c3,c4,c5],
        [f"{len(df):,}",df['Company'].nunique(),df['Country'].nunique(),df['Sector'].nunique(),f"{df['Date'].min().year}–{df['Date'].max().year}"],
        ["Total Records","Companies","Countries","Sectors","Date Range"]):
        col.markdown(f"<div class='metric-card'><div class='metric-value'>{val}</div><div class='metric-label'>{label}</div></div>", unsafe_allow_html=True)
    st.markdown("")
    col1,col2 = st.columns(2)
    with col1:
        st.markdown("<div class='section-title'>Records per Country</div>", unsafe_allow_html=True)
        cc = df['Country'].value_counts().reset_index(); cc.columns=['Country','Count']
        fig = px.bar(cc,x='Count',y='Country',orientation='h',color='Count',color_continuous_scale='Blues',template=PLOTLY_THEME)
        fig.update_layout(height=380,showlegend=False,coloraxis_showscale=False,margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig,use_container_width=True)
    with col2:
        st.markdown("<div class='section-title'>Avg Daily Return by Sector</div>", unsafe_allow_html=True)
        sr = df.groupby('Sector')['Daily_Return'].mean().sort_values().reset_index(); sr.columns=['Sector','Avg_Return']
        sr['Color'] = sr['Avg_Return'].apply(lambda x: '#10b981' if x>=0 else '#ef4444')
        fig = px.bar(sr,x='Avg_Return',y='Sector',orientation='h',color='Color',color_discrete_map='identity',template=PLOTLY_THEME)
        fig.update_layout(height=380,showlegend=False,margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig,use_container_width=True)
    ps_data = [
        ("PS1","📈","Price Prediction","Linear Reg · RF · GB","#00d4ff"),
        ("PS2","🎯","Buy/Sell Signal","Logistic · DT · RF","#7c3aed"),
        ("PS3","🌍","Market Analysis","K-Means · Heatmap · PCA","#f59e0b"),
        ("PS4","💭","Sentiment Analysis","Correlation · Rolling · RF","#10b981"),
        ("PS5","💼","Portfolio Optimizer","Monte Carlo · Sharpe","#ef4444"),
        ("PS6","📉","Volatility Forecast","Ridge · RF · Rolling","#06b6d4"),
        ("PS7","🚨","Anomaly Detection","Isolation Forest · LOF","#8b5cf6"),
        ("PS8","📈","Trend Classification","RF · KNN · DT","#f97316"),
        ("PS9","⚔️","War Period Impact","T-Test · Mann-Whitney","#14b8a6"),
        ("PS10","🔄","Sector Rotation","Momentum · Backtest","#ec4899"),
    ]
    st.markdown("<div class='section-title'>10 Problem Statements</div>", unsafe_allow_html=True)
    r1 = st.columns(5)
    for col,(ps,icon,title,methods,color) in zip(r1,ps_data[:5]):
        col.markdown(f"<div style='background:#111827;border:1px solid {color}33;border-top:3px solid {color};border-radius:10px;padding:1rem;text-align:center;height:150px;'><div style='font-size:1.5rem;'>{icon}</div><div style='font-family:Space Mono,monospace;font-size:0.7rem;color:{color};font-weight:700;'>{ps}</div><div style='font-weight:600;font-size:0.82rem;color:#e8eaf6;margin:0.3rem 0;'>{title}</div><div style='font-size:0.7rem;color:#8892a4;'>{methods}</div></div>",unsafe_allow_html=True)
    st.markdown("")
    r2 = st.columns(5)
    for col,(ps,icon,title,methods,color) in zip(r2,ps_data[5:]):
        col.markdown(f"<div style='background:#111827;border:1px solid {color}33;border-top:3px solid {color};border-radius:10px;padding:1rem;text-align:center;height:150px;'><div style='font-size:1.5rem;'>{icon}</div><div style='font-family:Space Mono,monospace;font-size:0.7rem;color:{color};font-weight:700;'>{ps}</div><div style='font-weight:600;font-size:0.82rem;color:#e8eaf6;margin:0.3rem 0;'>{title}</div><div style='font-size:0.7rem;color:#8892a4;'>{methods}</div></div>",unsafe_allow_html=True)

# ── PS1 ───────────────────────────────────────────────────────────────────────
elif page == "📊 PS1 — Price Prediction":
    st.markdown("<div class='main-header'><h1>📊 Stock Price Prediction</h1><p>Predict next-day closing prices using Linear Regression, Random Forest & Gradient Boosting</p></div>", unsafe_allow_html=True)
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    with st.sidebar:
        st.markdown("**PS1 Settings**")
        co1 = st.selectbox("Company", sorted(df['Company'].unique()))
        n_est = st.slider("N Estimators", 20, 150, 50, 10)
        ts = st.slider("Test Split %", 10, 40, 20, 5)
    @st.cache_data
    def prepare_ps1(company, n_est, ts):
        d = df[df['Company']==company].copy().sort_values('Date').reset_index(drop=True)
        d['Lag1_Close']=d['Close'].shift(1); d['Lag2_Close']=d['Close'].shift(2)
        d['Lag1_Return']=d['Daily_Return'].shift(1)
        d['MA5']=d['Close'].rolling(5).mean(); d['MA10']=d['Close'].rolling(10).mean()
        d['Next_Close']=d['Close'].shift(-1); d=d.dropna()
        features=['Open','High','Low','Close','Volume','Lag1_Close','Lag2_Close','Lag1_Return','MA5','MA10','Price_Range','BuySell_Ratio']
        X,y=d[features],d['Next_Close']
        Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=ts/100,shuffle=False)
        sc=StandardScaler(); Xtr_s=sc.fit_transform(Xtr); Xte_s=sc.transform(Xte)
        lr=LinearRegression().fit(Xtr_s,ytr)
        rf=RandomForestRegressor(n_estimators=n_est,random_state=42,n_jobs=-1).fit(Xtr_s,ytr)
        gb=GradientBoostingRegressor(n_estimators=n_est,random_state=42).fit(Xtr_s,ytr)
        results={}
        for name,m in [("Linear Reg",lr),("Random Forest",rf),("Grad Boost",gb)]:
            p=m.predict(Xte_s)
            results[name]={'preds':p,'MAE':mean_absolute_error(yte,p),'RMSE':np.sqrt(mean_squared_error(yte,p)),'R2':r2_score(yte,p)}
        imp=pd.Series(rf.feature_importances_,index=features).sort_values(ascending=False)
        return results,yte.values,imp
    results,yte,imp=prepare_ps1(co1,n_est,ts)
    c1,c2,c3=st.columns(3)
    clrs={'Linear Reg':'#00d4ff','Random Forest':'#7c3aed','Grad Boost':'#f59e0b'}
    for col,(name,r) in zip([c1,c2,c3],results.items()):
        col.markdown(f"<div class='metric-card' style='border-top:3px solid {clrs[name]};'><div style='font-family:Space Mono,monospace;font-size:0.75rem;color:{clrs[name]};'>{name}</div><div class='metric-value' style='font-size:1.4rem;'>{r['R2']:.4f}</div><div class='metric-label'>R² Score | MAE: {r['MAE']:.2f}</div></div>",unsafe_allow_html=True)
    st.markdown("")
    col1,col2=st.columns([2,1])
    with col1:
        st.markdown("<div class='section-title'>Actual vs Predicted (Random Forest)</div>", unsafe_allow_html=True)
        n=min(200,len(yte))
        fig=go.Figure()
        fig.add_trace(go.Scatter(y=yte[:n],name='Actual',line=dict(color='#00d4ff',width=2)))
        fig.add_trace(go.Scatter(y=results['Random Forest']['preds'][:n],name='Predicted',line=dict(color='#f59e0b',width=2,dash='dash')))
        fig.update_layout(template=PLOTLY_THEME,height=350,margin=dict(l=0,r=0,t=10,b=0),legend=dict(orientation='h',y=1.1))
        st.plotly_chart(fig,use_container_width=True)
    with col2:
        st.markdown("<div class='section-title'>Feature Importance</div>", unsafe_allow_html=True)
        fig=px.bar(x=imp.values,y=imp.index,orientation='h',color=imp.values,color_continuous_scale='Plasma',template=PLOTLY_THEME)
        fig.update_layout(height=350,showlegend=False,coloraxis_showscale=False,margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig,use_container_width=True)

# ── PS2 ───────────────────────────────────────────────────────────────────────
elif page == "🎯 PS2 — Buy/Sell Signal":
    st.markdown("<div class='main-header'><h1>🎯 Buy / Sell Signal Classifier</h1><p>Classify each trading day as BUY or SELL</p></div>", unsafe_allow_html=True)
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report,roc_auc_score,roc_curve,confusion_matrix
    with st.sidebar:
        st.markdown("**PS2 Settings**")
        co2=st.selectbox("Company",sorted(df['Company'].unique()),key='ps2')
        md2=st.slider("DT Max Depth",2,15,6)
    @st.cache_data
    def prepare_ps2(company,md):
        d=df[df['Company']==company].copy().sort_values('Date').reset_index(drop=True)
        d['Next_Close']=d['Close'].shift(-1); d['Signal']=(d['Next_Close']>d['Close']).astype(int)
        d['Lag1_Close']=d['Close'].shift(1); d['Lag1_Return']=d['Daily_Return'].shift(1)
        d['MA5']=d['Close'].rolling(5).mean(); d['MA10']=d['Close'].rolling(10).mean(); d=d.dropna()
        feat=['Open','High','Low','Close','Volume','Lag1_Close','Lag1_Return','MA5','MA10','Price_Range','BuySell_Ratio','Daily_Return']
        X,y=d[feat],d['Signal']
        Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,shuffle=False)
        sc=StandardScaler(); Xtr_s=sc.fit_transform(Xtr); Xte_s=sc.transform(Xte)
        lr=LogisticRegression(max_iter=500,random_state=42).fit(Xtr_s,ytr)
        dt=DecisionTreeClassifier(max_depth=md,random_state=42).fit(Xtr_s,ytr)
        rf=RandomForestClassifier(n_estimators=50,random_state=42,n_jobs=-1).fit(Xtr_s,ytr)
        res={}
        for name,m in [('Logistic Reg',lr),('Decision Tree',dt),('Random Forest',rf)]:
            p=m.predict(Xte_s); pr=m.predict_proba(Xte_s)[:,1]
            fpr,tpr,_=roc_curve(yte,pr)
            res[name]={'preds':p,'auc':roc_auc_score(yte,pr),'fpr':fpr,'tpr':tpr,'cm':confusion_matrix(yte,p),'acc':classification_report(yte,p,output_dict=True)['accuracy']}
        return res,yte
    res2,yte2=prepare_ps2(co2,md2)
    c1,c2,c3=st.columns(3)
    cm2={'Logistic Reg':'#00d4ff','Decision Tree':'#7c3aed','Random Forest':'#10b981'}
    for col,(name,r) in zip([c1,c2,c3],res2.items()):
        col.markdown(f"<div class='metric-card' style='border-top:3px solid {cm2[name]};'><div style='font-family:Space Mono,monospace;font-size:0.75rem;color:{cm2[name]};'>{name}</div><div class='metric-value' style='font-size:1.4rem;'>{r['auc']:.4f}</div><div class='metric-label'>AUC | Acc: {r['acc']:.2%}</div></div>",unsafe_allow_html=True)
    st.markdown(""); col1,col2=st.columns(2)
    with col1:
        st.markdown("<div class='section-title'>ROC Curve</div>", unsafe_allow_html=True)
        fig=go.Figure()
        for name,r in res2.items():
            fig.add_trace(go.Scatter(x=r['fpr'],y=r['tpr'],name=f"{name} (AUC={r['auc']:.3f})",line=dict(width=2.5)))
        fig.add_trace(go.Scatter(x=[0,1],y=[0,1],line=dict(color='gray',dash='dash'),name='Random'))
        fig.update_layout(template=PLOTLY_THEME,height=380,xaxis_title='FPR',yaxis_title='TPR',margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig,use_container_width=True)
    with col2:
        st.markdown("<div class='section-title'>Confusion Matrix — Random Forest</div>", unsafe_allow_html=True)
        cm=res2['Random Forest']['cm']
        fig=go.Figure(go.Heatmap(z=cm,x=['SELL','BUY'],y=['SELL','BUY'],colorscale='Blues',text=cm,texttemplate='%{text}',textfont_size=18,showscale=False))
        fig.update_layout(template=PLOTLY_THEME,height=380,xaxis_title='Predicted',yaxis_title='Actual',margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig,use_container_width=True)
    st.markdown("<div class='section-title'>🔴 Live Signal Predictor</div>", unsafe_allow_html=True)
    c1,c2,c3,c4=st.columns(4)
    open_p=c1.number_input("Open",value=100.0); high_p=c2.number_input("High",value=105.0)
    low_p=c3.number_input("Low",value=98.0); close_p=c4.number_input("Close",value=102.0)
    lag1_c=st.number_input("Yesterday's Close",value=100.0)
    if st.button("🔮 Predict Signal"):
        signal="BUY 📈" if close_p>lag1_c else "SELL 📉"
        css="signal-buy" if close_p>lag1_c else "signal-sell"
        st.markdown(f"<div style='text-align:center;margin:1rem 0;'><span class='{css}'>{signal}</span></div>",unsafe_allow_html=True)

# ── PS3 ───────────────────────────────────────────────────────────────────────
elif page == "🌍 PS3 — Market Analysis":
    st.markdown("<div class='main-header'><h1>🌍 Cross-Market Analysis</h1><p>K-Means Clustering, Heatmaps & Box Plots across sectors and countries</p></div>", unsafe_allow_html=True)
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    cs=df.groupby(['Country','Sector']).agg(Avg_Return=('Daily_Return','mean'),Volatility=('Daily_Return','std'),Avg_Volume=('Volume','mean'),Avg_Close=('Close','mean')).reset_index().dropna()
    tab1,tab2,tab3=st.tabs(["🗺️ Return Heatmap","📦 Volatility Box Plot","🔵 K-Means Clusters"])
    with tab1:
        pivot=cs.pivot_table(values='Avg_Return',index='Country',columns='Sector',fill_value=0)
        fig=go.Figure(go.Heatmap(z=pivot.values,x=pivot.columns.tolist(),y=pivot.index.tolist(),colorscale='RdYlGn',zmid=0,text=pivot.round(2).values,texttemplate='%{text}',textfont_size=7))
        fig.update_layout(template=PLOTLY_THEME,height=500,xaxis_tickangle=-40,title='Avg Daily Return by Country × Sector')
        st.plotly_chart(fig,use_container_width=True)
    with tab2:
        country_order=df.groupby('Country')['Daily_Return'].std().sort_values(ascending=False).index.tolist()
        fig=go.Figure()
        for country in country_order:
            fig.add_trace(go.Box(y=df[df['Country']==country]['Daily_Return'],name=country,boxpoints=False))
        fig.add_hline(y=0,line_dash='dash',line_color='red')
        fig.update_layout(template=PLOTLY_THEME,height=450,title='Daily Return Distribution by Country')
        st.plotly_chart(fig,use_container_width=True)
    with tab3:
        with st.sidebar:
            st.markdown("**PS3 Settings**")
            k_val=st.slider("K Clusters",2,8,4)
        sc3=StandardScaler(); Xs3=sc3.fit_transform(cs[['Avg_Return','Volatility','Avg_Volume','Avg_Close']])
        km3=KMeans(n_clusters=k_val,random_state=42,n_init=10); cs['Cluster']=km3.fit_predict(Xs3)
        pca3=PCA(n_components=2); coords=pca3.fit_transform(Xs3)
        cs['PCA1']=coords[:,0]; cs['PCA2']=coords[:,1]
        cs['Label']=cs['Country'].str[:3]+'-'+cs['Sector'].str[:4]
        fig=px.scatter(cs,x='PCA1',y='PCA2',color=cs['Cluster'].astype(str),text='Label',hover_data=['Country','Sector','Avg_Return','Volatility'],template=PLOTLY_THEME,title=f'K-Means Clustering (K={k_val})')
        fig.update_traces(textfont_size=7,textposition='top center')
        fig.update_layout(height=500,legend_title='Cluster')
        st.plotly_chart(fig,use_container_width=True)
        st.dataframe(cs.groupby('Cluster')[['Avg_Return','Volatility','Avg_Volume']].mean().round(4),use_container_width=True)

# ── PS4 ───────────────────────────────────────────────────────────────────────
elif page == "💭 PS4 — Sentiment Analysis":
    st.markdown("<div class='main-header'><h1>💭 Investor Sentiment Analysis</h1><p>Analyse BUY/SELL investor flow as leading indicators for future price movement</p></div>", unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("**PS4 Settings**")
        co4=st.selectbox("Company",sorted(df['Company'].unique()),key='ps4')
        rw=st.slider("Rolling Window (days)",10,60,30)
    d4=df[df['Company']==co4].copy().sort_values('Date').reset_index(drop=True)
    d4['Net_Flow']=d4['BUY']-d4['SELL']; d4['Flow_MA5']=d4['Net_Flow'].rolling(5).mean()
    d4['BuySell_MA5']=d4['BuySell_Ratio'].rolling(5).mean()
    d4['Future_3d_Return']=d4['Daily_Return'].rolling(3).mean().shift(-3); d4=d4.dropna()
    if len(d4)<30:
        st.warning("Not enough data.")
    else:
        rolling_corr=d4['BuySell_Ratio'].rolling(rw).corr(d4['Future_3d_Return'])
        c1,c2,c3=st.columns(3)
        c1.markdown(f"<div class='metric-card'><div class='metric-value'>{d4['BuySell_Ratio'].mean():.2f}</div><div class='metric-label'>Avg BuySell Ratio</div></div>",unsafe_allow_html=True)
        c2.markdown(f"<div class='metric-card'><div class='metric-value'>{d4['Net_Flow'].mean():,.0f}</div><div class='metric-label'>Avg Net Flow</div></div>",unsafe_allow_html=True)
        c3.markdown(f"<div class='metric-card'><div class='metric-value'>{rolling_corr.dropna().mean():.3f}</div><div class='metric-label'>Avg Rolling Corr</div></div>",unsafe_allow_html=True)
        fig=make_subplots(rows=3,cols=1,shared_xaxes=True,subplot_titles=('Stock Price','Investor Sentiment (BUY/SELL Ratio)',f'{rw}-Day Rolling Correlation'))
        fig.add_trace(go.Scatter(x=d4['Date'],y=d4['Close'],name='Close',line=dict(color='#00d4ff',width=1.5)),row=1,col=1)
        fig.add_trace(go.Bar(x=d4['Date'],y=d4['BuySell_Ratio'],name='BuySell Ratio',marker_color='#10b981'),row=2,col=1)
        fig.add_hline(y=1,line_dash='dash',line_color='red',row=2,col=1)
        fig.add_trace(go.Scatter(x=d4['Date'],y=rolling_corr,name='Rolling Corr',line=dict(color='#f59e0b',width=1.5),fill='tozeroy'),row=3,col=1)
        fig.update_layout(template=PLOTLY_THEME,height=700,showlegend=False,margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig,use_container_width=True)

# ── PS5 ───────────────────────────────────────────────────────────────────────
elif page == "💼 PS5 — Portfolio Optimizer":
    st.markdown("<div class='main-header'><h1>💼 Portfolio Optimizer</h1><p>Monte Carlo Simulation, Efficient Frontier & Sharpe Ratio</p></div>", unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("**PS5 Settings**")
        n_co=st.slider("Number of Companies",5,20,15)
        n_sim=st.slider("Monte Carlo Simulations",500,5000,2000,500)
        rfr=st.slider("Risk-Free Rate (%)",0,10,2)/100
    @st.cache_data
    def run_portfolio(n_co,n_sim,rfr):
        top_co=df['Company'].value_counts().head(n_co).index.tolist()
        d5=df[df['Company'].isin(top_co)]
        pp=d5.pivot_table(index='Date',columns='Company',values='Close').ffill().bfill()
        dr=pp.pct_change().dropna()
        exp_ret=dr.mean()*252; cov_mat=dr.cov()*252; n_assets=len(top_co)
        np.random.seed(42)
        sim_ret=np.zeros(n_sim); sim_risk=np.zeros(n_sim); sim_sharpe=np.zeros(n_sim); sim_weights=np.zeros((n_sim,n_assets))
        for i in range(n_sim):
            w=np.random.dirichlet(np.ones(n_assets))
            ret=np.dot(w,exp_ret.values); risk=np.sqrt(np.dot(w.T,np.dot(cov_mat.values,w)))
            sim_ret[i]=ret; sim_risk[i]=risk; sim_sharpe[i]=(ret-rfr)/risk; sim_weights[i]=w
        max_idx=np.argmax(sim_sharpe); min_idx=np.argmin(sim_risk)
        return sim_ret,sim_risk,sim_sharpe,sim_weights,max_idx,min_idx,top_co,dr
    sim_ret,sim_risk,sim_sharpe,sim_weights,max_idx,min_idx,top_co,dr=run_portfolio(n_co,n_sim,rfr)
    c1,c2,c3,c4=st.columns(4)
    c1.markdown(f"<div class='metric-card'><div class='metric-value'>{sim_ret[max_idx]:.1%}</div><div class='metric-label'>Max Sharpe Return</div></div>",unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-card'><div class='metric-value'>{sim_risk[max_idx]:.1%}</div><div class='metric-label'>Max Sharpe Risk</div></div>",unsafe_allow_html=True)
    c3.markdown(f"<div class='metric-card'><div class='metric-value'>{sim_sharpe[max_idx]:.3f}</div><div class='metric-label'>Best Sharpe Ratio</div></div>",unsafe_allow_html=True)
    c4.markdown(f"<div class='metric-card'><div class='metric-value'>{sim_risk[min_idx]:.1%}</div><div class='metric-label'>Min Risk Portfolio</div></div>",unsafe_allow_html=True)
    col1,col2=st.columns([3,2])
    with col1:
        st.markdown("<div class='section-title'>Efficient Frontier</div>", unsafe_allow_html=True)
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=sim_risk,y=sim_ret,mode='markers',marker=dict(color=sim_sharpe,colorscale='Plasma',size=3,opacity=0.4,colorbar=dict(title='Sharpe',thickness=12)),name='Portfolios',showlegend=False))
        fig.add_trace(go.Scatter(x=[sim_risk[max_idx]],y=[sim_ret[max_idx]],mode='markers',marker=dict(symbol='star',size=20,color='gold',line=dict(color='black',width=1)),name='Max Sharpe'))
        fig.add_trace(go.Scatter(x=[sim_risk[min_idx]],y=[sim_ret[min_idx]],mode='markers',marker=dict(symbol='diamond',size=14,color='#ef4444',line=dict(color='black',width=1)),name='Min Risk'))
        fig.update_layout(template=PLOTLY_THEME,height=420,xaxis_title='Annual Risk',yaxis_title='Annual Return',margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig,use_container_width=True)
    with col2:
        st.markdown("<div class='section-title'>Optimal Portfolio Weights</div>", unsafe_allow_html=True)
        opt_w=sim_weights[max_idx]; sorted_idx=np.argsort(opt_w)[::-1][:8]
        other_w=opt_w[np.argsort(opt_w)[::-1][8:]].sum()
        labels=[top_co[i] for i in sorted_idx]+(['Others'] if other_w>0.001 else [])
        sizes=list(opt_w[sorted_idx])+([other_w] if other_w>0.001 else [])
        fig=go.Figure(go.Pie(labels=labels,values=sizes,hole=0.35,textfont_size=9))
        fig.update_layout(template=PLOTLY_THEME,height=420,margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig,use_container_width=True)

# ── PS6 ───────────────────────────────────────────────────────────────────────
elif page == "📉 PS6 — Volatility Forecast":
    st.markdown("<div class='main-header'><h1>📉 Stock Volatility Forecasting</h1><p>Predict 7-day future volatility using Ridge Regression & Random Forest</p></div>", unsafe_allow_html=True)
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error,r2_score
    with st.sidebar:
        st.markdown("**PS6 Settings**")
        co6=st.selectbox("Company",sorted(df['Company'].unique()),key='ps6')
    d6=df[df['Company']==co6].copy().sort_values('Date').reset_index(drop=True)
    d6['Vol_7d_Future']=d6['Daily_Return'].shift(-1).rolling(7).std()
    d6['Vol_7d_Past']=d6['Daily_Return'].rolling(7).std()
    d6['Vol_14d']=d6['Daily_Return'].rolling(14).std()
    d6['Vol_30d']=d6['Daily_Return'].rolling(30).std()
    d6['Avg_Range_7d']=d6['Price_Range'].rolling(7).mean()
    d6['Avg_Return_7d']=d6['Daily_Return'].rolling(7).mean()
    d6['Lag1_Vol']=d6['Vol_7d_Past'].shift(1); d6=d6.dropna()
    if len(d6)<50:
        st.warning("Not enough data.")
    else:
        features_vol=['Vol_7d_Past','Vol_14d','Vol_30d','Avg_Range_7d','Avg_Return_7d','Lag1_Vol','BuySell_Ratio','Close','Volume']
        X6,y6=d6[features_vol],d6['Vol_7d_Future']
        Xtr,Xte,ytr,yte=train_test_split(X6,y6,test_size=0.2,shuffle=False)
        sc6=StandardScaler(); Xtr_s=sc6.fit_transform(Xtr); Xte_s=sc6.transform(Xte)
        ridge=Ridge(alpha=1.0).fit(Xtr_s,ytr); rf6=RandomForestRegressor(n_estimators=80,max_depth=10,random_state=42,n_jobs=-1).fit(Xtr_s,ytr)
        r_preds=ridge.predict(Xte_s); rf_preds=rf6.predict(Xte_s)
        c1,c2,c3,c4=st.columns(4)
        c1.markdown(f"<div class='metric-card'><div class='metric-value'>{r2_score(yte,rf_preds):.4f}</div><div class='metric-label'>RF R²</div></div>",unsafe_allow_html=True)
        c2.markdown(f"<div class='metric-card'><div class='metric-value'>{r2_score(yte,r_preds):.4f}</div><div class='metric-label'>Ridge R²</div></div>",unsafe_allow_html=True)
        c3.markdown(f"<div class='metric-card'><div class='metric-value'>{mean_absolute_error(yte,rf_preds):.4f}</div><div class='metric-label'>RF MAE</div></div>",unsafe_allow_html=True)
        c4.markdown(f"<div class='metric-card'><div class='metric-value'>{d6['Vol_7d_Past'].mean():.4f}</div><div class='metric-label'>Avg Volatility</div></div>",unsafe_allow_html=True)
        fig=make_subplots(rows=1,cols=2,subplot_titles=('Actual vs Predicted Volatility','Feature Importance'))
        n=min(150,len(yte))
        fig.add_trace(go.Scatter(y=yte.values[:n],name='Actual',line=dict(color='#00d4ff',width=2)),row=1,col=1)
        fig.add_trace(go.Scatter(y=rf_preds[:n],name='RF Predicted',line=dict(color='#f59e0b',width=2,dash='dash')),row=1,col=1)
        fig.add_trace(go.Scatter(y=r_preds[:n],name='Ridge Predicted',line=dict(color='#10b981',width=2,dash='dot')),row=1,col=1)
        imp6=pd.Series(rf6.feature_importances_,index=features_vol).sort_values()
        fig.add_trace(go.Bar(x=imp6.values,y=imp6.index,orientation='h',marker_color='#7c3aed',showlegend=False),row=1,col=2)
        fig.update_layout(template=PLOTLY_THEME,height=420,margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig,use_container_width=True)

# ── PS7 ───────────────────────────────────────────────────────────────────────
elif page == "🚨 PS7 — Anomaly Detection":
    st.markdown("<div class='main-header'><h1>🚨 Anomaly Detection</h1><p>Find unusual trading days using Isolation Forest & Local Outlier Factor</p></div>", unsafe_allow_html=True)
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.preprocessing import StandardScaler
    with st.sidebar:
        st.markdown("**PS7 Settings**")
        cont=st.slider("Anomaly %",1,20,5)/100
        co7=st.selectbox("Filter Country",["All"]+sorted(df['Country'].unique().tolist()),key='ps7')
    d7=df.copy()
    if co7!="All": d7=d7[d7['Country']==co7]
    d7['Vol_Change']=d7.groupby('Company')['Daily_Return'].transform(lambda x: x.rolling(5).std())
    d7['Return_Abs']=d7['Daily_Return'].abs(); d7=d7.dropna()
    features_a=['Daily_Return','Volume','Price_Range','BuySell_Ratio','Vol_Change','Return_Abs','Close']
    X7=d7[features_a]; sc7=StandardScaler(); X7_s=sc7.fit_transform(X7)
    iso=IsolationForest(contamination=cont,random_state=42); d7=d7.copy()
    d7['IF_Anomaly']=iso.fit_predict(X7_s)==-1
    lof=LocalOutlierFactor(n_neighbors=20,contamination=cont); d7['LOF_Anomaly']=lof.fit_predict(X7_s)==-1
    c1,c2,c3=st.columns(3)
    c1.markdown(f"<div class='metric-card'><div class='metric-value'>{d7['IF_Anomaly'].sum()}</div><div class='metric-label'>IF Anomalies</div></div>",unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-card'><div class='metric-value'>{d7['LOF_Anomaly'].sum()}</div><div class='metric-label'>LOF Anomalies</div></div>",unsafe_allow_html=True)
    c3.markdown(f"<div class='metric-card'><div class='metric-value'>{(d7['IF_Anomaly']&d7['LOF_Anomaly']).sum()}</div><div class='metric-label'>Detected by Both</div></div>",unsafe_allow_html=True)
    col1,col2=st.columns(2)
    with col1:
        st.markdown("<div class='section-title'>Daily Return — Anomalies</div>", unsafe_allow_html=True)
        normal=d7[~d7['IF_Anomaly']]; anomal=d7[d7['IF_Anomaly']]
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=normal.index,y=normal['Daily_Return'],mode='markers',marker=dict(color='#00d4ff',size=3,opacity=0.3),name='Normal'))
        fig.add_trace(go.Scatter(x=anomal.index,y=anomal['Daily_Return'],mode='markers',marker=dict(color='#ef4444',size=6,opacity=0.8),name='Anomaly'))
        fig.update_layout(template=PLOTLY_THEME,height=380,margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig,use_container_width=True)
    with col2:
        st.markdown("<div class='section-title'>Anomalies by Country</div>", unsafe_allow_html=True)
        abc=d7.groupby('Country')['IF_Anomaly'].sum().reset_index(); abc.columns=['Country','Anomalies']
        fig=px.bar(abc.sort_values('Anomalies',ascending=True),x='Anomalies',y='Country',orientation='h',color='Anomalies',color_continuous_scale='Reds',template=PLOTLY_THEME)
        fig.update_layout(height=380,showlegend=False,coloraxis_showscale=False,margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig,use_container_width=True)
    st.markdown("<div class='section-title'>Top Anomalous Days</div>", unsafe_allow_html=True)
    st.dataframe(d7[d7['IF_Anomaly']][['Date','Company','Country','Sector','Daily_Return','Volume','Price_Range']].sort_values('Daily_Return').head(15),use_container_width=True)

# ── PS8 ───────────────────────────────────────────────────────────────────────
elif page == "📈 PS8 — Trend Classification":
    st.markdown("<div class='main-header'><h1>📈 Market Trend Classification</h1><p>Classify each stock as BULLISH 🟢, SIDEWAYS 🟡 or BEARISH 🔴</p></div>", unsafe_allow_html=True)
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report,confusion_matrix
    with st.sidebar:
        st.markdown("**PS8 Settings**")
        co8=st.selectbox("Company",sorted(df['Company'].unique()),key='ps8')
    @st.cache_data
    def prepare_ps8(company):
        d=df[df['Company']==company].copy().sort_values('Date').reset_index(drop=True)
        d['Forward_5d']=d['Daily_Return'].shift(-1).rolling(5).mean()
        d['Trend_Label']=d['Forward_5d'].apply(lambda x: 2 if x>1.0 else (0 if x<-1.0 else 1))
        d['MA5']=d['Close'].rolling(5).mean(); d['MA20']=d['Close'].rolling(20).mean()
        d['MA_Cross']=d['MA5']-d['MA20']; d['Volatility']=d['Daily_Return'].rolling(7).std()
        d['Vol_Trend']=d['Volume'].rolling(5).mean(); d['BuySell_R']=d['BuySell_Ratio']; d=d.dropna()
        feat=['Close','Volume','MA5','MA20','MA_Cross','Volatility','Vol_Trend','Price_Range','BuySell_R','Daily_Return']
        X,y=d[feat],d['Trend_Label']
        Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,shuffle=False)
        sc=StandardScaler(); Xtr_s=sc.fit_transform(Xtr); Xte_s=sc.transform(Xte)
        rf8=RandomForestClassifier(n_estimators=50,random_state=42,n_jobs=-1).fit(Xtr_s,ytr)
        knn8=KNeighborsClassifier(n_neighbors=7).fit(Xtr_s,ytr)
        dt8=DecisionTreeClassifier(max_depth=6,random_state=42).fit(Xtr_s,ytr)
        res={}
        for name,m in [('Random Forest',rf8),('KNN',knn8),('Decision Tree',dt8)]:
            p=m.predict(Xte_s); rep=classification_report(yte,p,target_names=['BEARISH','SIDEWAYS','BULLISH'],output_dict=True)
            res[name]={'preds':p,'report':rep,'cm':confusion_matrix(yte,p)}
        return res,yte,d['Trend_Label'].value_counts()
    res8,yte8,dist8=prepare_ps8(co8)
    c1,c2,c3=st.columns(3)
    cm8={'Random Forest':'#10b981','KNN':'#f59e0b','Decision Tree':'#7c3aed'}
    for col,(name,r) in zip([c1,c2,c3],res8.items()):
        col.markdown(f"<div class='metric-card' style='border-top:3px solid {cm8[name]};'><div style='font-family:Space Mono,monospace;font-size:0.75rem;color:{cm8[name]};'>{name}</div><div class='metric-value' style='font-size:1.4rem;'>{r['report']['accuracy']:.2%}</div><div class='metric-label'>Accuracy</div></div>",unsafe_allow_html=True)
    col1,col2=st.columns(2)
    with col1:
        st.markdown("<div class='section-title'>Trend Distribution</div>", unsafe_allow_html=True)
        fig=go.Figure(go.Bar(x=['BEARISH','SIDEWAYS','BULLISH'],y=[dist8.get(0,0),dist8.get(1,0),dist8.get(2,0)],marker_color=['#ef4444','#f59e0b','#10b981']))
        fig.update_layout(template=PLOTLY_THEME,height=380,margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig,use_container_width=True)
    with col2:
        st.markdown("<div class='section-title'>Confusion Matrix — Random Forest</div>", unsafe_allow_html=True)
        cm=res8['Random Forest']['cm']
        fig=go.Figure(go.Heatmap(z=cm,x=['BEAR','SIDE','BULL'],y=['BEAR','SIDE','BULL'],colorscale='Greens',text=cm,texttemplate='%{text}',textfont_size=16,showscale=False))
        fig.update_layout(template=PLOTLY_THEME,height=380,xaxis_title='Predicted',yaxis_title='Actual',margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig,use_container_width=True)

# ── PS9 ───────────────────────────────────────────────────────────────────────
elif page == "⚔️ PS9 — War Period Impact":
    st.markdown("<div class='main-header'><h1>⚔️ War Period Impact Analysis</h1><p>Statistically test whether the war significantly impacted global stock returns</p></div>", unsafe_allow_html=True)
    from scipy import stats
    d9=df.copy(); d9['Period']=d9['War_Period'].apply(lambda x: 'Post-War' if 'Post' in str(x) else 'Pre-War')
    pre=d9[d9['Period']=='Pre-War']; post=d9[d9['Period']=='Post-War']
    t_stat,p_val_t=stats.ttest_ind(pre['Daily_Return'].dropna(),post['Daily_Return'].dropna(),equal_var=False)
    u_stat,p_val_u=stats.mannwhitneyu(pre['Daily_Return'].dropna(),post['Daily_Return'].dropna(),alternative='two-sided')
    pooled_std=np.sqrt((pre['Daily_Return'].std()**2+post['Daily_Return'].std()**2)/2)
    cohens_d=abs(pre['Daily_Return'].mean()-post['Daily_Return'].mean())/pooled_std if pooled_std>0 else 0
    c1,c2,c3,c4=st.columns(4)
    c1.markdown(f"<div class='metric-card'><div class='metric-value'>{pre['Daily_Return'].mean():.4f}</div><div class='metric-label'>Pre-War Avg Return</div></div>",unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-card'><div class='metric-value'>{post['Daily_Return'].mean():.4f}</div><div class='metric-label'>Post-War Avg Return</div></div>",unsafe_allow_html=True)
    c3.markdown(f"<div class='metric-card'><div class='metric-value'>{p_val_t:.4f}</div><div class='metric-label'>T-Test P-Value</div></div>",unsafe_allow_html=True)
    c4.markdown(f"<div class='metric-card'><div class='metric-value'>{cohens_d:.4f}</div><div class='metric-label'>Cohen's D</div></div>",unsafe_allow_html=True)
    sig="✅ Significant" if p_val_t<0.05 else "❌ Not Significant"
    color_sig="#10b981" if p_val_t<0.05 else "#ef4444"
    st.markdown(f"<div style='text-align:center;margin:1rem 0;'><span style='background:rgba(0,0,0,0.3);border:1px solid {color_sig};color:{color_sig};padding:0.5rem 2rem;border-radius:50px;font-family:Space Mono,monospace;font-weight:700;'>{sig}</span></div>",unsafe_allow_html=True)
    col1,col2=st.columns(2)
    with col1:
        st.markdown("<div class='section-title'>Return Distribution by Period</div>", unsafe_allow_html=True)
        fig=go.Figure()
        for period,color in [('Pre-War','#00d4ff'),('Post-War','#ef4444')]:
            fig.add_trace(go.Histogram(x=d9[d9['Period']==period]['Daily_Return'].dropna(),name=period,opacity=0.6,marker_color=color,nbinsx=80,histnorm='probability density'))
        fig.update_layout(template=PLOTLY_THEME,height=380,barmode='overlay',xaxis_title='Daily Return (%)',margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig,use_container_width=True)
    with col2:
        st.markdown("<div class='section-title'>Avg Return by Country & Period</div>", unsafe_allow_html=True)
        cp=d9.groupby(['Country','Period'])['Daily_Return'].mean().reset_index()
        fig=px.bar(cp,x='Country',y='Daily_Return',color='Period',barmode='group',color_discrete_map={'Pre-War':'#00d4ff','Post-War':'#ef4444'},template=PLOTLY_THEME)
        fig.update_layout(height=380,xaxis_tickangle=-30,margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig,use_container_width=True)
    st.dataframe(pd.DataFrame({'Test':['T-Test','Mann-Whitney U'],'Statistic':[round(t_stat,4),round(u_stat,2)],'P-Value':[round(p_val_t,6),round(p_val_u,6)],'Significant':['Yes' if p_val_t<0.05 else 'No','Yes' if p_val_u<0.05 else 'No']}),use_container_width=True)

# ── PS10 ──────────────────────────────────────────────────────────────────────
elif page == "🔄 PS10 — Sector Rotation":
    st.markdown("<div class='main-header'><h1>🔄 Sector Rotation Strategy</h1><p>Identify momentum sectors and backtest a rotation strategy vs benchmark</p></div>", unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("**PS10 Settings**")
        top_n=st.slider("Top N Sectors",1,5,3)
        mom_window=st.slider("Momentum Window (months)",1,6,3)
    d10=df.copy(); d10['YearMonth']=d10['Date'].dt.to_period('M')
    ms=d10.groupby(['YearMonth','Sector'])['Daily_Return'].mean().reset_index()
    ms.columns=['YearMonth','Sector','Monthly_Return']; ms['Date']=ms['YearMonth'].dt.to_timestamp()
    ms=ms.sort_values(['Sector','YearMonth'])
    ms['Momentum']=ms.groupby('Sector')['Monthly_Return'].transform(lambda x: x.rolling(mom_window).mean()); ms=ms.dropna()
    strat_ret=[]; bench_ret=[]; dates_list=[]
    unique_months=sorted(ms['YearMonth'].unique())
    for month in unique_months[mom_window:]:
        prev=ms[ms['YearMonth']<month]
        if len(prev)==0: continue
        latest=prev.groupby('Sector')['Momentum'].last()
        top_sectors=latest.nlargest(top_n).index.tolist()
        curr=ms[ms['YearMonth']==month]
        if len(curr)==0: continue
        strat_ret.append(curr[curr['Sector'].isin(top_sectors)]['Monthly_Return'].mean())
        bench_ret.append(curr['Monthly_Return'].mean())
        dates_list.append(month.to_timestamp())
    res10=pd.DataFrame({'Date':dates_list,'Strategy':strat_ret,'Benchmark':bench_ret}).dropna()
    res10['Cum_Strategy']=(1+res10['Strategy']/100).cumprod()
    res10['Cum_Benchmark']=(1+res10['Benchmark']/100).cumprod()
    strat_total=(res10['Cum_Strategy'].iloc[-1]-1)*100 if len(res10)>0 else 0
    bench_total=(res10['Cum_Benchmark'].iloc[-1]-1)*100 if len(res10)>0 else 0
    out=strat_total-bench_total
    c1,c2,c3=st.columns(3)
    c1.markdown(f"<div class='metric-card'><div class='metric-value'>{strat_total:.1f}%</div><div class='metric-label'>Strategy Return</div></div>",unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-card'><div class='metric-value'>{bench_total:.1f}%</div><div class='metric-label'>Benchmark Return</div></div>",unsafe_allow_html=True)
    out_color='#10b981' if out>0 else '#ef4444'
    c3.markdown(f"<div class='metric-card'><div class='metric-value' style='color:{out_color};'>{out:+.1f}%</div><div class='metric-label'>Outperformance</div></div>",unsafe_allow_html=True)
    col1,col2=st.columns(2)
    with col1:
        st.markdown("<div class='section-title'>Cumulative Returns</div>", unsafe_allow_html=True)
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=res10['Date'],y=res10['Cum_Strategy'],name=f'Top {top_n} Strategy',line=dict(color='#10b981',width=2.5)))
        fig.add_trace(go.Scatter(x=res10['Date'],y=res10['Cum_Benchmark'],name='Benchmark',line=dict(color='#ef4444',width=2.5,dash='dash')))
        fig.update_layout(template=PLOTLY_THEME,height=400,xaxis_title='Date',yaxis_title='Cumulative Return',margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig,use_container_width=True)
    with col2:
        st.markdown("<div class='section-title'>Avg Momentum by Sector</div>", unsafe_allow_html=True)
        avg_mom=ms.groupby('Sector')['Momentum'].mean().sort_values(ascending=True).reset_index()
        avg_mom['Color']=avg_mom['Momentum'].apply(lambda x: '#10b981' if x>=0 else '#ef4444')
        fig=px.bar(avg_mom,x='Momentum',y='Sector',orientation='h',color='Color',color_discrete_map='identity',template=PLOTLY_THEME)
        fig.update_layout(height=400,showlegend=False,margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig,use_container_width=True)
