import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import datetime

# ----------------------------
# FloatChat — Enhanced Prototype Streamlit App
# Single-file prototype with: Dashboard, Chat, Data Explorer, Satellite View
# Uses synthetic/dummy Argo-like float data and synthetic satellite SST grid.
# ----------------------------

st.set_page_config(page_title="FloatChat — Enhanced Prototype", layout="wide", page_icon="🌊")

# ----------------------------
# Utility: Generate richer dummy dataset
# ----------------------------
@st.cache_data
def generate_dummy_floats(n_floats=30, profiles_per_float=10, seed=42):
    rng = np.random.default_rng(seed)
    floats = []
    profiles = []
    docs = []

    for i in range(n_floats):
        fid = f"F{2900000 + i}"
        lat = float(rng.uniform(-15, 25))
        lon = float(rng.uniform(40, 100))
        deploy = datetime.date(2021 + int(rng.integers(0,3)), int(rng.integers(0,12))+1, int(rng.integers(0,27))+1)
        params = ["TEMP", "PSAL"]
        if rng.random() < 0.5:
            params.append("DOXY")
        status = "active" if rng.random() < 0.85 else "inactive"

        floats.append({
            "float_id": fid,
            "lat": round(lat,3),
            "lon": round(lon,3),
            "deploy_date": str(deploy),
            "params": ", ".join(params),
            "status": status
        })

        # create profiles for this float
        for p in range(profiles_per_float):
            prof_date = datetime.date(2023, int(rng.integers(0,12))+1, min(int(rng.integers(0,27))+1,28))
            depths = np.array([0, 10, 25, 50, 100, 200, 500])
            surf_temp = float(rng.uniform(20, 30))
            temp = surf_temp - 0.02*depths + rng.normal(0,0.25,size=depths.shape)
            surf_sal = float(rng.uniform(33.5, 36.5))
            sal = surf_sal - 0.001*depths + rng.normal(0,0.05,size=depths.shape)
            oxy = None
            if "DOXY" in params:
                oxy = rng.uniform(2,6,size=depths.shape) - 0.0006*depths + rng.normal(0,0.15,size=depths.shape)

            for d,t,s,o in zip(depths,temp,sal, oxy if oxy is not None else [None]*len(depths)):
                profiles.append({
                    "float_id": fid,
                    "profile_date": str(prof_date),
                    "lat": round(lat + float(rng.normal(0,0.05)),4),
                    "lon": round(lon + float(rng.normal(0,0.05)),4),
                    "depth_m": float(d),
                    "TEMP_C": float(round(float(t),3)),
                    "PSAL_PSU": float(round(float(s),3)),
                    "DOXY_umolkg": float(round(float(o),3)) if o is not None else None
                })

            # doc summary
            doc = (
                f"Profile {fid} on {prof_date}: location ~{lat:.2f}N,{lon:.2f}E. "
                f"Measured {', '.join(params)}. Surface temp {surf_temp:.1f} C, surface sal {surf_sal:.2f} PSU."
            )
            docs.append({"id": f"doc_{fid}_{prof_date}", "float_id": fid, "date": str(prof_date), "text": doc})

    return pd.DataFrame(floats), pd.DataFrame(profiles), pd.DataFrame(docs)

# create data
DF_FLOATS, DF_PROFILES, DF_DOCS = generate_dummy_floats()

# ----------------------------
# Synthetic satellite SST grid (dummy)
# ----------------------------
@st.cache_data
def generate_sst_grid(lon_min=40, lon_max=100, lat_min=-20, lat_max=30, nx=200, ny=120, seed=7):
    rng = np.random.default_rng(seed)
    lons = np.linspace(lon_min, lon_max, nx)
    lats = np.linspace(lat_min, lat_max, ny)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    # base SST pattern: warmer near equator, cooler towards edges + random variability
    sst_base = 28 - 0.05 * np.abs(lat_grid) + 2 * np.sin((lon_grid - 60) / 10)
    sst_noise = rng.normal(0, 0.6, size=sst_base.shape)
    sst = sst_base + sst_noise
    return lons, lats, sst

LONS, LATS, SST = generate_sst_grid()

# ----------------------------
# Vector-like TF-IDF index for demo semantic retrieval
# ----------------------------
@st.cache_resource
def build_tfidf_index(docs_df):
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_features=1024)
    mat = vectorizer.fit_transform(docs_df['text'].values)
    return vectorizer, mat

VECT, DOC_MAT = build_tfidf_index(DF_DOCS)

def semantic_retrieve(query, topk=4):
    qv = VECT.transform([query])
    sims = cosine_similarity(qv, DOC_MAT).flatten()
    idx = np.argsort(-sims)[:topk]
    res = DF_DOCS.iloc[idx].copy()
    res['score'] = sims[idx]
    return res

# ----------------------------
# Simulated LLM response that combines float + satellite context
# ----------------------------
def simulate_llm_answer(query):
    hits = semantic_retrieve(query, topk=5)
    # compute a small satellite summary (avg SST in bounding box if region mentioned)
    region_hint = None
    q = query.lower()
    if 'arabian' in q:
        region_hint = 'Arabian Sea'
        lon_min, lon_max, lat_min, lat_max = 55, 75, 5, 25
    elif 'bay of bengal' in q or 'bay' in q:
        region_hint = 'Bay of Bengal'
        lon_min, lon_max, lat_min, lat_max = 80, 100, 5, 22
    elif 'equator' in q:
        region_hint = 'Equatorial band'
        lon_min, lon_max, lat_min, lat_max = 40, 100, -5, 5
    else:
        # default whole domain
        lon_min, lon_max, lat_min, lat_max = 40, 100, -20, 30

    # find indices
    xi = np.where((LONS >= lon_min) & (LONS <= lon_max))[0]
    yi = np.where((LATS >= lat_min) & (LATS <= lat_max))[0]
    if len(xi) == 0 or len(yi) == 0:
        sst_avg = float('nan')
    else:
        sub = SST[np.min(yi):np.max(yi)+1, np.min(xi):np.max(xi)+1]
        sst_avg = float(np.nanmean(sub))

    # float summary: top docs
    float_lines = []
    for _, r in hits.iterrows():
        float_lines.append(f"- {r['id']}: {r['text']}")
    float_summaries = "\n".join(float_lines)

    # build multi-line answer safely
    answer_lines = []
    answer_lines.append("Summary (simulated):")
    if region_hint:
        answer_lines.append(f"Region: {region_hint}. Estimated mean satellite SST in region: {sst_avg:.2f} °C.")
    else:
        answer_lines.append(f"Estimated mean satellite SST for domain: {sst_avg:.2f} °C.")
    answer_lines.append("")
    answer_lines.append("Relevant float profiles found:")
    answer_lines.append(float_summaries)
    answer_lines.append("")
    answer_lines.append("Note: This response is a static simulation combining semantic retrieval and a synthetic satellite summary.")
    answer = "\n".join(answer_lines)
    return answer

# ----------------------------
# UI: Header + top metrics
# ----------------------------
st.title("FloatChat — Enhanced Ocean Data Hub (Prototype)")

# Top metric cards
col1, col2, col3, col4, col5 = st.columns([1.2,1,1,1,1])
with col1:
    st.metric("Total floats", int(DF_FLOATS.shape[0]))
with col2:
    avg_surf_temp = DF_PROFILES[DF_PROFILES['depth_m']==0]['TEMP_C'].mean()
    st.metric("Avg surface temp (°C)", f"{avg_surf_temp:.2f}")
with col3:
    avg_surf_sal = DF_PROFILES[DF_PROFILES['depth_m']==0]['PSAL_PSU'].mean()
    st.metric("Avg surface salinity (PSU)", f"{avg_surf_sal:.2f}")
with col4:
    bgc_count = DF_FLOATS['params'].str.contains('DOXY').sum()
    st.metric("BGC-capable floats", int(bgc_count))
with col5:
    st.metric("Latest profile", str(DF_PROFILES['profile_date'].max()))

# Tabs for navigation
tab_dash, tab_query, tab_data, tab_sat = st.tabs(["Dashboard", "Query (Chat)", "Data Explorer", "Satellite View"])

# ----------------------------
# Dashboard Tab
# ----------------------------
with tab_dash:
    st.header("Dashboard")
    st.markdown("Overview of float activity and satellite comparisons")

    # region selector
    reg_col1, reg_col2 = st.columns([2,3])
    with reg_col1:
        region_select = st.selectbox("Select Region", ["All", "Arabian Sea", "Bay of Bengal", "Equatorial band"])
    with reg_col2:
        compare_vars = st.multiselect("Compare variables (chart)", ["TEMP_C","PSAL_PSU","DOXY_umolkg"], default=["TEMP_C","PSAL_PSU"])

    # filter profiles by region selection
    df_dash = DF_PROFILES.copy()
    if region_select == 'Arabian Sea':
        df_dash = df_dash[(df_dash['lat']>=5)&(df_dash['lat']<=25)&(df_dash['lon']>=55)&(df_dash['lon']<=75)]
    elif region_select == 'Bay of Bengal':
        df_dash = df_dash[(df_dash['lat']>=5)&(df_dash['lat']<=22)&(df_dash['lon']>=80)&(df_dash['lon']<=100)]
    elif region_select == 'Equatorial band':
        df_dash = df_dash[(df_dash['lat']>=-5)&(df_dash['lat']<=5)]

    # Profiles per month
    df_dash['profile_date_dt'] = pd.to_datetime(df_dash['profile_date'])
    ts = df_dash.groupby(pd.Grouper(key='profile_date_dt', freq='M')).size().reset_index(name='count')
    fig_ts = px.bar(ts, x='profile_date_dt', y='count', title='Profiles per month')
    st.plotly_chart(fig_ts, use_container_width=True)

    # Avg profiles by depth for chosen vars
    if compare_vars:
        fig = go.Figure()
        for var in compare_vars:
            # aggregate mean by depth across profiles
            agg = df_dash.groupby('depth_m')[[var]].mean().reset_index()
            fig.add_trace(go.Scatter(x=agg[var], y=agg['depth_m'], mode='lines+markers', name=var))
        fig.update_yaxes(autorange='reversed', title_text='Depth (m)')
        fig.update_xaxes(title_text='Value')
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Float Locations")
    left_map, right_list = st.columns([3,1])
    with left_map:
        layer = pdk.Layer(
            "ScatterplotLayer",
            DF_FLOATS,
            get_position='[lon, lat]',
            get_radius=50000,
            get_fill_color=[120, 100, 200],
            pickable=True,
            auto_highlight=True
        )
        view = pdk.ViewState(latitude=float(DF_FLOATS['lat'].mean()), longitude=float(DF_FLOATS['lon'].mean()), zoom=3)
        tooltip = {"html": "<b>Float:</b> {float_id} <br/> <b>Params:</b> {params} <br/> <b>Status:</b> {status}", "style": {"color": "#fff"}}
        deck = pdk.Deck(layers=[layer], initial_view_state=view, tooltip=tooltip)
        st.pydeck_chart(deck)
    with right_list:
        st.dataframe(DF_FLOATS[['float_id','lat','lon','params','status']].head(15))

# ----------------------------
# Query / Chat Tab
# ----------------------------
# ----------------------------
# Query / Chat Tab
# ----------------------------
with tab_query:
    st.header("Ask FloatChat")
    st.markdown("Type a question in natural language or pick from static examples. This prototype returns a simulated contextual answer combining float documents and satellite summary.")

    # Static query options
    example_queries = [
        "Show me recent profiles in Arabian Sea",
        "What’s the average SST in Bay of Bengal?",
        "Give me float profiles near the equator",
        "Which floats measured oxygen?",
    ]
    ex_q = st.selectbox("Example queries", ["-- Select --"] + example_queries)

    user_q = st.text_area("Your question...", height=120, value=ex_q if ex_q != "-- Select --" else "")

    if st.button("Ask"):
        if not user_q.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Generating answer (simulated)..."):
                ans = simulate_llm_answer(user_q)
            st.subheader("Answer")
            st.write(ans)

            # Supporting documents
            st.markdown("---")
            st.subheader("Supporting evidence (top documents)")
            hits = semantic_retrieve(user_q, topk=6)
            for _,h in hits.iterrows():
                st.write(f"**{h['id']}** — {h['text']} (score: {h['score']:.3f})")

            # ----------------------------
            # Visualization: Parameter graph
            # ----------------------------
            st.markdown("---")
            st.subheader("Related Parameter Profiles")

            # Try to infer region from query
            dfq = DF_PROFILES.copy()
            q = user_q.lower()
            if 'arabian' in q:
                dfq = dfq[(dfq['lat']>=5)&(dfq['lat']<=25)&(dfq['lon']>=55)&(dfq['lon']<=75)]
            elif 'bay' in q:
                dfq = dfq[(dfq['lat']>=5)&(dfq['lat']<=22)&(dfq['lon']>=80)&(dfq['lon']<=100)]
            elif 'equator' in q:
                dfq = dfq[(dfq['lat']>=-5)&(dfq['lat']<=5)]

            if len(dfq) > 0:
                fig = go.Figure()
                for var in ["TEMP_C","PSAL_PSU"]:
                    agg = dfq.groupby('depth_m')[[var]].mean().reset_index()
                    fig.add_trace(go.Scatter(x=agg[var], y=agg['depth_m'], mode='lines+markers', name=var))
                fig.update_yaxes(autorange='reversed', title_text='Depth (m)')
                fig.update_xaxes(title_text='Value')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No relevant profiles found for visualization.")

            # ----------------------------
            # Visualization: Float locations on map
            # ----------------------------
            st.subheader("Float Locations (for query)")
            if len(dfq) > 0:
                layer = pdk.Layer(
                    "ScatterplotLayer",
                    dfq.drop_duplicates("float_id"),
                    get_position='[lon, lat]',
                    get_radius=60000,
                    get_color='[0, 150, 250]',
                    pickable=True,
                )
                view = pdk.ViewState(
                    latitude=float(dfq['lat'].mean()),
                    longitude=float(dfq['lon'].mean()),
                    zoom=3
                )
                tooltip = {"html": "<b>Float:</b> {float_id} <br/> <b>Date:</b> {profile_date}", "style": {"color": "#fff"}}
                deck = pdk.Deck(layers=[layer], initial_view_state=view, tooltip=tooltip)
                st.pydeck_chart(deck)

# ----------------------------
# Data Explorer Tab
# ----------------------------
with tab_data:
    st.header("Data Explorer")
    dfq = DF_PROFILES.copy()

    # Filters
    f1, f2, f3, f4 = st.columns([1,1,1,1])
    with f1:
        sel_float = st.selectbox("Float ID", options=["All"] + DF_FLOATS['float_id'].tolist())
    with f2:
        sel_param = st.selectbox("Parameter", options=['TEMP_C','PSAL_PSU','DOXY_umolkg'])
    with f3:
        date_from = st.date_input("From", value=pd.to_datetime(DF_PROFILES['profile_date']).min())
    with f4:
        date_to = st.date_input("To", value=pd.to_datetime(DF_PROFILES['profile_date']).max())

    if sel_float != 'All':
        dfq = dfq[dfq['float_id']==sel_float]
    dfq['profile_date_dt'] = pd.to_datetime(dfq['profile_date'])
    dfq = dfq[(dfq['profile_date_dt']>=pd.to_datetime(date_from)) & (dfq['profile_date_dt']<=pd.to_datetime(date_to))]

    st.markdown(f"Showing {len(dfq)} rows")
    st.dataframe(dfq.head(500))

    # Plot profiles
    if len(dfq) > 0:
        fig = go.Figure()
        for fid, grp in dfq.groupby('float_id'):
            agg = grp.groupby('depth_m')[[sel_param]].mean().reset_index()
            fig.add_trace(go.Scatter(x=agg[sel_param], y=agg['depth_m'], mode='lines+markers', name=fid))
        fig.update_yaxes(autorange='reversed', title_text='Depth (m)')
        fig.update_xaxes(title_text=sel_param)
        st.plotly_chart(fig, use_container_width=True)

    csv = dfq.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "profiles_export.csv", "text/csv")

# ----------------------------
# Satellite View Tab
# ----------------------------
with tab_sat:
    st.header("Satellite View — Synthetic SST Overlay")
    st.markdown("Synthetic sea surface temperature (SST) grid displayed as a heatmap. Floats are overlaid on the same geographic extent.")

    # show SST heatmap using plotly imshow
    sst_img = SST
    fig_sst = px.imshow(sst_img, x=LONS, y=LATS, aspect='auto', origin='lower', labels={'x':'Longitude','y':'Latitude','color':'SST (°C)'})
    fig_sst.update_layout(height=600)

    # overlay float locations as scatter on same figure
    fig_sst.add_scatter(x=DF_FLOATS['lon'], y=DF_FLOATS['lat'], mode='markers', marker=dict(size=8, color='black'), name='Floats')

    st.plotly_chart(fig_sst, use_container_width=True)

    # Controls: toggle grid or contours
    st.write("Controls")
    col_a, col_b = st.columns(2)
    with col_a:
        show_contours = st.checkbox("Show contour lines", value=True)
    with col_b:
        show_grid = st.checkbox("Show grid", value=False)

    if show_contours:
        # add contours as separate figure
        fig_cont = go.Figure(data=go.Contour(z=SST, x=LONS, y=LATS, contours_coloring='heatmap'))
        fig_cont.add_trace(go.Scatter(x=DF_FLOATS['lon'], y=DF_FLOATS['lat'], mode='markers', marker=dict(size=6, color='white'), name='Floats'))
        fig_cont.update_layout(height=500)
        st.plotly_chart(fig_cont, use_container_width=True)

# End of app
