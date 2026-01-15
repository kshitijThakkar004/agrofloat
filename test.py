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
