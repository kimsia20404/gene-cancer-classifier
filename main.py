        box_df = df[["Class", selected_gene]].dropna().copy()
    box_df["암 종류"] = box_df["Class"].map(lambda x: friendly_class(x))
    fig_gene = px.box(
        box_df, x="암 종류", y=selected_gene, color="암 종류",
        points="outliers", title=f"{selected_gene} 발현량의 암 종류별 분포"
    )
    fig_gene.update_layout(showlegend=False)
    st.plotly_chart(fig_gene, use_container_width=True)
    st.caption("상자 그림의 중앙값과 분포 범위를 비교해 암 종류에 따른 발현량 차이를 탐색합니다.")

    st.divider()
    st.subheader("PCA로 유전자 발현 패턴을 2차원으로 보기")
    st.caption("PCA는 수많은 유전자 정보를 두 개의 축으로 요약하는 탐색용 시각화입니다.")
    pca_input = SimpleImputer(strategy="median").fit_transform(X)
    pca_scaled = StandardScaler().fit_transform(pca_input)
    coords = PCA(n_components=2, random_state=42).fit_transform(pca_scaled)
    pca_df = pd.DataFrame({
        "주성분 1": coords[:, 0],
        "주성분 2": coords[:, 1],
        "암 종류": y.map(lambda x: friendly_class(x))
    })
    fig_pca = px.scatter(
        pca_df, x="주성분 1", y="주성분 2", color="암 종류",
        title="PCA 2차원 산점도", opacity=0.75
    )
    st.plotly_chart(fig_pca, use_container_width=True)
    st.caption("점들이 암 종류별로 모이는지 확인합니다. 가까이 모인다고 해서 임상 진단이 가능하다는 뜻은 아닙니다.")

with tab3:
    st.header("③ 모델링")
    st.write("정답 열인 `Class`를 예측하도록 두 가지 분류 모델을 학습합니다.")
    st.markdown("""
    - **독립 변수(X):** `gene_0`부터 `gene_20530`까지의 유전자 발현량
    - **종속 변수(y):** `Class`에 기록된 암 종류
    - **제외하는 열:** `Sample` 등 샘플을 구분하는 ID
    """)
    k = st.select_slider(
        "모델이 사용할 유전자 수 선택",
        options=[10, 25, 50, 100, 200, 500],
        value=50
    )
    st.caption("선택한 유전자 수만큼 학습 데이터에서 중요한 특성을 고릅니다. 테스트 데이터는 학습 과정에 사용하지 않습니다.")
    if st.button("두 모델 학습하기", type="primary", use_container_width=True):
        st.session_state["model_k"] = k
        st.session_state["models_ready"] = True

    if st.session_state.get("models_ready") and st.session_state.get("model_k") == k:
        results = train_models(X, y, k)
        metrics = pd.DataFrame([
            {"모델": name, "정확도": result["accuracy"], "Macro F1": result["macro_f1"]}
            for name, result in results.items()
        ])
        st.success("모델 학습이 완료되었습니다.")
        st.dataframe(
            metrics.style.format({"정확도": "{:.3f}", "Macro F1": "{:.3f}"}),
            use_container_width=True, hide_index=True
        )
        st.session_state["model_results"] = results
        st.session_state["model_classes"] = classes
    else:
        st.info("유전자 수를 선택한 뒤 **두 모델 학습하기** 버튼을 눌러 주세요.")

with tab4:
    st.header("④ 모델 평가")
    results = st.session_state.get("model_results")
    if not results:
        st.info("먼저 ③ 모델링 탭에서 두 모델을 학습해 주세요.")
    else:
        model_name = st.selectbox("평가할 모델 선택", list(results.keys()))
        result = results[model_name]
        m1, m2 = st.columns(2)
        m1.metric("정확도 (Accuracy)", f"{result['accuracy']:.3f}")
        m2.metric("Macro F1 점수", f"{result['macro_f1']:.3f}")

        st.subheader("혼동 행렬")
        cm = confusion_matrix(result["y_test"], result["pred"], labels=classes)
        cm_df = pd.DataFrame(cm, index=classes, columns=classes)
        cm_df.index.name = "실제 정답"
        cm_df.columns.name = "모델 예측"
        st.dataframe(cm_df, use_container_width=True)

        fig_cm = px.imshow(
            cm, x=classes, y=classes, text_auto=True,
            labels={"x": "예측한 암 종류", "y": "실제 암 종류", "color": "샘플 수"},
            title=f"{model_name} 혼동 행렬", aspect="auto"
        )
        st.plotly_chart(fig_cm, use_container_width=True)

        st.subheader("분류 결과 보고서")
        report = classification_report(
            result["y_test"], result["pred"], labels=classes,
            output_dict=True, zero_division=0
        )
        report_df = pd.DataFrame(report).transpose().round(3)
        st.dataframe(report_df, use_container_width=True)

        st.subheader("테스트 샘플 예측 체험")
        test_indices = list(result["test_indices"])
        row_pos = st.selectbox(
            "테스트 샘플 선택",
            options=range(len(test_indices)),
            format_func=lambda i: str(df.loc[test_indices[i], id_col]) if id_col in df.columns else f"테스트 샘플 {i+1}"
        )
        sample_index = test_indices[row_pos]
        sample_x = X.loc[[sample_index]]
        predicted = result["model"].predict(sample_x)[0]
        actual = y.loc[sample_index]
        st.write(f"모델 예측: **{friendly_class(predicted)}**")
        st.write(f"실제 정답: **{friendly_class(actual)}**")
        if predicted == actual:
            st.success("이 테스트 샘플은 모델이 올바르게 분류했습니다.")
        else:
            st.error("이 테스트 샘플은 모델이 다르게 분류했습니다.")

        st.caption("테스트 샘플은 학습에 사용하지 않은 데이터 중에서 선택합니다. 이 결과는 공개된 연구용 데이터셋의 분류 실습 결과이며 의료 진단 결과가 아닙니다.")

st.divider()
st.caption("프로젝트 질문: 유전자 발현 데이터를 활용하여 5가지 암 종류를 예측하고 분류할 수 있을까?")
