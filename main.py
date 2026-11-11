   
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from pathlib import Path
import io

# --------------------------------------------------
# 1. 기본 설정
# --------------------------------------------------

st.set_page_config(
    page_title="유전자 발현 암 분류 실습실",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 유전자 발현 암 분류 실습실")

st.markdown("""
**프로젝트 질문**

유전자 발현 데이터를 활용하여 5가지 암의 종류를 예측하고 분류할 수 있을까?

이 웹앱은 공개된 연구용 RNA-Seq 데이터로 분류 모델을 학습하고
성능을 평가하는 교육용 프로젝트입니다.
""")

# --------------------------------------------------
# 2. 암 종류 한글 이름
# --------------------------------------------------

CLASS_NAMES = {
    "BRCA": "유방암",
    "KIRC": "신장 투명세포암",
    "COAD": "대장암",
    "LUAD": "폐 선암",
    "PRAD": "전립선암"
}


def friendly_class(value):
    return CLASS_NAMES.get(str(value), str(value))


# --------------------------------------------------
# 3. 데이터 읽기
# --------------------------------------------------

@st.cache_data
def load_data(data_bytes, labels_bytes):

    data = pd.read_csv(io.BytesIO(data_bytes))
    labels = pd.read_csv(io.BytesIO(labels_bytes))

    # Sample 열 이름 확인
    if "Sample" not in data.columns:
        data = data.rename(columns={data.columns[0]: "Sample"})

    if "Sample" not in labels.columns:
        labels = labels.rename(columns={labels.columns[0]: "Sample"})

    # 두 파일 연결
    df = pd.merge(data, labels, on="Sample", how="inner")

    # 유전자 열 찾기
    gene_cols = [
        col for col in df.columns
        if str(col).startswith("gene_")
    ]

    # 유전자 값 숫자로 변환
    for col in gene_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Class"] = df["Class"].astype(str).str.strip()

    return df, gene_cols


# --------------------------------------------------
# 4. 데이터 업로드
# --------------------------------------------------

st.sidebar.header("📂 데이터 준비")

st.sidebar.write(
    "data.csv와 labels.csv를 각각 업로드해 주세요."
)

data_file = st.sidebar.file_uploader(
    "data.csv 업로드",
    type=["csv"],
    key="data_upload"
)

labels_file = st.sidebar.file_uploader(
    "labels.csv 업로드",
    type=["csv"],
    key="labels_upload"
)

if data_file is None or labels_file is None:

    st.info(
        "왼쪽 메뉴에서 data.csv와 labels.csv를 모두 업로드하면 "
        "분석이 시작됩니다."
    )

    # GitHub 저장소에 파일이 있을 경우 자동 읽기
    if Path("data.csv").exists() and Path("labels.csv").exists():

        data_bytes = Path("data.csv").read_bytes()
        labels_bytes = Path("labels.csv").read_bytes()

    else:
        st.warning(
            "데이터 파일이 아직 준비되지 않았습니다. "
            "두 CSV 파일을 업로드해 주세요."
        )
        st.stop()

else:
    data_bytes = data_file.getvalue()
    labels_bytes = labels_file.getvalue()


# --------------------------------------------------
# 5. 데이터 불러오기 및 전처리
# --------------------------------------------------

try:

    df, gene_cols = load_data(data_bytes, labels_bytes)

except Exception as e:

    st.error(f"데이터를 읽는 중 오류가 발생했습니다: {e}")
    st.stop()


if "Class" not in df.columns or len(gene_cols) == 0:
    st.error(
        "데이터 구조를 확인할 수 없습니다. "
        "data.csv와 labels.csv의 열 이름을 확인해 주세요."
    )
    st.stop()


# 실제 암 종류 확인
classes = sorted(df["Class"].dropna().unique())

if len(classes) != 5:
    st.warning(
        f"현재 데이터에서 {len(classes)}개 암 종류가 발견되었습니다. "
        "원본 데이터가 모두 포함되었는지 확인해 주세요."
    )


# Sample ID 제외하고 유전자만 사용
id_col = "Sample"

X = df[gene_cols].copy()
y = df["Class"].copy()

# 결측치 제거: 정답이 없는 행만 제거
valid = y.notna()

X = X.loc[valid].reset_index(drop=True)
y = y.loc[valid].reset_index(drop=True)
df = df.loc[valid].reset_index(drop=True)

# 모두 결측인 유전자 열 제외
gene_cols = [
    col for col in gene_cols
    if X[col].notna().any()
]

X = X[gene_cols]

# --------------------------------------------------
# 6. 세션 상태
# --------------------------------------------------

if "model_results" not in st.session_state:
    st.session_state["model_results"] = None

if "models_ready" not in st.session_state:
    st.session_state["models_ready"] = False


# --------------------------------------------------
# 7. 상단 데이터 요약
# --------------------------------------------------

st.divider()
st.header("📊 데이터 기본 정보")

c1, c2, c3, c4 = st.columns(4)

c1.metric("전체 샘플 수", f"{len(df):,}개")
c2.metric("유전자 수", f"{len(gene_cols):,}개")
c3.metric("암 종류 수", f"{len(classes)}개")
c4.metric("결측 정답 수", f"{df['Class'].isna().sum()}개")

st.caption(
    "Sample은 샘플 식별자이며, 유전자 발현량만 모델의 입력 변수로 사용합니다."
)

# --------------------------------------------------
# 8. 탭 구성
# --------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "① 데이터 살펴보기",
    "② 유전자 탐색",
    "③ 모델링",
    "④ 모델 평가"
])


# ==================================================
# TAB 1. 데이터 살펴보기
# ==================================================

with tab1:

    st.header("① 데이터 살펴보기")

    st.subheader("암 종류별 샘플 개수")

    class_counts = (
        df["Class"]
        .value_counts()
        .reindex(classes, fill_value=0)
        .reset_index()
    )

    class_counts.columns = ["암 종류 코드", "샘플 수"]

    class_counts["암 종류"] = (
        class_counts["암 종류 코드"].map(friendly_class)
    )

    fig_count = px.bar(
        class_counts,
        x="암 종류",
        y="샘플 수",
        color="암 종류",
        text="샘플 수",
        title="암 종류별 데이터 분포"
    )

    st.plotly_chart(fig_count, use_container_width=True)

    st.caption(
        "암 종류마다 샘플 수가 다르므로 데이터 불균형 여부를 확인할 수 있습니다."
    )

    st.divider()

    st.subheader("데이터 미리보기")

    preview = df[["Sample", "Class"] + gene_cols[:8]]

    st.dataframe(
        preview.head(10),
        use_container_width=True
    )

    st.caption(
        "전체 유전자 중 일부 열만 화면에 표시합니다. 실제 분석에는 전체 유전자 데이터를 사용합니다."
    )

    st.divider()

    st.subheader("결측치 확인")

    missing = X.isna().sum().sum()

    st.metric("유전자 발현량 결측값", f"{missing:,}개")

    st.write("암 종류별 샘플 수")

    st.dataframe(
        class_counts[["암 종류", "샘플 수"]],
        use_container_width=True,
        hide_index=True
    )


# ==================================================
# TAB 2. 유전자 탐색
# ==================================================

with tab2:

    st.header("② 유전자 발현 탐색")

    st.subheader("비교할 유전자 선택")

    selected_gene = st.selectbox(
        "유전자 선택",
        gene_cols,
        index=0
    )

    box_df = df[["Class", selected_gene]].dropna().copy()

    box_df["암 종류"] = box_df["Class"].map(
        lambda x: friendly_class(x)
    )

    fig_gene = px.box(
        box_df,
        x="암 종류",
        y=selected_gene,
        color="암 종류",
        points="outliers",
        title=f"{selected_gene} 발현량의 암 종류별 분포"
    )

    fig_gene.update_layout(showlegend=False)

    st.plotly_chart(
        fig_gene,
        use_container_width=True
    )

    st.caption(
        "상자 그림의 중앙값과 분포 범위를 비교해 암 종류에 따른 발현량 차이를 탐색합니다."
    )

    st.divider()

    st.subheader("PCA로 유전자 발현 패턴을 2차원으로 보기")

    st.caption(
        "PCA는 수많은 유전자 정보를 두 개의 축으로 요약하는 탐색용 시각화입니다."
    )

    if len(X) >= 2:

        pca_input = SimpleImputer(
            strategy="median"
        ).fit_transform(X)

        pca_scaled = StandardScaler().fit_transform(
            pca_input
        )

        coords = PCA(
            n_components=2,
            random_state=42
        ).fit_transform(pca_scaled)

        pca_df = pd.DataFrame({
            "주성분 1": coords[:, 0],
            "주성분 2": coords[:, 1],
            "암 종류": y.map(
                lambda x: friendly_class(x)
            )
        })

        fig_pca = px.scatter(
            pca_df,
            x="주성분 1",
            y="주성분 2",
            color="암 종류",
            title="PCA 2차원 산점도",
            opacity=0.75
        )

        st.plotly_chart(
            fig_pca,
            use_container_width=True
        )

        st.caption(
            "점들이 암 종류별로 모이는지 확인합니다. 가까이 모인다고 해서 임상 진단이 가능하다는 뜻은 아닙니다."
        )

    else:
        st.warning("PCA를 실행하기에 샘플 수가 부족합니다.")


# ==================================================
# TAB 3. 모델링
# ==================================================

with tab3:

    st.header("③ 모델링")

    st.write(
        "정답 열인 Class를 예측하도록 두 가지 분류 모델을 학습합니다."
    )

    st.markdown("""
    - **독립 변수(X):** gene_0부터 gene_20530까지의 유전자 발현량
    - **종속 변수(y):** Class에 기록된 암 종류
    - **제외하는 열:** Sample 등 샘플을 구분하는 ID
    """)

    k = st.select_slider(
        "모델이 사용할 유전자 수 선택",
        options=[10, 25, 50, 100, 200, 500],
        value=50
    )

    st.caption(
        "선택한 유전자 수만큼 학습 데이터에서 중요한 특성을 고릅니다. 테스트 데이터는 학습 과정에 사용하지 않습니다."
    )

    if st.button(
        "두 모델 학습하기",
        type="primary",
        use_container_width=True
    ):

        st.session_state["model_k"] = k
        st.session_state["models_ready"] = True
        st.session_state["model_results"] = None

    if (
        st.session_state.get("models_ready")
        and st.session_state.get("model_k") == k
    ):

        if st.session_state["model_results"] is None:

            with st.spinner("두 모델을 학습하고 있습니다..."):

                try:

                    # 모델 학습용 함수
                    def train_models(X, y, k):

                        X_train, X_test, y_train, y_test = train_test_split(
                            X,
                            y,
                            test_size=0.2,
                            random_state=42,
                            stratify=y
                        )

                        k_actual = min(k, X.shape[1])

                        common_steps = [
                            (
                                "imputer",
                                SimpleImputer(strategy="median")
                            ),
                            (
                                "select",
                                SelectKBest(
                                    score_func=f_classif,
                                    k=k_actual
                                )
                            )
                        ]

                        models = {
                            "로지스틱 회귀": Pipeline(
                                common_steps + [
                                    ("scaler", StandardScaler()),
                                    (
                                        "model",
                                        LogisticRegression(
                                            max_iter=3000,
                                            class_weight="balanced"
                                        )
                                    )
                                ]
                            ),

                            "랜덤 포레스트": Pipeline(
                                common_steps + [
                                    (
                                        "model",
                                        RandomForestClassifier(
                                            n_estimators=200,
                                            random_state=42,
                                            class_weight="balanced",
                                            n_jobs=-1
                                        )
                                    )
                                ]
                            )
                        }

                        results = {}

                        for name, model in models.items():

                            model.fit(X_train, y_train)

                            pred = model.predict(X_test)

                            results[name] = {
                                "model": model,
                                "accuracy": accuracy_score(
                                    y_test, pred
                                ),
                                "macro_f1": f1_score(
                                    y_test,
                                    pred,
                                    average="macro"
                                ),
                                "y_test": y_test,
                                "pred": pred,
                                "test_indices": X_test.index
                            }

                        return results

                    results = train_models(X, y, k)

                    st.session_state["model_results"] = results
                    st.session_state["model_classes"] = sorted(y.unique())

                except Exception as e:

                    st.error(f"모델 학습 중 오류가 발생했습니다: {e}")

        results = st.session_state.get("model_results")

        if results:

            metrics = pd.DataFrame([
                {
                    "모델": name,
                    "정확도": result["accuracy"],
                    "Macro F1": result["macro_f1"]
                }
                for name, result in results.items()
            ])

            st.success("모델 학습이 완료되었습니다.")

            st.dataframe(
                metrics.style.format({
                    "정확도": "{:.3f}",
                    "Macro F1": "{:.3f}"
                }),
                use_container_width=True,
                hide_index=True
            )

            st.session_state["model_results"] = results

    else:

        st.info(
            "유전자 수를 선택한 뒤 **두 모델 학습하기** 버튼을 눌러 주세요."
        )


# ==================================================
# TAB 4. 모델 평가
# ==================================================

with tab4:

    st.header("④ 모델 평가")

    results = st.session_state.get("model_results")

    if not results:

        st.info(
            "먼저 ③ 모델링 탭에서 두 모델을 학습해 주세요."
        )

    else:

        model_name = st.selectbox(
            "평가할 모델 선택",
            list(results.keys())
        )

        result = results[model_name]

        m1, m2 = st.columns(2)

        m1.metric(
            "정확도 (Accuracy)",
            f"{result['accuracy']:.3f}"
        )

        m2.metric(
            "Macro F1 점수",
            f"{result['macro_f1']:.3f}"
        )

        st.subheader("혼동 행렬")

        cm = confusion_matrix(
            result["y_test"],
            result["pred"],
            labels=classes
        )

        cm_df = pd.DataFrame(
            cm,
            index=classes,
            columns=classes
        )

        cm_df.index.name = "실제 정답"
        cm_df.columns.name = "모델 예측"

        st.dataframe(
            cm_df,
            use_container_width=True
        )

        fig_cm = px.imshow(
            cm,
            x=classes,
            y=classes,
            text_auto=True,
            labels={
                "x": "예측한 암 종류",
                "y": "실제 암 종류",
                "color": "샘플 수"
            },
            title=f"{model_name} 혼동 행렬",
            aspect="auto"
        )

        st.plotly_chart(
            fig_cm,
            use_container_width=True
        )

        st.subheader("분류 결과 보고서")

        report = classification_report(
            result["y_test"],
            result["pred"],
            labels=classes,
            output_dict=True,
            zero_division=0
        )

        report_df = pd.DataFrame(
            report
        ).transpose().round(3)

        st.dataframe(
            report_df,
            use_container_width=True
        )

        st.subheader("테스트 샘플 예측 체험")

        test_indices = list(
            result["test_indices"]
        )

        row_pos = st.selectbox(
            "테스트 샘플 선택",
            options=range(len(test_indices)),
            format_func=lambda i: str(
                df.loc[test_indices[i], id_col]
            ) if id_col in df.columns
            else f"테스트 샘플 {i+1}"
        )

        sample_index = test_indices[row_pos]

        sample_x = X.loc[[sample_index]]

        predicted = result["model"].predict(
            sample_x
        )[0]

        actual = y.loc[sample_index]

        st.write(
            f"모델 예측: **{friendly_class(predicted)}**"
        )

        st.write(
            f"실제 정답: **{friendly_class(actual)}**"
        )

        if predicted == actual:

            st.success(
                "이 테스트 샘플은 모델이 올바르게 분류했습니다."
            )

        else:

            st.error(
                "이 테스트 샘플은 모델이 다르게 분류했습니다."
            )

        st.caption(
            "테스트 샘플은 학습에 사용하지 않은 데이터 중에서 선택합니다. "
            "이 결과는 공개된 연구용 데이터셋의 분류 실습 결과이며 의료 진단 결과가 아닙니다."
        )


# --------------------------------------------------
# 마지막 안내
# --------------------------------------------------

st.divider()

st.caption(
    "프로젝트 질문: 유전자 발현 데이터를 활용하여 5가지 암 종류를 예측하고 분류할 수 있을까?"
)

st.caption(
    "이 웹앱은 교육 및 연구 실습용이며 실제 환자의 암 진단에 사용할 수 없습니다."
)
