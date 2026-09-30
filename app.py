import warnings
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

st.set_page_config(page_title="Iris Classifier", page_icon="🌸", layout="centered")

MODEL_PATH = Path(__file__).parent / "iris_model.pkl"

SPECIES = {
    0: ("Iris setosa", "ไอริส เซโทซา", "🌸"),
    1: ("Iris versicolor", "ไอริส เวอร์ซิคัลเลอร์", "🌺"),
    2: ("Iris virginica", "ไอริส เวอร์จินิกา", "🌷"),
}

# (min, max, default) ตามช่วงข้อมูลจริงของชุดข้อมูล Iris
RANGES = {
    "sepal length (cm)": (4.0, 8.5, 5.8),
    "sepal width (cm)": (1.5, 5.0, 3.0),
    "petal length (cm)": (0.5, 7.5, 4.3),
    "petal width (cm)": (0.0, 3.0, 1.3),
}
LABELS_TH = {
    "sepal length (cm)": "ความยาวกลีบเลี้ยง (Sepal length)",
    "sepal width (cm)": "ความกว้างกลีบเลี้ยง (Sepal width)",
    "petal length (cm)": "ความยาวกลีบดอก (Petal length)",
    "petal width (cm)": "ความกว้างกลีบดอก (Petal width)",
}
PRESETS = {
    "กำหนดเอง": None,
    "ตัวอย่าง Setosa": (5.1, 3.5, 1.4, 0.2),
    "ตัวอย่าง Versicolor": (6.0, 2.9, 4.5, 1.5),
    "ตัวอย่าง Virginica": (6.5, 3.0, 5.8, 2.2),
}


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()
features = list(model.feature_names_in_)

st.title("🌸 Iris Species Classifier")
st.caption("ทำนายสายพันธุ์ดอกไอริสจากขนาดกลีบ ด้วยโมเดล Random Forest")

tab_single, tab_batch, tab_about = st.tabs(["ทำนายทีละดอก", "ทำนายจากไฟล์ CSV", "เกี่ยวกับโมเดล"])

# ---------------------------------------------------------------- single
with tab_single:
    for f in features:  # ค่าเริ่มต้นของสไลเดอร์
        st.session_state.setdefault(f, float(RANGES.get(f, (0, 10, 5.0))[2]))

    preset = st.selectbox("เลือกตัวอย่าง", list(PRESETS.keys()))
    if preset != "กำหนดเอง" and st.session_state.get("_preset") != preset:
        for f, v in zip(features, PRESETS[preset]):
            st.session_state[f] = float(v)
    st.session_state["_preset"] = preset

    values = {}
    col1, col2 = st.columns(2)
    for i, f in enumerate(features):
        lo, hi, _ = RANGES.get(f, (0.0, 10.0, 5.0))
        with (col1 if i % 2 == 0 else col2):
            values[f] = st.slider(
                LABELS_TH.get(f, f), lo, hi, step=0.1, key=f,
            )

    X = pd.DataFrame([values], columns=features)
    proba = model.predict_proba(X)[0]
    pred = int(model.classes_[proba.argmax()])
    en, th, emoji = SPECIES[pred]

    st.divider()
    st.subheader(f"{emoji} {en}")
    st.write(f"{th} — ความมั่นใจ **{proba.max():.1%}**")

    prob_df = pd.DataFrame(
        {"ความน่าจะเป็น": proba},
        index=[SPECIES[int(c)][0] for c in model.classes_],
    )
    st.bar_chart(prob_df, horizontal=True)

# ----------------------------------------------------------------- batch
with tab_batch:
    st.write("อัปโหลด CSV ที่มีคอลัมน์ดังนี้ (ชื่อต้องตรงกัน):")
    st.code(", ".join(features))
    template = pd.DataFrame([PRESETS["ตัวอย่าง Setosa"], PRESETS["ตัวอย่าง Versicolor"],
                             PRESETS["ตัวอย่าง Virginica"]], columns=features)
    st.download_button("ดาวน์โหลดไฟล์ตัวอย่าง", template.to_csv(index=False),
                       "iris_template.csv", "text/csv")

    up = st.file_uploader("เลือกไฟล์ CSV", type="csv")
    if up is not None:
        df = pd.read_csv(up)
        missing = [c for c in features if c not in df.columns]
        if missing:
            st.error(f"ไม่พบคอลัมน์: {', '.join(missing)}")
        else:
            data = df[features]
            if data.isna().any().any():
                st.error("ข้อมูลมีค่าว่าง กรุณาตรวจสอบไฟล์")
            else:
                p = model.predict_proba(data)
                out = df.copy()
                out["species"] = [SPECIES[int(model.classes_[i])][0] for i in p.argmax(axis=1)]
                out["confidence"] = p.max(axis=1).round(4)
                st.dataframe(out, use_container_width=True)
                st.download_button("ดาวน์โหลดผลลัพธ์", out.to_csv(index=False),
                                   "iris_predictions.csv", "text/csv")

# ----------------------------------------------------------------- about
with tab_about:
    st.write(
        f"โมเดล: **{type(model).__name__}** ({model.n_estimators} ต้นไม้) "
        f"| คลาส: {', '.join(SPECIES[int(c)][0] for c in model.classes_)}"
    )
    st.write("ความสำคัญของแต่ละฟีเจอร์ (Feature importance)")
    imp = pd.Series(model.feature_importances_, index=features, name="importance")
    st.bar_chart(imp.sort_values())
