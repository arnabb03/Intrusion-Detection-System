import streamlit as st
import pandas as pd
import joblib
import numpy as np
import base64

def set_bg(img_path):
    with open(img_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("data:image/jpeg;base64,{encoded_string}") top center / cover
                        no-repeat fixed;
        }}

        .block-container {{
            background: rgba(0, 0, 0, 0.65);
            padding: 2rem;
            border-radius: 12px;
        }}
        h1,h2,h3,h4,h5,h6,p,label, .stMarkdown {{ color:#fff; }}

        header {{
            background-color: rgba(0, 0, 0, 0) !important;
        }}

        header * {{
            color: white !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

image_path = r"C:\Users\baner\fyp1\images\ai-cybersecurity-virus-protection-machine-learning.jpg"
set_bg(image_path)

model_path = r"C:\Users\baner\ids fyp\models\StackingEnsemble.joblib"
scaler_path = r"C:\Users\baner\ids fyp\models\scaler.joblib"

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

label_mapping = {
    0: 'BENIGN', 1: 'Bot', 2: 'DDoS', 3: 'DoS GoldenEye', 4: 'DoS Hulk',
    5: 'DoS Slowhttptest', 6: 'DoS slowloris', 7: 'FTP-Patator', 8: 'Heartbleed',
    9: 'Infiltration', 10: 'PortScan', 11: 'SSH-Patator',
    12: 'Web Attack : Brute Force', 13: 'Web Attack : Sql Injection', 14: 'Web Attack : XSS'
}

st.title("Intrusion Detection System (IDS)")

feature_names = [
    'Destination Port',
    'Init_Win_bytes_backward',
    'Bwd Packets/s',
    'Subflow Bwd Bytes',
    'Bwd Packet Length Mean',
    'Init_Win_bytes_forward',
    'Avg Bwd Segment Size',
    'Total Length of Bwd Packets',
    'Fwd Packet Length Max',
    'Flow Packets/s',
    'min_seg_size_forward',
    'Flow IAT Max',
    'Bwd Packet Length Max',
    'Bwd Header Length'
]

tab1, tab2 = st.tabs([" Single Prediction ", " Batch Prediction "])

with tab1:
    st.header("Enter Values for Prediction")

    user_input = []
    for i in range(0, len(feature_names), 3):
        cols = st.columns(3)
        for j in range(3):
            if i + j < len(feature_names):
                feature = feature_names[i + j]
                with cols[j]:
                    value = st.number_input(f"{feature}", step=0.0000001, format="%.7f")
                    user_input.append(value)

    if st.button("Predict"):
        input_array = np.array(user_input).reshape(1, -1)
        normalized_input = scaler.transform(input_array)
        prediction = model.predict(normalized_input)[0]
        st.success(f"Prediction: **{label_mapping[int(prediction)]}**")

with tab2:
    st.header("Upload CSV for Batch Prediction")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            if not all(col in df.columns for col in feature_names):
                st.error("Uploaded CSV must contain the exact 14 required feature columns.")
            else:
                df = df[feature_names]
                normalized_df = scaler.transform(df)
                predictions = model.predict(normalized_df)
                df['Prediction'] = [label_mapping[int(p)] for p in predictions]

                st.success("Predictions completed successfully!")
                st.dataframe(df)

                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Predictions", data=csv, file_name='predictions.csv', mime='text/csv')

        except Exception as e:
            st.error(f"Error: {e}")
