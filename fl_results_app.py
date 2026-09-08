import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Federated Learning Results",
    page_icon="🔗",
    layout="wide"
)

st.markdown("""
<style>
    .stApp { background-color: #1e1e2e; }
    section[data-testid="stSidebar"] { background-color: #181825; }
    h1, h2, h3, h4, p, label, span { color: #cdd6f4 !important; }
</style>
""", unsafe_allow_html=True)

N_SAMPLE_WEIGHTS = 10
CSV_GZ_PATH = "embedding_full_lifecycle.csv.gz"

st.markdown(
    "<h2 style='color:#89b4fa;'>Federated Learning Dashboard</h2>",
    unsafe_allow_html=True
)

# ============================================
# Load data once (cached across reruns/sessions)
# ============================================
@st.cache_data(show_spinner="Loading embedding lifecycle data...")
def load_lifecycle_csv():
    return pd.read_csv(CSV_GZ_PATH, compression="gzip")

try:
    df = load_lifecycle_csv()
except FileNotFoundError:
    st.error(
        f"'{CSV_GZ_PATH}' not found in the app's root folder. "
        "Make sure this file is uploaded to the same GitHub repo as this script."
    )
    st.stop()

# ============================================
# 1) Client Weights Comparison — auto-extracted from the CSV
# ============================================
st.markdown(
    f"### Client Weights Comparison (sample of {N_SAMPLE_WEIGHTS} raw weights)"
)

word_options = df["Word"].astype(str).tolist()
selected_word = st.selectbox(
    "Word / token to inspect",
    word_options,
    index=0,
    help="Pick which embedding row to compare Client 1 vs Client 2 on."
)

row = df[df["Word"].astype(str) == selected_word].iloc[0]

c1_vals = [row[f"AfterClient1_dim_{i}"] for i in range(N_SAMPLE_WEIGHTS)]
c2_vals = [row[f"AfterClient2_dim_{i}"] for i in range(N_SAMPLE_WEIGHTS)]

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Client 1")
    t1 = pd.DataFrame({
        "#": range(1, N_SAMPLE_WEIGHTS + 1),
        "Layer": [f"embedding_dim_{i}" for i in range(N_SAMPLE_WEIGHTS)],
        "Weight Value": [f"{v:.6f}" for v in c1_vals],
    })
    st.dataframe(t1, hide_index=True, use_container_width=True)

with col2:
    st.markdown("#### Client 2")
    t2 = pd.DataFrame({
        "#": range(1, N_SAMPLE_WEIGHTS + 1),
        "Layer": [f"embedding_dim_{i}" for i in range(N_SAMPLE_WEIGHTS)],
        "Weight Value": [f"{v:.6f}" for v in c2_vals],
    })
    st.dataframe(t2, hide_index=True, use_container_width=True)

if [round(v, 6) for v in c1_vals] == [round(v, 6) for v in c2_vals]:
    st.warning("Client 1 and Client 2 weights are identical for this word.")
else:
    st.success("Client 1 and Client 2 weights differ, as expected for independently trained clients.")

st.divider()

# ============================================
# 2) Clients Weights Update — full lifecycle table
# ============================================
st.markdown("### Clients Weights Update")
st.caption(
    "Full embedding lifecycle (Before Training → After Client 1 → "
    "After Client 2 → After FedAvg) for the selected word."
)

lifecycle_cols = (
    ["Word"]
    + [f"BeforeTraining_dim_{i}" for i in range(N_SAMPLE_WEIGHTS)]
    + [f"AfterClient1_dim_{i}" for i in range(N_SAMPLE_WEIGHTS)]
    + [f"AfterClient2_dim_{i}" for i in range(N_SAMPLE_WEIGHTS)]
    + [f"AfterFedAvg_Global_dim_{i}" for i in range(N_SAMPLE_WEIGHTS)]
)
preview = df[df["Word"].astype(str) == selected_word][lifecycle_cols]
st.dataframe(preview, hide_index=True, use_container_width=True)

st.caption(f"Full dataset: {len(df):,} words × {df.shape[1]} columns.")

csv_bytes = df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download full embedding_full_lifecycle.csv",
    data=csv_bytes,
    file_name="embedding_full_lifecycle.csv",
    mime="text/csv",
)
