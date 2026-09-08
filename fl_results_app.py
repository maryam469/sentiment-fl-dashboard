import streamlit as st
import numpy as np
import pandas as pd
import io

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
    .stFileUploader label { color: #cdd6f4 !important; }
</style>
""", unsafe_allow_html=True)

N_SAMPLE_WEIGHTS = 10

st.markdown(
    "<h2 style='color:#89b4fa;'>Federated Learning Dashboard</h2>",
    unsafe_allow_html=True
)
st.caption(
    "Upload the client weight files produced by your FL training run "
    "(client_0_all_weights.npz and client_1_all_weights.npz from your "
    "final_model/ folder) to compare Client 1 vs Client 2 raw weights."
)

# ============================================
# Helpers
# ============================================
def sample_weights_from_npz(file_bytes, n=N_SAMPLE_WEIGHTS):
    """Flatten every saved layer array (in save order) and return the
    first n raw values, paired with which layer each came from."""
    data = np.load(io.BytesIO(file_bytes), allow_pickle=True)
    rows = []
    for key in data.files:
        arr = np.asarray(data[key]).flatten()
        for v in arr:
            rows.append((key, float(v)))
            if len(rows) >= n:
                return rows
    return rows if rows else None


def render_client_table(label, rows):
    st.markdown(f"#### {label}")
    if rows is None:
        st.info("No file uploaded yet.")
        return
    df = pd.DataFrame(rows, columns=["Layer", "Weight Value"])
    df.insert(0, "#", range(1, len(df) + 1))
    df["Weight Value"] = df["Weight Value"].map(lambda v: f"{v:.6f}")
    st.dataframe(df, hide_index=True, use_container_width=True)


# ============================================
# 1) Client Weights Comparison
# ============================================
st.markdown(
    f"### Client Weights Comparison (sample of {N_SAMPLE_WEIGHTS} raw weights)"
)

col1, col2 = st.columns(2)

with col1:
    file0 = st.file_uploader(
        "Client 1 — client_0_all_weights.npz", type=["npz"], key="client0_npz"
    )
    rows0 = sample_weights_from_npz(file0.getvalue()) if file0 else None
    render_client_table("Client 1", rows0)

with col2:
    file1 = st.file_uploader(
        "Client 2 — client_1_all_weights.npz", type=["npz"], key="client1_npz"
    )
    rows1 = sample_weights_from_npz(file1.getvalue()) if file1 else None
    render_client_table("Client 2", rows1)

if rows0 and rows1:
    same = [round(v, 6) for _, v in rows0] == [round(v, 6) for _, v in rows1]
    if same:
        st.warning(
            "Client 1 and Client 2's sampled weights are identical — "
            "double check you uploaded two different clients' files."
        )
    else:
        st.success("Client 1 and Client 2 weights differ, as expected for independently trained clients.")

st.divider()

# ============================================
# 2) Clients Weights Update — embedding lifecycle CSV
# ============================================
st.markdown("### Clients Weights Update")
st.caption(
    "Full embedding lifecycle (Before Training → After Client 1 → "
    "After Client 2 → After FedAvg). Upload embedding_full_lifecycle.csv "
    "to preview and download it here."
)

csv_file = st.file_uploader(
    "embedding_full_lifecycle.csv", type=["csv"], key="lifecycle_csv"
)
if csv_file:
    df_csv = pd.read_csv(csv_file)
    st.dataframe(df_csv.head(50), hide_index=True, use_container_width=True)
    st.caption(f"Showing first 50 of {len(df_csv):,} rows.")
    st.download_button(
        "Download full CSV",
        data=csv_file.getvalue(),
        file_name="embedding_full_lifecycle.csv",
        mime="text/csv",
    )
else:
    st.info("No CSV uploaded yet.")
