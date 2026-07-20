import streamlit as st

st.set_page_config(page_title="BrainVision AI", page_icon="🧠", layout="wide")

st.markdown("""
<style>
.main {background:#f4f7fb;}
.block-container {padding-top:1.5rem;}
div[data-testid="stSidebar"] {background:#0b1f3a;color:white;}
.card {padding:1rem;border-radius:12px;background:white;
box-shadow:0 2px 8px rgba(0,0,0,.08);}
</style>
""", unsafe_allow_html=True)

st.title("🧠 BrainVision AI")
st.caption("Explainable Brain MRI Tumor Classification (Educational)")

left,right=st.columns([1,1])
with left:
    uploaded=st.file_uploader("Upload Brain MRI",type=["png","jpg","jpeg"])
    if uploaded:
        st.image(uploaded,use_container_width=True)
with right:
    st.markdown('<div class="card"><h3>Prediction</h3><p>Model not connected yet.</p></div>',unsafe_allow_html=True)
    st.markdown('<div class="card"><h3>Roadmap</h3><ul><li>EfficientNetB0</li><li>Grad-CAM</li><li>PDF Report</li></ul></div>',unsafe_allow_html=True)
