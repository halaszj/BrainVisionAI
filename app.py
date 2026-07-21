import streamlit as st
from brainvision.version import VERSION, BUILD
st.set_page_config(page_title="BrainVision AI",layout="wide")
st.title("🧠 BrainVision AI")
st.caption(f"Version {VERSION} • Build {BUILD}")
st.sidebar.title("Navigation")
page=st.sidebar.radio("Go to",["Home","Prediction","About"])
if page=="Home":
    st.write("Repository scaffold for Build 5.")
elif page=="Prediction":
    st.file_uploader("Upload an MRI image",type=["png","jpg","jpeg"])
    st.info("Inference will be connected in Delivery 2.")
else:
    st.write("Educational project. Not for medical diagnosis.")
