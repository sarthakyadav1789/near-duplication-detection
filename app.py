import streamlit as st
from streamlit_lottie import st_lottie
import json

@st.cache_data
def load_lottie(path):
    with open(path, "r") as f:
        return json.load(f)
hero_lottie = load_lottie('search.json')





st.set_page_config(
    page_title="Near-Duplicate Image Detection",
    page_icon="🖼️",

    layout="wide"
)
st.markdown("""
<style>
.stApp {
    background: linear-gradient(
        135deg,
        #f5f7fa 0%,
        #eef2ff 50%,
        #f8fafc 100%
    );
}
</style>
""", unsafe_allow_html=True)



st.markdown("""
<style>
.typewriter {
    font-size: 42px;
    font-weight: 800;
    color: #6b4f4f;   /* light + dark safe */
    overflow: hidden;
    white-space: nowrap;
    border-right: 3px solid rgba(79,70,229,0.8);
    width: 0;
    animation:
        typing 3s steps(35, end) forwards;
        border-right: none;


}

/* typing animation */
@keyframes typing {
    from { width: 0 }
    to { width: 100% }
}

/* cursor blink */
@keyframes blink {
    50% { border-color: transparent }
}
</style>
""", unsafe_allow_html=True)


col1, col2 = st.columns([3, 1])

with col1:
    st.markdown(
        "<div class='typewriter'>🖼️ Near-Duplicate Image Detection</div>",
        unsafe_allow_html=True
    )

with col2:
    st.markdown("<div class='lottie-wrapper'>", unsafe_allow_html=True)

    st_lottie(
      hero_lottie,
      height=80,
      speed=0.9,
      loop=True)



st.markdown("</div>", unsafe_allow_html=True)
st.markdown("""
<style>
/* Make Lottie background transparent */
div[data-testid="stLottie"] {
    background: transparent !important;
}
</style>
""", unsafe_allow_html=True)





st.write("")
with st.sidebar:
    st.header("WELCOME!!")
    st.file_uploader("Upload query image", type=["jpg","jpeg","png"])
    st.slider("Top-K results", 1, 5, 5)
    st.button("🔍 Run Search")
