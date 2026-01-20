import streamlit as st

st.title("Hello Chai App")
st.subheader("Brewed with streamlit")
st.text("welcome to your first interactive app")
st.write("Choose your favorite variety of chai")

chai=st.selectbox("Your fav chai:",["masala","lemon","kesar"])
