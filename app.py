import streamlit as st

st.title("Hello Chai App")
st.subheader("Brewed with streamlit")
st.text("welcome to your first interactive app")
st.write("Choose your favorite variety of chai")

chai=st.selectbox("Your fav chai:",["masala","lemon","kesar"])
st.write("You chose",chai)
st.success("Your chai has been brewed")
if st.button("Make chai"):
  st.success("Your chai is being brewed")
add_masala=st.checkbox("Add masala")
if add_masala:
  st.write("Masala added")
tea_type=st.radio("Pick your chai base:"",["Milk","Water",])
st.write("Your selected base:",tea_type)
favour=st.selectbox(["adrak","tulsi"])

sugar=st.slider("Sugar level(spoon)",0,5,2)
st.write("slected sugar level",sugar)
cups=st.number_input("How many cups?",min_value=1,max_value=10)
name=st.text_input("What is your name?")
if name:
     st.write("Welcome!",name)
