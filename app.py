import tensorflow as tf
import streamlit as st
saved_model = tf.keras.models.load_model("saved_model/1")
label = [
    "Politics",
    "Sport",
    "Technology",
    "Entertainment"
    "Business"
]

st.header("STUDY JAMS WEEK 7 ML ")

if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

text_input = st.text_input(
    "Enter some text 👇", 
    label_visibility=st.session_state.visibility,
    disabled=st.session_state.disabled,
    placeholder="document text"
)

predictions = saved_model.predict([text_input])
max = tf.math.argmax(predictions[0])
result = tf.keras.backend.eval(max)

if text_input:
    st.write("Your Text Document")
    st.write(text_input)
    st.write("The document talk about {}".format(label[result]))