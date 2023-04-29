import streamlit as st

# Create a selectbox with some options
option = st.selectbox('Select an option', ['Option 1', 'Option 2', 'Option 3'])

# Print the selected option
st.write('You selected:', option)