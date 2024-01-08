import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load data
@st.cache
def load_data(location):
    return pd.read_csv(f'path_to_your_dataset/{location}.csv')

# Sidebar for user inputs
location = st.sidebar.selectbox('Select Location', ('Location1', 'Location2', 'Location3', 'Location4'))
data = load_data(location)

# Main content
st.title(f'Wind Power Generation Analysis: {location}')

# Plotting Power Production Over Time
st.subheader('Power Production Over Time')
fig, ax = plt.subplots()
ax.plot(pd.to_datetime(data['Time']), data['Power'])
st.pyplot(fig)
