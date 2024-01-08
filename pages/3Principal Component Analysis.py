import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
import numpy as np


st.header('Principal Component Analysis (PCA) for Wind Power Generation Data')

# Function to load data
@st.cache_data  # Corrected decorator here
def load_data(location):
    df = pd.read_csv(f'dataset/{location}.csv')
    df['Time'] = pd.to_datetime(df['Time'])  # Ensure Time is in datetime format
    return df

# Sidebar - User Inputs
st.sidebar.header("PCA Settings")
location = st.sidebar.selectbox('Select Location', ('Location1', 'Location2', 'Location3', 'Location4'))
data = load_data(location)

# Exclude 'Time' for PCA and allow user to select other variables
exclude_power = st.sidebar.checkbox('Exclude Power from PCA', True)
selected_variables = [col for col in data.columns if col != 'Time' and (col != 'Power' or not exclude_power)]

# Preprocessing
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[selected_variables])

# PCA
n_components = min(len(selected_variables), 10)
pca = PCA(n_components=n_components)
pca_result = pca.fit_transform(scaled_data)

# Explained Variance Ratio Plot
fig = px.bar(x=[f'PC{i+1}' for i in range(n_components)], y=pca.explained_variance_ratio_)
fig.update_layout(title='Explained Variance Ratio by Principal Component',
                  xaxis_title='Principal Components',
                  yaxis_title='Explained Variance Ratio')
st.plotly_chart(fig)

# Combine PCA results with Power for correlation analysis
if exclude_power:
    pca_df = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(n_components)])
    combined_df = pd.concat([pca_df, data['Power'].reset_index(drop=True)], axis=1)
    corr_matrix = combined_df.corr()

    # Display correlation matrix
    st.subheader("Correlation Matrix with Power")
    st.write("Correlation of each Principal Component with 'Power' output:")
    st.dataframe(corr_matrix.iloc[:-1, -1])  # Corrected slicing to show all PCs correlation with Power

st.subheader("Interpretation of Results")
st.write("""
- The correlation matrix provides insights into how each principal component relates to the 'Power' output.
- A higher absolute value indicates a stronger relationship between the component and power generation.
""")