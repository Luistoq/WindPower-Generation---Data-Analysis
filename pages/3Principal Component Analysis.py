import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import numpy as np

st.header('Principal Component Analysis (PCA) for Wind Power Generation Data')

# Function to load data
@st.cache_data 

def load_data(location):
    df = pd.read_csv(f'dataset/{location}.csv')
    # Explicitly convert 'Time' to datetime format if it's not already
    df['Time'] = pd.to_datetime(df['Time'], errors='coerce')  # 'coerce' will set invalid parsing as NaT
    return df.drop(columns=['Time'], errors='ignore')  # Drop the 'Time' column here


# Sidebar - User Inputs
st.sidebar.header("PCA Settings")
location = st.sidebar.selectbox('Select Location', ('Location1', 'Location2', 'Location3', 'Location4'))
data = load_data(location)

# Exclude 'Power' from PCA, but keep it for correlation analysis
selected_variables = [col for col in data.columns if col != 'Power']

# Preprocessing
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[selected_variables])

# PCA
n_components = min(len(selected_variables), 10)
pca = PCA(n_components=n_components)
pca_result = pca.fit_transform(scaled_data)

# After PCA fitting
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
loadings_df = pd.DataFrame(loadings, columns=[f'PC{i+1}' for i in range(n_components)], index=selected_variables)

# Find the variable with the highest absolute loading for each principal component
most_influential_vars = loadings_df.abs().idxmax()

# Explained Variance Ratio Plot
fig = px.bar(x=[f'PC{i+1}' for i in range(n_components)], y=pca.explained_variance_ratio_)
fig.update_layout(title='Explained Variance Ratio by Principal Component',
                  xaxis_title='Principal Components',
                  yaxis_title='Explained Variance Ratio')
st.plotly_chart(fig)

# Prepare PCA results for correlation analysis with 'Power'
pca_df = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(n_components)])
combined_df = pd.concat([pca_df, data['Power'].reset_index(drop=True)], axis=1)
corr_matrix = combined_df.corr()

# Display correlation matrix with 'Power'
st.subheader("Correlation Matrix with Power")
st.write("Correlation of each Principal Component with 'Power' output:")
st.dataframe(corr_matrix.loc['Power'][:-1].to_frame())  # Corrected to show a DataFrame of correlations with 'Power'

# Display the most influential variables for each PC
st.subheader("Most Influential Variables for Each Principal Component")
# Convert the series to a DataFrame
most_influential_vars_df = most_influential_vars.reset_index()

# Rename the columns
most_influential_vars_df.columns = ["Principal Components", "Variables"]

# Now display the table with column titles
st.dataframe(most_influential_vars_df, hide_index=True)

st.subheader("Interpretation of Results")
st.write("""
- The correlation matrix provides insights into how each principal component relates to the 'Power' output.
- A higher absolute value indicates a stronger relationship between the component and power generation.
- Principal components are new variables created by PCA that are linear combinations of the original variables. They are constructed in a way that the first principal component (PC1) explains the most variance in the dataset, followed by the second (PC2), and so on.
- The variables with the highest loadings (both positive and negative) on the first few principal components are typically the most significant in explaining the variation in the dataset. These loadings can be interpreted as the weight or 'influence' of each variable on the principal component.
- The most influential variable for each principal component is the one that has the largest absolute loading. It is the variable that most strongly drives that component.
- By examining the most influential variables and their corresponding loadings on the principal components that show strong correlation with 'Power', we can identify which features are most predictive or indicative of power generation.
- For instance, if temperature has a high loading on PC1, and PC1 has a strong positive correlation with 'Power', this suggests that temperature is an important predictor of power output.
- It is important to consider both the size of the loadings and the correlation of the principal components with 'Power' to draw conclusions about the factors that contribute to power generation.
""")

