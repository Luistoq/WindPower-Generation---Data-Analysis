import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


st.header('Correlation Analysis for WindPower Data')

# Function to load data
@st.cache_data  # Corrected decorator here
def load_data(location):
    df = pd.read_csv(f'dataset/{location}.csv')
    df['Time'] = pd.to_datetime(df['Time'])  # Ensure Time is in datetime format
    return df

# Sidebar - Select Location
st.sidebar.header("Select Parameters")
location = st.sidebar.selectbox('Select Location', ('Location1', 'Location2', 'Location3', 'Location4'))

# Load Data
data = load_data(location)

# Correlation Analysis
st.header(f'{location}')
st.write("Correlation matrix showing the relationships between different variables and power output.")

# Compute Correlation Matrix
corr_matrix = data.corr()

# Plotting the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
st.pyplot(plt)

# Interpretation of results
st.subheader("Interpretation of Results")
st.write("""
- Positive values indicate a positive correlation: as one variable increases, so does the other.
- Negative values indicate an inverse relationship: as one variable increases, the other decreases.
- Values close to 0 suggest little to no linear relationship.
- Focus on the 'Power' row/column to see which factors most strongly correlate with power output.
""")

