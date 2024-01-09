import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
from tensorflow.keras.callbacks import Callback
from tqdm import tqdm

st.header('Power Output Prediction Model interface')

st.write('This tool uses a deep learning LSTM model to predict power output based on your data.')


# Custom callback for displaying a tqdm progress bar
class TqdmProgressCallback(Callback):
    def __init__(self, total_epochs):
        super().__init__()
        self.total_epochs = total_epochs
        self.progress_bar = None

    def on_train_begin(self, logs=None):
        self.progress_bar = tqdm(total=self.total_epochs, unit='epoch')

    def on_epoch_end(self, epoch, logs=None):
        self.progress_bar.update(1)

    def on_train_end(self, logs=None):
        self.progress_bar.close()


# Function to load data
@st.cache_data
def load_data(location):
    df = pd.read_csv(f'dataset/{location}.csv')
    df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
    return df.drop(columns=['Time'], errors='ignore')

@st.cache_data
def load_val_data(location):
    df_val = pd.read_csv(f'dataset/{location}.csv')
    df_val['Time'] = pd.to_datetime(df_val['Time'], format='%d/%m/%Y %H:%M', errors='coerce')
    df_val = df_val.sort_values('Time')
    return df_val

# Sidebar - User Inputs
st.sidebar.header("Deep Learning Model Settings")
location = st.sidebar.selectbox('Select Location', ('Location1', 'Location2', 'Location3', 'Location4'))
data = load_data(location)

# Exclude 'Power' from features, but keep it for target
selected_variables = [col for col in data.columns if col != 'Power']


st.info('Traning model, this could take up to 1 mintue...')

st.write("""
### Model Explanation

The Long Short-Term Memory (LSTM) model is a type of recurrent neural network (RNN) that is well-suited to predict time series data due to its ability to capture temporal dependencies and sequences in data. This is particularly useful in predicting power output, which is often dependent on previous readings.

#### Main Hyperparameters:

- **Number of LSTM Layers**: Two LSTM layers have been used to provide a deeper understanding of the time series patterns.
  
- **Units in LSTM Layers**: Each LSTM layer contains 50 units. This number represents the dimensionality of the output space and can be thought of as the 'memory' of the network.
  
- **Batch Size**: The model is trained in batches of 32 records, which balances the speed of computation with the model's ability to generalize.
  
- **Epochs**: We train the model for 30 epochs, meaning the entire dataset is passed forward and backward through the neural network 30 times.
  
- **Loss Function**: Mean Squared Error (MSE) is used as the loss function, which measures the average squared difference between the estimated values and the actual value.
  
- **Optimizer**: Adam optimizer is used for its efficiency in both computation and memory requirement.

This configuration of hyperparameters has been chosen based on initial experimentation and provides a good starting point. However, for improved performance, hyperparameter tuning may be necessary based on the specific characteristics of the dataset.

### Model Evaluation

The plot 'Model Loss During Training' shows the training and validation loss over each epoch. A decreasing trend in these plots indicates that the model is learning and improving its prediction accuracy over time.

The 'Actual vs Predicted Power Output' plot provides a visual comparison between the actual values and the model's predictions. To enhance readability, we plot a subset of points by displaying one point and skipping the next 24. This approach helps us to visualize the trend without overcrowding the plot.
""")

# Prepare data for the model
X = data[selected_variables]
y = data['Power']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Reshape input for LSTM layer
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Define the model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(50),
    Dense(1)
])

#Compile model
model.compile(optimizer='adam', loss='mean_squared_error')

# Define the total number of epochs
total_epochs = 15

# Create an instance of the custom callback
progress_bar_callback = TqdmProgressCallback(total_epochs)

# Train the model
history = model.fit(X_train, 
                    y_train, 
                    epochs=total_epochs, 
                    batch_size=32, 
                    validation_data=(X_test, y_test), 
                    verbose=0, 
                    callbacks=[progress_bar_callback])

# Evaluate the model
loss = model.evaluate(X_test, y_test, verbose=0)

fig = go.Figure()
fig.add_trace(go.Scatter(y=history.history['loss'], mode='lines', name='Train Loss'))
fig.add_trace(go.Scatter(y=history.history['val_loss'], mode='lines', name='Validation Loss'))
fig.update_layout(title='Model Loss During Training', xaxis_title='Epochs', yaxis_title='Loss')
st.plotly_chart(fig)
st.subheader(f"Test Loss: {loss}")
st.write("""
The "Test Loss" represents the model's performance on the unseen test dataset. It's a numeric value that quantifies the difference between the actual power output values and the values predicted by the model. Specifically, we are using Mean Squared Error (MSE) as the loss function, which calculates the average of the squares of the errors—that is, the average squared difference between the estimated values (predictions) and the actual value (ground truth).
Here's what you need to know about the Test Loss:
- **Lower values are better**: A lower MSE value indicates that the model's predictions are closer to the actual data, which means better performance.
- **Zero is the ideal**: A test loss of zero means the model predicts the output perfectly, which in practice is highly unlikely, especially with complex data like power output over time.
- **Context is important**: The value of the test loss should be considered in the context of the data. For example, a test loss of 10 may be very low if the actual values range in the thousands, but quite high if the values are typically between 0 and 1.
The Test Loss provides a straightforward measure of how well the model is expected to perform in practical applications when making predictions on new, unseen data.
This value is an average across all predictions in the test set and provides a benchmark for evaluating improvements or regressions in the model's performance over time.
""")

# Plot some predictions
predictions = model.predict(X_test)
fig = go.Figure()
fig.add_trace(go.Scatter(y=y_test, mode='markers', name='Actual'))
fig.add_trace(go.Scatter(y=predictions.flatten(), mode='markers', name='Predicted'))
fig.update_layout(title='Actual vs Predicted Power Output', xaxis_title='Sample', yaxis_title='Power Output')
st.plotly_chart(fig)

# If y_test is a numpy array or a list:
plot_y_test = y_test[::25]
plot_predictions = predictions.flatten()[::25]

# If y_test is a pandas Series or DataFrame column, you should use .iloc for slicing:
plot_y_test = y_test.iloc[::25]
plot_predictions = predictions.flatten()[::25]

#Upsample Prediction plots for ease visualisation
fig = go.Figure()
fig.add_trace(go.Scatter(y=plot_y_test, mode='markers', name='Actual'))
fig.add_trace(go.Scatter(y=plot_predictions, mode='markers', name='Predicted'))
fig.update_layout(title='Actual vs Predicted Power Output Sampled Down', xaxis_title='Sample', yaxis_title='Power Output')
st.plotly_chart(fig)

st.write("""
         ### Thank you for using the Power Output Prediction Model.
        You can select different locations and adjust the hyperparameters from the sidebar to refine the model's predictions. The current settings aim to provide a balance between accurate predictions and computational efficiency.
         """)