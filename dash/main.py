# Import relevant libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import joblib

# Load dataset
data = pd.read_csv("/app/data/winequality.csv")
# Load the model
model = joblib.load("/app/data/model_dtc.joblib")
# Check for missing values
data.isna().sum()
# Remove duplicate data
data.drop_duplicates(keep='first', inplace=True)  # Add inplace=True to modify the DataFrame in place
# Calculate the correlation matrix
corr_matrix = data.corr()
# Label quality into Good (1) and Bad (0)
data['quality'] = data['quality'].apply(lambda x: 1 if x >= 6.0 else 0)
# Drop the target variable
X = data.drop('quality', axis=1)
# Set the target variable as the label
y = data['quality']

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


# Create the Dash app
app = dash.Dash(__name__)
server = app.server

# Define the layout of the dashboard
# Define the layout of the dashboard
app.layout = html.Div(
    children=[
        html.H1('Calidad del Vino'),
        html.Div([
            html.H3('Analisis exploratorio'),
            html.Label('Variable 1 (Eje - X)'),
            dcc.Dropdown(
                id='x_feature',
                options=[{'label': col, 'value': col} for col in data.columns],
                value=data.columns[0]
            )
        ], style={'width': '30%', 'display': 'inline-block'}),

        html.Div([
            html.Label('variable 2 ( Eje - Y)'),
            dcc.Dropdown(
                id='y_feature',
                options=[{'label': col, 'value': col} for col in data.columns],
                value=data.columns[1]
            )
        ], style={'width': '30%', 'display': 'inline-block'}),

        dcc.Graph(id='correlation_plot'),

        # Wine quality prediction based on input feature values
        html.H3("Wine Quality Prediction"),
        html.H2("Califica del 1 al 10 tu vino en cuestión de:"),
        html.Div([
            html.Label("Densidad"),
            dcc.Input(id='density', type='number', required=True),
            html.Label("Alcohol"),
            dcc.Input(id='alcohol', type='number', required=True),
            html.Label("Citrico"),
            dcc.Input(id='citric_acid', type='number', required=True),
            html.Label("Azucar"),
            dcc.Input(id='residual_sugar', type='number', required=True),
            html.Label("Acidez"),
            dcc.Input(id='ph', type='number', required=True),
            html.Label('¿Es Blanco? 1=Si, 0=No'),
            dcc.Input(id='type', type='number', min=0, max=1, step=1, value=1, placeholder='Enter value'),
            html.Br()
        ]),

        html.Div([
            html.Button('Predict', id='predict-button', n_clicks=0),
        ]),

        html.Div(id='calidad-output')  # New div for calidad output
    ]
)

# Define the callback to update the correlation plot
@app.callback(
    Output('correlation_plot', 'figure'),
    [Input('x_feature', 'value'),
     Input('y_feature', 'value')]
)
def update_correlation_plot(x_feature, y_feature):
    fig = px.scatter(data, x=x_feature, y=y_feature)
    fig.update_layout(title=f"Correlation between {x_feature} and {y_feature}")
    return fig

# Define the callback function to predict wine quality
@app.callback(
    Output('calidad-output', 'children'),
    [Input('predict-button', 'n_clicks')],
    [State('density', 'value'),
     State('alcohol', 'value'),
     State('citric_acid', 'value'),
     State('residual_sugar', 'value'),
     State('ph', 'value'),
     State('type', 'value')]
)
def update_calidad_output(n_clicks, density, alcohol, citric_acid, residual_sugar, ph, type):
    # Perform predictions based on the input values
    prediction = model.predict([[density, alcohol, citric_acid, residual_sugar, ph, type]])
    calidad = "Es un buen vino" if prediction == 1 else "Podría ser de mejor calidad"
    
    # Return the calidad output
    return html.Div([
        html.Label("Calidad Predecida:"),
        html.Span(calidad)
    ])


if __name__ == '__main__':
    app.run_server(debug=True)