# Importing required packages
import pandas as pd
import plotly.express as px
import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output

# Opening Data
file_name = 'airline_data.csv'
airline_data = pd.read_csv(file_name,
                           encoding="ISO-8859-1",
                           dtype={'Div1Airport': str, 'Div1TailNum': str,
                                  'Div2Airport': str, 'Div2TailNum': str})

# Creating dash object
app = dash.Dash()

# Layout
app.layout = html.Div(
    children=[
        html.H1('Airline Dashboard', style={
            'textAlign': 'center', 'color': '#503D36',
            'font-size': 40}),
        # Now we set up the input whose modification will become the
        # trigger for updates
        html.Div(["Input: ", dcc.Input(id='input-yr',
                                       value='2010',
                                       type='number',
                                       style={
                                           'height': '50px',
                                           'font-size': 35})],
                 style={'font-size': 40}),
        html.Div(["State Abbreviation: ",
                  dcc.Input(id='input-ab',
                            value='AL',
                            type='text',
                            style={
                                'height': '50px',
                                'font-size': 35
                            })],
                 style={'font-size': 40}),
        html.Br(),
        html.Br(),
        html.Div(dcc.Graph(id='bar-plot'))  # We are setting an
        # interactive graph with
        # id = bar-plot
    ]
)


# We are setting our output as a graph that will be placed in the
# interactive graph object we set up before whose id = bar-plot
@app.callback(
    Output(component_id='bar-plot', component_property='figure'),
    # We set up that the inputs that trigger change when updated are
    # the html inputs whose id are input-year when and input-ab its
    # property value is modified we will call this function, execute it
    # and replace the interactive graph bar-plot with the one returned ç
    # in this function
    [Input(component_id='input-yr', component_property='value'),
     Input(component_id='input-ab', component_property='value')]
)
def get_graph(entered_year, entered_state):
    # Select data
    df = airline_data[(airline_data['Year'] == int(entered_year)) &
                      (airline_data['OriginState'] == entered_state)]
    # Top 10 airline carrier in terms of number of flights
    g1 = df.groupby(['Reporting_Airline'])['Flights'].\
        sum().nlargest(10).reset_index()
    # Plot the graph
    fig1 = px.bar(g1, x='Reporting_Airline', y='Flights',
                  title='Top 10 airline carrier in  year ' +
                        str(entered_year) + ' in terms of number of flights')
    fig1.update_layout()
    # Return that graph that will be placed in interactive graph
    # with id bar-plot set up before
    return fig1


if __name__ == "__main__":
    app.run_server(port=8002, host='127.0.0.1', debug=True)
