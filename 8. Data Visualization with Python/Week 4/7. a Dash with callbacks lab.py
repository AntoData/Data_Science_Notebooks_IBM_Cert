import pandas as pd
import plotly.express as px
import dash
import dash.html as html
import dash.dcc as dcc
from dash.dependencies import Input, Output

# Opening Data
file_eu_gdp: pd.DataFrame = pd.read_excel('EU GDP.xlsx')
file_eu_population: pd.DataFrame = pd.read_excel('EU population.xlsx')
# Cleaning data
# Getting years in both datasets
years: set = set(list(file_eu_gdp.columns)[1:]).intersection(
    list(file_eu_population.columns)[1:])
# Casting to list
years: list = list(years)
# We sort the list
years.sort()
# Add the column TIME
years.insert(0, "TIME")
# Filtering the data to include only common years and the column with
# countries
file_eu_gdp_filtered: pd.DataFrame = file_eu_gdp[years]
file_eu_population_filtered: pd.DataFrame = file_eu_population[years]
# Now we need to filter by country so data matches in both dataframes
countries: set = set(list(file_eu_gdp_filtered["TIME"])).intersection(
    list(file_eu_population_filtered["TIME"]))
# Casting to list
countries: list = list(countries)
# We sort the list
countries.sort()
# Now we filter by common countries
file_eu_gdp_filtered: pd.DataFrame = file_eu_gdp_filtered[
    file_eu_gdp_filtered["TIME"].isin(countries)]
file_eu_population_filtered: pd.DataFrame = file_eu_population_filtered[
    file_eu_population_filtered["TIME"].isin(countries)]

# Creating the object for dash
app = dash.Dash()

# Building the HTML of dash
app.layout = html.Div(children=[
    # H1 heading
    html.H1("European Union: GDP vs Population",
            style={'textAlign': 'center', 'background-color': '#0b0178',
                   'color': '#f6fa05', 'font-size': 40}),
    # Creating input for the country that will filter data by country
    html.Div(["Country: ", dcc.Input(id="id_country",
                                     value="Spain",
                                     type="text",
                                     style={
                                         'height': '50px',
                                         'font-size': 35}
                                     ),
              ], style={'font-size': 40}),
    # Separations
    html.Br(),
    html.Br(),
    # H1 heading
    html.H1("GDP in country",
            style={'textAlign': 'center', 'background-color': '#0b0178',
                   'color': '#f6fa05', 'font-size': 40}),
    # Letting Dash know where in the HTML the first bar plot is placed
    html.Div(dcc.Graph(id='bar-plot')),
    # H1 heading
    html.H1("Population",
            style={'textAlign': 'center', 'background-color': '#0b0178',
                   'color': '#f6fa05', 'font-size': 40}),
    # Letting Dash know where in the HTML the second bar plot is placed
    html.Div(dcc.Graph(id="bar-plot2")),
    # H1 heading for the year input that affects pie plots
    html.H1("Country's data against EU",
            style={'textAlign': 'center', 'background-color': '#0b0178',
                   'color': '#f6fa05', 'font-size': 40}),
    # Input for years that will filter and create dataframes for the
    # pie plots
    html.Div(["Year: ", dcc.Input(id="id_year",
                                     value="2022",
                                     type="text",
                                     style={
                                         'height': '50px',
                                         'font-size': 35}
                                     ),
              ], style={'font-size': 40}),
    # H1 heading for first pie plot
    html.H1("Country's population against EU",
            style={'textAlign': 'center', 'background-color': '#0b0178',
                   'color': '#f6fa05', 'font-size': 30}),
    # Letting Dash know where in the HTML the first pie plot is placed
    html.Div(dcc.Graph(id="pie-plot1")),
    # H1 heading for the second pie plot
    html.H1("Country's GDP against EU",
            style={'textAlign': 'center', 'background-color': '#0b0178',
                   'color': '#f6fa05', 'font-size': 30}),
    # Letting Dash know where in the HTML the second pie plot is placed
    html.Div(dcc.Graph(id="pie-plot2"))
])


# In order to get the dynamic information, we first need to describe
# the inputs that our input fields will use to filter data and what
# graphs we will return as outputs to be placed in HTML
@app.callback(
    # Output here is describing that we will return 4 different
    # graphs like in our HTML, we need to define the same IDs that
    # we used in the HTML describe in app.layout, that is how the code
    # sends the info to the HTML
    Output(component_id='bar-plot', component_property='figure'),
    Output(component_id='bar-plot2', component_property='figure'),
    Output(component_id='pie-plot1', component_property='figure'),
    Output(component_id='pie-plot2', component_property='figure'),
    # Input here is describing that we will receive two different inputs
    # these need to have the same IDs as the ones we placed in the HTML
    # that is how the HTML sends info to the code
    Input(component_id='id_country', component_property='value'),
    Input(component_id='id_year', component_property='value')
)
# Therefore, our function needs to input parameters
def get_graph(input_country, input_year):
    """
    Given the input parameters defined in the HTML, returns a bar plot
    that displays the GDP of a country by year and another for
    the population and one pie chart for the percentage of that
    country's population in the EU in a certain year and another for
    the percentage of the GDP in the EU for that country in a certain
    year

    :param input_country: Country whose data we want to display
    :param input_year: Year we want to compare country's data to the
    rest of the EU

    :return: 2 bar plots and 2 pie charts explained above
    """

    # We filter the data set whose data has been cleaned by the
    # country selected in the HTML and set as input parameter
    df_gpd_country: pd.DataFrame = \
        file_eu_gdp_filtered[file_eu_gdp_filtered["TIME"] == input_country]
    # We need to set the index so TIME (so when it is transposed and
    # reset is available as a column)
    df_gpd_country.set_index(keys=["TIME"], inplace=True)
    # We build the first bar plot that will show the GPD every year
    # for that country (we need to transpose the dataframe)
    bar1 = px.bar(df_gpd_country.transpose().reset_index(),
                  # We set which column should become the axis x,
                  # in this case, the years
                  x="index",
                  # We set which should be the axis y, in this case
                  # the GDP of the selected country
                  y=input_country,
                  range_y=[df_gpd_country.min()-1000,
                           df_gpd_country.max()+1000],
                  # This will update the names of the axis x and y
                  labels={
                      "index": "Year",
                      input_country: "GPD"
                    }
                  )

    # We filter the data set whose data has been cleaned by the
    # country selected in the HTML and set as input parameter
    df_population_country: pd.DataFrame = \
        file_eu_population_filtered[file_eu_population_filtered["TIME"] ==
                                    input_country]
    # We need to set the index so TIME (so when it is transposed and
    # reset is available as a column)
    df_population_country.set_index(keys=["TIME"], inplace=True)
    # We build the second bar plot that will show the population every
    # year for that country (we need to transpose the dataframe)
    bar2 = px.bar(df_population_country.transpose().reset_index(),
                  # We set which column should become the axis x,
                  # in this case, the years
                  x="index",
                  # We set which should be the axis y, in this case
                  # the GDP of the selected country
                  y=input_country,
                  range_y=[df_population_country.min() - 1000,
                           df_population_country.max() + 1000],
                  # This will update the names of the axis x and y
                  labels={
                      "index": "Year",
                      input_country: "Population"
                    }
                  )
    bar1.update_layout()
    bar2.update_layout()

    # We filter now the dataframe whose data we cleaned by country and
    # year
    country_year_population = \
        file_eu_population_filtered[
            file_eu_population_filtered["TIME"] == input_country][input_year]
    # We filter now the dataframe whose data we cleaned so it contains
    # all the data for any country by the one we selected the year
    # we selected
    not_country_year_population = \
        file_eu_population_filtered[
            file_eu_population_filtered["TIME"] != input_country][input_year]
    # We get the population for the country selected in the year
    # selected
    sum_population_country = country_year_population.sum()
    # We get the total population in the rest of the European Union
    sum_population_not_country = not_country_year_population.sum()

    # We need to create now a new dataframe with a row with the
    # population in the country we selected and another row with the
    # population in the rest of the European Union during the selected
    # year
    df_population_pie: pd.DataFrame = pd.DataFrame(
        data={input_country: [sum_population_country],
              "EU": [sum_population_not_country]})
    # To use it later to create the oue chart, we need to transpose it
    df_population_pie = df_population_pie.transpose()
    # We also need to reset the indexes so we have the two columns with
    # the name of the country and its population
    df_population_pie.reset_index(inplace=True)
    # We rename the columns to make it more readable
    df_population_pie.columns = ["Country", "Population"]

    # We build the first pie chart that displays the percentage
    # of the population in the country we selected vs the population
    # in the rest of the European Union in the year selected
    pie1 = px.pie(df_population_pie, values="Population", names="Country")

    # We filter now the dataframe whose data we cleaned by country and
    # year
    country_year_gdp = \
        file_eu_gdp_filtered[
            file_eu_gdp_filtered["TIME"] == input_country][input_year]
    # We filter now the dataframe whose data we cleaned so it contains
    # all the data for any country by the one we selected the year
    # we selected
    not_country_year_gdp = \
        file_eu_gdp_filtered[
            file_eu_gdp_filtered["TIME"] != input_country][input_year]
    # We get the GDP of the country and the year we selected
    sum_gdp_country = country_year_gdp.sum()
    # We get the GDP of every country in the European Union but the
    # country we selected during the year we selected
    sum_gpd_not_country = not_country_year_gdp.sum()

    # We build a new dataframe that contains in a row the GDP in the
    # country we selected and in the other the sum of the GDP in every
    # country but the one we selected during the year we selected
    df_gdp_pie: pd.DataFrame = pd.DataFrame(
        data={input_country: [sum_gdp_country],
              "EU": [sum_gpd_not_country]})
    # We rename the columns to reflect their content
    df_gdp_pie.columns = ["Country", "GDP"]
    # We build the second pie chart that displays the percentage of
    # a country's GDP vs the GDP of every other country in the EU
    # during the year we selected
    pie2 = px.pie(df_gdp_pie, values="GDP", names="Country")
    # We return the 4 charts so they are displayed in the HTML
    return bar1, bar2, pie1, pie2


if __name__ == "__main__":
    app.run_server(port=8002, host='127.0.0.1', debug=True)
