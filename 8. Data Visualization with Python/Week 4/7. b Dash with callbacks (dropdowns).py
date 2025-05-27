import dash
import pandas as pd
import seaborn as sns
import dash.html as html
import dash.dcc as dcc
import plotly.express as px
from dash.dependencies import Input, Output

# We load the data (example dataset provided by Seaborn)
# Dataset about Titanic passengers
df_titanic = sns.load_dataset("titanic")


# We need to get the possible values for dropdown sex
values_dropdown_sex: [str] = df_titanic["sex"].unique()

# We need to get the possible values for dropdown class
values_dropdown_class: [str] = df_titanic["class"].unique().tolist()

# Getting survivors vs deaths data for first pie chart
df_survivors_deaths: pd.DataFrame = \
    df_titanic.groupby(by=["survived"]).count()["who"].reset_index()
df_survivors_deaths["Survivor/Dead"] = ["Dead", "Survivor"]
df_survivors_deaths.rename(columns={"who": "Total"}, inplace=True)

# Getting survivors grouped by gender for second pie chart
df_survivors_by_gender: pd.DataFrame = df_titanic.groupby(
    by=["sex"]).count()["who"].reset_index()
df_survivors_by_gender.rename(
    columns={"sex": "Gender", "who": "Total"}, inplace=True)

# Getting survivors data grouped by class for third pie chart
df_survivors_by_class: pd.DataFrame = \
    df_titanic[
        df_titanic["survived"] == 1].groupby(
        by=["class"], observed=False).count()["pclass"].reset_index()
df_survivors_by_class.rename(columns={"class": "Class", "pclass": "Total"},
                             inplace=True)

app = dash.Dash(__name__)
app.layout = html.Div(children=[
    html.H1("TITANIC SURVIVORS STUDY", style={"textAlign": "center",
                                              "background-color": "#554fa8",
                                              "color": "#c3bff2",
                                              "font-size": 42}, ),
    html.Div(children=[
        html.Div(
            children=[
                html.H6("Survivors vs dead",
                        style={"font-size": 24}),
                html.Div(dcc.Graph(
                    id="survivors_vs_dead",
                    style={
                        'height': '400px',
                        'font-size': 35}))
            ]),
        html.Div(
            children=[
                html.H6("Survivors by gender",
                        style={"font-size": 24}),
                html.Div(dcc.Graph(
                    id="survivors_by_sex",
                    style={
                        'height': '400px',
                        'font-size': 35})),
            ]),
        html.Div(
            children=[
                html.H6("Survivors by class",
                        style={"font-size": 24}),
                html.Div(dcc.Graph(
                    id="survivors_by_class",
                    style={
                        'height': '400px',
                        'font-size': 35})),
            ]),
    ],
        style={'columnCount': 3}),
    html.Div(children=[
        html.Div(["From: ", dcc.Input(id="id_from_age",
                                      value="18",
                                      type="text",
                                      style={
                                          'height': '50px',
                                          'font-size': 35}
                                      )]),
        html.Div(["To: ", dcc.Input(id="id_to_age",
                                    value="99",
                                    type="text",
                                    style={
                                        'height': '50px',
                                        'font-size': 35}
                                    )]),
    ], style={'columnCount': 5}),
    html.Div(children=[
        html.Div(["Gender: ", dcc.Dropdown(id="id_sex",
                                           options=
                                           df_titanic["sex"].unique().tolist(),
                                           value=
                                           df_titanic["sex"].
                                           unique().tolist()[0], )]),
        html.Div(["Class: ", dcc.Dropdown(id="id_class",
                                          options=
                                          df_titanic[
                                              "class"].unique().tolist(),
                                          value=
                                          df_titanic["class"].
                                          unique().tolist()[0], )]),
        html.Div(["Survived?: ", dcc.Dropdown(id="id_survived",
                                              options=["Yes", "No"],
                                              value="Yes", )]),

    ], style={'columnCount': 1}),
    html.Div(dcc.Graph(id="survivors_by_age"))
])


@dash.callback(
    Output(component_id="survivors_vs_dead", component_property="figure"),
    Output(component_id="survivors_by_sex", component_property="figure"),
    Output(component_id="survivors_by_class", component_property="figure"),
    Output(component_id="survivors_by_age", component_property="figure"),
    Input(component_id="id_from_age", component_property="value"),
    Input(component_id="id_to_age", component_property="value"),
    Input(component_id="id_sex", component_property="value"),
    Input(component_id="id_class", component_property="value"),
    Input(component_id="id_survived", component_property="value")
)
def get_graph(from_age, to_age, gender, class_, survived):
    print(gender)
    print(type(gender))
    print(class_)
    print(type(class_))
    print(survived)
    print(type(survived))
    if survived == "Yes":
        survived_ = 1
    elif survived == "No":
        survived_ = 0
    else:
        survived_ = -1
    print(survived_)
    pie_chart1 = \
        px.pie(df_survivors_deaths, names="Survivor/Dead", values="Total")
    pie_chart2 = \
        px.pie(df_survivors_by_gender, names="Gender", values="Total")
    pie_chart3 = px.pie(df_survivors_by_class, names="Class", values="Total")
    df_age: pd.DataFrame = df_titanic[(df_titanic["survived"] == survived_) &
                                      (df_titanic["age"] > float(from_age)) &
                                      (df_titanic["age"] < float(to_age)) &
                                      (df_titanic["class"] == class_) &
                                      (df_titanic["sex"] == gender)]
    print(df_age)
    histogram_plot = px.histogram(df_age, x="age", y="survived", histfunc="count")
    return pie_chart1, pie_chart2, pie_chart3, histogram_plot


if __name__ == "__main__":
    app.run_server(port=8002, host='127.0.0.1', debug=True)
