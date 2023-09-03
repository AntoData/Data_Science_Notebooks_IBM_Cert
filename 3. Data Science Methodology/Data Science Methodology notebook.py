#!/usr/bin/env python
# coding: utf-8
"""
Self-made practical example of the application of the Data Science
Methodology explained in the course in
https://www.coursera.org/learn/data-science-methodology/
"""
import json
import requests
import pandas as pd
from IPython.display import Markdown as Md
from IPython.display import Image, display
import matplotlib.pyplot as plt
import plotly.express as px

# # Data Science Methodology

# ##### A practical example
# __NOTE__: This is an exercise using some real data. Some of the data
# will not come from real sources but estimated by the author.
# This is not a real tool to make any decision.
# <font color='red'>Author is not responsible for the use given to
# this notebook. It is not a real tool, it is an example
# of the methodology to follow</font>.

# ### 1. Business Understanding

# __Client__: We are an IT consulting company that takes projects from
# around the world. However, for legal reasons we only can have teams
# in the following countries:
# + United States
# + United Kingdom
# + France
# + Germany
# + India
# 
# Our volume of projects increases at a very big rate. Our HR teams
# needs a tool to decide where to base our teams for each project.
# And also provide those teams with the adecuate equipment
# (laptop, appropiate OS, plug type...).
# Usually technology used in the project and budget are key elements
# to decide where our team will be based for a project.
# 
# __Data Science Team__: We understand that the deciding factors to
# make this decision are main programming language used in that project
# and budget for salary of a developer. That determines which country
# the team will be baased in. Once the country is decided, the OS for
# their mobile devices and plugs for the laptops is decided based on
# the popularity of the OS in that country and the plug type.

# ### 2. Analytical approach

# __Data Science Team__: We can design an algorithm that given the
# following inputs:
# 1. Programming language
# 2. Budget for developers
# 
# will select one of the 5 countries in which our teams will be based.
# The popularity of the programming language in that country and the
# average salary of a developer. The algorithm will return the country
# and the highest experience level of developer within that budget
# (Entry, Junior, Senior).
# The country will select the OS for their mobile devices based on the
# popularity of the OS in the country and the plug for their devices.

# ### 3. Data Requirements

# 1. For starters, we need statistics about the popularity of
# programming languages in the United States, United Kingdom, France,
# Germany and India
# 2. We need also statistics about the salaries of developers expert in
# different technologies in those countries:
# 
# _Important Note_:<font color='red'> We will estimate some of the
# salaries when data is not found </font>. This is just an example
# using as much real data as possible. But it is not a real tool that
# should be used for these decisions
# 
# 3. Then, we need statistics about the preferred OS in different
# countries
# 4. Finally, a dataset with the different plug types used in those
# countries

# ### 4. Data Collection

# 1. Statistics about the popularity of programming languages in the
# United States, United Kingdom, France, Germany and India
# + We will take it from
# [PYPL PopularitY of Programming Language](https://pypl.github.io/PYPL.html)

# 2. Statistics about the salaries of developers expert in different
# technologies in those countries:
# + We will take it from the site [payscale](https://www.payscale.com/)
# 
# _Important Note_:<font color='red'> We will estimate some of the
# salaries when data is not found </font>. This is just an example
# using as much real data as possible. But it is not a real tool that
# should be used for these decisions
# 
# 3. Statistics about the preferred OS in different countries
# + We can get it from the site
# [statscounter]
# (https://gs.statcounter.com/os-market-share#monthly-202207-202307)

# 4. Dataset with the different plug types used in those countries
# + We can get if from the site of the
# [International Electrotechnical Commission]
# (https://www.iec.ch/world-plugs)

# ### 5. Data Understanding

# #### Is our data representative?
# 1. Statistics about the popularity of programming languages in the
# United States, United Kingdom, France, Germany and India
# from [PYPL PopularitY of Programming Language]
# (https://pypl.github.io/PYPL.html)
# + This is the most trusted and quoted survey about popularity and use
# of programming languages around the world. This will sort our 5
# possible countries in order from the one with the biggest share for a
# certain programming language to the one with the lowest share.
# 
# 2. Statistics about the salaries of developers expert in different
# technologies in those countries from the site [payscale]
# (https://www.payscale.com/)
# + Apart from a couple of positions where we had to estimate,
# that database offers salaries for all possible positions. We compared
# some in European countries like France and Germany to real job offers
# and those estimations matched those real salaries.
# 
# _Remember_:<font color='red'> We will estimate some of the salaries
# when data is not found </font>. This is just an example using as
# much real data as possible. But it is not a real tool that should be
# used for these decisions
# 
# 3. Statistics about the preferred OS in different countries from the
# site [statscounter]
# (https://gs.statcounter.com/os-market-share#monthly-202207-202307)
# + This is the most complete database about OS use around the world
# with trusted statistics

# 4. Dataset with the different plug types used in those countries from
# the site of the [International Electrotechnical Commission]
# (https://www.iec.ch/world-plugs).
# + This is an official organisation so the information is completely
# accurate

# ### 6. Data preparation

# ##### 1. Statistics about the popularity of programming languages in
# the United States, United Kingdom, France, Germany and India
# from [PYPL PopularitY of Programming Language]
# (https://pypl.github.io/PYPL.html)
pypl_dataframe: pd.DataFrame = pd.read_csv("./PYPL.csv")
print("PYPL PopularitY of Programming Language:")
print(pypl_dataframe)
print("")

# Languages to consider:
languages_to_select = list(
    pypl_dataframe[~pypl_dataframe.duplicated('Language')]["Language"])
print("Possible programming languages: ")
print(languages_to_select)
print("")

# ##### 2. Statistics about the salaries of developers expert in
# different technologies in those countries from the site
# [payscale](https://www.payscale.com/)
salaries_dollar_dataframe: pd.DataFrame = pd.read_csv("./salaries_dollars.csv")
print("Salaries in different countries by technology and experience")
print(salaries_dollar_dataframe)
print("")

#  ##### 3. Statistics about the preferred OS in different countries
#  from the site [statscounter]
#  (https://gs.statcounter.com/os-market-share#monthly-202207-202307)
OS_share_dataframe: pd.DataFrame = pd.read_csv("./OS_share_countries.csv")
print("OS (desktop and mobile) share per country")
print(OS_share_dataframe)

# ##### 4. Dataset with the different plug types used in those
# countries from the site of the
# [International Electrotechnical Commission]
# (https://www.iec.ch/world-plugs).
plug_types_dataset: pd.DataFrame = pd.read_csv(
    "./plug_types_countries.csv")
print("Plug types by country")
print(plug_types_dataset)
print("")


# ### 7. Modelling

# The model works as follows:
# 1. The user provides the following parameters:
# + Technology (programming language) desired for the project
# + Salary budget (for a developer, in dollars)
# 
# 2. The desired technology will sort the countries that are candidates
# to hold the team from the one with the highest share of that language
# to the one with the lowest
# 3. We iterate by country and sort the different salaries depending on
# the level of experience of developers
# + Once we find the first salary that is lower than the budget, we
# will stop the loop and
# __that will be the country where the team will be based__
# 
# 4. The country will determine the OS.
# + We will select the OS with the highest share
# for both desktop and mobile devices in that country
# 
# 5. Finally, the country will determine the plug typefor the laptop
# and devices

def team_selection_model(tech_pl: str, salary_budget: int) -> []:
    """
    Returns which country a team should be based in and the OS and plug
    types their devices should have
    :param tech_pl: Desired programming language the project should use
    :type tech_pl: str
    :param salary_budget: Budget for the salary of one developer in
    dollars
    :return: Country where the team should be based, OS and plug type
    their devices should have
    """

    # For starters, we have to import the datasets with all the
    # information

    # Dataframe with the popularity of programming languages by country
    pypl_df: pd.DataFrame = pd.read_csv("./PYPL.csv")

    # Dataframe with the information about average salaries by
    # programming language in different countries depending on
    # experience level in dollars
    salaries_dollar_df: pd.DataFrame = pd.read_csv(
        "./salaries_dollars.csv")

    # Dataframe with the share of different operating systems (desktop
    # and mobile) by country
    os_share_df: pd.DataFrame = pd.read_csv(
        "OS_share_countries.csv")

    # Dataframe with the information about the plug types used in
    # different countries
    plug_types_df: pd.DataFrame = pd.read_csv(
        "./plug_types_countries.csv")

    # First, we filter the dataframe with the information about the
    # popularity of a programming language in a certain country
    # by the programming language, sort it by the field share (share
    # of developers that use that language in that country) and
    # convert the result to a list
    candidate_countries_sorted: [str] = list(pypl_df[(
            pypl_df["Language"] == tech_pl)].sort_values(
        by="Share", ascending=False)["Country"])
    if len(candidate_countries_sorted) == 0:
        raise ValueError("The programming language {0} is not in our list".
                         format(tech_pl))
    print("We will check the countries in the following order: {0}".
          format(candidate_countries_sorted))

    # We iterate by country in order of popularity of the desired
    # language in that country
    team_country: str = ""
    developer_experience_level: str = ""
    for country in candidate_countries_sorted:
        # We get the salaries of a developer using that programming
        # language in that country with all 3 levels of experience
        salaries_country_pl: pd.DataFrame = salaries_dollar_df[
            (salaries_dollar_df["Country"] == country) &
            (salaries_dollar_df["Language"] == tech_pl)]

        # Now we create a list with the different levels of experience
        # in order from most expensive to cheapest
        experience_levels: [str] = ["Senior", "Average", "Entry"]
        # And we use them to get the salary for a developer with that
        # level of experience that uses the programming language
        # given by the user in the country in this iteration
        for exp_level in experience_levels:
            it_salary: float = salaries_country_pl[exp_level].iloc[0]
            # As we are in order from most country with the most number
            # of developers that use that programming language and
            # from more experienced and therefore more expensive
            # developer to less, once we find the first combination of
            # country and developer experience that is cheaper than
            # our budget, that should be the country where our team
            # should be based in and the level of experience of the
            # developer
            if salary_budget > it_salary:
                team_country = country
                developer_experience_level = exp_level
                print("We found the solution: ")
                print("- The country where the team should be based is: {0}".
                      format(team_country))
                print("- The level of the developer should be {0}".format(
                    developer_experience_level))
                break
        else:
            continue
        break

    if developer_experience_level == "":
        raise ValueError("Budget of {0} is not enough to hire a {1} developer "
                         "in any country".format(salary_budget, tech_pl))

    # At this point we select the OS for desktop and mobile devices that
    # are most popular in the country we selected
    types_of_devices: [str] = ["Desktop", "Mobile"]
    os_devices: [str] = []
    for type_device in types_of_devices:
        suggested_os_var: str = os_share_df[
            (os_share_df["Country"] == team_country) &
            (os_share_df["Type"] == type_device)].sort_values(
            by="Share", ascending=False)["OS"].iloc[0]
        os_devices.append(suggested_os_var)
        print("- The suggested OS for their {0} devices is {1}".format(
            type_device, suggested_os_var))

    # Finally, we will use the country to select the plug those devices
    # should have
    plug_types: [str] = list(plug_types_df[(plug_types_df["Country"] ==
                                            team_country)]["Plug Type"])
    print("- Plug types should be one of this list: {0}".format(plug_types))
    return team_country, developer_experience_level, os_devices, plug_types


# ### 8. Evaluation

# ##### We are going to look into the datasets and pick 3 test cases to
# check the information returned is correct

# 1. If select __Rust__ as _programming language_ and give a _budget_
# of __$30000__, the results should be:
# + Country: United Kindgom
# + Experience Level: Entry
# + OS: Desktop -> Windows, Mobile -> iOS
# + Plug types: G


print(team_selection_model("Rust", 30000))

# 2. If I select __Swift__ as _programming language_ and give a
# _budget_ of __$50000__ the results should be:
# + Country: France
# + Experience Level: Average
# + OS: Desktop -> Windows, Mobile -> Android
# + Plug types: C, E


print(team_selection_model("Swift", 50000))

# 3. If I select __C#__ as _programming language_ and give a _budget_
# of __$28000__ the results should be:
# + Country: India
# + Experience Level: Senior
# + OS: Desktop -> Windows, Mobile -> Android
# + Plug types: C, D, M


team_selection_model("C#", 28000)

# ### 9. Deployment

# This will be deployed as a notebook where we will ask the HR
# specialist to provide a desired programming language for the team and
# a budget for the salary of a developer in dollars.
# The notebook will be placed in a server where the HR team will have
# access to it:
# Let's pretend this is that notebook in the server. We will add a
# couple of presentation features


desired_programming_language: str = \
    input("Which programming language or technology is the project "
          "going to be developed in? ")
budget_developer_salary: int = \
    int(input("What is the intended salary for a single developer in "
              "dollars? "))

choosen_country, selected_experience_level, suggested_os, needed_plug_types \
    = team_selection_model(desired_programming_language,
                           budget_developer_salary)

countries_flags: dict = {
    "US": "https://upload.wikimedia.org/wikipedia/commons/a/a9/"
          "Flag_of_the_United_States_%28DoS_ECA_Color_Standard%29.svg",
    "UK": "https://upload.wikimedia.org/wikipedia/commons/a/a5/"
          "Flag_of_the_United_Kingdom_%281-2%29.svg",
    "France": "https://upload.wikimedia.org/wikipedia/en/c/c3/"
              "Flag_of_France.svg",
    "Germany": "https://upload.wikimedia.org/wikipedia/en/b/ba/"
               "Flag_of_Germany.svg",
    "India": "https://upload.wikimedia.org/wikipedia/en/4/41/"
             "Flag_of_India.svg"
}
desktop_operating_system_icons: dict = {
    "Windows": "https://upload.wikimedia.org/wikipedia/commons/e/e2/"
               "Windows_logo_and_wordmark_-_2021.svg",
    "OSX": "https://upload.wikimedia.org/wikipedia/commons/e/e2/"
           "Windows_logo_and_wordmark_-_2021.svg",
    "Linux": "https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg",
    "Chrome OS": "https://upload.wikimedia.org/wikipedia/commons/e/e1/"
                 "Google_Chrome_icon_%28February_2022%29.svg"}
mobile_operating_system_icons: dict = {
    "Android": "https://upload.wikimedia.org/wikipedia/commons/6/64/"
               "Android_logo_2019_%28stacked%29.svg",
    "iOS": "https://upload.wikimedia.org/wikipedia/commons/6/63/"
           "IOS_wordmark_%282017%29.svg"}
plug_type_icons: dict = {
    "A": "https://www.iec.ch/themes/custom/iec/images/world-plugs/types/"
         "A/A-button.png",
    "B": "https://www.iec.ch/themes/custom/iec/images/world-plugs/types/"
         "B/B-button.png",
    "C": "https://www.iec.ch/themes/custom/iec/images/world-plugs/types/"
         "C/C-button.png",
    "D": "https://www.iec.ch/themes/custom/iec/images/world-plugs/types/"
         "D/D-button.png",
    "E": "https://www.iec.ch/themes/custom/iec/images/world-plugs/types/"
         "E/E-button.png",
    "F": "https://www.iec.ch/themes/custom/iec/images/world-plugs/types/"
         "F/F-button.png",
    "G": "https://www.iec.ch/themes/custom/iec/images/world-plugs/types/"
         "G/G-button.png",
    "H": "https://www.iec.ch/themes/custom/iec/images/world-plugs/types/"
         "H/H-button.png",
    "I": "https://www.iec.ch/themes/custom/iec/images/world-plugs/types/"
         "I/I-button.png",
    "J": "https://www.iec.ch/themes/custom/iec/images/world-plugs/types/"
         "J/J-button.png",
    "K": "https://www.iec.ch/themes/custom/iec/images/world-plugs/types/"
         "K/K-button.png",
    "L": "https://www.iec.ch/themes/custom/iec/images/world-plugs/types/"
         "L/L-button.png",
    "M": "https://www.iec.ch/themes/custom/iec/images/world-plugs/types/"
         "M/M-button.png",
    "N": "https://www.iec.ch/themes/custom/iec/images/world-plugs/types/"
         "N/N-button.png",
    "O": "https://www.iec.ch/themes/custom/iec/images/world-plugs/types/"
         "O/O-button.png"
}

Md("#### The team should be based in __{0}__".format(choosen_country))
Image(url=countries_flags[choosen_country], width=100, height=50)

Md("#### Desired experience of developers: __{0}__".format(
    selected_experience_level))

Md("#### Recommended operating system for desktop devices: __{0}__".format(
    suggested_os[0]))
Image(url=desktop_operating_system_icons[suggested_os[0]], width=200,
      height=100)

Md("#### Recommended operating system for mobile devices: __{0}__".format(
    suggested_os[1]))
Image(url=mobile_operating_system_icons[suggested_os[1]], width=200,
      height=100)

Md("#### Plug type should be one of the following:")
for plug_type in needed_plug_types:
    display(Md("#### - __{0}__".format(plug_type)))
    display(Image(url=plug_type_icons[plug_type], width=50, height=50))


def get_bar_chart_salaries_technology_in_country_by_experience_level(
        programming_language: str, country: str) -> None:
    """
    Displays a bar chart plot where we compare the salaries of a
    developer in a given technology in a given country by experience
    level
    :param programming_language: Programming language of the developers
    whose salary we will compare
    :type programming_language: str
    :param country: Country of the developers whose salary we will
    compare
    :type country: str
    :return: Displays bar chart
    """
    # We create matplotlib subplots
    fig, ax = plt.subplots()

    # We define the experience levels and set them as labels
    experience_levels: [str] = ['Entry', 'Average', 'Senior']
    bar_labels: [str] = experience_levels

    # We filter our data frame with the salaries in dollars to get
    # the average salary of a developer in the country and programming
    # language passed as parameters
    df_salaries_country_technology: pd.DataFrame = salaries_dollar_dataframe[
        (salaries_dollar_dataframe["Country"] == country) & (
                salaries_dollar_dataframe[
                    "Language"] == programming_language)]

    # We set the values of every experience level of a developer as
    # the values to display (turn them into float)
    counts: [float] = [float(df_salaries_country_technology["Entry"]),
                       float(df_salaries_country_technology["Average"]),
                       float(df_salaries_country_technology["Senior"])]

    # By default, all bars are orange
    bar_colors: [str] = ['tab:orange', 'tab:orange', 'tab:orange']

    # The bar with the experience levels the algorithm recommended will
    # turn blue
    if selected_experience_level == "Entry":
        bar_colors[0] = 'tab:blue'
    elif selected_experience_level == "Average":
        bar_colors[1] = 'tab:blue'
    elif selected_experience_level == "Senior":
        bar_colors[2] = 'tab:blue'

    # We set the bar plot
    ax.bar(experience_levels, counts, label=bar_labels, color=bar_colors)

    # We set the labels, title and legend
    ax.set_ylabel('Salary in dollars')
    ax.set_title('Experience level')
    ax.legend(title='Salaries for {0} developers in {1}'.format(
        programming_language, country))

    # We display the plot
    plt.show()


Md("#### Salaries for __{0}__ developers in __{1}__ by experience "
   "level".format(desired_programming_language, choosen_country))

get_bar_chart_salaries_technology_in_country_by_experience_level(
    desired_programming_language, choosen_country
)


def get_bar_chart_salaries_developer_country_experience_level_by_technology(
        experience_level: str, country: str) -> None:
    """
    Displays a bar chart plot where we compare the salaries of a
    developer with a given experience in a given country by technology
    level
    :param experience_level: Experience level of the developers
    :type experience_level: str
    :param country: Country of the developers
    :type country: str
    :return: Displays bar chart
    """
    # We create matplotlib subplots
    fig, ax = plt.subplots()

    # We filter the dataframe that contains salaries to get the average
    # salaries of a developer in a country using a technology passed
    # as parameters to this function
    df_salaries_country_experience: pd.DataFrame = salaries_dollar_dataframe[
        (salaries_dollar_dataframe["Country"] == country)][
        (["Language", experience_level])]
    df_salaries_country_experience = df_salaries_country_experience.dropna(
        subset=[experience_level])

    # The values represented in the bar will be the salaries
    counts: [float] = df_salaries_country_experience[
        experience_level].to_list()

    # The axis x will be the different programming languages we offer
    # in that given country
    programming_languages: [str] = df_salaries_country_experience[
        "Language"].to_list()

    # All bars will be orange except the one for the desired programming
    # language who will be blue
    bar_colors: [str] = [
        'tab:blue' if pl == desired_programming_language else 'tab:orange' for
        pl in programming_languages]

    # Programming languages offered will be the labels
    bar_labels: [str] = programming_languages

    # We set up the bar plot
    ax.bar(programming_languages, counts, label=bar_labels, color=bar_colors)

    # We set the labels, title and legend
    ax.set_ylabel('Salary in dollars')
    ax.set_title('Programming languages')
    ax.legend(title='Salaries for {0} developers in {1}'.format(
        experience_level, country))

    # We display the plot
    plt.show()


Md("#### Salaries for developers with experience level __{0}__ in __{1}__".
   format(selected_experience_level, choosen_country))

get_bar_chart_salaries_developer_country_experience_level_by_technology(
    selected_experience_level, choosen_country
)


def display_map_salaries_technology_experience_level_by_country(
        programming_language: str, experience_level: str) -> None:
    """
    Displays coloured map with the different average salaries of
    developers of a given programming language with a given experience
    level by country
    :param programming_language: Programming language used by the
    developers
    :type programming_language: str
    :param experience_level: Experience level of developers
    :type experience_level: str
    :return: Displays map
    """
    # We filter the dataframe to get the average salaries of developers
    # in a programming language with a level of experience passed
    # as parameters
    df_salaries_technology_experience: pd.DataFrame = \
        salaries_dollar_dataframe[(
                salaries_dollar_dataframe[
                    "Language"] == programming_language)][
            (["Country", experience_level])]

    # We need to change the values in the column Country to the ids
    # of the countries in the geo JSON we are going to use to display
    # the country borders in the map
    countries_correction: dict = {'US': 'USA', 'UK': 'GBR', 'India': 'IND',
                                  'Germany': 'DEU', 'France': 'FRA'}
    for v_key, v_value in countries_correction.items():
        df_salaries_technology_experience.replace(v_key, v_value, inplace=True)

    # We get the geo json with the borders of every country
    json_countries_req: requests.Response = requests.get(
        "https://raw.githubusercontent.com/johan/world.geo.json/master/"
        "countries.geo.json")
    countries_json: dict = json.loads(json_countries_req.text)

    # We get the minimum and maximum salaries to be used as range
    # of the colour scale we will use in the map
    salary_min: float = df_salaries_technology_experience[
        experience_level].min()
    salary_max: float = df_salaries_technology_experience[
        experience_level].max()

    # We set up the map
    fig = px.choropleth(df_salaries_technology_experience,
                        geojson=countries_json,
                        locations='Country', color=experience_level,
                        color_continuous_scale="Viridis",
                        range_color=(salary_min, salary_max),
                        scope="world",
                        labels={'Salary': 'unemployment rate'}
                        )
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

    # We display the map
    fig.show()


Md("#### Salaries for developers with experience level "
   "__{0}__ in __{1}__".format(
    selected_experience_level, desired_programming_language))

display_map_salaries_technology_experience_level_by_country(
    desired_programming_language, selected_experience_level)

# ### 10. Feedback

# __Client__; Model is correct and application is useful. However,
# data has to be updated manually. We need to automate the process
# to get periodic updates automatically. Also, we need more information
# about salaries for developers with skills in other technologies
