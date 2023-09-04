pypl_dataframe <- read.csv("./PYPL.csv", header=TRUE, stringsAsFactors=FALSE)
print(pypl_dataframe)

languages_to_select <- unique(pypl_dataframe$Language)
print(languages_to_select)

salaries_dollar_dataframe <- read.csv("salaries_dollars.csv")
print(salaries_dollar_dataframe)

OS_share_dataframe <- read.csv("./OS_share_countries.csv")
print(OS_share_dataframe)

plug_types_dataset <- read.csv("plug_types_countries.csv")
print(plug_types_dataset)

team_selection_model <- function(tech_pl, salary_budget){
    # Returns which country a team should be based in and the OS and plug types their devices should have
    # tech_pl: A string with the desired programming language the project should use
    # salary_budget: An interger with the budget to hire a developer in dollars
    
    # For starters, we have to import the datasets with all the
    # information
    
    # Dataframe with the popularity of programming languages by country
    pypl_df <- read.csv("./PYPL.csv", header=TRUE, stringsAsFactors=FALSE)
    
    # Dataframe with the information about average salaries by
    # programming language in different countries depending on
    # experience level in dollars
    salaries_dollar_df <- read.csv("./salaries_dollars.csv", header=TRUE, stringsAsFactors=FALSE)
    
    # Dataframe with the share of different operating systems (desktop and mobile) by country
    os_share_df <- read.csv("OS_share_countries.csv", header=TRUE, stringsAsFactors=FALSE)
    
    # Dataframe with the information about the plug types used in
    # different countries
    plug_types_df <- read.csv("./plug_types_countries.csv", header=TRUE, stringsAsFactors=FALSE)
    
    # First, we filter the dataframe with the information about the
    # popularity of a programming language in a certain country
    # by the programming language, sort it by the field share (share
    # of developers that use that language in that country) and
    # convert the result to a list
    countries_use_language <- pypl_df[pypl_df$Language == tech_pl, ]
    countries_use_language <- (countries_use_language[order(countries_use_language$Share, decreasing=TRUE),])
    
    if(length(countries_use_language)==0){
        print(paste0("The programming language %s is not in our list", tech_pl))
        stop("The programming language is not in our list")
    }
    
    print("We will check the countries in the following order:")
    print(countries_use_language$Country)
    
    # We iterate by country in order of popularity of the desired
    # language in that country
    team_country <- ""
    developer_experience_level <- ""
    found <- FALSE
    for(country in countries_use_language$Country){
        # We get the salaries of a developer using that programming
        # language in that country with all 3 levels of experience
        if(found){
            break
        }
        salaries_country_pl <- salaries_dollar_df[
            (salaries_dollar_df["Country"] == country & salaries_dollar_df["Language"] == tech_pl), ]
        
        # Now we create a list with the different levels of experience in order from most expensive to cheapest
        experience_levels <- c("Senior", "Average", "Entry")
        
        # And we use them to get the salary for a developer with that level of experience that uses the programming language
        # given by the user in the country in this iteration
        for(exp_level in experience_levels){
            it_salary <- salaries_country_pl[1, exp_level]
            if(as.integer(it_salary) < as.integer(salary_budget)){
                team_country <- country
                developer_experience_level <- exp_level
                found <- TRUE
                print("We found the solution: ")
                print(paste0("- The country where the team should be based is: ", team_country))
                print(paste0("- The level of the developer should be ", developer_experience_level))
                break
            } 
        }
    }
    if(developer_experience_level == ""){
        stop("Budget is not enough to hire a developer in any country with the selected technology")
    }
    
    # At this point we select the OS for desktop and mobile devices that 
    # are most popular in the country we selected
    types_of_devices <- c("Desktop", "Mobile")
    os_devices <- c()
    for(type_device in types_of_devices){
        suggested_os_df <- os_share_df[(os_share_df["Country"] == team_country & os_share_df["Type"] == type_device),]
        suggested_os_df <- suggested_os_df[order(suggested_os_df$Share, decreasing=TRUE), ]
        suggested_os <- suggested_os_df[1, 2]
        os_devices <- append(os_devices, suggested_os)
        cat("- The suggested OS for", type_device, "devices is", suggested_os,"\n",sep=" ")
    }
    
    # Finally, we will use the country to select the plug those devices should have
    plug_types <- plug_types_df[(plug_types_df["Country"] == team_country), ]
    plug_types <- plug_types$Plug.Type
    
    cat("- Plug types should be one of this list: ", plug_types, "\n",sep=" ")
    
    return(c(team_country, developer_experience_level, os_devices, plug_types))
}

team_selection_model("Python", 30000)

team_selection_model("Rust", 30000)

team_selection_model("Swift", 50000)

team_selection_model("C#", 28000)

desired_programming_language <- readline(prompt="Which programming language or technology is the project going to be developed in?")
budget_developer_salary <- as.integer(readline(prompt="What is the intended salary for a single developer in dollars?"))

result <-  team_selection_model(desired_programming_language, budget_developer_salary)
choosen_country <- result[1] 
selected_experience_level <- result[2]
suggested_desktop_os <- result[3]
suggested_mobile_os <- result[4]
plug_types_needed <- c()
for(i in 5:(length(result))){
    plug_type <- result[i]
    plug_types_needed <- c(plug_types_needed, plug_type)
}

library(imager)

countries_flags <- c("US" = "https://cdn.countryflags.com/thumbs/united-states-of-america/flag-400.png",
                    "UK" = "https://cdn.countryflags.com/thumbs/united-kingdom/flag-400.png",
                    "France" = "https://cdn.countryflags.com/thumbs/france/flag-400.png",
                    "Germany" = "https://cdn.countryflags.com/thumbs/germany/flag-400.png",
                    "India" = "https://cdn.countryflags.com/thumbs/india/flag-400.png")

desktop_operating_system_icons <- c("Windows" = "https://www.iconarchive.com/download/i58702/dakirby309/windows-8-metro/Folders-OS-Windows-8-Metro.256.png",
                                   "OSX" = "https://www.iconarchive.com/download/i60265/mat-u/camill/Drive-macos.256.png",
                                   "Linux" = "https://www.iconarchive.com/download/i45763/tatice/operating-systems/Linux.256.png",
                                   "Chrome OS" = "https://www.iconarchive.com/download/i44717/morcha/browsers/Chrome.256.png")

mobile_operating_system_icons <- c("Android" = "https://www.iconarchive.com/download/i54020/danleech/simple/android.512.png",
                                  "iOS" = "https://www.iconarchive.com/download/i150039/simpleicons-team/simple/ios.512.png")

plug_type_icons <- c("A" = "https://www.iec.ch/themes/custom/iec/images/world-plugs/types/A/A-button.png",
    "B" = "https://www.iec.ch/themes/custom/iec/images/world-plugs/types/B/B-button.png",
    "C" = "https://www.iec.ch/themes/custom/iec/images/world-plugs/types/C/C-button.png",
    "D" = "https://www.iec.ch/themes/custom/iec/images/world-plugs/types/D/D-button.png",
    "E" = "https://www.iec.ch/themes/custom/iec/images/world-plugs/types/E/E-button.png",
    "F" = "https://www.iec.ch/themes/custom/iec/images/world-plugs/types/F/F-button.png",
    "G" = "https://www.iec.ch/themes/custom/iec/images/world-plugs/types/G/G-button.png",
    "H" = "https://www.iec.ch/themes/custom/iec/images/world-plugs/types/H/H-button.png",
    "I" = "https://www.iec.ch/themes/custom/iec/images/world-plugs/types/I/I-button.png",
    "J" = "https://www.iec.ch/themes/custom/iec/images/world-plugs/types/J/J-button.png",
    "K" = "https://www.iec.ch/themes/custom/iec/images/world-plugs/types/K/K-button.png",
    "L" = "https://www.iec.ch/themes/custom/iec/images/world-plugs/types/L/L-button.png",
    "M" = "https://www.iec.ch/themes/custom/iec/images/world-plugs/types/M/M-button.png",
    "N" = "https://www.iec.ch/themes/custom/iec/images/world-plugs/types/N/N-button.png",
    "O" = "https://www.iec.ch/themes/custom/iec/images/world-plugs/types/O/O-button.png")

IRdisplay::display(paste0("The team should be based in ", choosen_country))

country_flag <- load.image(countries_flags[choosen_country])
plot(country_flag)

IRdisplay::display(paste0("Desired experience of developers: ", selected_experience_level))

IRdisplay::display(paste0("Recommended operating system for desktop devices: ", suggested_desktop_os))

desktop_OS <- load.image(desktop_operating_system_icons[suggested_desktop_os])
plot(desktop_OS)

IRdisplay::display(paste0("Recommended operating system for mobile devices: ", suggested_mobile_os))

mobile_OS <- load.image(mobile_operating_system_icons[suggested_mobile_os])
plot(mobile_OS)

for(plug_type in plug_types_needed){
    IRdisplay::display(paste0("- ", plug_type))
    plug_type_image <- load.image(plug_type_icons[plug_type])
    plot(plug_type_image)
}

# We import ggplot2 so we can plot charts
library(ggplot2)

get_bar_chart_salaries_technology_in_country_by_experience_level <- function(programming_language, country){
    # Displays a bar chart plot where we compare the salaries of a developer in a given technology in a given country by experience level
    # programming_language: Programming language of the developers whose salary we will compare
    # country: Country of the developers whose salary we will compare
    
    # We filter the dataset that contains the salary of developers to get average salaries in our country and programming language
    df_salaries_country_technology <- salaries_dollar_dataframe[
        (salaries_dollar_dataframe["Country"] == choosen_country & salaries_dollar_dataframe["Language"] == desired_programming_language),]
    
    # We keep only the columns with the salaries
    df_salaries_country_technology <- df_salaries_country_technology[c("Entry", "Average", "Senior")]
    
    # By default, all bars are orange
    bar_colors <- c('orange', 'orange', 'orange')
    
    # The column that represents the experience level we recommended previously will become blue
    if(selected_experience_level == "Entry"){
        bar_colors[1] <- "blue" 
    }else if(selected_experience_level == "Average"){
        bar_colors[2] <- "blue" 
    }else{
        bar_colors[3] <- "blue" 
    }
    
    barplot(c(df_salaries_country_technology$Entry, df_salaries_country_technology$Average, df_salaries_country_technology$Senior), 
            col = bar_colors, names.arg = names(df_salaries_country_technology))
    
}

IRdisplay::display(paste0("- Salaries for ", desired_programming_language, " developers in ", choosen_country, " by experience level "))

get_bar_chart_salaries_technology_in_country_by_experience_level(desired_programming_language, choosen_country)

get_bar_chart_salaries_developer_country_experience_level_by_technology <- function(experience_level, country){
    # Displays a bar chart plot where we compare the salaries of a developer with a given experience in a given country by technology level
    # experience_level: Experience level of the developers
    # country: Country of the developers
    
    # To start, we filter the dataframe with the average salaries to get only the ones for the programming language and experience level passed
    # as parameters
    df_salaries_technology_experience <- salaries_dollar_dataframe[
            (salaries_dollar_dataframe["Country"] == country), c("Language", experience_level)]
    
    # We drop all row that have value NA
    df_salaries_technology_experience <- na.omit(df_salaries_technology_experience)

    # All columns will be orange, except the one suggested by us will be blue
    bar_colors <- c()
    for(language in df_salaries_technology_experience$Language){
        if(language == desired_programming_language){
            bar_colors <- append(bar_colors, 'blue')
        }else{
            bar_colors <- append(bar_colors, 'orange')
        }
    }
    
    # We set up the bar chart
    barplot(df_salaries_technology_experience$Senior, 
            col = bar_colors, names.arg = df_salaries_technology_experience$Language)
    
}

IRdisplay::display(paste0("- Salaries for developers with experience level ", selected_experience_level, " in ", choosen_country))

get_bar_chart_salaries_developer_country_experience_level_by_technology(selected_experience_level, choosen_country)

# We import the libraries need to execute this function
library(rjson)
library(stringr)
library(plotly)

display_map_salaries_technology_experience_level_by_country <- function(programming_language, experience_level){
    # Displays coloured map with the different average salaries of developers of a given programming language with a given experience
    # level by country
    # programming_language: Programming language used by the developers
    # experience_level: Experience level of developers
    
    # To start, we filter the dataframe with the average salaries to get only the ones for the programming language and experience level passed
    # as parameters
    df_salaries_technology_experience <- salaries_dollar_dataframe[
            (salaries_dollar_dataframe["Language"] == programming_language), c("Country", experience_level)]
    
    # We define a vector with keys to correct the names of the countries to match the ids in the json we will use to display countries in the map
    countries_correction <- c("US" = "USA", "India" = "IND", "Germany" = "DEU", "UK" = "GBR", "France" = "FRA")
    
    # We correct the names of the countries in the dataframe to match the ids in the geo json
    for(v_key in names(countries_correction)){
        df_salaries_technology_experience$Country <- str_replace(df_salaries_technology_experience$Country, v_key, countries_correction[v_key])
    }
    
    # We get the geo json to display countries in a map
    url = "https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json"
    countries <- rjson::fromJSON(file=url)
    
    # We set up the map
    fig <- plot_ly() 
    fig <- fig %>% add_trace(
        type="choroplethmapbox",
        geojson=countries,
        locations=df_salaries_technology_experience$Country,
        z=df_salaries_technology_experience[, selected_experience_level],
        colorscale="Viridis",
        zmin=min(df_salaries_technology_experience[, selected_experience_level], na.rm = TRUE),
        zmax=max(df_salaries_technology_experience[, selected_experience_level], na.rm = TRUE),
        marker=list(line=list(width=0),opacity=0.5))
    fig <- fig %>% plotly::layout(
        mapbox=list(
            style="carto-positron",
            zoom =1,
            center=list(lon=0, lat=30))
    )
    fig
}

IRdisplay::display(paste0("Salaries for developers with experience level ", selected_experience_level, " in ", desired_programming_language))

display_map_salaries_technology_experience_level_by_country(desired_programming_language, selected_experience_level)


