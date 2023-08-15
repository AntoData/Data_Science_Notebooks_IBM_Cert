# We import the library for data visualization
library(ggplot2)

# DATA IMPORT
# We import the csv with the data
# We ask for the location of the file
# File name is: pob_provinces_csv_tab.csv
# While we were writing this the file path was:
# "C:\\Users\\ingov\\Documents\\R Scripts\\pob_provinces_csv_tab.csv"
file_path <- readline(prompt="Enter location of the file: ")
spain_provinces_population <- read.csv(file_path)

# We display and explore the data
View(spain_provinces_population)

# DATA TRANSFORMATION
# We define a function to convert each column (which contains the population
# series in that province) to integer.
column_data_to_integer <- function(column) {
  # Replaces "." with "" which removes the "."
  number_points = length(unlist(gregexpr('\\.', column)))
  if (number_points == 1){
    to_integer <- as.numeric(as.character(column))
    to_integer = to_integer * 1000
    # print(to_integer)
  }else{
    # The figure has or more points, so it is taken as an Integer.
    # In order to turn it into a integer, we have to remove the
    # "." and then turn the column to integer
    removed_points <- gsub("\\.", "", column)
    # print(removed_points)
    # We convert that result to integer
    to_integer <- as.numeric(as.character(removed_points))
    # print(to_integer)
    # We reverse the vector
  }
  # Finally, we reverse the vector as the data starts in 2021 and ends in 1996
  # (in case it is needed)
  to_integer_reversed <- rev(to_integer)
  # print(to_integer_reversed)
  result <- to_integer_reversed
}
# Now we use the function to get a valid integer vector with the population
# series in Seville
population_evolution_seville <- column_data_to_integer(
  spain_provinces_population$X41.Sevilla)
# We explore the vector
View(population_evolution_seville)

# Now we use the function to get a valid integer vector with the population
# series in Malaga
population_evolution_malaga <- column_data_to_integer(
  spain_provinces_population$X29.Málaga)
# We explore the vector
View(population_evolution_malaga)

# We also use the function to normalize the series of years
years <- column_data_to_integer(spain_provinces_population$Año)
View(years)

# We compile into a dataframe the data we need for our bar plot
# We need years, population in Seville and population in Malaga
data_bar_plot <- data.frame(years=years, seville=population_evolution_seville, 
                  malaga=population_evolution_malaga)

# Because of how barplot works, we need to transpose the dataframe so
# each year is a column and the population in each city is a row and convert
# the row with the years in the different columns (each year is a column)
data_bar_plot <- setNames(data.frame(t(data_bar_plot[ , - 1])), 
                          as.integer(data_bar_plot[ , 1]))
print(data_bar_plot)

# Now we use as.matrix with our dataframe to display the information in a bar
# plot 
# as.matrix(data_bar_plot) makes the function take our dataframe as 
# data correctly. 
# xlim makes room for the legend
# beside = True makes possible to have multiple variables being plotted in the
# same barplot (each row in this case is a different bar series in a different
# colour)
bar_plot_sev_vs_ma <- barplot(as.matrix(data_bar_plot),
             main='Population evolution Seville vs Malaga', 
        ylab='Population', xlab='Year', beside = TRUE, 
         col=c("red", "blue"), legend = TRUE, xlim = c(0, 95))
print(bar_plot_sev_vs_ma)

print("")
print("Study of the evolution of the population in the province of Seville")
# What was the largest population recorded in Seville? In which year?
# In order to reply to this question, we can use the dataframe data_bar_plot
# in which every row represents a different city and each column represents
# a different year

# So to get what the max population in Seville was, we use the method apply
# The apply family functions in R are a well-known set of R vectorized functions 
# that allows you to perform complex tasks over arrays, avoiding the use of 
# for loops. 
# Syntax
# apply(X,       # Array, matrix or data frame
# MARGIN,  # 1: rows, 2: columns, c(1, 2): rows and columns
# FUN,     # Function to be applied
# ...)     # Additional arguments to FUN
# So to get the maximum population we use this function like this:
# apply(data_for_seville, 1 -> which means apply in the row, max, na.rm )
# na.rm means remove NA values
# So basically we are telling apply, return the max value in the row after
# removing NA values
max_population_seville <- apply(data_bar_plot["seville", ], 1, max, na.rm=TRUE)
# print(max_population_seville)
print("")
print(paste0("The highest population in Seville was ", 
             max_population_seville["seville"]))
# To get the year in which the population was the highest in Seville, we use
# the function apply like this:
# colnames(data_in_seville)[apply...]
# We get an array with the names of the years and select a position on it
# The position we have to return now is the position of the column with the
# highest value, the row with the highest value
# To get that: apply(data_in_seville, 1 -> apply to rows, which.max)
# Where which.max, returns the position of the element with the maximal value 
# in a vector.
# in this case the max value in our row
# so that index, applied to colnames(data_in_seville) which contains the list
# of years as a vector will return the year with the highest population
year_max_population_seville <- 
  colnames(data_bar_plot["seville", ])[apply(data_bar_plot["seville", ],1,
                                             which.max)]
# print(year_max_population_seville)
print(paste0("In the year ", year_max_population_seville))

print("")
print("Study on population growth in the province of Seville")
# POPULATION GROWTH
# In order to get the population growth between consecutive years we will have
# the following strategy
# We will create two empty vector, one to contain the years for which we have
# data (it means the previous and current year have numeric value)
# Another one that contains the difference between the population the previous
# and current year
population_growth_seville = c()
years_population_growth_seville = c()

# We create a j index that we will use every time we want to add a new value
# to both vectors
j = 1

# Now, we iterate through our vector that contains the population data for 
# Seville (data_bar_plot["seville", ]) but we will use the indexes
# for (i in x:z) means first value x, last value z
# We use the indexes so we add also use the vector years that is synced with
# our vector to create the vector also with the years where we have enough data
for (i in 2:length(data_bar_plot["seville", ]))
{
  # If both the current and previous years are not NA, it means we can proceed
  if (!is.na(data_bar_plot["seville", i-1]) & 
      !is.na(data_bar_plot["seville", i])){
    # We add this years (most recent of the two) to the vector of years
    years_population_growth_seville[j] <- years[i]
    # We add the difference between populations during the previous and current 
    # to the other vector
    population_growth_seville[j] <- 
      data_bar_plot["seville", i] - data_bar_plot["seville", i-1]
    # We increase the index of the vector
    j = j + 1
  }
}

print(years_population_growth_seville)
print(population_growth_seville)

# Which was the highest growth in the series? Which year?
max_increase_population_seville <- max(population_growth_seville)
max_increase_population_year_seville <- 
  years_population_growth_seville[which.max(population_growth_seville)]
print("")
print("Which was the highest growth in the series")
print(paste0("Highest growth was this number of people more: ", 
             max_increase_population_seville))
print("Which year?")
print(paste0("The year ", 
             max_increase_population_year_seville))

# Which was the lowest or negative growth in the series? Which year?
min_increase_population_seville <- min(population_growth_seville)
min_increase_population_year_seville <- 
  years_population_growth_seville[which.min(population_growth_seville)]
print("Which was the lowest growth in the series")
print(paste0("Lowest or negative growth was this number of people: ", 
             min_increase_population_seville))
print("Which year?")
print(paste0("The year ", 
             min_increase_population_year_seville))

plot(years_population_growth_seville, population_growth_seville, 
     main = "Population variation every year",
     xlab = "Years", ylab = "Population variation",
     pch = 19, frame = FALSE)
abline(h=0, col = "blue")
abline(v=2008, col = "red")
abline(v=2020, col = "purple")
legend(2017, 24000, legend=c("Zero", "Fin. Crisis", "Covid"), 
                                       fill = c("blue","red", "purple"))

# Which province experienced the biggest decrease (or smallest growth) in 
# population between 2020 and 2021?
print("")
print("Which province experienced the biggest decrease (or smallest growth) 
      in population between 2020 and 2021?")

# For starters, we need to get the population in every province in 2020 and 2021

spain_provinces_population_covid = 
  spain_provinces_population[spain_provinces_population$Año %in% c(2020,2021), ]
print("Population in every province in 2020 and 2021")
print(spain_provinces_population_covid)
spain_provinces_population_covid <- spain_provinces_population_covid[, -c(1,2)]
View(spain_provinces_population_covid)

provinces = colnames(spain_provinces_population_covid)
# print(provinces)
diff_population_prov_spain = data.frame(matrix(nrow = 1, ncol = length(provinces))) 
colnames(diff_population_prov_spain) = provinces
print("Population difference in every province in Spain in 2021 and 2020")
#print(df)

# Now we go through every column and substract each row (the population each
# of those years)
for (i in 1:length(spain_provinces_population_covid))
{
  # print(column_data_to_integer(spain_provinces_population_covid[1, i]))
  # print(column_data_to_integer(spain_provinces_population_covid[2, i]))
  diff = column_data_to_integer(spain_provinces_population_covid[1, i]) - 
    column_data_to_integer(spain_provinces_population_covid[2, i])
  diff_population_prov_spain[i] <- diff
  
}

print.data.frame(diff_population_prov_spain)

# We get the lowest population growth (or highest negative growth) like this:
min_population_growth <- apply(diff_population_prov_spain, 1, min, na.rm=TRUE)
prov_min_population_growth <- 
  colnames(diff_population_prov_spain)[apply(diff_population_prov_spain,1,
                                             which.min)]
print("")
print("Which was the lowest growth between 2020 and 2021?")
print(paste0("Lowest or negative growth was this number of people: ", 
             min_population_growth))
print("Which province?")
print(paste0("In the province of ", 
             prov_min_population_growth))
print("")
print("What about percentages?")
# In order to do so, first we define a function that works out percentages
percentage_calc <-function(part, whole){
  percentage <- 100 * as.numeric(part)/as.numeric(whole)
  result <- percentage
}

# Now we should create another empty dataframe that will contain the percentage
# of growth in every province
percentage_growth_population_prov_spain = 
  data.frame(matrix(nrow = 1, ncol = length(provinces))) 
colnames(percentage_growth_population_prov_spain) = provinces
#print(percentage_growth_population_prov_spain)
# Now we have to go through both the dataframe that contains the population
# in every province selecting year 2020 and the difference in population
for (i in 1:length(spain_provinces_population_covid)){
  whole = column_data_to_integer(spain_provinces_population_covid[2, i])
  part = diff_population_prov_spain[i]
  percentage_growth_province <- percentage_calc(part, whole)
  percentage_growth_population_prov_spain[i] <- percentage_growth_province
}
print("")
print("Growth percentage in every Spanish province between 2020 and 2021")
print.data.frame(percentage_growth_population_prov_spain)

# Now we have to get the column with the lowest value
min_population_growth_percentage <- 
  apply(percentage_growth_population_prov_spain, 1, min, na.rm=TRUE)
# And the column name with the lowest value
prov_min_population_growth_percentage <- 
  colnames(diff_population_prov_spain)[apply(
    percentage_growth_population_prov_spain,1, which.min)]
print("")
print(paste0("Lowest or negative percentage growth was: ", 
             min_population_growth_percentage))
print("Which province?")
print(paste0("In the province of ", 
             prov_min_population_growth_percentage))

print("")
print("Let's represent the population in every province in 2021")
# First, we create a vector with the population in 2021 in every province
vector_population_provinces_2021 <- 
  column_data_to_integer(rev(spain_provinces_population_covid[1,]))
# We remove the names of the rows
rownames(vector_population_provinces_2021)<-NULL
# We create now a dataset with where the provinces are a column and the 
# population in every province is another column
population_provinces_2021 <- data.frame(Provinces=provinces,
  Population=vector_population_provinces_2021)

# Because we have too many provinces, that will mean labels will collapse
# so we have to create a vector where only the names of certain provinces
# will be included

# First, we create an empty vector
labels_pie_chart <- c()

# Then we use the vector provinces that contains all Spanish provinces
# to get the indexes to add elements to the vector with the labels for the pie
# char but also to iterate through the vector vector_population_provinces_2021
# to check if the population in that province is larger than 1000000. If it is
# we add the name of the province to the vector of labels. If it is not we add
# an empty string
for(i in 1:length(provinces)){
  province <- ""
  if(vector_population_provinces_2021[i] > 1000000){
    province <- provinces[i]
  }
  labels_pie_chart[i] <- province
}

# Now we have the data to create the pie chart

# However, RStudio has a limitation. Color sets for charts don't have enough
# colours for 52 provinces
# So we need to create a palette
# First we get the number of colours we will need
colourCount = length(unique(population_provinces_2021$Provinces))
# Then, we use colorRampPalette to extend the palette Set1
getPalette = colorRampPalette(brewer.pal(9, "Set1"))

# We use the function pie to create the pue chart
# First parameter is the population (because it is what we are representing)
# Then, we add the vector with the labels to display
# Finally, the colours and the name
pie(population_provinces_2021$Population, labels_pie_chart, 
    col = getPalette(colourCount), 
    main="Population per province in 2021")
# Then, we add a legend
# inset displaces the legend so it does not collapse against the chart
legend("left", population_provinces_2021$Provinces, inset=c(0.7,-1), cex = 0.8,
       fill = getPalette(colourCount), ncol = 2, text.width=0.5, y.intersp=0.5)

# REGRESSION
print("Does the population in Seville follow a linear growth?")
# For starters, we create a dataset with the population in Seville every year
# except 1997 (as there is no data for it)
population_evolution_seville_reg <- population_evolution_seville[-2]
print(population_evolution_seville_reg)
years_reg <- years[-2]
regression_dataset <- data.frame(years_v = years_reg,
                                 population_v = population_evolution_seville_reg)
# Now we get if its linear correlation has acceptable values using cor
cor_coef <- cor(years_reg, population_evolution_seville_reg)
print(paste0("The correlation coef between the population and the year in the 
             province of seville is: ", 
             cor_coef))

if(cor_coef > 0.95){
  print("Acceptable linear correlation")
}else{
  print("No linear correlation")
}


if(cor_coef > 0.95){
  print("Let's create the model for linear regression")
  # We use the method lm to create a model for linear regression using our data
  # formula = column that will become y (values to predict) ~ column that will 
  # become x, data = dataset -> This dataset has to be our training set
  # The dataset with the data we will use to train the model, it has to contain
  # values x and y for every record in the dataset
  lm_reg_model <- 
    lm(formula = population_v ~ years_v, data = regression_dataset)
  print("We have created a new linear model for our data")
  print(lm_reg_model)
  # Now we will create our test dataset or vector
  # A test dataset contains the values in x but does not need values in y 
  # (as these ones are the predicted values)
  # In our case, x is the year so we create a new vector that contains all
  # years from 1996 (first year we have records) till 2030 (so we predict´
  # the population from 2022 to 2030)
  years_test <- 1996:2030
  # As we don't have data for 1997, we have to remove it from the test set
  # as we don't have this for our training set
  years_test <- years_test[-2]
  # We get use the model we create to predict the population between 1996 and
  # 2030 if the correlation was linear using the method predict
  # First parameter is the model and new data are values for the variable x
  predicted_population_seville <- predict(
    lm_reg_model, newdata = data.frame(years_v = years_test))
  print("Predictions for population in Seville")
  # We quickly create a dataset just to print the predictions we made with
  # the year
  dataset_predictions_population_seville <- data.frame(
    years = c(1996:2030)[-2], 
    population = predicted_population_seville)
  print(dataset_predictions_population_seville)
  # Now we represent the our predictions as a linear function against the real
  # values of population displayed as a point plot
  linear_reg_pop_seville_plot <- ggplot() + geom_point(aes(x = years_reg,
                            y = population_evolution_seville_reg), colour = 'red') +
    geom_line(aes(x = years_test,
                  y = predicted_population_seville), colour = 'blue') +
    
    ggtitle('Population in province of Seville prediction') +
    xlab('Year') +
    ylab('Population')
  print(linear_reg_pop_seville_plot)
  
}
