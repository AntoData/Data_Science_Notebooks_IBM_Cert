import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats

"""
Datasets columns are:

track_name: The name of the track.
artist(s)_name: The name(s) of the artist(s) who created the track.
artist_count: The number of artists associated with the track.
released_year: The year when the track was released.
released_month: The month when the track was released.
released_day: The day when the track was released.
in_spotify_playlists: Indicates whether the track is included in Spotify 
playlists.
in_spotify_charts: Indicates whether the track is present in Spotify 
charts.
streams: The total number of streams the track has accumulated.
in_apple_playlists: Indicates whether the track is included in Apple 
Music playlists.
in_apple_charts: Indicates whether the track is present in Apple Music 
charts.
in_deezer_playlists: Indicates whether the track is included in Deezer 
playlists.
in_deezer_charts: Indicates whether the track is present in Deezer 
charts.
in_shazam_charts: Indicates whether the track is present in Shazam 
charts.
bpm: Beats per minute - a measure of tempo in music.
key: The musical key in which the track is composed.
mode: Indicates whether the track is in a major or minor key.
danceability_: A measure of how suitable a track is for dancing.
valence_: The musical positiveness conveyed by a track.
energy_: The perceived energy of a track.
acousticness_: A measure of how acoustic a track is.
instrumentalness_: A measure of whether a track contains vocals.
liveness_speechiness_: A measure of presence of live elements or spoken 
words in a track.
"""

df: pd.DataFrame = pd.read_csv('Popular_Spotify_Songs.csv',
                               encoding="ISO-8859-1")
print("Dataframe columns: ")
print(df.columns)
print("")

# We will keep the fields: track_name, artist(s)_name, released_year,
# in_spotify_playlists, bpm, key, mode, danceability_

df_tests: pd.DataFrame = df[['track_name', 'artist(s)_name',
                             'released_year', 'in_spotify_playlists',
                             'bpm', 'key', 'mode', 'danceability_%',
                             'streams']]

# Let´s rename artist name and danceability columns
# NOTE: inplace = True returns a warning here
df_tests: pd.DataFrame = df_tests.rename(
    columns={"artist(s)_name": "artist", "danceability_%": "danceability"})
print("Subset and renamed dataset")
print(df_tests.columns)
print("")

# Let's get some descriptive statistics of numerical columns
print("Results of describe()")
print(df_tests.describe())
print("")

# For instance:
# That the average year of the songs is 2018 means probably most of them
# are close to 2018
# However, if we look at the quarters 25&=2020-2022=75% at least %50
# are between 2020 and 2022
# There are 953 rows or songs

# Let's get some descriptive statistics of all columns
print("Results of describe including categorical variables")
print(df_tests.describe(include='all'))
print("")

# Here we can see that there are 645 different artists in this dataset
# Taylor Swift is the most frequent artist
# She appears 34 times (we assume that with 34 songs)
# There are 943 songs with a unique name, apparently the max times a
# song title is repeated is 2
# Most songs are in key C#, 120

# Let's count now categorical variables like song titles and artists
print("Counting song + artist combinations")
print(df_tests[['track_name', 'artist']].value_counts())
print("")

# We can see that some songs are repeated twice, same artist, same song
# For instance About Damn Time by Lizzo

# Let's count now how many songs in every key and mode
print("Number of songs in every key and mode")
print(df_tests[['key', 'mode']].value_counts())
print("")

# Let's make a box plot to compare the distribution of bpm in every key
sns.boxplot(data=df_tests, x='key', y='bpm')
plt.show()

# It looks like F# has the biggest difference but G and A# are the only
# ones to have outliers

# Let's make now a box plot to see the distribution of the bpm of a
# subgroup of the 5 more frequent

# First, let's get the top 5 artist

df_artists_sorted_by_frequency: pd.DataFrame = \
    df_tests[["artist", "track_name"]].groupby(
        ["artist"]).count().sort_values(by=["track_name"], ascending=False)

top_5_artists: pd.DataFrame = \
    df_artists_sorted_by_frequency.head().reset_index()["artist"]
print("Top 5 artists")
print(top_5_artists)
print("")

df_top_5_artists: pd.DataFrame = \
    df_tests[df_tests['artist'].isin(top_5_artists)]

sns.boxplot(data=df_top_5_artists, x='artist', y='bpm')
plt.show()

# We are going to create a line plot where the 10 most streams songs are
# displayed
# We will display the position in the list in x and the streams of the
# song in that position in y

# For starters let´s convert the column streams to integer, first
# removing all non-numeric values
df_tests = df_tests[pd.to_numeric(df_tests['streams'],
                                  errors='coerce').notnull()]
df_tests['streams'] = df_tests['streams'].astype('int64')

# Now let´s sort the data set by number of streams from more to less
print("Most streamed songs")
print(df_tests.sort_values(by='streams', ascending=False))
print("")
df_top_10_most_streamed_songs: pd.DataFrame = \
    df_tests.sort_values(by='streams', ascending=False).head(10)
print(df_top_10_most_streamed_songs)

# Now we use plot to display a series of number in x-axis and the number
# of streams in y-axis
plt.plot(list([x for x in range(1, 11)]),
         df_top_10_most_streamed_songs['streams'])
plt.xlabel("Chart position")
plt.ylabel("Streams in billions")
plt.show()

# Now let´s display a scatter plot of bpms vs streams
plt.scatter(x=df_tests['bpm'], y=df_tests['streams'])
plt.xlabel("bpm")
plt.ylabel("streams")
plt.show()

# Now let´s group danceability in 10 different ranges and diplay the number of
# streams in total of those groups

# First, we get danceability and the streams
df_g: pd.DataFrame = df_tests[["danceability", "streams"]]

# We display 10 ranges of streams and danceability in histogram
plt.hist(x=df_g["danceability"], bins=10)
plt.xlabel("Danceability")
plt.ylabel("Streams")
plt.title("Danceability vs Streams")
plt.show()

# We display the streams of the top 5 artist in a bar plot

# First, we get to get the variables artists and streams, we need to
# group by artist to use sum() to get the total of streams of every
# artist
df_top_5_artists_by_streams: pd.DataFrame = \
    df_tests[['artist', 'streams']].groupby(by='artist').sum()
df_top_5_artists_by_streams = \
    df_top_5_artists_by_streams.sort_values(by='streams',
                                            ascending=False).head()
df_top_5_artists_by_streams.reset_index(inplace=True)
print("Top 5 artists by number of strams")
print(df_top_5_artists_by_streams)
print("")

bar_colors = ['tab:red', 'tab:pink', 'tab:orange', 'tab:grey',
              'tab:blue']

plt.bar(x=df_top_5_artists_by_streams['artist'],
        height=df_top_5_artists_by_streams['streams'], color=bar_colors)
plt.xlabel("Artist")
plt.ylabel("Streams")
plt.title("Top 5 artists")
plt.show()

# We want to use a pseudo color plot to display the number of songs in
# every key and mode

# For starters, we need to count the number of songs in every
# combination of key and mode, then we count the number of values of
# each combination
df_key_mode: pd.DataFrame = \
    df_tests[["key", "mode"]].groupby(by=["key", "mode"]).value_counts()
# We turn this series into a matrix by resetting the index
# NOTE: We can´t use inplace=True because it changes the type of the
# variable from series to dataframe, we need to assign the result to a
# variable
df_key_mode = df_key_mode.reset_index()

# Now we use pivot to get a pivot table where x is the key (index) and
# y is the mode (columns) the value of each cell is the number of songs
# with that combination of key and mode
df_key_mode_pivot: pd.DataFrame = \
    df_key_mode.pivot(index="key", columns="mode")

# We can use that matrix to display a pseudo color plot using pcolor
plt.pcolor(df_key_mode_pivot)
plt.show()

sns.heatmap(df_key_mode_pivot, cmap="RdBu")
plt.show()

# We will use now a regplot to display the same variables as when
# we displayed the scatter plot
sns.regplot(x="bpm", y="streams", data=df_tests)
plt.show()

# Now, let's use the same variables in a residual plot
sns.residplot(x="bpm", y="streams", data=df_tests)
plt.show()

# Let´s display a KDE plot to see if the distribution of number of songs
# and danceability is close to the normal distribution
sns.kdeplot(x="danceability", data=df_tests)
plt.show()
# It actually matches that distribution but songs with more danceability
# tend to be a little more frequent

sns.displot(x="danceability", data=df_tests, kde=True)
plt.show()

# Let´s print the correlation coefficient between all numeric variables
print(df_tests[["released_year", "streams", "bpm", "danceability"]].corr())
# We can see there is not much correlation between variables

scipy.stats.pearsonr()