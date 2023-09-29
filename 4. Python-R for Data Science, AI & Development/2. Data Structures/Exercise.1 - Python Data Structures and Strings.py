#!/usr/bin/env python
# coding: utf-8
import os
import re
import statistics
from pdfquery import PDFQuery

# # Practice Exercise

# __<font color='red'>NOTE</font>__: This is just an exercise to
# practise data structures in Python. It is not a real problem. We will
# apply the previously defined __data science methodology__ to tackle
# the proposed problem. But both solution and problem are just an excuse
# to use basics and data structures of Python

# ### 1. Business Understanding

# As a university, we need to compare texts to identify keywords and
# topics. Especially in the case of ancient texts, we need a tool that
# analyses different texts and compares them to get statistics about
# topics of the texts, genders...

# ### 2. Analytical Approach

# We can build a tool where we analyse texts and provide statistics that
# can be saved for studies. Those statistics can be used to make changes
# in the texts, books, ... that are taught and identify possible biases

# ### 3. Data requirements

# We need a collection of public domain texts to test the application
# and start the statistics.
# We need pdfs of these texts to analyse them and get statistics
# We need to install the package: __pdfquery__

# ### 4. Data collection

# We can get those public domain texts and books from the library of
# the university.
# <font color='red'>For learning purposes </font> we will pretend that
# the url of the library of university is [wikisource]
# (https://en.wikisource.org/wiki/Main_Page)

# Those pdf files are located in the folder _books_

# ### 5. Data Understanding

# The pdf files contain texts from which we can extract information
# about the topics using most common words. We can extract the
# following information:
# - We can extract all the words contained in each text fragment
# - We would have to remove common words like "of", "the"...
# - We can get the frequency of each word in the text
# - We can compare two texts getting the words that are contained in
# both
# - Later, we can compare the frequency of the words in the two texts
# - We can get a set of all words contained in all our texts and count
# the number of different words (to check how rich vocabulary is)

# ### 6. Data Preparation

# We need to get the text from the pdf file to string

pdf = PDFQuery('./books/An_Account_of_the_Battle_of_Megiddo.pdf')
pdf.load()

# Use CSS-like selectors to locate the elements
text_elements = pdf.pq('LTTextLineHorizontal')

# Extract the text from the elements
text = [t.text for t in text_elements]

print(text)


# As you can see, _we have to extract the text page by page_ so we have
# to define a __function to join all the pages to the same string
# variable__

def join_pages_text_in_file(filepath: str) -> str:
    """
    Returns a string variable that contains all the text contained in
    the pdf file in the path

    :param filepath: Path to the pdf file whose text we want to turn
    into a string
    :type filepath: str
    :return: Text in the file in string format
    :rtype: str
    """

    pdf_file = PDFQuery(filepath)
    pdf_file.load()

    # Use CSS-like selectors to locate the elements
    text_elements_file = pdf_file.pq('LTTextLineHorizontal')
    # Extract the text from the elements
    file_text: str = ""
    for t in text_elements_file:
        file_text += t.text
    return file_text


print(join_pages_text_in_file(
    "./books/An_Account_of_the_Battle_of_Megiddo.pdf"))


# The strategy will be __splitting our text by " "__ as words are
# separated by spaces, that will transform our data _from text to a set
# where each word is an element_

# In order to use that strategy successfully, we need to __clean our
# data__ from other possible simbols or special characters that could
# change the result (for instance "." can be appended to a word making
# it a new register later

def remove_spaces_returns(text_to_clean: str) -> str:
    """
    Removes every possible character that might cause the same word
    to be considered two different words (special characters,
    commas, periods, numbers attached to a word, parenthesis...)

    :param text_to_clean: Text we want to clean to be explored later
    :type text_to_clean: str
    :return: Text without special characters, commas, periods, ...
    :rtype: str
    """

    upper_text: str = text_to_clean.upper()
    clean_text: str = upper_text.replace("\n", " ")
    clean_text = clean_text.replace("\\n", " ")
    clean_text = clean_text.replace(".", "")
    clean_text = clean_text.replace("\"", "")
    clean_text = clean_text.replace("'", "")
    clean_text = clean_text.replace(",", "")
    clean_text = clean_text.replace(":", "")
    clean_text = clean_text.replace(";", "")
    clean_text = clean_text.replace("\\[[0-9]*\\]", "")
    clean_text = clean_text.replace("(", "")
    clean_text = clean_text.replace(")", "")
    clean_text = clean_text.replace("'S", "")
    clean_text = clean_text.replace("-", "")
    clean_text = clean_text.replace("—", "")
    return clean_text


text = join_pages_text_in_file(
    "./books/An_Account_of_the_Battle_of_Megiddo.pdf")
print(remove_spaces_returns(text))


# So now we have to go __from a filepath to a list that contains in
# its elements all the words__ contained within the text. We have to:
# 1. Open the file and join all paragraphs ->
# __join_pages_text_in_file__
# 2. Clean the text of special characters and symbol, so it can be split
# by empty spaces to get the words <- __remove_spaces_returns__
# 3. Get the words in the paragraph <- _Function that will be described
# below_ (will contain the other two steps too)

def get_words_in_text(filepath: str) -> [str]:
    """
    Returns all the words in the text contained in the pdf file
    in the form of a list

    :param filepath: Path to the pdf file
    :type filepath: str
    :return: A list that contains all the words in the text
    :rtype: str
    """

    text_file: str = join_pages_text_in_file(filepath)
    cleaned_text: str = remove_spaces_returns(text_file)
    words_in_text: [str] = cleaned_text.split(" ")
    return words_in_text


print(get_words_in_text("./books/An_Account_of_the_Battle_of_Megiddo.pdf"))


# Finally, as words can be repeated, it would be useful to get a set
# with each word just once (not repeated)

def get_unique_words_in_pdf(filepath: str) -> {str}:
    """
    Returns a set that only contains the words contained in the text
    in the pdf file once (as elements in sets can only appear once)

    :param filepath:
    :return: A set that contains all the words that appear in the text
    in the pdf file
    :rtype: {str}
    """

    words_in_text: [str] = get_words_in_text(filepath)
    unique_words: {str} = set(words_in_text)
    # We have to remove the element "" (empty string) in case it
    # was added
    try:
        unique_words.remove("")
    except KeyError:
        # If it was not part of the set, we move on
        pass
    return unique_words


get_unique_words_in_pdf("./books/An_Account_of_the_Battle_of_Megiddo.pdf")


# ##### Now, we define several functions to get statistics on our texts

# 1. We define a function that returns the __frequency__ of each word
# in text

def get_frequency_words_in_text(filepath: str) -> {str: int}:
    """
    Returns a dictionary where the key is the word and the value
    the amount of times that word appears in the text

    :param filepath: Path to the pdf file with the text
    :type filepath: str
    :return: A dictionary with word: number of appearances
    :rtype: {str: int}
    """

    # We first get a set with the words in the text (so words only
    # appear once)
    unique_words: {str} = get_unique_words_in_pdf(filepath)

    # Now we get the full text from the file and we clean it
    full_text: str = join_pages_text_in_file(filepath)
    cleaned_text: str = remove_spaces_returns(full_text)

    # We create an empty dictionary that will contain the frequencies
    frequencies: dict = {}
    # We go through every word in the set (in the text)
    for unique_word in unique_words:
        # And search how many times it appears in the text
        frequency: int = len([el.start() for el in re.finditer(unique_word,
                                                               cleaned_text)])
        if frequency > 0:
            # And add it to the dictionary
            frequencies[unique_word] = frequency

    # We sort the dictionary by value (most frequent words first)
    # and we return it
    sorted_frequencies: [tuple] = sorted(frequencies.items(),
                                         key=lambda item: item[1],
                                         reverse=True)
    sorted_frequencies_dict = {}
    for key, value in sorted_frequencies:
        sorted_frequencies_dict[key] = value
    return sorted_frequencies_dict


get_frequency_words_in_text("./books/An_Account_of_the_Battle_of_Megiddo.pdf")


# 2. A function that return a list of words that are __contained in
# every text in the list__:

def get_words_in_every_text_list_of_files(list_texts: [str]) -> {str}:
    """
    Returns a set that contains each word that appears in each text
    in the list of pdf files
    
    :param list_texts: List of the path to the pdf file we want to 
    analyze
    :type list_texts: [str]
    :return: A set with each word that appears in those texts
    :rtype: {str}
    """

    unique_words_in_files: {str} = set()
    for file_text in list_texts:
        if type(file_text) == str:
            unique_words_text = get_unique_words_in_pdf(file_text)
            unique_words_in_files = unique_words_in_files.union(
                unique_words_text)

    return sorted(unique_words_in_files)


get_words_in_every_text_list_of_files(
    ["./books/An_Account_of_the_Battle_of_Megiddo.pdf",
     "./books/Le_Morte_d'Arthur_Volume_I_Book_I_Chapter_I.pdf"])


# 3. Function that will return __words that are in text 1 but not
# in text 2__

def words_in_text_a_not_in_text_b(file_path1: str, file_path2: str) -> {str}:
    """
    Returns a set that contains only the words in the first text that
    don't appear in the second text
    
    :param file_path1: Path to the first pdf file
    :param file_path2: Path to the second pdf file (the one whose words
    we will remove from the final set)    
    :return: A set that contains only the words in the first text that
    don't appear in the second text
    :rtype: {str}
    """
    unique_words_text1: {str} = get_unique_words_in_pdf(file_path1)
    unique_words_text2: {str} = get_unique_words_in_pdf(file_path2)

    words_in_text1_not_in_text2: {str} = unique_words_text1.difference(
        unique_words_text2)
    return words_in_text1_not_in_text2


words_in_text_a_not_in_text_b(
    "./books/An_Account_of_the_Battle_of_Megiddo.pdf",
    "./books/Le_Morte_d'Arthur_Volume_I_Book_I_Chapter_I.pdf")


# 4. A function to find the __frequency of a word in a text__

def find_frequency_of_word_in_text(filepath: str, word: str) -> int:
    """
    Return the number of times a word appears in the text in the file
    
    :param filepath: Path to the pdf file we want to analyze
    :type filepath: str
    :param word: Word we are searching for
    :type word: str
    
    :return: Number of times the word appears in the text
    :rtype: int 
    """

    text_in_file: str = join_pages_text_in_file(filepath)
    cleaned_text: str = remove_spaces_returns(text_in_file)
    times: int = len([el.start() for el in re.finditer(word, cleaned_text)])
    return times


find_frequency_of_word_in_text(
    "./books/An_Account_of_the_Battle_of_Megiddo.pdf", "I")


# 5. A function to __compare the frequency of words__ between two texts:

def compare_frequency_of_common_words(file_path1: str, file_path2: str) -> \
        dict:
    """
    Returns a dictionary whose keys are the words present in both texts
    and value is True if the word appears more times in the first text
    or False otherwise

    :param file_path1: Path to the first PDF file
    :param file_path2: Path to the second PDF file
    :return: Dictionary described above
    :rtype: dict
    """

    freq_words_text1: {str: int} = get_frequency_words_in_text(file_path1)
    freq_words_text2: {str: int} = get_frequency_words_in_text(file_path2)
    common_words: {str} = set(freq_words_text1.keys()) & set(
        freq_words_text2.keys())
    more_freq_in_1: {str: bool} = {}

    for word in common_words:
        freq_word_1: int = freq_words_text1[word]
        freq_word_2: int = freq_words_text2[word]
        more_freq_in_1[word] = freq_word_1 > freq_word_2

    sorted_words: [tuple] = sorted(more_freq_in_1.items(),
                                   key=lambda item: item[0])
    sorted_words_dict = {}
    for key, value in sorted_words:
        sorted_words_dict[key] = value

    return sorted_words_dict


compare_frequency_of_common_words(
    "./books/An_Account_of_the_Battle_of_Megiddo.pdf",
    "./books/Le_Morte_d'Arthur_Volume_I_Book_I_Chapter_I.pdf")


# 6. Function to return the words that __appear more frequently than
# average__:


def return_words_more_common_than_mean(filepath: str) -> {str}:
    """
    Return a set that contains the words that appear more times than
    average in the text contained in the file

    :param filepath: Path to the pdf file
    :type filepath: str
    :return: Set with the words that appear more times than average
    :rtype: {str}
    """

    freq_words_text: dict = get_frequency_words_in_text(filepath)
    mean_frequency: float = statistics.mean(freq_words_text.values())
    print(
        "The average/mean frequency of a word in this text is {0}".format(
            mean_frequency))
    result_words: [tuple] = [item for item in sorted(freq_words_text.items(),
                                                     key=lambda item: item[0])
                             if item[1] > mean_frequency]
    return result_words


# In[22]:


return_words_more_common_than_mean(
    "./books/Le_Morte_d'Arthur_Volume_I_Book_I_Chapter_I.pdf")


# 7. Function that returns the __most frequent and least frequent words
# in a text__:


def get_most_and_least_frequent_word_in_text(filepath: str) -> []:
    """
    Returns a list with two dictionaries that contains as key the
    frequency of the words in value. First element contains dictionary
    with the most frequent words and the second the one with the least
    frequent
    
    :param filepath: Path to the pdf file
    :type filepath: str
    :return: Returns a list with two dictionaries that contains as key
    the
    frequency of the words in value
    :rtype: []
    """

    freq_words_text: dict = get_frequency_words_in_text(filepath)
    lowest_index: int = len(freq_words_text.keys()) - 1
    least_frequent_words: [str] = [item[0] for item in freq_words_text.items()
                                   if item[1] == freq_words_text[
                                       list(freq_words_text.keys())[
                                           lowest_index]]]
    most_frequent_words: [str] = [item[0] for item in freq_words_text.items()
                                  if item[1] == freq_words_text[
                                      list(freq_words_text.keys())[0]]]

    print("Most frequent words appear {0} times and are: {1}".format(
        freq_words_text[list(freq_words_text.keys())[0]], most_frequent_words))
    print("Lest frequent words appear {0} times and are: {1}".format(
        freq_words_text[list(freq_words_text.keys())[lowest_index]],
        least_frequent_words))

    return [{freq_words_text[
                 list(freq_words_text.keys())[0]]: most_frequent_words}, {
                freq_words_text[list(freq_words_text.keys())[
                    lowest_index]]: least_frequent_words}]


get_most_and_least_frequent_word_in_text(
    "./books/Le_Morte_d'Arthur_Volume_I_Book_I_Chapter_I.pdf")

# ### 7. Modelling

# We will design an interactive program where we will list the files
# in the directory __books__ that contains the pdf and then offer the
# different methods

print("Welcome to BOOKS DATA:")
print("We can perform the following operations: ")
print("1. Get frequency of the words a the text")
print("2. Get every word in the files in our current directory")
print("3. Get words in one text but not in another one")
print("4. Find how many times a word appears in a text")
print("5. Compare the frequency of common words in two files")
print(
    "6. Return the words that are more common than average in a certain text")
print("7. Get the most and the least frequent word in a text")
function_picked: str = str(
    input("Give me the number of the function you want to perform: "))

print("Welcome to BOOKS DATA:")
print("This are the books we can analyze in our directory:")

book_files: [str] = os.listdir(os.getcwd() + '/books')
for i, file in enumerate(book_files):
    print("{0}. - {1}".format(i, file))

file_choice: int = int(
    input("Give me the index of the file you want to explore: "))

chosen_file = book_files[file_choice]
print("The file chosen was {0}".format(chosen_file))
filepath1: str = os.getcwd() + '/books/' + chosen_file

if function_picked == "1":
    print("Frequency of words in the text")
    print(get_frequency_words_in_text(filepath1))
elif function_picked == "2":
    chosen_files: [str] = [filepath1]
    file_choice: int = -2
    while file_choice != -1:
        file_choice: int = int(input(
            "Pick another file, give me the index of the file you want "
            "to explore or select -1: "))
        if file_choice == -1:
            break
        chosen_file: str = book_files[file_choice]
        filepath2: str = os.getcwd() + '/books/' + chosen_file
        chosen_files.append(filepath2)
    print("Let's perform the operation them")
    print("The words in every file in the list are:")
    print(get_words_in_every_text_list_of_files(chosen_files))
elif function_picked == "3":
    file_choice: int = int(input(
        "Pick another file, give me the index of the file you want to "
        "explore: "))
    chosen_file2: str = book_files[file_choice]
    filepath2: str = os.getcwd() + '/books/' + chosen_file2
    print("The words in first file we selected but not in the second are:")
    print(words_in_text_a_not_in_text_b(filepath1, filepath2))
elif function_picked == "4":
    chosen_word = str(
        input("Give me a word whose frequency we can search for: "))
    print("The frequency of that word in the text is: ")
    print(find_frequency_of_word_in_text(filepath1, chosen_word))
elif function_picked == "5":
    file_choice: int = int(input(
        "Pick another file, give me the index of the file you want to "
        "explore: "))
    chosen_file2: str = book_files[file_choice]
    filepath2: str = os.getcwd() + '/books/' + chosen_file2
    print(
        "The words in first file we selected that are more frequent than "
        "in the second are:")
    print(compare_frequency_of_common_words(filepath1, filepath2))
elif function_picked == "6":
    print("Returning words that appear more than the mean in the text: ")
    print(return_words_more_common_than_mean(filepath1))
elif function_picked == "7":
    print("Most and least frequent words in the selected text are: ")
    print(get_most_and_least_frequent_word_in_text(filepath1))
