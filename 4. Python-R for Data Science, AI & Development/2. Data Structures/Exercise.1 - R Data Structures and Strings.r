install.packages("pdftools")

library(stringr)

var <- "./books/Le_Morte_d'Arthur_Volume_I_Book_I_Chapter_I.pdf"

library(pdftools)
text <- pdftools::pdf_text(pdf = "./books/Le_Morte_d'Arthur_Volume_I_Book_I_Chapter_I.pdf")
typeof(text)

print(text)

join_paragraphs <- function(filepath){
    # Given a pdf file path, we will open the file. The result is a vector that contains the different paragraphs in each element. 
    # We will join each paragraph to the same text variable"
    text_array <- pdftools::pdf_text(pdf = filepath)
    text <- "\n "
    for(paragraph in text_array){
        text <- paste(text, paragraph)
    }
    return(text)
}

print(join_paragraphs("./books/Le_Morte_d'Arthur_Volume_I_Book_I_Chapter_I.pdf"))

remove_spaces_returns <- function(text_to_clean){
    # Given a text, it removes special characters like '\n' or '.' and replaces them 
    # by an empty space (so we can split the text) later
    upper_text <- toupper(text_to_clean)
    clean_text <- stringr::str_replace_all(upper_text, "\n", " ")
    clean_text <- stringr::str_replace_all(clean_text, "\\n", " ")
    clean_text <- stringr::str_replace_all(clean_text, "\\.", "")
    clean_text <- stringr::str_replace_all(clean_text, ",", "")
    clean_text <- stringr::str_replace_all(clean_text, ":", "")
    clean_text <- stringr::str_replace_all(clean_text, ";", "")
    clean_text <- stringr::str_replace_all(clean_text, "\\[[0-9]*\\]", "")
    clean_text <- stringr::str_replace_all(clean_text, "\\(", "")
    clean_text <- stringr::str_replace_all(clean_text, "\\)", "")
    clean_text <- stringr::str_replace_all(clean_text, "'s", "")
    clean_text <- stringr::str_replace_all(clean_text, "-", "")
    clean_text <- stringr::str_replace_all(clean_text, "—", "")
    return(clean_text)
}

get_words_in_text <- function(filepath){
    # Given a pdf file path, it returns all the words contained in the text as elements in a vector
    text <- join_paragraphs(filepath)
    cleaned_text <- remove_spaces_returns(text)
    words_in_text <- unlist(strsplit(cleaned_text, " "))
    # We need to remove "" elements in the result
    words_vector <- unlist(words_in_text[words_in_text != ""])
    return(words_vector)
}

get_words_in_text("./books/Le_Morte_d'Arthur_Volume_I_Book_I_Chapter_I.pdf")

get_unique_words_in_pdf <- function(filepath){
    # Given a pdf file path, it returns a vector that contains the words in the text without repetitions
    words_in_text <- get_words_in_text(filepath)
    unique_words <- unique(words_in_text)
    return(unique_words)
}

print(get_unique_words_in_pdf("./books/Le_Morte_d'Arthur_Volume_I_Book_I_Chapter_I.pdf"))

get_frequency_words_in_text <- function(filepath){
    # Given a pdf file path, it returns the frequency of each word (number of times that a word appears in the text)
    
    # First, we need to get the words that appear in the text without repetitions
    unique_words <- get_unique_words_in_pdf(filepath)
    # Now, we get the text in the pdf file (first we open it, then we join the different parts in a string)
    full_text <- join_paragraphs(filepath)
    # We clean our data and turn the string to upper case
    full_text <- remove_spaces_returns(full_text)
    full_text <- toupper(full_text)
    
    # We create an empty vector that will contain the word as name and how many times appears in the text as value
    frequencies <- c()
    i <- 1
    for(word_vector in unique_words){
        freq <- length(unlist(gregexpr(word_vector, full_text)))
        frequencies <- append(frequencies, freq)
        names(frequencies)[i] <- word_vector
        i <- i + 1
    }
    # We order the vector by value from higher to lower frequency
    frequencies <- frequencies[order(-frequencies)]
    return(frequencies)
}

get_frequency_words_in_text("./books/Le_Morte_d'Arthur_Volume_I_Book_I_Chapter_I.pdf")

get_words_in_every_text_list_of_files <- function(list_texts){
    # Given a list of pdf file paths, returns all the words that are contained in every text and without repetitions
    # The word has to be contained in every text in the list
    unique_words_in_files <- c()
    i <- 1
    # We iterate through the file paths
    for(file_text in list_texts){
        if(typeof(file_text) == 'character'){
            # We get the unique words contained in that file
            unique_words_text <- get_unique_words_in_pdf(file_text)
            # If this is the first time we are performing this operation, we add the vector with the unique words in this file
            # to the vector that will contain the result
            if(i == 1){
                unique_words_in_files <- append(unique_words_in_files, unique_words_text)
                i <- i + 1
            }else{
                # Otherwise, we use intersect to return only words that are contained in both texts
                unique_words_in_files <- intersect(unique_words_in_files, unique_words_text)
            }
        }else{
            stop("Input parameter has to be a list/vector of character variables")
        }
    }
    return(unique_words_in_files)
}

var1 <- "./books/Le_Morte_d'Arthur_Volume_I_Book_I_Chapter_I.pdf"
var2 <- "./books/An_Account_of_the_Battle_of_Megiddo.pdf"
list_files <- list(var1, var2)
words_in_texts <- get_words_in_every_text_list_of_files(list_files)

print(words_in_texts)

var_text <- get_unique_words_in_pdf("./books/An_Account_of_the_Battle_of_Megiddo.pdf")

length(var_text)

var_text

get_frequency_words_in_text("./books/An_Account_of_the_Battle_of_Megiddo.pdf")

words_in_text_a_not_in_text_b <- function(filepath_text1, filepath_text2){
    # Given a two pdf file paths, this function will return the words contain in the first file but not in the second
    unique_words_text1 <- get_unique_words_in_pdf(filepath_text1)
    unique_words_text2 <- get_unique_words_in_pdf(filepath_text2)
    words_in_text1_not_in_text2 <- setdiff(unique_words_text1, unique_words_text2)
    return(words_in_text1_not_in_text2)
}

words_in_text_a_not_in_text_b(var1, var2)

find_frequency_of_word_in_text <- function(filepath, word){
    # Given a pdf file path and a word, this function returns how many times this word appears in the text
    text_in_file <- pdftools::pdf_text(pdf = filepath)
    found_times <- 0
    for(paragraph in text_in_file){
        list_found_indexes = stringr::str_locate_all(paragraph, word)
        found_times <- found_times + length(unlist(list_found_indexes))
    }
    return(found_times)
}

print(find_frequency_of_word_in_text(var1, "I"))

compare_frequency_of_common_words <- function(filepath1, filepath2){
    # Given two pdf file paths, it will return a vector with the common words in both text as names. If the value is True,
    # it means the words is more frequent in the first text and if it is false, it means the word is more frequent in 
    # the second text
    freq_words_text1 <- get_frequency_words_in_text(filepath1)
    freq_words_text2 <- get_frequency_words_in_text(filepath2)
    common_words <- intersect(names(freq_words_text1), names(freq_words_text2))
    more_freq_in_1 <- c()
    for(word in common_words){
        freq_word_1 <- as.integer(freq_words_text1[word])
        freq_word_2 <- as.integer(freq_words_text2[word])
        freq_word_diff <- freq_word_1 - freq_word_2
        if(freq_word_diff > 0){
            freq_word_diff <- as.logical(freq_word_diff)
        }else{
            freq_word_diff <- FALSE
        }
        more_freq_in_1 <- append(more_freq_in_1, freq_word_diff)
    }
    names(more_freq_in_1) <- common_words
    more_freq_in_1 <- more_freq_in_1[sort(names(more_freq_in_1))]
    return(more_freq_in_1)
}

compare_frequency_of_common_words(var1, var2)

return_words_more_common_than_mean <- function(filepath){
    # Given a pdf file path, it returns a vector with only that words that appearç
    # more times than the average times a word appears in the text in the file
    freq_words_text <- get_frequency_words_in_text(filepath)
    mean_freq <- mean(freq_words_text)
    print(sprintf("The average/mean frequency of a word in this text is %f", mean_freq))
    result_words <- which(freq_words_text > mean_freq)
    return <- result_words
}

freq <- return_words_more_common_than_mean(var1)
names(freq)

print(return_words_more_common_than_mean(var2))

get_most_and_least_frequent_word_in_text <- function(filepath){
    # Given a pdf file path, it returns the words that appears the least and the most number of times
    # (we say words because several words might appear the same number times and those could be the most
    # and least frequent in the text)
    freq_words_text <- get_frequency_words_in_text(filepath)
    sorted_freq <- freq_words_text[order(freq_words_text)]
    sorted_freq <- sorted_freq[-(length(sorted_freq))]
    result_list <- list()
    least_freq_list <- list(frequency = unname(sorted_freq[1]), words = names(sorted_freq[sorted_freq == sorted_freq[1]]))
    most_freq_list <- list(frequency = unname(sorted_freq[length(sorted_freq)]), words = names(sorted_freq[sorted_freq == sorted_freq[length(sorted_freq)]]))
    result_list <- append(result_list, least_freq_list)
    result_list <- append(result_list, most_freq_list)
    least_frequent_message <- sprintf("The lowest frequency of a word in the text is %d by words: %s", least_freq_list$frequency, paste(least_freq_list$word, collapse = ", "))
    print(least_frequent_message)
    most_frequent_message <- sprintf("The highest frequency of a word in the text is %d by words: %s", most_freq_list$frequency, paste(most_freq_list$word, collapse = ", "))
    print(most_frequent_message)
    return(result_list)
}

print(get_most_and_least_frequent_word_in_text(var1))

print("Welcome to BOOKS DATA:")
print("We can perform the following operations: ")
print("1. Get frequency of the words a the text")
print("2. Get every word in the files in our current directory")
print("3. Get words in one text but not in another one")
print("4. Find how many times a word appears in a text")
print("5. Compare the frequency of common words in two files")
print("6. Return the words that are more common than average in a certain text")
print("7. Get the most and the least frequent word in a text")
function_picked <- readline("Give me the number of the function you want to perform")

print("Welcome to BOOKS DATA:")
print("This are the books we can analyze in our directory:")

files_list <- unlist(list.files(path=paste0(as.character(getwd()),"/books")))
files_index <- c()
i <- 1
for(filepath in files_list){
    print(paste0(i, ": ", filepath))
    i <- i + 1
}

file_choice <- as.numeric(readline("Give me the index of the file you want to explore: "))
chosen_file <- files_list[file_choice]
print(sprintf("The file chosen was '%s'", chosen_file))

if(function_picked == "1"){
    print("Frequency of words in the text")
    print(get_frequency_words_in_text(paste0(as.character(getwd()),"/books/",chosen_file)))
}else if(function_picked == "2"){
    chosen_files <- c()
    chosen_file <- paste0(as.character(getwd()),"/books/",chosen_file)
    chosen_files <- append(chosen_files, chosen_file)
    file_choice <- "-1"
    while(file_choice != "0"){
        file_choice <- as.numeric(readline("Pick another file, give me the index of the file you want to explore or select 0: "))
        if(file_choice == "0"){
            break
        }
        chosen_file <- files_list[file_choice]
        chosen_file <- paste0(as.character(getwd()),"/books/",chosen_file)
        chosen_files <- append(chosen_files, chosen_file)
        print(sprintf("The file chosen was '%s'", chosen_file))
    }
    print("Let's perform the operation them")
    print("The words in every file in the list are:")
    print(get_words_in_every_text_list_of_files(chosen_files))
}else if(function_picked == "3"){
    file_choice <- as.numeric(readline("Pick another file, give me the index of the file you want to explore: "))
    chosen_file2 <- files_list[file_choice]
    chosen_file <- paste0(as.character(getwd()),"/books/",chosen_file)
    chosen_file2 <- paste0(as.character(getwd()),"/books/",chosen_file2)
    print("The words in first file we selected but not in the second are:")
    print(words_in_text_a_not_in_text_b(chosen_file, chosen_file2))
}else if(function_picked == "4"){
    chosen_word <- as.character(readline("Give me a word whose frequency we can search for: "))
    chosen_file <- paste0(as.character(getwd()),"/books/",chosen_file)
    print("The frequency of that word in the text is: ")
    print(find_frequency_of_word_in_text(chosen_file, chosen_word))
}else if(function_picked == "5"){
    file_choice <- as.numeric(readline("Pick another file, give me the index of the file you want to explore: "))
    chosen_file2 <- files_list[file_choice]
    chosen_file <- paste0(as.character(getwd()),"/books/",chosen_file)
    chosen_file2 <- paste0(as.character(getwd()),"/books/",chosen_file2)
    print("The words in first file we selected but not in the second are:")
    print(compare_frequency_of_common_words(chosen_file, chosen_file2))
}else if(function_picked == "6"){
    chosen_file <- paste0(as.character(getwd()),"/books/",chosen_file)
    print("Returning words that appear more than the mean in the text: ")
    print(return_words_more_common_than_mean(chosen_file))
}else if(function_picked == "7"){
    chosen_file <- paste0(as.character(getwd()),"/books/",chosen_file)
    print("Most and least frequent words in the selected text are: ")
    print(get_most_and_least_frequent_word_in_text(chosen_file))
}


