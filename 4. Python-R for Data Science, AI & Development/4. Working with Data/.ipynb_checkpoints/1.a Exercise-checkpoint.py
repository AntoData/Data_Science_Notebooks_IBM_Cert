#!/usr/bin/env python
# coding: utf-8
import datetime
import shutil

# #### Exercise 1:
# 
# We have this file where words after a period are not capitalized. Copy
# and correct that file in just one iteration, then without closing the
# file and creating a new stream test that the book has correctly copied
# and fixed.
# 
# __File name__: file_exercise1.txt

# We first create a generator of names for our files
file_name: str = "./file_example_{0}.txt".format(int(datetime.datetime.now().
                                                     timestamp()))

# This array will control if we could open the original file and the
# file who will contain the copy of the file capitalized
opened_files: [bool] = [False, False]
try:
    # We create the file that will contain the capitalized copy, we need
    # mode w+
    # In mode w+, the file does not have to exist (so it will be
    # created) and the file can be read
    # so we can test that the copy is correct without closing the file
    file_copy = open(file_name, mode="w+")
    opened_files[0] = True
    # We open the original file in mode r (so it is protected against
    # being written)
    file_original = open("file_exercise1.txt", mode="r")
    opened_files[1] = True
    # We iterate through the lines of the original file
    for line in file_original:
        # We split all the sentences (sentences are separated by .), we
        # have to capitalize the word after .
        sentences: [str] = line.split(".")
        # We go through all the sentences in this line
        for count, sentence in enumerate(sentences):
            # If the first character is a blank space, we have to
            # capitalize the following character
            if sentence[0] == " ":
                capital_letter: str = sentence[1].upper()
                # If count == 0, it means it is the first sentence in
                # the line, so no blank space after . is needed
                if count == 0:
                    capitalized_sentence: str = capital_letter + sentence[2:]
                else:
                    # Otherwise, we need a blank space after . is needed
                    capitalized_sentence: str = ". " + capital_letter + \
                                                sentence[2:]
            else:
                # Otherwise, we capitalize the first character
                capital_letter: str = sentence[0].upper()
                if count == 0:
                    capitalized_sentence: str = capital_letter + sentence[1:]
                else:
                    capitalized_sentence: str = ". " + capital_letter + \
                                                sentence[1:]
            # Now we write the capitalized sentence to the copy file
            file_copy.write(capitalized_sentence)
except IndexError as e:
    # We raise an IndexError when we get to the end of the file, so the
    # line's length is 0
    print("End of file: {0}".format(e))
except FileNotFoundError as e:
    print("One of the files was not found, could not be found")
else:
    print("No exceptions, did we get to the end of file")
finally:
    if opened_files[1]:
        file_original.close()
    file_copy.seek(0, 0)
    for line in file_copy:
        sentences: [str] = line.split(".")
        try:
            for sentence in sentences:
                print(sentence)
                if sentence[0] == " ":
                    original_letter: str = sentence[1]
                    capital_letter: str = sentence[1].upper()
                else:
                    capital_letter: str = sentence[0].upper()
                    original_letter: str = sentence[0]
                if capital_letter != original_letter:
                    raise ValueError("Sentence was not capitalized")
                else:
                    print("Tested successfully")
        except IndexError as e:
            print("End of File while checking: {0}".format(e))
    if opened_files[0]:
        file_copy.close()


# #### Exercise 2:
# 
# Same as above but in this case, we can't create a new file. We have to
# modify the current file
# 
# __File name__: file_exercise1.txt -> Execute the function below to copy it

file_to_work_with: str = "./file_example_{0}.txt".format(int(
    datetime.datetime.now().timestamp()))

shutil.copyfile("file_exercise1.txt", file_to_work_with)

with open(file_to_work_with, mode="r+") as file_var:
    try:
        # We create the file that will contain the capitalized copy, we
        # need mode w+
        # In mode w+, the file does not have to exist (so it will be
        # created) and the file can be read
        # so we can test that the copy is correct without closing the
        # file
        capitalized_text: str = ""
        line = file_var.readline()
        pos_to_go_back: int = 0
        while line:
            # We split all the sentences (sentences are separated by .),
            # we have to capitalize the word after .
            sentences: [str] = line.split(".")
            # We go through all the sentences in this line
            for count, sentence in enumerate(sentences):
                # If the first character is a blank space, we have to
                # capitalize the following character
                if sentence[0] == " ":
                    capital_letter: str = sentence[1].upper()
                    # If count == 0, it means it is the first sentence
                    # in the line, so no blank space after . is needed
                    if count == 0:
                        capitalized_sentence: str = capital_letter + \
                                                    sentence[2:]
                    else:
                        # Otherwise, we need a blank space after . is needed
                        capitalized_sentence: str = ". " + capital_letter + \
                                                    sentence[2:]
                else:
                    # Otherwise, we capitalize the first character
                    capital_letter: str = sentence[0].upper()
                    if count == 0:
                        capitalized_sentence: str = capital_letter + \
                                                    sentence[1:]
                    else:
                        capitalized_sentence: str = ". " + capital_letter + \
                                                    sentence[1:]
                # Now we write the capitalized sentence to the copy file
                capitalized_text += capitalized_sentence
            line = file_var.readline()
    except IndexError as e:
        # We raise an IndexError when we get to the end of the file, so
        # the line's length is 0
        print("End of file: {0}".format(e))
    except FileNotFoundError as e:
        print("One of the files was not found, could not be found")
    else:
        print("No exceptions, did we get to the end of file")
    print(capitalized_text)
    file_var.seek(0, 0)
    file_var.write(capitalized_text)


# #### Exercise 3:
# 
# Replace the word four by five in the file. You need to do this in the
# file itself (can't create a new one)
# 
# __File name__: file_exercise3.txt -> Execute the function below to
# copy it

file_to_work_with: str = "./file_example_{0}.txt".format(
    int(datetime.datetime.now().timestamp()))

shutil.copyfile("file_exercise3.txt", file_to_work_with)

with open(file_to_work_with, mode="r+") as file_var:
    line = file_var.readline()
    pos_to_go_back: int = 0
    while line:
        if "four" in line.lower():
            file_var.seek(pos_to_go_back)
            file_var.write("Five")
        pos_to_go_back = file_var.tell()
        line = file_var.readline()
