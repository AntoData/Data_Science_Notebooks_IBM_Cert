#!/usr/bin/env python
# coding: utf-8
import io
import errno

# # Files in Python

# ### Reading Files With Open

# #### File names

# Before we explain how to open and work with files in Python. We need
# to state several issues that will affect our ability to work with
# them.
# 
# __File names and path are different depending on the Operating
# System__
#
# Windows
# -  __Uses \\___: For instance c:\path\file
# - As \ is a special character, it has to be written \\
# - Names are __not case sensitive__
# - __Line End__: RF or LF \r\n
#
# Linux
# - __Uses /__: For instance c:/path/file
# - No need to convert the symbol
# -  Names are __case sensitive__
# - __Line End__: CF \n
#
# Python can convert Linux to Windows format
# __Portability:__ This problem is solved by Python

# When we are working with __files in Python we use handles or streams__
# to communicate with the files.
# - Operations are done in the stream -> The Operating System will
# handle the change in the real file
# - 1ª we need to connect the file to the stream -> __open__
# - Finally, disconnect the file from stream, so changes and saved ->
# __close__

# ### Opening modes
# 
# - __Read__ mode: Only to read the file. Exception if trying to write
# the file when in this mode
#     - UnsupportedOperation:
#         - OSError
#         - ValueError
# - __Write__ mode: Only to write the file. Exception if trying to read
# the file
#     - UnsupportedOperation
# - __Update__ mode: Write and read

# ### Types of Streams
#
# Text:
# - Structure __in lines__
# - Typographical characters arranged in lines
# - Read __line by line__ or __character by character__
#
# Binary:
# - Sequence of __bytes__ of any value
# - Reads and write relate to portions of data of any size
# - Read __byte by byte__ block by block
#
# ### Function open
# To open a file we use the following function:
# 
# __open__(file_path, mode = "r", encoding = None)
# 
# Returns a stream connected to the file or __FileNotFoundError__ if the
# file was not found
# 
# #### Modes:
# 
# __Mode__:
# - __Read__: __r__: File has to exist -> Text: __rt__ or __r__ |
# Binary: __rb__
# - __Write__: __w__: File does not have to exist and it will be
# truncated to 0 -> Text: __wt__ or __w__ Binary: __wb__
# - __Append__: __a__: File does not have to exist, writes at the end of
# the file -> Text: __at__ or __a__ | Binary: __ab__
# - __Read plus__: __r+__ -> Read and update, file has to exist and data
# is appended and the end of the file -> Text: __r+t__ or __r+__ |
# Binary: __r+b__
# - __Write plus__: __w+__ -> Write and update, file does not have to
# exist and it will be truncated to 0 -> Text: __w+t__ or __w+__ |
# Binary: __w+b__

file_example = open("example.txt", mode="r")
print(type(file_example))
print(file_example)


# ### Function close
# To close a file we use the following function:
# 
# file.__close__()
# 
# This will commit the changes we have made in the file, we have to
# close the streams to files once we are no longer going to use it

# ### Attributes of the stream:
# 
# - file.__name__: Name of the file
# - file.__mode__: Mode in which the file was opened
# - file.__closed__: If file was closed

print(file_example.name)
print(file_example.mode)
print(file_example.closed)


# ### Reading from the stream
# 
# - file.__read(__n__)__ -> n is the number of characters to read

print(file_example.read(30))  # We read more than a line, we continue to
# other lines


# - file.__read()__ -> Reads the rest of the file (whole file if
# executed first)

print(file_example.read())


# - file.__readline__(n) -> n max number of characters we will read in
# a line

file_example.close()

file_example = open("example.txt", mode="r")

print(file_example.readline(1000))  # As you can see we read only the
# line. We don´t keep reading other lines
print(file_example.readline(10))


# - file.__readlines()__; Returns an array where each line is an
# of the array

lines: [str] = file_example.readlines()
print(lines)
print(lines[1])


# - __Iteration line by line__: for __line__ in file:

file_example.close()

file_example = open("example.txt", mode="r")

for line in file_example:
    print(line)


# - file.__tell__() -> Returns the position of the pointer in bytes

file_example.read(4)
print("Current position of the pointer: {0}".format(file_example.tell()))

file_example.close()

file_example = open("example.txt", mode="r")

print("Current position of the pointer after opening: {0}".format(
    file_example.tell()))


# - file.__seek__(offset, from) -> Changes the position by 'offset'
# bytes with respect to 'from'. From can take the value of 0,1,2
# corresponding to beginning, relative to current position and end

# Resetting the pointer
file_example.seek(0, 0)

# Read the first character
print(file_example.read(1))

# Setting the pointer two byte later
file_example.seek(2)

# Read third character
print(file_example.read(1))  # The pointer was in the following
# character


# - file.__truncate()__: It will remove all the data after the pointer


print(file_example.tell())
try:
    file_example.truncate()
except io.UnsupportedOperation as e:
    print("Can't update file in read mode: {0}".format(e))

file_example.seek(0, 0)

for line in file_example:
    print(line)
    
file_example.close()


# ### with open
# 
# There is a way to make sure the file is closed once we finish the
# execution of a piece of code
# 
# __with__ open(file_name, mode = mod) __as__ file:
# 
#     code...
#     
# Once we leave the portion called "code", the file is closed

with open("example.txt", mode="r") as file_example:
    print(file_example.read(10))
   
try:
    file_example.read()
except NameError as e:
    print("Stream variable is no longer available: {0}".format(e))
except ValueError as e:
    print("ValueError: {0}".format(e))


# ### errno library and exceptions
# 
# __errno__ is a library that provides some __exceptions to handle
# files__
# 
# - errno.__EACCES__: Permission denied
# - errno.__EBADF__: Bad file number (for example operate with unopened
# stream)
# - errno.__EEXIST__: File exists
# - errno.__EFBIG__: File is too big
# - errno.__EISDIR__: It is a directory
# - errno.__EMFILE__: Too many opened files
# - errno.__ENOENT__: No such file or directory
# - errno.__ENOSPC__: No space left on device

try:
    file = open("../../../../../../Downloads", mode="r")
except IOError as e:
    if e.errno == errno.EACCES:
        print("Permision denied: {0}".format(e))
    if e.errno == errno.EISDIR:
        print("We are trying to open a folder: {0}".format(e))
    if e.errno == errno.ENOSPC:
        print("No more space in disk: {0}".format(e))


# ### Another example of a structure to deal with files

try:
    file_example = open("example.txt", mode="r")
    # Operations
except IOError as e:
    if e.errno == errno.EACCES:
        print("Permission denied: {0}".format(e))
    if e.errno == errno.EISDIR:
        print("We are trying to open a folder: {0}".format(e))
    if e.errno == errno.ENOSPC:
        print("No more space in disk: {0}".format(e))
else:
    # Operations where we need the file to be opened
    pass
finally:
    file_example.close()
