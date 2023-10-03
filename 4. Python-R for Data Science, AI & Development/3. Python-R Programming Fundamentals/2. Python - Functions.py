#!/usr/bin/env python
# coding: utf-8

import random

# # Functions
# __Functions__ are one of the foundational elements in every
# programming language. Basically, __functions are called from any point
# in the code, variables or values called arguments are passed on to
# them and a piece of code located somewhere else is executed, the
# function might return a value(s).__ After that we resume the execution
# from the following line.

# __print()__ is an example of a function. It receives a string and
# executes some code (located somewhere) that displays that text in our
# terminal

print("Hello functions!!!!")


# __abs()__ is another example of a function. It receives one argument
# and returns the absolute value of it

print(abs(-3))


# So there are two types of functions:
# - __Predefined/built-in functions__: Functions that are __already
# defined and provided__ by Python itself, one of its libraries or a
# third party library. Like the examples above
# - __Custom/User-defined functions__: These are functions that are
# __created and defined by developers__. The syntax is as follows:
#     - def: To start defining a function we have to use this keyword
#     - __Function name__: After def, we have to provide a name for the
#     function
#     - __Argument list__: Between () we have to provide the list of
#     parameters the function will receive and therefore will be
#     available in the function.
#     This is __optional__. The () can be empty if we don't want to
#     provide any arguments to the function
#     -code: Now we can write the code that will be executed after
#     calling our function
#     - return: This one is optional and only needed if we want the
#     function to return a value
#
# For example, we can define a function that takes a string argument
# and prints it in reverse:
# - It will take one argument only of type string
# - Statement Return is not needed here

def print_reverse(message):
    print(message[::-1])


print_reverse("Hello World!!!!")


# For example, the following one won't receive any arguments, but will
# return a random char string

def random_char_string():  # As it does not receive any parameters, ()
    # is empty in the signature/header of the function
    random_string = ""
    length = random.randint(1, 10)  # We get a random number between 1
    # and 10 to become the length of the new random string
    for _ in range(length):
        unicode_num = random.randint(0, 128)  # We get a random number
        # between 0 and 128, which are codes of characters in unicode
        unicode_char = chr(unicode_num)  # We convert the code into the
        # character
        random_string += unicode_char  # We add the character to the
        # string
    return random_string  # We use return to return our random string


random_char_string()


# Finally, the following is an example that includes both

def random_char_string(length):
    random_string = ""
    for _ in range(length):
        unicode_num = random.randint(0, 128)  # We get a random number
        # between 0 and 128, which are codes of characters in unicode
        unicode_char = chr(unicode_num)  # We convert the code into the
        # character
        random_string += unicode_char  # We add the character to the
        # string
    return random_string  # We use return to return our random string


random_char_string(10)


# __NOTE__: When a __variable is passed__ as an argument and is updated
# in the function -> the __update won't affect the collection
# outside__ the function

v_list1 = ["First", "Second", "Third"]
v_list2 = ["Fourth", "Fifth", "Sixth"]


def add_and_reverse_lists(list1, list2):
    list1 = list1[::-1]
    list2 = list2[::-1]
    list2.extend(list1)
    return list2


print(add_and_reverse_lists(v_list1, v_list2))
print(v_list1)
print(v_list2)

a = 2
b = 3


def add_square(value1, value2):
    value1 = value1**2
    value2 = value2**2
    return value1 + value2


print(add_square(a, b))
print(a)
print(b)


# ### Scope
# A __scope__ is a section of the code in which certain variables are
# or not visible or are shadowed by other ones (meaning both have the
# same name, so you will use and update the one in your scope).
#
# __Functions__ define their own scope. The __parameters defined in the
# signature of the function and the variables created in the function
# are only accessible/visible inside it__. If we try to access them
# outside the function an exception will be raised _NameError_

# In[8]:


def random_char_string(length):
    random_string = ""
    for _ in range(length):
        unicode_num = random.randint(0, 128)  # We get a random number
        # between 0 and 128, which are codes of characters in unicode
        unicode_char = chr(unicode_num)  # We convert the code into
        # the character
        random_string += unicode_char  # We add the character to the
        # string
    return random_string  # We use return to return our random string


random_char_string(10)
# print(random_string) -> Will be a compilation error


# Same for the input parameters:

# print(length) -> Will be a compilation error


# Also this is one of the reasons why as we explained previously:
# When a __variable is passed__ as an argument and is updated in the
# function -> the __update won't affect the collection outside__ the
# function. The update is done __in the scope of the function__

# __Shadowing__: To put it simply, it is when a variable inside a
# function has the same name as a variable in the main scope:
#     - In this case, when we perform operations in such variable ->
#     The updates will only affect the variable inside the function
#     - The variable outside the function will remain the same

random_string = "I am not going to change"


def random_char_string(length):
    random_string = ""
    for _ in range(length):
        unicode_num = random.randint(0, 128)  # We get a random number
        # between 0 and 128, which are codes of characters in unicode
        unicode_char = chr(unicode_num)  # We convert the code into the
        # character
        random_string += unicode_char  # We add the character to the
        # string
    return random_string  # We use return to return our random string


print(random_char_string(10))
print(random_string)


# __Global variables__: Global variables __are variables defined in the
# main scope__. These are visible everywhere, __they can be accessed
# everywhere__. However, changes or __updates made to them inside a
# function will not be applied outside__ that function

words = []


def count_words(text):
    words = text.split(" ")
    return len(words)


print(count_words("Hello World here we go"))
print(words)


# __Global__: The keyword _global_ allows us to do a number of things:
# - To refer to a variable in the main scope inside a function so when
# we update it, the update is also visible outside the function

words = []


def count_words(text):
    global words
    words = text.split(" ")
    return len(words)


print(count_words("Hello World here we go"))
print(words)


# - To create a global variable, a variable whose scope is the main
# scope outside a function inside it

def count_words(text):
    global words
    words = text.split(" ")
    return len(words)


print(count_words("Hello World here we go"))
print(words)


# ### Arguments
# - __Default arguments__: Python allows developers to assign default
# values to input parameters in their functions:
#     - We can modify the value of the parameter
#     - We can skip that argument, and it will get the value defined as
#     default


def random_char_string(length=10):  # To set a default value to a
    # variable we just give it the value like this
    random_string = ""
    for _ in range(length):
        unicode_num = random.randint(0, 128)  # We get a random number
        # between 0 and 128, which are codes of characters in unicode
        unicode_char = chr(unicode_num)  # We convert the code into the
        # character
        random_string += unicode_char  # We add the character to the
        # string
    return random_string  # We use return to return our random string


print(len(random_char_string()))


# The value of length was 10 because it used the default value

print(len(random_char_string(12)))


# We overwrote the value of length to 12

# __NOTE__: All default arguments must follow non-default arguments.
# Otherwise, we will get a __SyntaxError__

# def get_character_move_to_final(text = "Hello World!", index):
#  This won't compile because:
#  non-default parameter follows default parameter
#     character_var = text[index]
#     first_half = text[: index]
#     second_half = text[index + 1: len(text)]
#     return first_half + second_half + character_var
#
# get_character_move_to_final(3)

def get_character_move_to_final(text, index=1):
    character_var = text[index]
    first_half = text[: index]
    second_half = text[index + 1: len(text)]
    return first_half + second_half + character_var


get_character_move_to_final("Hello World!!!!")


# - __Positional arguments__: When we provide the arguments by order
# of their declaration in the function signature (we have sent the
# variables values to the function and the function takes it in order)

def get_character_move_to_final(text, index):
    character_var = text[index]
    first_half = text[: index]
    second_half = text[index + 1: len(text)]
    return first_half + second_half + character_var


get_character_move_to_final("Hello World!!!!", 3)


# The above was an example of __positional arguments__, the values were
# taken by the function and assigned to the input parameters of the
# function by order

# - __Keyword arguments__: When we use the name of the input parameter
# to give it the value. Therefore, we can switch the order of the
# arguments

get_character_move_to_final(index=3, text="Hello World!!!!")


# In the above example we switched the order of the arguments, but the
# function worked correctly because we were using the input parameters
# as keyword arguments

# - __Packing arguments__: Python allows us to define a function with
# an undetermined number of parameters:
#     - __Positional arguments__: In order to define an undetermined
#     number of positional arguments we use __*args__: That parameter
#     will contain all values passed as undetermined arguments in a list


def add_elements_to_list(list_var, *args):  # We have defined one
    # argument and then an undetermined number of arguments
    for element in args:
        list_var.append(element)

    return list_var


v_list = ["a", 2, "z"]
add_elements_to_list(v_list, "b", 3, "c")


# We could have not added any more positional arguments than the one
# we defined

v_list_2 = [1, 2, 3]
add_elements_to_list(v_list_2)


# - Continuation:
#     - __Keyword arguments__: We can define and unlimited number of
#     keyword parameters using __**kwargs__. This means, we can add as
#     many tuples or var = value, as we want. The parameter _kwargs_
#     will contain all these parameters in a dictionary with key the
#     name of the variable and value the value given to that variable

def add_elements_to_list(list_var, **kwargs):  # We have defined one
    # argument and then an undetermined number of arguments
    for key_element in kwargs:
        list_var.append(kwargs[key_element])

    return list_var


v_list = ["a", 2, "z"]
add_elements_to_list(v_list, destination="Malta", plane=3,  door="c")


# __NOTE__: Keyword parameter must follow positional parameters

# In[23]:


def travel_organizer(kms, country="Scotland", seat_class="Tourist",
                     *args, **kwargs):
    print("We are traveling to {0} which will be a {1} km trip in {2}".
          format(country, kms, seat_class))
    if len(args) > 0:
        print("We are stopping in: ")
        for stop in args:
            print("- {0}".format(stop))

    if len(kwargs.keys()) > 0:
        print("Additional information")
        for v_key in kwargs.keys():
            print("- {0}: {1}".format(v_key, kwargs[v_key]))


travel_organizer(100)

travel_organizer(200, "France")

travel_organizer(300, "Norway", "First", "France", "Switzerland", "Germany")

travel_organizer(300, "Norway", "First", company="RyanAir")

travel_organizer(300, "Norway", "First", "Denmark", "Sweden",
                 company="RyanAir", sale=True)

# travel_organizer(300, "Norway", "First", company = "RyanAir",
# sale = True, "Denmark", "Sweden") -> This won't compile because:
# Positional argument after keyword argument


# __Remember__: Positional arguments must be declared before keyword
# parameters

# __Return__: We can return more than one parameter and __unpack them__

def get_first_and_last_element(collection):
    return collection[0], collection[len(collection) - 1]


first_el, second_el = get_first_and_last_element("Hello World")
print(first_el)
print(second_el)


# In the example above, we have __unpacked__ the return into two
# different variables

elements = get_first_and_last_element("Hello World")

print(elements)


# Here, both variables were __packed__ in the same variable (a tuple)
