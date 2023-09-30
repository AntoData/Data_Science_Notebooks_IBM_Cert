#!/usr/bin/env python
# coding: utf-8

# # Loops
# __Loops__ are used to iterate through the __elements__ that are
# contained in a variable whose type is __a collection__
# (_lists_, _sets_. _tuples_) or simple __to repeat the same algorithm
# a number of times__

# ### While
# __while__ is a _loop_ that will execute the code inside until a
# certain __condition is false__

i: int = 0
while i < 5:  # The condition here is "i < 5"
    i += 1    # Be mindful that the condition has to change
    print("Iteration: n={0}".format(i))


# __break__ is a keyword that stops a _loop_ and continue the execution
# after the loop

i: int = 100
while i < 200:
    print("Iteration: n={0}".format(i-99))
    if i % 13 == 0:
        print("We found the first multiple of 13 between 100 and 200: {0}"
              .format(i))
        break
    else:
        i += 1
print("The loop has finished because we found our number")


# If we analyze the code above, we are looking for an element between to
# boundaries __[100, 200]__. Our goal is to find an element in that
# collection that meets a condition. But we don't know which element
# will meet it. So we have to go through them, one by one and check if
# the condition. Once we found the condition, in this case we don't have
# to keep iterating, so we can use __break__ to stop the loop and
# continue the execution after the loop

# __else__ can be used also with __while__ _loops_. The code in __else__
# will be executed if and only if the condition in __while__ has
# returned __False__. Therefore, if we use __break__, the code in
# __else__ won't be executed

i: int = 0

while i < 5:
    if i % 7 == 6:
        print("How come? This is impossible")
        break
    i += 1
    print("Iteration: n={0}".format(i))
else:
    print("As expected break was not executed, so we execute else ")

print("We continue the execution from here")


# As previously mentioned, here the _loop_ was executed until the
# condition returned __False__ and then we moved on to the branch
# __else__


i: int = 0

while i < 7:
    if (i + 5) % 3 == 1:
        print("Breaking the loop")
        break
    i += 1
    print("Iteration: {0}".format(i))
else:
    print("Else won't be executed")

print("Executing main flow")


# ### For
# __for__ is a _loop_ that will __iterate through a collection__


cities: [str] = ["Edinburgh", "Milan", "Eindhoven", "Piraeus", "Marseille",
                 "Valletta"]

for city in cities:    # The variable next to for (in this case city)
    # will contain the element we are getting from the collection
    # (cities)
    print("We are travelling to {0}".format(city))


# The __elements in the list are not modified if we modify the
# variable__. The variable contains a copy of the element in the
# collection


cities: [str] = ["Edinburgh", "Milan", "Eindhoven", "Piraeus", "Marseille",
                 "Valletta"]

for city in cities:   # city contains a copy of the element in list
    print("We are travelling to {0}".format(city))
    if city == "Milan":
        city = "Rome"  # Won´t modify the list
        
print(cities)  # List is not modified


# However, __the list can be modified while we are executing the for
# loop__


cities: [str] = ["Edinburgh", "Milan", "Eindhoven", "Piraeus", "Marseille",
                 "Valletta"]

for city in cities:    # We can add elements to the list
    print("We are travelling to {0}".format(city))
    if city == "Milan":
        cities.append("Rome")  # This element ("Rome") will be appended
        
print(cities)  # Rome was appended at the end of the list


# However, be mindful of changes. For instance, we can __remove an
# element__ from the collection. However, _the loop will skip the
# element that follows the eliminated element_


cities: [str] = ["Edinburgh", "Milan", "Eindhoven", "Piraeus",
                 "Marseille", "Valletta"]

i: int = 0
for city in cities:    # We can remove elements to the list
    print("We are travelling to {0}".format(city))
    if city == "Milan":
        del cities[i]  # This element will be removed. However, the
        # following element "Eindhoven" will be skipped
    i += 1
print(cities)  # Milan was removed from the list


# We can __modify__ _the elements of the list_ without problems like
# this:

cities: [str] = ["Edinburgh", "Milan", "Eindhoven", "Piraeus", "Marseille",
                 "Valletta"]

i: int = 0
for city in cities:    # We can modify elements to the list
    print("We are travelling to {0}".format(city))
    if city == "Milan":
        cities[i] = "Rome"  # Milan will become Rome
    i += 1
print(cities)  # Milan now is Rome


# __enumerate__ applied to the _collection_ will return a tuple of
# element, the __index__ of the element in the collection and the
# __element__ itself


cities: [str] = ["Edinburgh", "Milan", "Eindhoven", "Piraeus", "Marseille",
                 "Valletta"]


for i, city in enumerate(cities):    # enumerate will return the index
    # of the element in the collection and the element
    print("Element index: {0}".format(i))
    print("We are travelling to {0}".format(city))


# __break__ as explained before, break will stop the loop and continue
# execution after it


cities: [str] = ["Edinburgh", "Milan", "Eindhoven", "Piraeus", "Marseille",
                 "Valletta"]


for city in cities:
    print("We are travelling to {0}".format(city))
    if city == "Milan":
        print("Milan is the final destination, breaking the loop")
        break

print("Back to the main branch")


# __else__ can be used also with __for__ _loops_. The code in __else__
# will be executed if and only if we __have gone through all the
# elements in the collection__. Therefore, if we use __break__, the
# code in __else__ won't be executed

cities: [str] = ["Edinburgh", "Milan", "Eindhoven", "Piraeus", "Marseille",
                 "Valletta"]


for city in cities:
    print("We are travelling to {0}".format(city))
    if city == "Malaga":
        print("This won't be executed")
        break
else:
    print("We have travelled to all the cities")

print("Back to the main branch")


cities: [str] = ["Edinburgh", "Milan", "Eindhoven", "Piraeus", "Marseille",
                 "Valletta"]


for city in cities:
    print("We are travelling to {0}".format(city))
    if city == "Piraeus":
        print("This is the last stop")
        break
else:
    print("This won't be executed")

print("Back to the main branch")


# __range__ is a _function_ that returns a collection of numbers like
# this:
# - __range__(n): From 0 to n-1

for i in range(5):
    print(i)


# - __range(n, m)__: From n to m-1:

for i in range(3, 7):  # From 3 and 6
    print(i)


# - __range(n, m, step)__: From n to m-1 stepping "step" (n, n+step,
# n+2*step,...) while < m

print("Printing pairs")
for i in range(0, 10, 2):
    print(i)


# ### Generators/Iterators
# Piece of specialized code __able to produce a series of values__ and
# to control the iteration process
# __range__ is a __generator

# __Iterator protocol__: It's a way in which an object should behave to
# conform to the rules imposed by the context of the for and in
# statements

# In other words, we code how these objects should behave in the _loop_
# __for__ (using __in__)

# The __class__ (we will explain this concept later) needs to implement
# the methods __ __iter__() __ and __ __next__() __
# - __iter__(): Has to return the object itself (in this case the member
# __self__ of the class, we will find out more about this later)
# - __next__(): Has to generate and return the following value in the
# generator
#     - When we want to finish the _loop_ we have to __raise__ the
#     exception __StopIteration__ (we will explore exceptions later)
# 
# The following is an example of a generator that given a number _n_
# and a _limit_ will return all the even numbers between _n_ and _limit_

class EvenNumbers:
    def __init__(self, n: int, limit: int):
        self.n: int = n   # Number to start looking for even numbers
        self.limit: int = limit  # Limit of the search
        
    def __iter__(self):  # It has to return the object itself (self)
        return self
    
    def __next__(self):  # It has to return the following element
        if self.n < self.limit:
            while True:  # This is a way to create an infinite loop
                if self.n % 2 == 0:
                    n_next: int = self.n
                    self.n += 1
                    return n_next  # Return is executed, breaks the loop
                else:
                    self.n += 1
        raise StopIteration("Final value")  # Once we don't want to
        # continue the loop (we reach the limit), we raise StopIteration


even_numbers: EvenNumbers = EvenNumbers(6, 20)

for number in even_numbers:
    print(number)


# ### List comprehension
# We can use the _loop_ __for__ to fill a collection in creation
# 
# - _Simple_: We use the loop __for__ only:

even_numbers: list = [x for x in range(0, 10, 2)]


# This will add all the elements that are return by the for loop to the
# list _even_numbers_

print(even_numbers)


# - __If__ + __for__: We can combine both like this

even_numbers: list = [x if x % 2 == 0 else False for x in range(0, 10)]


# This will return the element returned by the for loop if the condition
# is met (x % 2 == 0) or False if the condition is not met

print(even_numbers)


# - __Multidimensional__: We can use it to fill multidimensional collections

matrix: [int] = [[i+j if (i+j) % 2 == 0 else -(i+j) for j in range(0, 3)]
                 for i in range(0, 3)]


print(matrix)


# This returns a matrix where each element is the sum of its indexes,
# but the element is negative if it is an odd number
