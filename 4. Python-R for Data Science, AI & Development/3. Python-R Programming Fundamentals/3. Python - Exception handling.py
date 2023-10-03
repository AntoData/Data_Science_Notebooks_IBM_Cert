#!/usr/bin/env python
# coding: utf-8
import math
import traceback

# # Exception handling

# An __exception__ is an _unexpected error_ that occurs during
# _runtime_. So when Pyhon finds one of these situations it cannot cope
# with:
# - __Stops__ your program:
# - Creates a special type of object:
# 
# Basically, it __raises an exception__
# 
# _Raised exception_ expects something to take care of it
# - If _exception is not caught_ -> Our program will be _terminated_
# and an error will be sent to our console
# - But if we _catch the exception_ properly -> Execution can resume
# 
# In order to catch an exception, we have to surround the piece of code
# with:
# 
# try:
#     code that can raise a exception...
#     ....
# except:
#     code to execute if there is an exception...
#     ...
#

# For instance, the following operations will raise an exception

zero_div = 1 / 0


# __ZeroDivisionError__ is raised because we can't divide a number by 0

a = [1, 2, 3, 4]
print(a[4])


# __IndexError__ is raised because in our array besides being
# length = 4, indexes only get to 3, index 4 does not exist

# We can apply __exception handling__ to these snippets of code to
# control the execution:

try:
    a = 1 / 0
except:
    print("It seems like we were trying to divide by 0, but that is "
          "not possible")

a = [1, 2, 3, 4]
try:
    print(a[4])
except:
    print("It looks like our array is shorter than excepted")


# __Exceptions are organised in a tree-shaped hierarchy__: 
# 
# Some exceptions are parents of other more specific exceptions.
# For instance: __ArithmeticError__ is the parent exception of
# __ZeroDivisionError__
# We can use __ArithmeticError__ for cases in which we are trying to
# divide by 0 too (but it gives us less information as other errors
# will be caught by this exception

# - __Catching specific exceptions__:
# 
# The above examples use only the keyword __except__ which will catch
# any exceptions that is raised. However, this is discouraged by
# __PEP8__ (Python's code style guide)
# To catch specific exceptions we have to use the following syntax:
#
# try:
#     code that can raise a exception...
#     ....
# except ExceptionClass:
#     code to execute if there is an exception...
#     ...

# In all these cases, we will only catch the exception that has been
# declared. If any other exception is raised, it will stop the
# execution:

try:
    a = 1 / 0
except ZeroDivisionError:
    print("We were trying to divide by 0")


# Here the exception raised was ZeroDivisionError so the exception was
# caught and we could continue execution

a = [1, 2, 3, 4]

try:
    a[4]
except ZeroDivisionError:
    print("Exception was caught")


# However, in the example above, the exception we wanted to catch was
# again ZeroDivisionError but the one raised was IndexError. So the
# exception was raised and stopped execution

# If we want to __combine the general exception handling with specific
# exception handling__ we have to do it in the following order:
# 1. __Specific__ exception handling: All the branches for the
# different particular exceptions we want to catch have to go first.
# Actually Python will start trying to catch the exception __from the
# one in the top to the one in the bottom__

try:
    a = 1 / 0
except ArithmeticError:
    print("This will always be displayed")
except ZeroDivisionError:
    print("We will never get to this point")


# As explained before, __ArithmeticError__ is a parent exception (
# therefore more generic exception) to ZeroDivisionError. As it is the
# first branch after the try branch, this is the one where the exception
# is caught.

# 2. __General__ exception handling: It has to be __the last exception
# clause__.
# 
# It will be only executed if the exception raised was not any of the
# specific ones that we declared.
# 
# We can use the clause __except Exception__ also as a generic one

try:
    a = 1 / 0
except IndexError:
    print("We are trying to catch the error raised after trying to use "
          "an index that does not exist in the array")
except ZeroDivisionError:
    print("We tried to divide by 0 -> This is the one we are catching")
except:
    print("Generic exception -> Never displaying this")


# In this example the second branch caught the exception


try:
    a = chr(10**9)
except ZeroDivisionError:
    print("We will never get to this point")
except ArithmeticError:
    print("This will never be displayed")
except:
    print("This was not any of the exceptions above -> This will always "
          "be displayed")


# - __Getting the exception object__:
# 
# To get the __exception__ raised __as an object__ we only have to
# add _as var_ (usually _as e_) at the end of the exception clause
#
# try:
#     code that can raise a exception...
#     ....
# except ExceptionClass as e:
#     code to execute if this exception is caught...
#     ...
# except Exception as e:
#     code to execute if there is an exception that is not one of the
#     ones above...
#     ...
#
# 
# 

# The object __e__ will be of the type of the exception so _we can
# access the arguments/method of the class_

try:
    a = 1 / 0
except ZeroDivisionError as e:
    print("Exception args: {0}".format(e.args))
    print("This will always be displayed")
except ArithmeticError as e:
    print("We will never get to this point")
except Exception as e:
    print("This was not any of the exceptions above -> This will "
          "always be displayed")


# - __else__:
# 
# __else__ branch will be executed if no exception was raised.
# __else__ has to be placed after the last exception clause

upper_case: str = "UPPER CASE"

try:
    lower_case: str = upper_case.lower()
except ZeroDivisionError as e:
    print("ZeroDivisionError")
except ArithmeticError as e:
    print("ArithmeticError")
except Exception as e:
    print("General exception")
else:
    print("Result: {0}".format(lower_case))
    print("No exception was raised -> This will always be displayed")


# - __finally__:
# 
# __finally__ will always be executed no matter if an exception was
# raised or not
# 
# __finally__ has to be placed at the very end of the exception branch

upper_case: str = "UPPER CASE"

try:
    lower_case: str = upper_case.lower()
except ZeroDivisionError as e:
    print("ZeroDivisionError")
except ArithmeticError as e:
    print("ArithmeticError")
except Exception as e:
    print("General exception")
else:
    print("Result: {0}".format(lower_case))
    print("No exception was raised")
finally:
    print("Finally, we are finishing all this")

try:
    a = 1 / 0
except ZeroDivisionError as e:
    print("ZeroDivisionError")
except ArithmeticError as e:
    print("ArithmeticError")
except Exception as e:
    print("General exception")
else:
    print("Result: {0}".format(a))
    print("No exception was raised")
finally:
    print("Finally, we are finishing all this")


# ### Chained exceptions
# 
# This happens when __while trying to handle an exception,
# we raised an exception__. This might happen because of two reasons:
# 
# - We raised another exception in the branch -> __Implicitly__ raised
# exceptions -> __content__
# - We wanted to turn that exception into another type of exception ->
# __Explicitly__ raised exceptions -> __cause__
# 
# These two parameters described above are:
# 
# - e.__ context __ : Returns the __original__ exception raised ->
# __Implicitly__ raised exceptions
# - e.__ cause __ : Returns the message of the __original__ exception
# raise before it was turned into another one -> __Explicitly__
# raised exceptions

# - __Implicitly__ chained exceptions:
# 
# We raised an exception while handling another different exception:

arr = [1, 2, 3, 4]

try:
    a = 1 / 0
except ZeroDivisionError:
    try:
        print("ZeroDivisionError was raised")
        print(arr[4])
    except IndexError:
        print("IndexError was raised")

# Adding __context__ to the tree branch, we can see that context
# return the original exception

arr = [1, 2, 3, 4]

try:
    a = 1 / 0
except ZeroDivisionError as e:
    try:
        print("ZeroDivisionError was raised")
        print(arr[4])
    except IndexError as f:
        print(f.__cause__)
        print("IndexError was raised")


# - __Explicitly__ chained exceptions:
# 
# We try to convert an exception of a certain type to another type of
# exception

try:
    a = 1 / 0
except ZeroDivisionError as e:
    print("ZeroDivisionError was raised")
    raise ArithmeticError from e


def div_pi_by(n: int):
    try:
        div_result = math.pi / n
    except ZeroDivisionError as e1:
        raise ArithmeticError from e1
    return div_result


try:
    result: float = div_pi_by(0)
except ArithmeticError as f:
    print(f.__cause__)  # Exception returned by context was the
    # original one


# As we can see in the example above using __context__ we can get _the
# original exception that was raised_

# ### Traceback

# __traceback__ is a library that _helps you manage the traceback of
# your exceptions properly_
# 
# When we save an exception as a variable that variable has an
# attribute called __ traceback __

try:
    result: float = 1 / 0
except ZeroDivisionError as e:
    print(e.__traceback__)


# To make use of that object, the library __traceback__ contains a
# number of methods:
# 
# - __format_tb__: Saves the traceback to a list/array of string where
# each line is an element of the collection

try:
    result: float = 1 / 0
except ZeroDivisionError as e:
    trace_list: [str] = traceback.format_tb(e.__traceback__)
    print(trace_list[0])


# - __print_tb__: It prints formatted traceback

try:
    result: float = 1 / 0
except ZeroDivisionError as e:
    traceback.print_tb(e.__traceback__)
