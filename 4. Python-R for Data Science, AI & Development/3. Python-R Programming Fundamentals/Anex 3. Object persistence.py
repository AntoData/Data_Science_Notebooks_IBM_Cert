#!/usr/bin/env python
# coding: utf-8
import datetime
import random
import copy

# ### Object Persistence
# When we create a variable whether it is a __primitive__ type or an
# __object__, the variable is actually a _pointer to a memory address
# where the value of the variable is store_
# The process is:
# 
# 1. Object is created in memory
# 2. Object is populated (with values, other objects...)
# 3. The label is created and this label refers to an address in the
# computers memory

# - id() -> Returns an integer that identifies uniquely that variable
# during its lifespan (however, non-overlapping variables might have the
# same id)
# 
# __NOTE__: In Python C this returns the memory address of the variable

string_var: str = "Hello"
id(string_var)

int_var: int = 0
id(int_var)


class Patient:
    """
    Represents a patient of a health clinic. It contains the attributes

    p_id: Random and unique identification number
    :type p_id: int
    :param name: Patient's name
    :type name: str
    :param surname: Patient's surname
    :type surname: str
    :param height: Patient's height in metres
    :type height: float
    :param weight: Patient's weight in kgs
    :type weight: float
    :param birthdate: Patient's birthdate
    :type birthdate: datetime
    """

    instances: int = 0
    """
    Class attribute that will count how many instances/objects of 
    this class have been created
    """

    def __init__(self, name: str, surname: str, height: float,
                 weight: float, birthdate: datetime.datetime):
        """
        Constructor of the class, receives the following input
        parameters that will initialize a number of attributes of
        this class. Observe that attributes are declared like this:
        self.attribute_name = .... inside the constructor

        :param name: Patient's name
        :type name: str
        :param surname: Patient's surname
        :type surname: str
        :param height: Patient's height in metres
        :type height: float
        :param weight: Patient's weight in kgs
        :type weight: float
        :param birthdate: Patient's birthdate
        :type birthdate: datetime
        """

        self.p_id: int = random.randint(0, 10 ** 6)  # See how this was
        # not an input parameter of the constructor but it is
        # nonetheless an attribute of the class
        self.name: str = name
        self.surname: str = surname
        self.height: float = height
        self.weight: float = weight
        self.birthdate: datetime.datetime = birthdate
        # Constructors don't return anything, never have a clause return
        Patient.instances += 1
        # As you can see inside an instance member, we need to use the
        # name of the class to refer to the attribute

    def get_body_surface_area(self) -> float:
        """
        Return the measurement known as body surface area
        
        :return: Body surface area
        :rtype: float
        """
        # Self is passed as a parameter, so we have access to the
        # members of the class (attributes and other methods)
        body_surface_area: float = 1 / 6 * (self.weight * self.height) * 0.5
        # We get the attributes of this instance using the word self
        return body_surface_area

    def get_patient_age_in_date(self, comp_date: datetime.datetime) -> int:
        """
        Returns how many years old the patient was or will be in a
        certain date (past or present)

        :param comp_date: Date we will compare to our patient's
        birthdate to see how old he was at that time
        :type comp_date: datetime.datetime
        :return: Difference between comp_date and birthdate in years
        :rtype: int
        """

        # Methods can receive other input parameters besides self
        # And of course we can use them in the scope of the method
        difference: datetime.timedelta = comp_date - self.birthdate
        return difference.days // 365

    def get_patient_age(self):
        """
        Returns how old years old our patient is

        :return: Difference between current instant and birthdate
        in years
        :rtype: int
        """

        current_instant: datetime = datetime.datetime.now()
        # We can also call to other methods using self
        return self.get_patient_age_in_date(current_instant)


patient: Patient = Patient("James", "OHara", 1.81, 83,
                           datetime.datetime(year=1988, month=5, day=10))
id(patient)

# ### == vs is
# 
# They are different operators that are used for different things:
# 
# - ==: Returns True is both object have the same attributes and those
# attributes have the same values
# - is: Compares if both labels refer to the same object (return the
# same id).
# If both objects point to the same memory address

# Example 1: Both are different objects of type String with the same
# attributes:
# 
# - ==: True
# - is: False


string1: str = "Hello World"
string2: str = "Hello World"

print("string1 == string2: {0}".format(string1 == string2))
print("string1 is string2: {0}".format(string1 is string2))

# Example2: Both objects point to the same memory address


patient1: Patient = Patient("James", "OHara", 1.81, 83,
                            datetime.datetime(year=1988, month=5, day=10))
patient2: Patient = patient1

print("patient1 == patient2: {0}".format(patient1 == patient2))
print("patient1 is patient2: {0}".format(patient1 is patient2))

# - ### = does not copy
# 
# When you use the operator __=__ to assign a variable the value of
# another variable, _you are not copying that value into the new
# variable_. What you are doing is __pointing with the new variable to
# the same direction as this the variable passed on to__ by the operator
# 
# For example:
# 
# a = 2
# 
# b = a
# 
# __means that now b points to the same direction as a__
# 
# So:
# a == b -> True
# a is b -> True

a: int = 1
b: int = a

print("a == b: {0}".format(a == b))
print("a is b: {0}".format(a is b))


class NumberClass:
    def __init__(self, n: int):
        self.n: int = n

    def get_n(self) -> int:
        return self.n


n1: NumberClass = NumberClass(1)
n2 = n1

print("n1 == n2: {0}".format(n1 == n2))
print("n1 is n2: {0}".format(n1 is n2))

# - __Primitive types vs class and arrays__:
# 
# However, watch out for this: Primitives types and objects behave
# differently when we update their values after the situation presented
# above

# We start with the example above

a: int = 1
b: int = a
print("b = a")
print("a == b: {0}".format(a == b))
print("a is b: {0}".format(a is b))
print("")

# However, if we update b

b = 2
print("b = 2")
print("a == b: {0}".format(a == b))
print("a is b: {0}".format(a is b))
# a and b now have different values, b points to a different memory
# address than b (now be is an independent variable)

# However for objects

n1: NumberClass = NumberClass(1)
n2: NumberClass = n1
print("n2 = n1")
print("")
print("n1.n = {0}".format(n1.n))
print("n2.n = {0}".format(n2.n))
print("n1 == n2: {0}".format(n1 == n2))
print("n1 is n2: {0}".format(n1 is n2))
print("")

# If we update n2.n look what happens

n2.n = 2
print("n2.n = 2")
print("n1.n = {0}".format(n1.n))
print("n2.n = {0}".format(n2.n))
print("n1 == n2: {0}".format(n1 == n2))
print("n1 is n2: {0}".format(n1 is n2))

# n1 and n2 are still pointing to the same memory address so if we
# update n2, n1 is updated too


# ### Shadow copy
# 
# It is a copy where the new variable is independent. However, the
# contents of the object point to the same address. The copy __has just
# one level__


var: [] = [1, 2, [3, 4]]
var1: [] = var[:]

print("var1 = var[:]")
print("var: {0}".format(var))
print("var1: {0}".format(var1))

del var1[0]

print("del var1[0]")
print("var: {0}".format(var))
print("var1: {0}".format(var1))

# This did not affect var

var1[1][1] = 5

print("var1[1][1] = 5")
print("var: {0}".format(var))
print("var1: {0}".format(var1))

# However, if we update the array inside the array it affects both
# This is a shallow copy, the deeper level of the variable was not
# copied so the array inside the array was still the same for both


# ### Deep copy
# 
# The copy now is  an __independent__ object but __its contents are also
# independent__
# 
# __import copy__
# 
# copy.__deepcopy(var)__
# 
# __NOTE__: We don't call the constructor of the class for the copy


n1: NumberClass = NumberClass(1)
n2: NumberClass = copy.deepcopy(n1)
print("n2 = n1")
print("")
print("n1.n = {0}".format(n1.n))
print("n2.n = {0}".format(n2.n))
print("n1 == n2: {0}".format(n1 == n2))
print("n1 is n2: {0}".format(n1 is n2))
print("id(n1) = {0}".format(id(n1)))
print("id(n2) = {0}".format(id(n2)))
# In this case, n1 and n2 have the same values, but they are not the same
# variable.
# n1 and n2 point to different memory addresses
print("")

# If we update n2.n look what happens

n2.n = 2
print("n2.n = 2")
print("")
print("n1.n = {0}".format(n1.n))
print("n2.n = {0}".format(n2.n))
print("n1 == n2: {0}".format(n1 == n2))
print("n1 is n2: {0}".format(n1 is n2))

# As n1 and n2 are now independent variable, updating n2 does not affect
# n1
