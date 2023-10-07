#!/usr/bin/env python
# coding: utf-8

import datetime
import random


# # Classes and Objects
# 
# __Python__ is an _Object Oriented Programming Language_ (OOP). This is
# a programming paradigm where we define entities called __classes__.
# The classes can be considered like _new custom-made types_. When we
# define a class, we define its _members_:
# - which __data__ it will contain -> __attributes__: These are the
# variables that a new instance of that class will contain
# - which __operations__ it will perform -> __methods__: These are the
# functions the class will provide
# 
# An _instance_ of a class is called __object__. An object is just a
# variable whose type is a custom class that has been initialized.


class SimpleClass:
    def print_n(self, n):
        print("{0}".format(n))


# This was a very basic example. It did not contain attributes as we
# did not overwrite its constructor
# - __Constructor__: It is a reserved method that builds the instance
# of the class. We use it to initialize the attributes of the class.
# This method is called __ init __


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


# __Object__: To create an instance of a class, we call the name of the
# class
# - If the __constructor__ was overwritten, we have to provide the same
# arguments that are input parameters in the method __ init __

patient1: Patient = Patient("James", "OHara", 1.81, 83,
                            datetime.datetime(year=1988, month=5, day=10))

# In the above statement, we create a new variable called patient1 of
# type Patient. This patient's name is James, their surname is OHara,
# their height is 1.81m, their weight is 83 kgs, and they were born May
# 10th 1988

# To access the members (attributes and methods) of a class, you just
# need to:

# Getting attributes from the object
print("Patient ID: {0}".format(patient1.p_id))
print("Name: {0}".format(patient1.name))
print("Surname: {0}".format(patient1.surname))

# Calling methods from the object
print("Age: {0}".format(patient1.get_patient_age()))
print("Age in 2000: {0}".format(patient1.get_patient_age_in_date(
    datetime.datetime(year=2000, month=1, day=1))))
print("Body Surface Are: {0}".format(patient1.get_body_surface_area()))

# ### Class Members
# 
# There are two types of members:
# 
# - __Instance__ members: This only applies really to __attributes__.
# It means that this kind of member, this kind of attribute might have
# different values depending on the object. Even might have a different
# set of attributes and methods from one object to another:
# 
# The above examples are instance attributes, we gave them value during
# the instantiation of the class. But we can:
# - Change their value only for them: If we change their value, it will
# only be applied to that instance of the class to that object
# - Create a new object whose attributes have different values


patient2: Patient = Patient("Katya", "Colby", 1.71, 63,
                            datetime.datetime(year=1980, month=4, day=1))
# The attributes for this instance of Patient have different values

print(patient2.name)

patient1.name = "John"
print(patient1.name)
print(patient2.name)
# Updating the attribute name of patient1, only affects his instance,
# patient2 remains the same


# Also, __instance attributes can be added at any point during the life
# cycle of the member__. They are usually added during initialization in
# __ init __ but actually can be added at any point

patient1.country = "Greece"
print(patient1.country)

try:
    print(patient2.country)
except AttributeError as e:
    print("patient2 does not have the attribute country: {0}".format(e))


# Even methods can be added to an object this way:

def method_returns_two():
    return 2


patient1.new_method = method_returns_two
print(patient1.new_method())

# - To see the __instance members__ of an __object__ we can call the
# default attribute __ __dict__ __ :

print(patient1.__dict__)


# - __Class__ members: These are:
#     - __Attributes__: Attributes shared by all the instances of a
#     class. All of them share the same instance of the variable.
#     Therefore, if it changes, changes for all of them
#     - __Methods__: Method that don't use __instance members__, nor
#     methods, nor attributes. They only rely on __class attributes and
#     methods__

# - To define a __class attribute__ you only have to declare and
# initialize it after the signature of the class, _not in the
# constructor_
#

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


# You __access a class attribute__ using __the name of the class__

# In[11]:


patient1: Patient = Patient("James", "OHara", 1.81, 83,
                            datetime.datetime(year=1988, month=5, day=10))
patient2: Patient = Patient("Katya", "Colby", 1.71, 63,
                            datetime.datetime(year=1980, month=4, day=1))

# In[12]:


print(Patient.instances)

# You can also __access class attributes using the instances__ of that
# class.
# However, you can __only get them__.
# You __can't set or update them using the instances__ of that class.
# _Python will create an instance attribute with that name instead_
# 
# In that case also, you won't be allowed to get the class attribute using
# the instance of that class as the instance attribute will be the one
# called instead.

print(patient1.instances)
print(patient2.instances)

patient1.instances = 5
print(patient1.instances)
print(Patient.instances)


# See the example above, we created a new instance attribute with value
# 5 instead of updating the class attribute when using the object
# instead of the class

# - To define a __class method__:
#     - You have to use the annotation __@classmethod__
#     - Then, instead of _self_, the first parameter will be __cls__
#     which will _give you access to all class members_
#     - As mentioned above, inside our class method __we can't access
#     any instance member (nor variables, nor methods)__. Think about
#     this for a second, and you will realise it makes sense. These
#     methods belong to the class, before any instance is created.
#     Therefore, they can't use any instance member


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

    @classmethod
    def get_number_of_instances(cls) -> int:
        """
        Returns the number of instances created from this class

        :return: Number of instances created
        :rtype:  int
        """

        return cls.instances


# Now, we have a _class method_ called __get_number_of_instances__ whose
# first parameter is _cls_ and we can use it to get all the class
# members in our current class

# Again, we __use the class to call to class methods__

patient1: Patient = Patient("James", "OHara", 1.81, 83,
                            datetime.datetime(year=1988, month=5, day=10))
patient2: Patient = Patient("Katya", "Colby", 1.71, 63,
                            datetime.datetime(year=1980, month=4, day=1))
print(Patient.get_number_of_instances())

# We can also use the instances of the class to call the method

print(patient1.get_number_of_instances())

# __NOTE__: One curious thing about __cls__ is that it can be used as a
# __constructor__ inside a _class method_
# 
# This might be very useful, for instance we can create a default
# constructor that does not need parameters


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

    @classmethod
    def get_number_of_instances(cls) -> int:
        """
        Returns the number of instances created from this class

        :return: Number of instances created
        :rtype:  int
        """

        return cls.instances

    @classmethod
    def create_new_default_patient(cls) -> 'Patient':
        """
        Creates a dummy patient

        :return: new object Patient
        :rtype: Patient
        """

        return cls("John", "Doe", 1.70, 70, datetime.datetime.now())


patient3: Patient = Patient.create_new_default_patient()
print(patient3.name)

# As with instances of the class, we can use the object __ dict __ to
# get the members of a class.
# 
# In this case __we use the name of the class__ and the object dict and
# all class members (both attributes and methods) are returned

print(Patient.__dict__)

# We can also __add class members__ after the class has been declared


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

    @classmethod
    def get_number_of_instances(cls) -> int:
        """
        Returns the number of instances created from this class

        :return: Number of instances created
        :rtype:  int
        """

        return cls.instances

    @classmethod
    def create_new_default_patient(cls) -> 'Patient':
        """
        Creates a dummy patient

        :return: new object Patient
        :rtype: Patient
        """

        return cls("John", "Doe", 1.70, 70, datetime.datetime.now())

    @staticmethod
    def return_values_default_patient() -> list:
        """
        Returns a list with values we can use to create a new patient

        :return:
        """
        return ["John", "Doe", 1.70, 70, datetime.datetime.now()]


# This method will become a class method of Patient
@classmethod
def get_instances_multiplied_by(cls, n: int) -> int:
    """
    Will become a class method of class Patient and will return the
    number of current instances of the class multiplied by n
    :param cls: Class Patient
    :param n: Number by which we will multiply the number of instances

    :return: The number of instances of class Patient multiplied by n
    """

    return cls.get_number_of_instances() * n


patient1: Patient = Patient("James", "OHara", 1.81, 83,
                            datetime.datetime(year=1988, month=5, day=10))
patient2: Patient = Patient("Katya", "Colby", 1.71, 63,
                            datetime.datetime(year=1980, month=4, day=1))

Patient.hospital = "Class Hospital"
print("The hospital of our patients is: {0}".format(Patient.hospital))
print("")
print("Current state of dict")
print(Patient.__dict__)
print("")
Patient.new_class_method = get_instances_multiplied_by
print(
    "Number of instances multiplied by 2 using the new class method: {0}".
    format(Patient.new_class_method(2)))
print("")
print("Current state of dict")
print(Patient.__dict__)

# - __static methods__: These are methods that __don't use any of the
# attributes__ of the class, _nor instance nor class attributes_.
#     - To create a static method we just use the annotation
#     __staticmethod__
#     - Therefore, static methods __don't have self or cls as
#     paramaters__
#     - So we __can't access any member of the class__
#     - They are used to append functions might be useful but don't use
#     any member of the class.


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

    @classmethod
    def get_number_of_instances(cls) -> int:
        """
        Returns the number of instances created from this class

        :return: Number of instances created
        :rtype:  int
        """

        return cls.instances

    @classmethod
    def create_new_default_patient(cls) -> 'Patient':
        """
        Creates a dummy patient

        :return: new object Patient
        :rtype: Patient
        """

        return cls("John", "Doe", 1.70, 70, datetime.datetime.now())

    @staticmethod
    def return_values_default_patient() -> list:
        """
        Returns a list with values we can use to create a new patient

        :return:
        """
        return ["John", "Doe", 1.70, 70, datetime.datetime.now()]


# We can __only use the class to call static methods__

print(Patient.return_values_default_patient())

try:
    print(patient1.return_values_default_patient())
except AttributeError as e:
    print(e)
    print(
        "As said above, the object patient1 does not have access to "
        "static members")

# __static methods__ are also returned when using <b> __ dict __ in
# a class</b>

print(Patient.__dict__)
