#!/usr/bin/env python
# coding: utf-8
import abc
import datetime
import random
import math
import shutil
import scipy.constants as scg

# # Classes Advanced
# 
# - __Encapsulation__: IT is the bundling of data members and functions
# inside a single class. Bundling similar data members and functions
# inside a class also helps in __data hiding__
# 
# So we can actually _restrict the access_ to members of a class.
# 
# To do so, we have to __start the name of the attribute or method
# by__ "_ _".
# 
# These members __can't be accessed from outside the class__

# Following the example of the previous notebook, we will:
# - Turn the class attribute instances private
# - Turn the instance variable pid private
# - Turn the method get_patient_age_in_date private
# - Turn the class method return_values_default_patient private and use
# it in create_new_default_patient


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

    __instances: int = 0
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

        self.__p_id: int = random.randint(0, 10 ** 6)  # See how this was
        # not an input parameter of the constructor but it is
        # nonetheless an attribute of the class
        self.name: str = name
        self.surname: str = surname
        self.height: float = height
        self.weight: float = weight
        self.birthdate: datetime.datetime = birthdate
        # Constructors don't return anything, never have a clause return
        Patient.__instances += 1
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

    def __get_patient_age_in_date(self, comp_date: datetime.datetime) -> int:
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
        return self.__get_patient_age_in_date(current_instant)

    @classmethod
    def get_number_of_instances(cls) -> int:
        """
        Returns the number of instances created from this class

        :return: Number of instances created
        :rtype:  int
        """

        return cls.__instances

    @classmethod
    def create_new_default_patient(cls) -> 'Patient':
        """
        Creates a dummy patient

        :return: new object Patient
        :rtype: Patient
        """

        return cls(*cls.__return_values_default_patient())

    @staticmethod
    def __return_values_default_patient() -> list:
        """
        Returns a list with values we can use to create a new patient

        :return:
        """
        return ["John", "Doe", 1.70, 70, datetime.datetime.now()]


# Now if we try to get the class attribute _ _ instances we will
# raise an AttributeError exception

try:
    print(Patient.__instances)
except AttributeError as e:
    print("__instances is private: {0}".format(e))


# Now if we try to get the class method _ _
# return_values_default_patient we will raise an AttributeError
# exception

try:
    Patient.__return_values_default_patient()
except AttributeError as e:
    print("__return_values_default_patient is private: {0}".format(e))


# However, we can call to methods that use them:


patient1: Patient = Patient.create_new_default_patient()
print(patient1.name)

print(Patient.get_number_of_instances())


# - __Mangled names__:However, there is a way by which these members
# can still be called: _ className __ memberName
# 
# Following the example above:

print(Patient._Patient__instances)

print(patient1._Patient__return_values_default_patient())


# Now we can call to both using their __mangled names__

# - Annotation __@property__: It is used to become the method executed
# when we call a private attribute of the same name (without __ ). So we
# can control what happens when we call that member. We can raise a
# different exception, return the value of the attribute...
#
# We will create a method pid that uses that annotation to be able to
# get that attribute


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

    __instances: int = 0
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

        self.__p_id: int = random.randint(0, 10 ** 6)  # See how this was
        # not an input parameter of the constructor but it is
        # nonetheless an attribute of the class
        self.name: str = name
        self.surname: str = surname
        self.height: float = height
        self.weight: float = weight
        self.birthdate: datetime.datetime = birthdate
        # Constructors don't return anything, never have a clause return
        Patient.__instances += 1
        # As you can see inside an instance member, we need to use the
        # name of the class to refer to the attribute

    @property
    def p_id(self) -> int:
        """
        Gets the parameters __p_id
        
        :return: Parameter __p_id
        :rtype: int
        """
        return self.__p_id
    
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

    def __get_patient_age_in_date(self, comp_date: datetime.datetime) -> int:
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
        return self.__get_patient_age_in_date(current_instant)

    @classmethod
    def get_number_of_instances(cls) -> int:
        """
        Returns the number of instances created from this class

        :return: Number of instances created
        :rtype:  int
        """

        return cls.__instances

    @classmethod
    def create_new_default_patient(cls) -> 'Patient':
        """
        Creates a dummy patient

        :return: new object Patient
        :rtype: Patient
        """

        return cls(*cls.__return_values_default_patient())

    @staticmethod
    def __return_values_default_patient() -> list:
        """
        Returns a list with values we can use to create a new patient

        :return:
        """
        return ["John", "Doe", 1.70, 70, datetime.datetime.now()]


patient1: Patient = Patient.create_new_default_patient()
print(patient1.p_id)  # Look at how we call the attribute, not a method


# However, this is only valid to __get__ the attribute. But __we can´t
# set__ or _update_ the attribute still and the __same
# exception will be raised__


try:
    patient1.p_id = 1
except AttributeError as e:
    print("We can't update the attribute: {0}".format(e))


# - Annotation __@setter.property__: It is used, so we can control when
# an attribute (specially private ones) is __updated__. We can actually
# update the attribute, make operations before updating the attribute,
# raising an exception...

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

    __instances: int = 0
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

        self.__p_id: int = random.randint(0, 10 ** 6)  # See how this was
        # not an input parameter of the constructor but it is
        # nonetheless an attribute of the class
        self.name: str = name
        self.surname: str = surname
        self.height: float = height
        self.weight: float = weight
        self.birthdate: datetime.datetime = birthdate
        # Constructors don't return anything, never have a clause return
        Patient.__instances += 1
        # As you can see inside an instance member, we need to use the
        # name of the class to refer to the attribute

    @property
    def p_id(self) -> int:
        """
        Gets the parameters __p_id

        :return: Parameter __p_id
        :rtype: int
        """
        return self.__p_id

    @p_id.setter
    def p_id(self, new_p_id: int):
        """
        Setter for attribute __p_id
        
        :param new_p_id: New value to update the attribute
        :return: 
        """
        raise RuntimeError("p_id can't be updated, unique key value")
        
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

    def __get_patient_age_in_date(self, comp_date: datetime.datetime) -> int:
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
        return self.__get_patient_age_in_date(current_instant)

    @classmethod
    def get_number_of_instances(cls) -> int:
        """
        Returns the number of instances created from this class

        :return: Number of instances created
        :rtype:  int
        """

        return cls.__instances

    @classmethod
    def create_new_default_patient(cls) -> 'Patient':
        """
        Creates a dummy patient

        :return: new object Patient
        :rtype: Patient
        """

        return cls(*cls.__return_values_default_patient())

    @staticmethod
    def __return_values_default_patient() -> list:
        """
        Returns a list with values we can use to create a new patient

        :return:
        """
        return ["John", "Doe", 1.70, 70, datetime.datetime.now()]


# In[12]:


patient1: Patient = Patient.create_new_default_patient()
print(patient1.p_id)

try:
    patient1.p_id = 1
except RuntimeError as e:
    print(e)


# ### Inheritance
# 
# One of the main principles of computer programming is reusing pre-existing
# code. This has two benefits:
# - Makes updating our code easier cause ideally, we don't have the same
# piece of code twice, so we only have to find the place to update the
# code once
# - Code is more organised
# - Code is written faster
# 
# One of the main ways to do this is using __inheritance__, we will
# define some concepts:
# - __Superclass__: Also called  __parent class__. Class that is
# _inherited by_ another class. In this case, it is the class that is
# used by another class
# - __Subclass__: Also called __child class__. Class that _inherits
# from_ another class. This is the class that uses another class to
# define itself
# 
# When a class B inherits from class A, this means:
# - B gets all the methods and attributes that class A has
# - During __ init __ you need to also _initialize_ the super class, so
# you need to call its constructor using __super__ or the name of the
# parent class
# - To call the member of the super class you have to use the name of
# the class or __super__ (not _self_)
#
# We will create a simple example now using inheritance:
# 
# - class __File__: Represents any type of file
# - class __Audio__: Represents any audio file

class File:
    """
    Represents a file in our system
    
    :cvar __files_created: Counts the number of files created
    :type __files_created: int
    """

    __files_created: int = 0

    def __init__(self, path: str, name: str, mb_size: float, mode: int):
        """
        Initializes a File object

        :param path: Path to the file
        :type path: str
        :param name: File name
        :type name: str
        :param mb_size: Size of the file in MegaBytes
        :type mb_size: float
        :param mode: UNIX file permissions
        :type mode: int
        """

        self.mode_validation(mode)
        self.__file_id: int = random.randint(0, 100000)
        self.path: str = path
        self.name: str = name
        self.__mb_size: float = mb_size
        self.mode: int = mode
        File.__files_created += 1

    @staticmethod
    def mode_validation(mode: int) -> None:
        """
        Validates if the parameter mode is a valid value for a Unix file
        permission. If it is not valid we raise an exception

        :param mode: File mode
        :type mode: int
        :return: None
        :rtype: None
        """

        user_mode: int = mode // 100
        group_mode: int = (mode - user_mode * 100) // 10
        others_mode: int = (mode - user_mode * 100 - group_mode * 10)
        if user_mode not in range(0, 8):
            raise ValueError("This is not a valid file mode: "
                             "User permissions are not correct")
        elif group_mode not in range(0, 8):
            raise ValueError("This is not a valid file mode: "
                             "Group permissions are not correct")
        elif others_mode not in range(0, 8):
            raise ValueError("This is not a valid file mode: "
                             "Other permissions are not correct")

    @property
    def file_id(self) -> int:
        """
        Returns the ID of current file

        :return: attribute __file_id
        :rtype: int
        """

        return self.__file_id

    @file_id.setter
    def file_id(self, new_file_id: int) -> None:
        """
        Forbids users from changing the file ID

        :param new_file_id: New file ID
        :type new_file_id: int
        :return: None
        """

        raise OSError("File ID can't be changed")

    @property
    def mb_size(self) -> float:
        """
        Returns size of the current file in MegaByte

        :return: size of the current file in MB
        :rtype: float
        """

        return self.__mb_size

    @mb_size.setter
    def mb_size(self, new_size: float) -> None:
        """
        Updates the size of the current file. However, we raise an
        exception if the new size is lower than 0 or higher than 300

        :param new_size: New size of the current file
        :type new_size: float
        :return: None
        """

        if new_size < 0:
            raise ValueError("Size can't be lower than 0")
        elif new_size > 300:
            raise SystemError("File size can't be more than 300 MB")
        self.__mb_size = new_size

    @classmethod
    def get_number_of_files(cls) -> int:
        """
        Returns the number of files created in the system

        :return: Number of files created
        :rtype: int
        """

        return cls.__files_created

    def copy_file(self, new_path: str) -> None:
        """
        Copies the current file to the folder given as an argument

        :param new_path: Route to the folder where we want to copy the
        current file
        :type new_path: str
        :return: None
        """

        shutil.copyfile(self.path + "/" + self.name,
                        new_path + "/" + self.name)

    def get_full_path(self) -> str:
        """
        Returns the full path plus name of the file

        :return: Full path plus name of the file
        :rtype: str
        """

        return self.path + "/" + self.name

    @staticmethod
    def get_size_in_n_lower_unit(size: float, n: int) -> float:
        """
        Converts a file size to a unit n levels lower
        For instance to convert from MB to Bytes we have to use n=2
        (n=1 turns to KB, n=2 turns to Bytes)

        :param size: Size to convert

        :param n:
        :return:
        """

        return size * (2 ** 10) ** n

    def size_in_bytes(self) -> float:
        """
        Returns the size of the current file in Bytes

        :return: Size of the file in Bytes
        :rtype: float
        """

        return self.get_size_in_n_lower_unit(self.mb_size, 2)


class AudioFile(File):
    """
    Represents a file whose information is audio/sound. It inherits the
    traits of class File and adds new ones
    
    :cvar __audio_apps: Contains the applications that are installed in 
    the system and also can play audio
    :type __audio_apps: set
    :cvar __audio_files: Counter of audio files created
    :type __audio_files: int
    """

    __audio_apps: set = {"Windows Media Player", "WinAmp",
                         "Apple Music Player"}

    __audio_files: int = 0

    def __init__(self, path: str, name: str, mb_size: float, mode: int,
                 duration_in_secs: int, extension: str, default_app: str):
        """
        Constructor for this class
        
        :param path: Path to the file
        :type path: str
        :param name: Name of the file (includes extension)
        :type name: str
        :param mb_size: Size of the file in MegaBytes
        :type mb_size: float 
        :param mode: UNIX file mode
        :type mode: int
        :param duration_in_secs: Duration of the audio in seconds
        :type duration_in_secs: int 
        :param extension: File extension
        :type extension: str
        :param default_app: Application that will be executed by default
        when trying to open the file
        :type default_app: str
        """
        
        # We need to initialize the object in the super class
        super().__init__(path, name, mb_size, mode)

        self.duration_in_secs: int = duration_in_secs
        self.__extension: str = extension
        # We call this class method from the constructor to check that
        # the value of default_app is valid
        AudioFile.check_valid_audio_app(default_app)
        self.__default_app: str = default_app

    @classmethod
    def check_valid_audio_app(cls, new_audio_app: str) -> None:
        """
        Checks if the input parameter is an application installed in
        our system that can reproduce audio. If not, exception is raised
        
        :param new_audio_app: Candidate application to reproduce 
        an audio file in our system
        :type new_audio_app: str
        :return: None
        """
        
        if new_audio_app not in cls.__audio_apps:
            raise ValueError(
                "{0} is not installed in our system".format(cls.__audio_apps))

    @classmethod
    def get_number_of_audio_files(cls) -> int:
        """
        Returns number of audio files created
        
        :return: Number of audio files created
        :rtype: int
        """
        
        return cls.__audio_files

    @classmethod
    def add_audio_app(cls, new_audio_app: str) -> None:
        """
        After an application that can reproduce audio is installed
        in the system, this method is used to add it to the list of
        available apps that can reproduce audio
        
        :param new_audio_app: Newly installed application that can 
        reproduce audio
        :type new_audio_app: str
        :return: None
        """
        if new_audio_app in cls.__audio_apps:
            raise ValueError("This app is already installed in the system")

        cls.__audio_apps += new_audio_app

    def get_duration_in_minutes_and_seconds(self) -> (int, int):
        """
        Return the duration of the audio in a tuple of minutes, seconds
        
        :return: Tuple with the duration of this audio in minutes, 
        seconds
        :rtype: tuple
        """
        
        minutes: int = self.duration_in_secs // 60
        seconds: int = self.duration_in_secs - (minutes * 60)
        return minutes, seconds

    def get_bytes_per_second(self) -> float:
        """
        Returns the number of bytes per second of this file
        
        :return: Bytes per second
        """
        
        # Notice how we call the method size_in_bytes that belongs to
        # the superclass
        return self.size_in_bytes() / self.duration_in_secs

    @classmethod
    def percentage_audio_files(cls) -> float:
        """
        Returns which percentage of files created are audio files
        
        :return: Percentage of audio files created
        :rtype: float
        """
        # Notice how we class the class method get_number_of_files
        # that belongs to the parent class
        return cls.get_number_of_audio_files() / \
            cls.get_number_of_files() * 100


# Let's create several objects of these classes

# In[3]:


file1: File = File("downloads", "Example.txt", 1, 777)
audio1: AudioFile = AudioFile("Music", "Vampire.mp3", 7.6, 700, 355, "mp3",
                              "Apple Music Player")
file2: File = File("Documents", "Summary.pdf", 10, 777)
audio2: AudioFile = AudioFile("Desktop", "Silence.mp3", 8.6, 770, 405, "mp3",
                              "Apple Music Player")


# We can get the attributes from the parent class in object of type AudioFile

print(audio2.path)
print(audio2.name)
print(audio2.mb_size)
print(audio2.mode)
print(audio2.file_id)


# We can also execute the methods of the superclass

print(AudioFile.get_number_of_files())
print(audio1.size_in_bytes())


# - ### Abstract classes
# 
# There is a special type of classes were related to inheritance.
# __Abstract classes__ are classes that __can't be used to create
# objects.__
# You can use these classes only so other can inherit methods from them.
# 
# An abstract class is:
# - A __class that contains an abstract method__
#     - To define abstract methods, the class needs inherit
#     from __abc.ABC__
#     - We don't need to provide a definition for them (use __pass__)
#     - Declared using the __annotation abc.abstractmethod__
#     - If in a child class an abstract method has not been defined
#     -> Class is still considered abstract class
#     
# - An abstract class __can contain non-abstract methods__
#
# In the following example, we defined __an abstract class called
# CelestialBody__. This creates a blueprint for all the data and
# functionality any celestial body has to include in our code.
# However, a celestial body is not really something we can create objects of.
# We defined __child classes Planet and Star to use and extend the basis
# set in CelestialBody__

class CelestialBody(abc.ABC):
    """
    Represents a celestial body, the concept. Abstract class, objects
    of this class can't be created
    """

    def __init__(self, name: str, radius: float, mass: float, galaxy: str):
        """
        Constructor which will raise an exception is used but will be
        used by its child classes

        :param name: Name of the object
        :type name: str
        :param radius: Radius of the object
        :type radius: float
        :param mass: Mass of the object
        :type mass: float
        :param galaxy: Galaxy where the celestial object is
        :type galaxy: str
        """

        self.name: str = name
        self.radius: float = radius
        self.mass: float = mass
        self.galaxy: str = galaxy

    @abc.abstractmethod
    def get_light_time(self):
        """
        Gets how long light takes to get to the object

        :return: Time light takes to get to the object from its Sun
        :rtype: float
        """

        pass

    def get_object_mean_density(self) -> float:
        """
        Returns object mean density

        :return: Object mean density
        :rtype: float
        """

        return self.mass / (4/3 * math.pi * self.radius ** 3)

    def get_surface_gravity(self) -> float:
        """
        Returns gravity on the surface of the object

        :return: Gravity on the surface of the object
        :rtype: float
        """

        return scg.gravitational_constant * self.mass / self.radius ** 2

    def get_escape_velocity(self):
        """
        Speed needed to escape the celestial body by another object

        :return: Escape velocity
        :rtype: float
        """

        pass

    @classmethod
    @abc.abstractmethod
    def get_object_types(cls):
        """
        Returns the types of this class of celestial body

        :return: Types of this class of celestial body
        :rtype: [str]
        """

        pass

    @abc.abstractmethod
    def get_object_type(self):
        """
        Returns the object type of this instance

        :return: Object type of this instance
        :rtype: str
        """

        pass


class Star(CelestialBody):
    """
    Represents stars in the universe

    :cvar star_types: Types of stars
    :type: [str]
    """

    star_types: [str] = ["O", "B", "A", "F", "G", "K", "M"]

    def __init__(self, name: str, radius: float, mass: float, galaxy: str,
                 star_type: str, brightness: float):
        """
        Constructor for class Star

        :param name: Name of the star
        :type name: str
        :param radius: Radius of the star
        :type radius: float
        :param mass: Mass of the star
        :type mass: float
        :param galaxy: Galaxy where the star is
        :type galaxy: str
        :param star_type: Type of star
        :type star_type: str
        :param brightness: Brightness of the star
        :type brightness: float
        """

        super().__init__(name, radius, mass, galaxy)
        if star_type not in Star.star_types:
            raise ValueError("{0} can't be the value for sun_type,"
                             "it should be one of the following: {1}".
                             format(star_type, Star.star_types))
        else:
            self.sun_type: str = star_type

        self.brightness: float = brightness
        # This will contain the planets that orbit this star
        self.__planets_in_orbit: [Planet] = []

    @property
    def planets_in_orbit(self) -> []:
        """
        Returns planets orbiting this star

        :return: Planets orbiting this star
        :rtype: [Planet]
        """

        return self.__planets_in_orbit

    @planets_in_orbit.setter
    def planets_in_orbit(self, planet) -> None:
        """
        Adds a new planet to the list of planets orbiting this star

        :param planet: Planet that orbits this star
        :type planet: Planet
        :return: None
        """

        if not isinstance(planet, Planet):
            raise ValueError("We can only add planets to the orbit of "
                             "this star")
        self.__planets_in_orbit.append(planet)

    def get_light_time(self) -> float:
        # In this case, a star produces light so the time light takes
        # to get to the surface of the star is 0
        return 0

    def get_escape_velocity(self) -> float:
        return math.sqrt(2 * scg.gravitational_constant *
                         self.mass * self.radius)

    @classmethod
    def get_object_types(cls) -> [str]:
        """
        Returns the types of stars

        :return: [str]
        """

        return cls.star_types

    def get_object_type(self) -> str:
        return self.sun_type


class Planet(CelestialBody):
    """
    Represents a planet

    :cvar planet_types: Types of planets
    :type: [str]
    """

    planet_types: [str] = ["Rocky", "Gas", "Ice"]

    def __init__(self, name: str, radius: float, mass: float, galaxy: str,
                 planet_type: str, star_sun: Star, distance_from_sun: float):
        """
        Constructor for class Planet

        :param name: Name of the planet
        :type name: str
        :param radius: Radius of the planet
        :type radius: float
        :param mass: Mass of the planet
        :type mass: float
        :param galaxy: Galaxy where the planet is:
        :type galaxy: str
        :param planet_type: Type of planet
        :type planet_type: str
        :param star_sun: Sun the planet orbits around
        :type star_sun: Star
        :param distance_from_sun: Distance from its sun
        :type distance_from_sun: float
        """

        super().__init__(name, radius, mass, galaxy)

        if planet_type not in Planet.planet_types:
            raise ValueError("{0} is not a valid planet type, planet"
                             "types are {1}".format(planet_type,
                                                    Planet.planet_types))
        else:
            self.planet_type: str = planet_type
        self.sun: Star = star_sun
        # We defined the set for this attribute to append the new
        # element to the list instead of setting the argument to this
        # new value
        self.sun.planets_in_orbit = self
        self.distance_from_sun: float = distance_from_sun

    def get_escape_velocity(self) -> float:
        return math.sqrt(2 * scg.gravitational_constant *
                         self.mass * self.radius)

    @classmethod
    def get_object_types(cls) -> [str]:
        """
        Returns the different type of planets

        :return: Type of planets
        :rtype: [str]
        """

        return cls.planet_types

    def get_object_type(self) -> str:
        """
        Returns the type of this planet

        :return: Type of this planet
        :rtype: str
        """

        return self.planet_type

    def get_light_time(self) -> float:
        return scg.speed_of_light * self.distance_from_sun


class PlanetIncomplete(CelestialBody):
    """
    Represents a planet

    :cvar planet_types: Types of planets
    :type: [str]
    """

    planet_types: [str] = ["Rocky", "Gas", "Ice"]

    def __init__(self, name: str, radius: float, mass: float, galaxy: str,
                 planet_type: str, star_sun: Star, distance_from_sun: float):
        """
        Constructor for class Planet

        :param name: Name of the planet
        :type name: str
        :param radius: Radius of the planet
        :type radius: float
        :param mass: Mass of the planet
        :type mass: float
        :param galaxy: Galaxy where the planet is:
        :type galaxy: str
        :param planet_type: Type of planet
        :type planet_type: str
        :param star_sun: Sun the planet orbits around
        :type star_sun: Star
        :param distance_from_sun: Distance from its sun
        :type distance_from_sun: float
        """

        super().__init__(name, radius, mass, galaxy)

        if planet_type not in Planet.planet_types:
            raise ValueError("{0} is not a valid planet type, planet"
                             "types are {1}".format(planet_type,
                                                    Planet.planet_types))
        else:
            self.planet_type: str = planet_type
        self.sun: Star = star_sun
        self.distance_from_sun: float = distance_from_sun

    def get_escape_velocity(self) -> float:
        return math.sqrt(2 * scg.gravitational_constant *
                         self.mass * self.radius)

    @classmethod
    def get_object_types(cls) -> [str]:
        """
        Returns the different type of planets

        :return: Type of planets
        :rtype: [str]
        """

        return cls.planet_types


# If we try to create an object of class CelestialBody, a TypeError
# exception will be raised as it is an abstract class
# (it contains abstract methods)

try:
    celestial_body = CelestialBody("Andromeda", 10**20, 2*10*32, "Andromeda")
except TypeError as e:
    print("We can't create object of CelestialBody: {0}".format(e))


# However, we can create objects of the class star

sun = Star("Helios", 6.957 * 10 ** 6, 1.9885*10**30, "Milky Way", "G", -26.74)

print(sun.get_escape_velocity())
print(sun.get_surface_gravity())


# We can't create objects of the class PlanetIncomplete as we left some
# abstract methods undefined. That makes this an abstract class and
# therefore, when trying to create an object from it a TypeError
# exception will be raised

try:
    planet_incomplete = PlanetIncomplete("Incomplete", 1, 1 * 10**2,
                                         "Andromeda", "Rocky", sun, 10*10**2)
except TypeError as e:
    print("We can't create object of PlanetIncomplete: {0}".format(e))


# But we can create objects of the class Planet
venus = Planet("Venus", 6.0518 * 10 ** 6, 4.8675*10**24, "Milky Way", "Rocky",
               sun, 108.94 * 10 ** 6)

print(sun.planets_in_orbit[0].name)
print(venus.get_surface_gravity())
