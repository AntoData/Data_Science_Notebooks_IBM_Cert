#!/usr/bin/env python
# coding: utf-8
import abc
import datetime
import random

# ### Multiple Inheritance
#
# In Python, classes can inherit from more than one class which can be
# very useful.
# The syntax is the same as with regular inheritance

# There might be the case that two or several parent classes provide
# methods with the same name. In that case, which one will be the method
# that our new child class will provide?
#
# When a method is called Python's __Method Resolution Order__ will
# search for it in the following places:
#
# 1. The class
# 2. Parents: When there are _several parents classes_, we look for the
# __method parent class by parent class in the list of parent classes
# from left to right__
# 3. Finally, parents of parents...

# In the following example, we have:
#
# - An abstract class __Employee__
# - Classes __Developer, QA, TeamLeader__
# - Class __DeveloperTeamLeader__ that represents a team leader who
# leads a group of developers (and _therefore is both developer and
# team leader_)


class Employee(abc.ABC):
    """
    Abstract class that represents an employee of the company
    """

    def __init__(self, name: str, date_joined: datetime, yearly_salary: int):
        """
        Constructor of the class that represents employees of the company

        :param name: Name of the employee
        :type name: str
        :param date_joined: Date employee joined the company
        :type date_joined: datetime
        :param yearly_salary: Salary per year
        :type yearly_salary: int
        """

        # ID of the employee
        self.__eid: int = random.randint(0, 100000)
        self.name: str = name
        self.__date_joined: datetime = date_joined
        self.__yearly_salary: int = yearly_salary
        # Tasks employee has finished completely
        self.items_done: int = 0

    @property
    def eid(self) -> int:
        """
        Returns the ID of the employee

        :return: ID of the employee
        :rtype: int
        """

        return self.__eid

    @property
    def date_joined(self) -> datetime:
        """
        Returns the date the employee joined the company

        :return: Date the employee joined the company
        :rtype: datetime
        """

        return self.__date_joined

    @property
    def yearly_salary(self) -> int:
        """
        Returns salary per year

        :return: Salary per year
        :rtype: int
        """

        return self.__yearly_salary

    @yearly_salary.setter
    def yearly_salary(self, new_yearly_salary: int):
        """
        Sets new salary for employee. New salary has to be greater than
        current salary

        :param new_yearly_salary: New salary per year
        :type new_yearly_salary: int
        :return: None
        """

        if new_yearly_salary < self.__yearly_salary:
            raise ValueError("New salary can't be lower than current salary")
        else:
            self.__yearly_salary = new_yearly_salary

    @abc.abstractmethod
    def get_monthly_payslip(self) -> float:
        """
        Returns quantity to pay per month

        :return: Quantity to pay per month
        """
        pass

    @abc.abstractmethod
    def get_performance_review(self) -> float:
        """
        Works out the performance index for this employee

        :return: Performance index for this employee
        """
        pass

    def get_time_in_the_company_days(self) -> float:
        """
        Returns how many days has this employee work for the company
        since they joined

        :return: Hours worked by this employee
        :rtype: float
        """

        # We get how much time has passed since the employee joined
        time_in_company: datetime = datetime.datetime.now() - self.date_joined

        # Now we work out how many working days that is. First, we work
        # out how many days this employee has been working for the
        # company and then how many weeks and because we work 5 days per
        # week, we have now the solution
        time_in_company_days: float = time_in_company.days / 7 * 5 * 8
        return time_in_company_days


class Developer(Employee):
    """
    Represents an employee who is a developer in the company

    :cvar instances_of_developers: Number of developers in the company
    :type instances_of_developers: int
    """

    instances_of_developers: int = 0

    def __init__(self, name: str, date_joined: datetime, yearly_salary: int,
                 language: str):
        """
        Constructor of the class Developer

        :param name: Name of the employee
        :type name: str
        :param date_joined: Date the employee joined the company
        :type date_joined: datetime
        :param yearly_salary: Salary per year
        :type yearly_salary: int
        :param language: Main language used by the developer
        :type language: str
        """

        # We have to create the parent instance (Employee)
        Employee.__init__(self, name, date_joined, yearly_salary)
        self.language: str = language

        # The id of the developer is the current number of developers
        self.__dev_id: int = Developer.instances_of_developers
        # We have to increase the number of developers
        Developer.instances_of_developers += 1

    def get_monthly_payslip(self) -> float:
        """
        Returns the salary per month, in this case the salary per year
        is paid in 12 payslips (one per month)

        :return: Salary per year for developers (12 payslips)
        :rtype: float
        """

        return self.yearly_salary / 12

    def get_performance_review(self) -> float:
        """
        Works out the performance index of this developer

        :return: Performance index of this developer
        :rtype: float
        """

        time_in_company_hours: float = self.get_time_in_the_company_days()
        return self.items_done * 10 ** 5 / time_in_company_hours

    @property
    def dev_id(self) -> int:
        """
        Returns developer id

        :return: Developer id
        :rtype: int
        """

        return self.__dev_id


class QA(Employee):
    """
    Represents an employee that works as a QA

    :cvar instances_of_qa: Number of QAs working in the company
    :type instances_of_qa: int
    """

    instances_of_qa: int = 0

    def __init__(self, name: str, date_joined: datetime, yearly_salary: int,
                 field: str):
        """
        Constructor of class QA

        :param name: Name of the employee
        :type name: str
        :param date_joined: Date the employee joined the company
        :type date_joined: datetime
        :param yearly_salary: Salary per year
        :type yearly_salary: int
        :param field: Speciality of the QA (performance, automation...)
        :type field: str
        """

        # We have to create the instance of employee
        Employee.__init__(self, name, date_joined, yearly_salary)
        self.field: str = field
        # The ID of the QA is the current instances of QA
        self.__qa_id = QA.instances_of_qa
        # We need to increase the number of QAs working in the company
        QA.instances_of_qa += 1

    def get_monthly_payslip(self) -> float:
        """
        Returns the salary per month, in this case the salary per year
        is paid in 12 payslips (one per month)

        :return: Salary per year for QAs (12 payslips)
        :rtype: float
        """

        return self.yearly_salary / 12

    def get_performance_review(self) -> float:
        """
        Works out the performance index of this QA

        :return: Performance index of this QA
        :rtype: float
        """

        time_in_company_hours: float = self.get_time_in_the_company_days()
        return self.items_done * 10 ** 5 / time_in_company_hours * 0.4

    @property
    def qa_id(self) -> int:
        """
        Returns the id of the QA

        :return: ID of the QA
        :rtype: int
        """

        return self.__qa_id


class TeamLeader(Employee):
    """
    Represents an employee that is a team leader

    :cvar instances_of_team_leader: Number of team leaders in the
    company
    :type instances_of_team_leader: int
    """

    instances_of_team_leader: int = 0

    def __init__(self, name: str, date_joined: datetime, yearly_salary: int,
                 team: str):
        """
        Constructor of class TeamLeader

        :param name: Name of the employee
        :type name: str
        :param date_joined: Date when the employee joined the company
        :type date_joined: datetime
        :param yearly_salary: Salary per year
        :type yearly_salary: int
        :param team: Team this employee leads
        :type team: str
        """

        # We need to create the instance of Employee
        Employee.__init__(self, name, date_joined, yearly_salary)
        self.team: str = team

        # The ID of the team leader is the current number of team
        # leaders in the company
        self.__leader_id: int = TeamLeader.instances_of_team_leader
        # We need to increase the number of instances of team leaders
        TeamLeader.instances_of_team_leader += 1

    def get_performance_review(self) -> float:
        """
        Works out the performance index of this team leader

        :return: Performance index of this team leader
        :rtype: float
        """

        time_in_company_hours: float = self.get_time_in_the_company_days()
        return self.items_done * 10 ** 4 / time_in_company_hours * 0.9

    def get_monthly_payslip(self) -> float:
        """
        Returns the salary per month, in this case the salary per year
        is paid in 14 payslips (one per month and two extras)

        :return: Salary per year for QAs (14 payslips)
        :rtype: float
        """

        return self.yearly_salary / 14

    @property
    def leader_id(self) -> int:
        return self.__leader_id


class DeveloperTeamLeader(TeamLeader, Developer):
    """
    Represents a Developer who is also a team leader
    """

    def __init__(self, name: str, date_joined: datetime, yearly_salary: int,
                 language: str, team: str):
        """
        Constructor of class DeveloperTeamLeader

        :param name: Name of the employee
        :type name: str
        :param date_joined: Date the employee joined the company
        :type date_joined: datetime
        :param yearly_salary: Salary per year
        :type yearly_salary: int
        :param language: Programming language
        :type language: str
        :param team: Team this employee leads
        :type team: str
        """

        # We need to create the instance of TeamLeader
        TeamLeader.__init__(self, name, date_joined, yearly_salary, team)
        # We need to create the instance of Developer
        Developer.__init__(self, name, date_joined, yearly_salary, language)


# We can see that all 3 classes that inherit from __Employee__:
# - Developer
# - QA
# - TeamLeader
#
# Implement the methods:
#
# - get_monthly_payslip
# - get_performance_review
#
# We can create objects of these classes


dev_joined: datetime = datetime.datetime(2007, 7, 30)
dev1: Developer = Developer(
    "John Doe", dev_joined, 42000, "Python", )

dev_team_leader_joined: datetime = datetime.datetime(2007, 7, 30)
dev_team_leader: DeveloperTeamLeader = DeveloperTeamLeader(
    "John Doe", dev_team_leader_joined, 42000, "Python", "Solutions")

dev_team_leader.items_done = 200


# As in class DeveloperTeamLeader the first parent class from left to
# right is TeamLeader, the methods:
#
# - get_monthly_payslip
# - get_performance_review
#
# That are inherited from both parent classes will be offered from TeamLeader.
# So when we call them from an object of type DeveloperTeamLeader, we
# are calling the same version of the method as in TeamLeader


print(dev_team_leader.get_monthly_payslip())
print(dev_team_leader.get_performance_review())


# ### Metaclasses
#
# Metaclasses are class whose instances are other classes. So basically
# a metaclass builds classes and can be used to modify a class at class
# creation time.
#
# All classes that inherit from __type__ are __metaclasses__
#
# A class is defined by 3 args:
#
# - name: Name of the class
# - bases: Tuple with the parent classes of the class
# - dict: Object dict that contains the class methods and attributes of
# the class
#
# You declare a metaclass like a regular __class that inherits from type__
# You need to define the method __ new __ (mcs, name, bases, dict)
#
# We are going to provide an example of this.
# - We have a class called RandomStringId that contains a class method
# that creates a 5 digit string id (from 0 to 10000, we fill with 0s the
# remaining positions) -> All our new classes need to inherit from that
# one, so they all can create an id when creating a new object
# - We have a method called get_instances that will return all the
# instances created of the current class
# - We will add a parameter called instances that will count how many
# instances of each class we have created
#
# So we will:
# - add the class RandomStringId to the tuple bases
# - add the class method get_instances to the object dict
# - add the class attribute instances to the object dict

class RandomStringId:
    """
    Creates a 5 digit string id which has to be unique

    :cvar used_ids: Here we keep record of ids that have been already
    given to other instances
    """

    used_ids: [str] = []

    @classmethod
    def create_random_new_id(cls) -> str:
        """
        Returns a unique 5 digit string id for an object

        :return: Unique 5 digit string id
        """

        # We get a random number between 0 and 99999 and convert it to
        # string
        str_number_id: str = str(random.randint(0, 99999))
        len_str_id: int = len(str_number_id)
        len_zeros_id: int = 5 - len_str_id
        # We fill the string with 0s until the whole string (0s + id)
        # has length 5
        str_id: str = "0" * len_zeros_id + str_number_id
        # If the id had been already used, we start again and try to
        # get a new one
        if str_id in cls.used_ids:
            str_id = cls.create_random_new_id()
        else:
            # Otherwise, we add it to the list of used ids
            cls.used_ids.append(str_id)
        return str_id


@classmethod
def get_instances(cls):
    """
    Returns the class parameter instances

    :param cls: Current class
    :return: Class parameter instances
    """

    return cls.instances


class MetaClassAddInstancesCounter(type):
    """
    Metaclass that will make a class inherit from class RandomStringId
    adds the class attribute instances that should be used to count
    the number of instances of the class and a class method that returns
    that parameter
    """

    def __new__(mcs, name: str, bases: tuple, v_dict: dict):
        """
        Initializes the class

        :param name: Name of the class
        :type name: str
        :param bases: Tuple with the classes that are inherited from in
        the class we are creating
        :type bases: tuple
        :param v_dict: Object __dict__ of the class (contains all the
        class attributes and class methods
        :type v_dict: dict
        """

        # We create a new tuple that adds the class RandomStringId to
        # the parents of the class we are creating
        new_bases = bases + (RandomStringId,)
        instances: int = 0
        # We add the class attribute instances to this class
        v_dict["instances"] = instances
        # We add the class method get_instances to this class
        v_dict["get_instances"] = get_instances
        # We call the original method __new__ with the new parameters
        obj = super().__new__(mcs, name, new_bases, v_dict)
        return obj


# We will use this metaclass in the class Patient that represents a
# patient in a health insurance company

class Patient(metaclass=MetaClassAddInstancesCounter):
    """
    Represents a patient of a health insurance company
    """

    def __init__(self, name: str, surname: str, country: str, age: int,
                 date_insurance: datetime, active: bool):
        """
        Constructor of class Patient

        :param name: Name of the patient
        :type name: str
        :param surname: Surname of the patient
        :type surname: str
        :param country: Country where patient was born
        :type country: str
        :param age: Age of the patient
        :type age: int
        :param date_insurance: Date when the contract was signed
        :type date_insurance: datetime
        :param active: Is contract active
        :type active: bool
        """

        self.name: str = name
        self.surname: str = surname
        self.country: str = country
        self.age: int = age
        self.date_insurance: datetime = date_insurance
        self.active: bool = active

        # We use the attribute parameter that has been added in the
        # metaclass and increase it by 1
        Patient.instances += 1
        # We use the function create_random_new_id we should have access
        # to now as a class method thanks to the metaclass that added it
        # to the class member of this class
        self.v_id = Patient.create_random_new_id()


patient1: Patient = Patient("J", "K", "Spain", 11, datetime.datetime(
    year=2010, month=10, day=20), True)

# We can get the instance parameter v_id that was created using the
# method create_random_new_id from the class RandomStringId which was
# added as a parent class by the metaclass
print(patient1.v_id)
# We can use the class method get_instances added in the metaclass to
# the class Patient that uses the class attribute instances that
# was appended to our class the same way
print(Patient.get_instances())


# - ### type() to create classes
#
# __type(name, bases, dict)__ can be used also to create classes
# dynamically

# First we define a parent class that will be added to the tuple bases
# to be the parent class of our new class

class RandomStringId:
    """
    Creates a 5 digit string id which has to be unique

    :cvar used_ids: Here we keep record of ids that have been already
    given to other instances
    """

    used_ids: [str] = []

    @classmethod
    def create_random_new_id(cls) -> str:
        """
        Returns a unique 5 digit string id for an object

        :return: Unique 5 digit string id
        """

        # We get a random number between 0 and 99999 and convert it to
        # string
        str_number_id: str = str(random.randint(0, 99999))
        len_str_id: int = len(str_number_id)
        len_zeros_id: int = 5 - len_str_id
        # We fill the string with 0s until the whole string (0s + id)
        # has length 5
        str_id: str = "0" * len_zeros_id + str_number_id
        # If the id had been already used, we start again and try to
        # get a new one
        if str_id in cls.used_ids:
            str_id = cls.create_random_new_id()
        else:
            # Otherwise, we add it to the list of used ids
            cls.used_ids.append(str_id)
        return str_id


# Now we define:
# - __init__ for the new class
# - A method called get_days_insurance_contract that returns the number
# of days since the insurance contract was signed
# - A class parameter called instances that will count the number of
# instances of our class


def __init__(self, name: str, surname: str, country: str, age: int,
             date_insurance: datetime, active: bool):
    """
    Constructor of class Patient

    :param name: Name of the patient
    :type name: str
    :param surname: Surname of the patient
    :type surname: str
    :param country: Country where patient was born
    :type country: str
    :param age: Age of the patient
    :type age: int
    :param date_insurance: Date when the contract was signed
    :type date_insurance: datetime
    :param active: Is contract active
    :type active: bool
    """

    self.name: str = name
    self.surname: str = surname
    self.country: str = country
    self.age: int = age
    self.date_insurance: datetime = date_insurance
    self.active: bool = active
    DynamicPatient.instances += 1

    # We use the function create_random_new_id we should have access
    # to now as a class method thanks to the metaclass that added it
    # to the class member of this class
    self.v_id = RandomStringId.create_random_new_id()


def get_days_insurance_contract(self) -> int:
    """
    Returns number of days passed since contract was signed

    :return: Number of days passed since contract was signed
    """

    now: datetime = datetime.datetime.now()
    time_contract: datetime.timedelta = now - self.date_insurance
    return time_contract.days


instances: int = 0

# Now we create the parameter bases that will be used by __type__ to
# build our class

patient_bases: tuple = (RandomStringId,)

# Also now we create the parameter __dict__ for our new class

patient__dict__: dict = {"instances": instances, "__init__": __init__,
                         "get_days_insurance_contract":
                             get_days_insurance_contract}

# We can now create the class and create instances of it

DynamicPatient = type("Patient", patient_bases, patient__dict__)

dynamic_patient: DynamicPatient = DynamicPatient("John", "Doe", "Canada", 41,
                                                 datetime.datetime(year=2010,
                                                                   month=5,
                                                                   day=30),
                                                 True)

print(DynamicPatient.instances)
print(dynamic_patient.get_days_insurance_contract())