#!/usr/bin/env python
# coding: utf-8
import datetime
import math

# # Decorators
# 
# In summary, __decorators__ are __functions that modify other
# functions__.
# Python allows us to define these functions that will modify the
# behaviour of other functions.
# 
# Because of this decorators are __a design pattern__. When we see
# several functions
# have similar algorithms at the beginning or the end or need to do the
# same transformations
# in their arguments, we can create a decorator and use it to reuse code
# and avoid
# repetition
# 
# - __Decorator__: Function that modifies the algorithm of another
# function
# - __Decorated function__: Function that is modified by a decorator

# In the following example, we defined a decorator that gets the current
# time before executing the decorated function, executes it and then
# gets the time after execution and works out the execution time:


def simple_decorator(func):
    """
    This is the new callable object that will receive the function
    as parameter. This is used as the decorator, the annotation
    above the function that will be decorated (modified)

    :param func: Callable object
    :return: Inner decorator that will decorate func
    """

    def inner_wrapper(*args, **kwargs):
        """
        This function receives the parameters of the decorated function
        and here we will perform the modification of the decorated
        function. In this case, we just get the time before and print
        it, then execute the function, get the time after and print it
        and work out execution time

        :param args: Positional arguments of the decorated function
        :param kwargs: Keyword arguments of the decorated function
        :return: Result of executing the decorated function
        """

        before: datetime = datetime.datetime.now()
        print("Before execution: {0}".format(before))
        res = func(*args, **kwargs)
        after: datetime = datetime.datetime.now()
        print("After execution: {0}".format(after))
        execution_time = after - before
        print("Execution time: {0}".format(execution_time))
        return res
    return inner_wrapper


@simple_decorator
def get_area_of_sphere(radius: float) -> float:
    """
    Works out the area of a sphere
    :param radius: Radius of the sphere

    :return: Area of the sphere
    :rtype: float
    """
    
    return 4 * math.pi * (radius ** 2)


print(get_area_of_sphere(2))


# We can do more operations like modifying the result of the function
# In the following example we created a decorator that reverses the
# result of a function that returns the divisors of a certain number so
# we get the number from greater to lower


def decorator_reverse_results(func):
    """
    This is the new callable object that will receive the function
    as parameter. This is used as the decorator, the annotation
    above the function that will be decorated (modified)

    :param func: Callable object
    :return: Inner decorator that will decorate func
    """
    
    def inner_reverse_results(*args, **kwargs):
        """
        This function receives the parameters of the decorated function
        and here we will perform the modification of the decorated
        function. In this case, we modify the result. We reserve
        the order of the elements in the array that is returned

        :param args: Positional arguments of the decorated function
        :param kwargs: Keyword arguments of the decorated function
        :return: Result of executing the decorated function
        """
        
        res = func(*args, **kwargs)
        res = res[::-1]
        return res
    return inner_reverse_results


@decorator_reverse_results
def find_divisors_number(number: int) -> [int]:
    divisors: [int] = []
    for i in range(1, int(number/2)+1):
        if number % i == 0:
            divisors.append(i)
    return divisors


print(find_divisors_number(10))


# Another example, we can modify the input parameters. Imagine you use a
# function to work out the area of triangles. The information about the
# base and height of triangles is given in metres to that function but
# we need to return the result in inches so another
# provider can use this function. We can use a decorator to convert
# the arguments from metres to inches


def decorator_measurements(func):
    """
    This is the new callable object that will receive the function
    as parameter. This is used as the decorator, the annotation
    above the function that will be decorated (modified)

    :param func: Callable object
    :return: Inner decorator that will decorate func
    """

    def inner_from_metres_to_inches(*args, **kwargs):
        """
        This function receives the parameters of the decorated function
        and here we will perform the modification of the decorated
        function. In this case, we convert the arguments in metres to
        inches

        :param args: Positional arguments of the decorated function
        :param kwargs: Keyword arguments of the decorated function
        :return: Result of executing the decorated function
        """
        
        inches_conversion_rate = 39.3701
        args_inches: [] = []
        kwargs_inches: dict = {}
        
        # We get every positional argument and convert it to inches
        for arg in args:
            args_inches.append(arg * inches_conversion_rate)
        # We get every keyword argument and convert it to inches
        for key in kwargs.keys():
            kwargs_inches[key] = kwargs[key] * inches_conversion_rate
        
        # We use the arguments in inches to execute the function
        res = func(*args_inches, **kwargs_inches)
        return res

    return inner_from_metres_to_inches


@decorator_measurements
def get_area_of_triangle(base_metres: float, height_metres: float) -> float:
    """
    Return the area of a triangle in metres (but the decorator will
    convert it to inches)

    :param base_metres: Size of the base in metres
    :param height_metres: Height of the triangle in metres
    :return: Area of the triangle in m2 (because
    """
    return 1 / 2 * base_metres * height_metres


base_triangle_metres: float = \
    float(input("What's the size of the base in metres? "))
height_triangle_metres: float = \
    float(input("What's the height in metres? "))

area_inches: float = get_area_of_triangle(base_triangle_metres,
                                          height_triangle_metres)
print("The area of the triangle is {0} square inches".format(area_inches))


# In general, we can modify the function as we see fit. In this example,
# the formula to work out the slope of a line can return an exception if
# it is a vertical line. It will return a ZeroDivisionError. In the
# decorator we will handle that exception and return infinite instead


def decorator_handles_division_by_zero(func):
    """
    This is the new callable object that will receive the function
    as parameter. This is used as the decorator, the annotation
    above the function that will be decorated (modified)

    :param func: Callable object
    :return: Inner decorator that will decorate func
    """

    def inner_handles_division_by_zero(*args, **kwargs):
        """
        This function receives the parameters of the decorated function
        and here we will perform the modification of the decorated
        function. In this case, if the operation returns a
        ZeroDivisionError we handle it and return infinite

        :param args: Positional arguments of the decorated function
        :param kwargs: Keyword arguments of the decorated function
        :return: Result of executing the decorated function
        """

        try:
            res = func(*args, **kwargs)
            return res
        except ZeroDivisionError as e:
            print("Vertical line: {0}".format(e))
            print("We will return infinite")
        return math.inf
    return inner_handles_division_by_zero


@decorator_handles_division_by_zero
def get_slope_of_a_line(x1: int, y1: int, x2: int, y2: int):
    return (y2 - y1) / (x2 - x1)


print(get_slope_of_a_line(1, 1, 1, 2))
print(get_slope_of_a_line(1, 1, 2, 2))


# __Decorators can take arguments also__
# 
# These arguments can be applied in the namespace of the decorator and
# therefore used to modify the algorithm

# We will reuse the example we created where we convert the area of a
# triangle from metres to inches but in this case, we will provide
# the conversion rate ourselves, so we can convert from metres to any
# other unit


def conversion_decorator(conversion_rate: float):
    """
    Receives the parameter that can be accessed in the namespace
    of the decorator and therefore used as an argument to modify
    the algorithm of the decorated function. This is used as the
    decorator, the annotation above the function that will be
    decorated (modified)

    :param conversion_rate: Rate to convert the arguments of the
    decorated function to another unit

    :return: The result of the decorated function where the arguments
    have been converted to another unit using this conversion rate
    """

    def function_to_convert(func):
        """
        This is the new callable object that will receive the function
        as parameter.

        :param func: Callable object
        :return: Inner decorator that will decorate func
        """

        def inner_conversion_decorator(*args, **kwargs):
            """
            This function receives the parameters of the decorated
            function and here we will perform the modification of the
            decorated function. In this case, we convert the arguments
            in one unit to another unit using the parameter of the
            decorator called conversion_rate

            :param args: Positional arguments of the decorated function
            :param kwargs: Keyword arguments of the decorated function
            :return: Result of executing the decorated function
            """

            converted_args: [] = []
            converted_kwargs: dict = {}
            # Get every positional argument and apply conversion rate
            print("Positional arguments before conversion: {0}".format(args))
            for arg in args:
                converted_args.append(arg * conversion_rate)
            print("Positional arguments after conversion: {0}".format(
                converted_args))

            # Get every keyword argument and apply conversion rate
            print("Keyword arguments before conversion: {0}".format(kwargs))
            for v_key in converted_kwargs.keys():
                converted_kwargs[v_key] = kwargs[v_key] * conversion_rate
            print("Keyword arguments after conversion: {0}".format(
                converted_args))

            res = func(*converted_args, **converted_kwargs)
            print("Result of the function is: {0}".format(res))
            return res
        return inner_conversion_decorator
    return function_to_convert


# We apply the conversion rate from metres to inches:
# 1 metre is 39.3701 inches
@conversion_decorator(39.3701)
def get_area_of_triangle(base_metres: float, height_metres: float) -> float:
    """
    Return the area of a triangle in metres (but the decorator will
    convert it to inches)

    :param base_metres: Size of the base in metres
    :param height_metres: Height of the triangle in metres
    :return: Area of the triangle in m2 (because
    """
    return 1 / 2 * base_metres * height_metres


base_triangle_metres: float = \
    float(input("What's the size of the base in metres? "))
height_triangle_metres: float = \
    float(input("What's the height in metres? "))

area_inches: float = get_area_of_triangle(base_triangle_metres,
                                          height_triangle_metres)
print("The area of the triangle is {0} square inches".format(area_inches))


# - ### Decorators can be classes too
# 
# We can build classes that will act as decorators the same way that
# functions can.
# In the most simple case, we have to keep the following structure

# Reusing the example with the function that works out the area of a
# triangle

class ToInchesDecoratorClass:
    """
    Decorator class. This will be used as an annotation above a function
    that will modify the algorithm of the decorated function
    """

    def __init__(self, func):
        """
        Constructor for the decorator class, receives the decorated
        function

        :param func: Decorated function
        """

        self.func = func

    def __call__(self, *args, **kwargs):
        """
        Receives the arguments of the decorated function and here it is
        where we will modify its algorithm. In this case, we convert
        the parameters of the decorated function from metres to inches
        so the result is in square inches

        :param args: Positional arguments of the decorated function
        :param kwargs: Keyword arguments of the decorated function
        :return: Result of the function in inches
        """

        inches_conversion_rate = 39.3701
        args_inches: [] = []
        kwargs_inches: dict = {}

        # We get every positional argument and convert it to inches
        for arg in args:
            args_inches.append(arg * inches_conversion_rate)
        # We get every keyword argument and convert it to inches
        for key in kwargs.keys():
            kwargs_inches[key] = kwargs[key] * inches_conversion_rate

        # We use the arguments in inches to execute the function
        res = self.func(*args_inches, **kwargs_inches)
        return res


@ToInchesDecoratorClass
def get_area_of_triangle(base_metres: float, height_metres: float) -> float:
    """
    Return the area of a triangle in metres (but the decorator will
    convert it to inches)

    :param base_metres: Size of the base in metres
    :param height_metres: Height of the triangle in metres
    :return: Area of the triangle in m2 (because
    """
    return 1 / 2 * base_metres * height_metres


base_triangle_metres: float = \
    float(input("What's the size of the base in metres? "))
height_triangle_metres: float = \
    float(input("What's the height in metres? "))

area_inches: float = get_area_of_triangle(base_triangle_metres,
                                          height_triangle_metres)
print("The area of the triangle is {0} square inches".format(area_inches))


# - __Class decorators can take arguments__, same way that decorated
# functions can take arguments

class ConversionDecoratorClass:
    """
    Decorator class. This will be used as an annotation above a function
    that will modify the algorithm of the decorated function
    """

    def __init__(self, conversion_rate: float):
        """
        Constructor for the decorator class, receives the argument
        passed on to the decorator

        :param conversion_rate: Argument passed on to the decorator,
        rate by which we will turn the arguments of the decorated
        function from one unit to another
        """

        self.conversion_rate: float = conversion_rate

    def __call__(self, func):
        """
        Receives the function to decorate

        :param func: Function to decorate
        :return: inner_wrapper
        """

        def inner_wrapper(*args, **kwargs):
            """
            Receives the arguments of the decorated function and here it
             is where we will modify its algorithm. In this case, we
            convert the parameters of the decorated function from metres
             to another unit using the instance attribute of this class
             conversion_rate

            :param args: Positional arguments of the decorated function
            :param kwargs: Keyword arguments of the decorated function
            :return: Result of the function in inches
            """

            args_inches: [] = []
            kwargs_inches: dict = {}

            # We get every positional argument and convert it to inches
            for arg in args:
                args_inches.append(arg * self.conversion_rate)
            # We get every keyword argument and convert it to inches
            for key in kwargs.keys():
                kwargs_inches[key] = kwargs[key] * self.conversion_rate

            # We use the arguments in inches to execute the function
            res = func(*args_inches, **kwargs_inches)
            return res
        return inner_wrapper


# We apply the conversion rate from metres to inches:
# 1 metre is 39.3701 inches
@ConversionDecoratorClass(39.3701)
def get_area_of_triangle(base_metres: float, height_metres: float) -> float:
    """
    Return the area of a triangle in metres (but the decorator will
    convert it to inches)

    :param base_metres: Size of the base in metres
    :param height_metres: Height of the triangle in metres
    :return: Area of the triangle in m2 (because
    """
    return 1 / 2 * base_metres * height_metres


base_triangle_metres: float = \
    float(input("What's the size of the base in metres? "))
height_triangle_metres: float = \
    float(input("What's the height in metres? "))

area_inches: float = get_area_of_triangle(base_triangle_metres,
                                          height_triangle_metres)
print("The area of the triangle is {0} square inches".format(area_inches))


# - ### Decorator functions for classes
# 
# Decorator function can also decorate classes. In this case, we can
# modify for instance how a constructor works.

# Following the example above, we will convert the attributes of a class
# Triangle from metres to inches


def to_inches_decorator(class_):
    """
    Receives the class to be decorated and will be used as the 
    annotation above the class to be decorated
    
    :param class_: Class to be decorated
    :return: Modified class
    """
    
    # We save the constructor of the class, so we can access it inside
    # the function that modifies its algorithm
    class_.__init__new__ = class_.__init__

    def __decorated_init__(self, *args, **kwargs):
        """
        Modifies the algorithm of the constructor. This will replace the
        constructor of the decorated class
        
        :param self: Instance of the decorated class
        :param args: Positional arguments of the constructor
        :param kwargs: Keyword arguments of the constructor
        :return: New instance of the decorated class after the new
        decorated constructor has been executed
        """
        
        conversion_rate: float = 39.3701
        converted_args: list = list(args)
        converted_kwargs: dict = kwargs
        
        # We control if the arguments to convert to inches have
        # been passed like positional or keyword arguments
        if "base_metres" in kwargs.keys():
            # If they were passed as keyword arguments we convert them
            # to inches
            converted_kwargs["base_metres"] = kwargs["base_metres"] * \
                                              conversion_rate
            if "height_metres" in kwargs.keys():
                converted_kwargs["height_metres"] = kwargs["height_metres"] * \
                                                  conversion_rate
            else:
                # In this case, base_metres have been passed as keyword
                # argument, but not height_metres which means
                # height_metres is positional argument with index 0
                # and we have to convert it to inches
                height_metres: float = converted_args.pop(0)
                height_inches: float = height_metres * conversion_rate
                converted_args.insert(0, height_inches)
        else:
            # Otherwise, both arguments to be converted to inches have
            # been passed as positional arguments, so we have
            # to convert both. base_metres will have index 0 now
            base_metres: float = converted_args.pop(0)
            base_inches: float = base_metres * conversion_rate
            converted_args.insert(0, base_inches)
            if "height_metres" in kwargs.keys():
                converted_kwargs["height_metres"] = kwargs["height_metres"] * \
                                                  conversion_rate
            else:
                height_metres: float = converted_args.pop(1)
                height_inches: float = height_metres * conversion_rate
                converted_args.insert(1, height_inches)

        return class_.__init__new__(self, *converted_args, **converted_kwargs)
    
    # We replace the original constructor of the class with the new
    # constructor we created above
    class_.__init__ = __decorated_init__
    # We have to return the class
    return class_


@to_inches_decorator
class Triangle:
    def __init__(self, base_metres: float, height_metres: float, angle1: int,
                 angle2: int, angle3: int):
        self.base: float = base_metres
        self.height: float = height_metres
        self.angle1: int = angle1
        self.angle2: int = angle2
        self.angle3: int = angle3
        if angle1 + angle2 + angle3 != 180:
            raise ValueError("Angles of a triangle must be 180 total")

    def get_area(self):
        return 1 / 2 * self.base * self.height


# Now, the constructor takes the parameters in metres but will return
# the result of the area in inches

triangle = Triangle(2, 5, 90, 30, 60)
print(triangle.get_area())


# - ### Arguments to decorators of classes
# 
# As with decorators for functions, we can add arguments to the
# decorators that decorate classes

# Following the example of the triangle, in this case instead of
# converting directly to inches, we will use a conversion rate to
# convert to another unit


def unit_conversion_decorator(conversion_rate: float):
    """
    Receives the argument given to the decorator and used below to
    convert the parameters base_metres, height_metres to another
    unit

    :param conversion_rate: Conversion rate
    :return: Decorated class
    """

    def inner_unit_conversion_decorator(class_):
        """
        Receives the class to be decorated and will be used as the
        annotation above the class to be decorated

        :param class_: Class to be decorated
        :return: Modified class
        """

        # We save the constructor of the class, so we can access it
        # inside the function that modifies its algorithm
        class_.__init__new__ = class_.__init__

        def __decorated_init__(self, *args, **kwargs):
            """
            Modifies the algorithm of the constructor. This will replace
            the constructor of the decorated class

            :param self: Instance of the decorated class
            :param args: Positional arguments of the constructor
            :param kwargs: Keyword arguments of the constructor
            :return: New instance of the decorated class after the new
            decorated constructor has been executed
            """

            converted_args: list = list(args)
            converted_kwargs: dict = kwargs

            # We control if the arguments to convert to inches have
            # been passed like positional or keyword arguments
            if "base_metres" in kwargs.keys():
                # If they were passed as keyword arguments we convert
                # them to inches
                converted_kwargs["base_metres"] = kwargs["base_metres"] * \
                                                  conversion_rate
                if "height_metres" in kwargs.keys():
                    converted_kwargs["height_metres"] =\
                        kwargs["height_metres"] * conversion_rate
                else:
                    # In this case, base_metres have been passed as
                    # keyword argument, but not height_metres which
                    # means height_metres is positional argument with
                    # index we have to convert it to inches
                    height_metres: float = converted_args.pop(0)
                    height_inches: float = height_metres * conversion_rate
                    converted_args.insert(0, height_inches)
            else:
                # Otherwise, both arguments to be converted to inches
                # have been passed as positional arguments, so we have
                # to convert both. base_metres will have index 0 now
                base_metres: float = converted_args.pop(0)
                base_inches: float = base_metres * conversion_rate
                converted_args.insert(0, base_inches)
                if "height_metres" in kwargs.keys():
                    converted_kwargs["height_metres"] = \
                        kwargs["height_metres"] * conversion_rate
                else:
                    height_metres: float = converted_args.pop(1)
                    height_inches: float = height_metres * conversion_rate
                    converted_args.insert(1, height_inches)

            return class_.__init__new__(self, *converted_args,
                                        **converted_kwargs)

        # We replace the original constructor of the class with the new
        # constructor we created above
        class_.__init__ = __decorated_init__
        # We have to return the class
        return class_
    return inner_unit_conversion_decorator


# We apply the conversion rate from metres to inches:
# 1 metre is 39.3701 inches
@unit_conversion_decorator(39.3701)
class Triangle:
    def __init__(self, base_metres: float, height_metres: float, angle1: int,
                 angle2: int, angle3: int):
        self.base: float = base_metres
        self.height: float = height_metres
        self.angle1: int = angle1
        self.angle2: int = angle2
        self.angle3: int = angle3
        if angle1 + angle2 + angle3 != 180:
            raise ValueError("Angles of a triangle must be 180 total")

    def get_area(self):
        return 1 / 2 * self.base * self.height


triangle = Triangle(2, 5, 90, 30, 60)
print(triangle.get_area())
