import datetime
import math


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