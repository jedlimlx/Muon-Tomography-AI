import inspect
import sys


def get_custom_objects():
    clsmembers = inspect.getmembers(sys.modules['layers'], inspect.isclass)
    clsmembers += inspect.getmembers(sys.modules['metrics'], inspect.isclass)
    clsmembers += inspect.getmembers(sys.modules['misc'], inspect.isclass)
    return clsmembers
