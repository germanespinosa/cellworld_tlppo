'''
This class simplifies resource management by offering a unified interface for accessing files.
Whether you're developing locally or after installing via pip, accessing files through the
Resources class ensures seamless reading and maintenance. Below is an example demonstrating
how to import and utilize the class for file access:

from .resources import Resources

# Accessing a file using the Resources class
file_path = Resources.file('files/example.txt')

# Reading the file
with open(file_path, 'r') as file:
    content = file.read()
    print(content)
'''

import os

class Resources(object):
    path = os.path.dirname(os.path.abspath(__file__))
    @staticmethod
    def file(file_path: str) -> str:
        return os.path.join(Resources.path, file_path)
