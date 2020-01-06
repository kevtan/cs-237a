#!/usr/bin/env python

#This script will introduce us to the basics of using Python

#Imports
import math

#Variables
initial_integer = 1
initial_float = 1.0
cast_int_to_float = float(initial_integer)
cast_float_to_string = str(initial_float)

#Printing
print("Hello world!")
name = "Your Name"
print("Hello my name is {}.".format(name))

#Functions
print("\nFunctions:")
y = lambda x: math.sin(x)

def sin(x):
    return math.sin(x)

assert(y(1) == sin(1))
print("sin(1) = {}, {}.".format(y(1), sin(1)))

#Conditions
print("\nConditions:")
if type(initial_integer) == type(cast_int_to_float):
    print("{} == {}".format(initial_integer, cast_int_to_float))
else:
    print("{} != {}".format(initial_integer, cast_int_to_float))

string_equal = "{} == {}".format(initial_integer, cast_int_to_float)
string_notequal = "{} != {}".format(initial_integer, cast_int_to_float)
truth = string_equal if type(initial_integer) == type(cast_int_to_float) else string_notequal
print(truth)

# For loops
print("\nFor loops:")
number_list_one = range(10)
number_list_two = [i + 1 for i in range(10)]

print_string = ""
for number in number_list_one:
    print_string += str(number) + " "
print(print_string)

for i,number in enumerate(number_list_one):
    print("i: {}, number: {}".format(i, number))

for number_one, number_two in zip(number_list_one,number_list_two):
    print("number_one: {}, number_two: {}".format(number_one, number_two))

# Classes
print("\nClasses:")
class MyString():
    def __init__(self,string):
        self.string = string
    def print_string(self):
        print(self.string)

my_string = MyString("hello")
my_string.print_string()

# Data structures
print("\nData structures:")

my_tuple = ('hi','hello','hey')
print(my_tuple)

my_list = [1,2,3,4]
print("my_list: {}".format(my_list))
my_list[-1] = 5
print("my_list: {}".format(my_list))

my_dictionary = {'hi':1,'hello':2}
my_dictionary['hey'] = 3
print("my_dictionary: {}".format(my_dictionary))

my_set = set()
my_set.add("Hi")
my_set.add(1)
my_set.add(40151)
my_set.add("Hi")
print("my_set: {}".format(my_set))

# Equality vs Identity
print("\nEquality vs identity:")
list_one = [1,2,3]
list_two = [1,2,3]
list_three = list_one

print("list_one is list_two: {}".format(list_one is list_two))
print("list_one == list_two: {}".format(list_one == list_two))

print("list_one is list_three: {}".format(list_one is list_three))
print("list_one == list_three: {}".format(list_one == list_three))
