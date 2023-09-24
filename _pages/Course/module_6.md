---
layout: page
title: "Module 6 --- Python, Numpy, Pandas, Visualization"
classes: wide
---

## Why Python?
+ Python is a widely used, general-purpose programming language.
+ Easy to start working with.
+ Scientific computation functionality similar to Matlab and Octave.
+ Used by major deep learning frameworks such as PyTorch and TensorFlow.


## Python Installation

## Anaconda
- Anaconda is a popular Python environment/package manager  
- Install from ([here](https://www.anaconda.com/download/))
- We suggest using Python >= 3.8

## IDEs:
- IDLE
  - IDLE comes with Python installation by default.
- Jupyter Notebook and JupyterLab ([link](https://docs.jupyter.org/en/latest/))
  -  Jupyter Notebook and JupyterLab are two web-based notebooks of the bigger project called Project Jupyter Documentation. A notebook is a shareable document that combines computer code, plain language descriptions, data, rich visualizations like 3D models, charts, graphs and figures, and interactive controls.
  -  Once you installed Anaconda, you can install and launch Jupyter Notebook or JupyterLab, and start immediately writing and running Python codes.
- PyCharm
  - PyCharm is a widely used IDE ([link](https://www.jetbrains.com/pycharm/)).
- Spyder 
  - Spyder already exists in Anaconda once it is installed. Spider is very similar to Matlab ([link](https://www.spyder-ide.org/)).


## Printing a message


```python
print('Hello World')
```

    Hello World


## Arithmetics


```python
x = 10
y = 3
print(x + y)
print(x - y)
print(x ** y)
print(x / y) # in python 2 returns 3
print(x / float(y))
print(x // y)
print(str(x) + '' + '' + str(y))
```

    13
    7
    1000
    3.3333333333333335
    3.3333333333333335
    3
    103


## Logical


```python
print(True and False)            # & same as and
print(True or False)             # | same as or
print(not (True or False))
```

    False
    True
    False


## Relational


```python
print(3 == 3) 
print(1 == 5)
print(12 != 30)
print(-2.5 < 3)
print(1 <= 40)
print(0 > 10)
print(0 >= 10)
```

    True
    False
    True
    True
    True
    False
    False


## Bitwise


```python
print(format(10, '04b'))
print(format(7, '04b'))
print("============ Bitwise AND operation ===========")
x = 10 & 7
print(x, format(x, '04b'))
print("============ Bitwise OR operation ============")
x = 10 | 7
print(x, format(x, '04b'))
print("============ Bitwise XOR operation ===========")
x = 10 ^ 7
print(x, format(x, '04b'))
print("============ Bitwise Left Shift ===============")
x = x << 1
print(x, format(x, '04b'))
print("============ Bitwise Right Shift ==============")
x = x >> 2
print(x, format(x, '04b'))
```

    1010
    0111
    ============ Bitwise AND operation ===========
    2 0010
    ============ Bitwise OR operation ============
    15 1111
    ============ Bitwise XOR operation ===========
    13 1101
    ============ Bitwise Left Shift ===============
    26 11010
    ============ Bitwise Right Shift ==============
    6 0110


## Membership
* The ```range()``` function returns a sequence of numbers, starting from 0 by default, and increments by 1 (by default), and stops before a specified number.
  - range(start, stop, step)


```python
print('hell' in 'hello')
print(3 in range(5), 7 in range(5))
print('a' in dict(zip('abc', range(3))))
```

    True
    True False
    True


## Identity


```python
x = [2,3]
y = [2,3]
print(x == y, x is y)
print(id(x), id(y))
x = 'hello'
y = 'hello'
print(x == y, x is y)
print(id(x), id(y))
```

    True False
    140619199022464 140619199022016
    True True
    140619189642544 140619189642544


## Assignment


```python
x = 10
print(x)
x = x + 2
print(x)
x *= 2
print(x)
x += 5
print(x)
x -= 3
print(x)
```

    10
    12
    24
    29
    26


## Built-in Values


```python
True               # The boolean value True.
False              # The boolean value False.
None               # The singleton value None.
float("inf")       # The floating-point value inf.
float("-inf")      # The floating-point value -inf.
float("nan")       # The floating-point value nan.
complex('1+2j')    # Return a complex number with the value real + imag*1j or convert a string or number to a complex number 
```

## List of keywords in Python
### There cannot be used as a keyword as a variable name, function name, or any other identifier.

``` python
False	await	else	import	pass
None	break	except	in	raise
True	class	finally	is	return
and	for	lambda	try     continue	
as	def	from	while   nonlocal	
assert	del	global	not	with
async	elif	if	or	yield
```

### Special variables

*These variables are all reserved by Python and should not be used for other purposes.
* ```__author__``` The name of the author of the module
* ```__doc__``` A string that contains the documentation for the module
* ```__file__``` The path to the module file
* ```__name__``` The name of the module
* ```__package__``` The name of the package that the module is part of
* ```__main__``` The namespace that a Python module is running in
* ```__doc__``` Printing out the docstring that appears in a class or method
* ```__class__``` Returning the class of an instance
* ```__dict__``` Returning, as a dictionary, all attributes of a class instance:
* ```dir()``` Rreturning, as a list, every associated method or attribute

## String Format
### There are 3 ways to format strings. 
* Old method using "%" sign
* Using _format()_ method
* Using _f-String_

#### Old method using "%" sign
* Using **%n1.n2f**
  - n1 is the total minimum number of digits the string should contain
    - Filling with whitespace if the entire number does not have this many digits
  - n2 placeholder denotes the number of decimal point
  - Two methods %s and %r actually convert any Python object to a string


```python
s = 'DEAR'
print("Place another string with a mod and s: %s" %(s))
print("Floating point numbers: %1.2f" %(13.144567))
print("Floating point numbers: %2.4f" %(13.144567))
print("Here is a number: %s. Here is a string: %s" %(123.1,'hi')) 
print("Here is a number: %r. Here is a string: %r" %(123.1,'hi'))
```

    Place another string with a mod and s: DEAR
    Floating point numbers: 13.14
    Floating point numbers: 13.1446
    Here is a number: 123.1. Here is a string: hi
    Here is a number: 123.1. Here is a string: 'hi'


#### Using _format()_ method
* This approach is more cleaner to write


```python
print("This is a string with a {var}".format(var='DEAR'))
print("One: {var1}, Two: {var1}, Three: {var1}".format(var1='HI !!!'))                                   # Multiple times
print("Object 1: {var1}, Object 2: {var2}, Object 3: {var1}".format(var1=1,var2='two',var3=12.3))        # Several Objects
```

    This is a string with a DEAR
    One: HI !!!, Two: HI !!!, Three: HI !!!
    Object 1: 1, Object 2: two, Object 3: 1


#### Using _f-String_
* f-string is the modern way, and the shortest and best approach to format a string:
* Syntax \\(~\Longrightarrow\\) ``` python f"This is an f-string {var_name} and {var_name}." ```


```python
language = "Python"
school = "freeCodeCamp"
print(f"I'm learning {language} from {school}.")
num1 = 83.98765
num2 = 9.876543218765
print(f"The product of {num1} and {num2} is {num1 * num2}.")
print(f"The product of {num1:.2f} and {num2:.4f} is {num1 * num2:.2f}.")
```

    I'm learning Python from freeCodeCamp.
    The product of 83.98765 and 9.876543218765 is 829.5076550675084.
    The product of 83.99 and 9.8765 is 829.51.


### String methods
* find()
    - Returns the index of first occurrence of substring
* islower()
    - Checks if all Alphabets in a String are Lowercase
* isnumeric()
    - Checks Numeric Characters
* replace()
    - Replaces Substring Inside
* lstrip()
    - Removes Leading Characters
* rstrip()
    - Removes Trailing Characters
* split()
    - Splits String from Left
* join()
    - Returns a Concatenated String


```python
string = " !Hi man how are you?&"

print(string.find("man"), "\n")

print(string.islower(), "\n")

print(string.isnumeric(), "\n")

print(string.replace("man", "Sir"), "\n")

print(string.lstrip(" !"), "\n")

print(string.rstrip("&"), "\n")

print(string.split(" "), "\n")

a = string.split(" ")
" ".join(a)
```

    5 
    
    False 
    
    False 
    
     !Hi Sir how are you?& 
    
    Hi man how are you?& 
    
     !Hi man how are you? 
    
    ['', '!Hi', 'man', 'how', 'are', 'you?&'] 
    





    ' !Hi man how are you?&'



## Code blocks are created using indents.
#### Indents can be 2 or 4 spaces but should be consistent throughout the file.


```python
def fib(n):
    # Indent level 1: function body
    if n <= 1:
        # Indent level 2: if statement body
        return 1
    else:
        # Indent level 2: else statement body
        return fib(n-1)+fib(n-2)
```

## Zip/Unzip
* Zip allows to combine two collections.
* it returns back an iterator.
* We use "* for unzipping.


```python
first = ['Joe','Earnst','Thomas','Martin','Charles']
last = ['Schmoe','Ehlmann','Fischer','Walter','Rogan','Green']
age = [23, 65, 11, 36, 83]

for first_name, last_name, age in zip(first_name, last_name, age):
    print(f"{first_name} {last_name} is {age} years old")
```

    Joe Schmoe is 23 years old
    Earnst Ehlmann is 65 years old
    Thomas Fischer is 11 years old
    Martin Walter is 36 years old
    Charles Rogan is 83 years old



```python
full = [('Joe', 'Schmoe', 23),
      ('Earnst', 'Ehlmann', 65),
      ('Thomas', 'Fischer', 11),
      ('Martin', 'Walter', 36),
      ('Charles', 'Rogan', 83)]

first_name, last_name, age = list(zip(*full))

print(f"first name: {first_name}")
print(f"last name: {last_name}")
print(f"age: {age}")
```

    first name: ('Joe', 'Earnst', 'Thomas', 'Martin', 'Charles')
    last name: ('Schmoe', 'Ehlmann', 'Fischer', 'Walter', 'Rogan')
    age: (23, 65, 11, 36, 83)


## Loops
### For loops (If you want an index \\(\Longrightarrow\\) using enumerate()!)


```python
for i, name in enumerate(['Zack','Jay','Richard']):
    print('Hi ' + '! {0}: {1:.4f}'.format(name, i))
```

### While Loops


```python
while True:
  print('We are stuck in a loop...')
  break           # Break out of the while loop
```

    We are stuck in a loop...


### What about for (i=0; i<10; i++)? \\(\Longrightarrow\\) using range():


```python
for i in range(5):
    print('Line' + str(i))
```

    Line0
    Line1
    Line2
    Line3
    Line4


### Looping over a list, unpacking tuples:


```python
for x, y in [(1,10), (2,20), (3,30)]:
    print(x, y)
```

    1 10
    2 20
    3 30


### _Enumerate_ is a built-in function of Python, allowing us to loop over something and have an automatic counter


```python
for i, item in enumerate(["Ali", "John", "Zach"]):
    print(i, item)
```

    0 Ali
    1 John
    2 Zach


## Conditions
* Similar to other languages, for writing conditions we can use _if_, _elif_, _else_, or in _while argumnet_ as shown below:


```python
if 5 + 1 == 6:
    print("yeah!")
'correct' if 1 + 1  == 3 else 'incorrect'

if 1+1 == 3:
    print("oops")
else:
    print("yeah!")
for grade in [94, 79, 81, 57]:
    if grade > 90:
        print('A')
    elif grade > 80:
        print('B')
    elif grade > 70:
        print('C')
    else:
        print('Are you in the right class?')
i = 4
while i > 0:
    print(i)
    i -= 1
    
print("=======================")
for i in range(1, 4):
    if i % 2 == 0:
        continue
    print(i)
    
print("=======================")
for i in range(1, 4):
    if i % 2 == 0:
        break
    print(i)
    
print("=======================")
for i in range(1, 4):
    if i % 2 == 0:
        pass
    else:
        print(i)
```

    yeah!
    yeah!
    A
    C
    B
    Are you in the right class?
    4
    3
    2
    1
    =======================
    1
    3
    =======================
    1
    =======================
    1
    3


## Errors and Error Handling
* We can handle errors nicely using _try_ and _exception_ built-in:


```python
try:
    1 / 0
except ZeroDivisionError as e:
    print(e)
```

    division by zero


* ``` assert()```
* The assert keyword lets one test if a condition in yotheur code returns True, if not, the program will raise an AssertionError.


```python
x = "Good"

#if condition is False, AssertionError is raised:
assert x == "OK", "x should be 'OK'"
```


    ---------------------------------------------------------------------------

    AssertionError                            Traceback (most recent call last)

    /var/folders/nf/r9pb0kd107s0x03h3r5bgl6r0000gn/T/ipykernel_29083/2461160715.py in <module>
          2 
          3 #if condition is False, AssertionError is raised:
    ----> 4 assert x == "OK", "x should be 'OK'"
    

    AssertionError: x should be 'OK'


### List of possible exceptions ([link](https://www.programiz.com/python-programming/exceptions)): 

``` python
AssertionError	      Raised when an assert statement fails.
AttributeError	      Raised when attribute assignment or reference fails.
EOFError	      Raised when the input() function hits the end-of-file condition.
FloatingPointError    Raised when a floating point operation fails.
GeneratorExit	      Raise when close() method a generator is called.
ImportError	      Raised when the imported module is not found.
IndexError	      Raised when the index of a sequence is out of range.
KeyError	      Raised when a key is not found in a dictionary.
KeyboardInterrupt	Raised when the user hits the interrupt key (Ctrl+C or Delete).
MemoryError	      Raised when an operation runs out of memory.
NameError	      Raised when a variable is not found in local or global scope.
NotImplementedError	Raised by abstract methods.
OSError	              Raised when system operation causes system-related error.
OverflowError	      Raised when the result of an arithmetic operation is too large to be represented.
ReferenceError	      Raised when a weak reference proxy is used to access a garbage collected referent.
RuntimeError	      Raised when an error does not fall under any other category.
StopIteration	      Raised by next() function to indicate that there is no further item to be returned by iterator.
SyntaxError	      Raised by the parser when a syntax error is encountered.
IndentationError	   Raised when there is incorrect indentation.
TabError	      Raised when indentation consists of inconsistent tabs and spaces.
SystemError	      Raised when the interpreter detects an internal error.
SystemExit	      Raised by sys.exit() function.
TypeError	      Raised when a function or operation is applied to an object of incorrect type.
UnboundLocalError	 Raised when a reference is made to a local variable in a function or method, but no value has been bound to that variable.
UnicodeError	      Raised when a Unicode-related encoding or decoding error occurs.
UnicodeEncodeError    Raised when a Unicode-related error occurs during encoding.
UnicodeDecodeError    Raised when a Unicode-related error occurs during decoding.
UnicodeTranslateError	 Raised when a Unicode-related error occurs during translating.
ValueError	      Raised when a function gets an argument of the correct type but improper value.
ZeroDivisionError	 Raised when the second operand of division or modulo operation is zero.
```

## Python data structures
* **Sequence containers - list, tuple**
* **Mapping containers - set, dict**
* The **collections module** ([link](https://docs.python.org/2/library/collections.html))
  - **deque** (list-like container with fast appends and pops on either end)
  - **Counter** (dict subclass for counting hashable objects)
  - **OrderedDict** (dict subclass that remembers the order entries were added)
  - **defaultdict** (dict subclass that calls a factory function to supply missing values)

### List
* List is a collection that is ordered and changeable (Mutable). Allows duplicate members.


```python
empty_list = []                          # Empty list
empty_list = list()                      # Empty list
names = ['Zach','Jay']
print(names[0] == 'Zach')
names.append('Richard')
print(len(names) == 3)
print(names)

names.extend(['Abi','Kevin'])
print(names)

names = []                               # Creates an empty list
names = list()                           # Also creates an empty list
stuff = [1, ['hi','bye'], -0.12, None]   # Can mix types
print(stuff)
```

    True
    True
    ['Zach', 'Jay', 'Richard']
    ['Zach', 'Jay', 'Richard', 'Abi', 'Kevin']
    [1, ['hi', 'bye'], -0.12, None]


#### List slicing (indexing)
* **x[start:stop:step]**
- start - starting integer where the slicing of the object starts
- stop - integer until which the slicing takes place. The slicing stops at index stop - 1.
- step - integer value which determines the increment between each index for slicing


```python
numbers = [0, 1, 2, 3, 4, 5, 6]
print(numbers[0:3])            # numbers[:3]
print(numbers[5:])             # numbers[5:7]
print(numbers[:])              # numbers
print(numbers[-1])             # Negative index wraps around
print(numbers[-3:])
print(numbers[3:-2])           # Can mix and match
print(numbers[0:5:2])          # numbers[:5:2]
numbers[::-1]
```

    [0, 1, 2]
    [5, 6]
    [0, 1, 2, 3, 4, 5, 6]
    6
    [4, 5, 6]
    [3, 4]
    [0, 2, 4]





    [6, 5, 4, 3, 2, 1, 0]



### List Methods:
- sort(): Sorts the list in ascending order.
  - We can also use sorted() built-in function.
- append(): Adds a single element to a list.
- extend(): Adds multiple elements to a list.
- index(): Returns the first appearance of the specified value.
- max(): It returns an item from the list with max value.
- min(): It returns an item from the list with min value.
- len(): It gives the total length of the list.
- pop(): It removes and returns the item at index (default last).
- remove(): It removes the first occurrence of a value


```python
a = [5, 1, 7, -1, 4, 10, 3]
print("======= using sorted function =======")
print(sorted(a))

print("======== using sort() method ========")
a.sort()
print(a)
a.append(100)
print(a)

print("======== extend method ========")
b = [20, 20, 40]
a.extend(b)
print(a)
a = [5, 1, 7, -1, 4, 10, 3]
print(a.index(-1))
print(max(a))
print(min(a))
print(len(a))

print("======== pop method ===========")
print(a.pop(5))
print(a)

print("======== remove method ========")
a.remove(4)
print(a)
```

    ======= using sorted function =======
    [-1, 1, 3, 4, 5, 7, 10]
    ======== using sort() method ========
    [-1, 1, 3, 4, 5, 7, 10]
    [-1, 1, 3, 4, 5, 7, 10, 100]
    ======== extend method ========
    [-1, 1, 3, 4, 5, 7, 10, 100, 20, 20, 40]
    3
    10
    -1
    7
    ======== pop method ===========
    10
    [5, 1, 7, -1, 4, 3]
    ======== remove method ========
    [5, 1, 7, -1, 3]


### Tuple
* A tuple is a collection that is ordered and unchangeable (Immutable).
  - Tuples cannot be modified. This makes a good choice for _key_ in hashtables or dictionaries.


```python
empty_tuple = ()              # Empty tuple
empty_tuple = tuple()         # Empty tuple
names = ('Zach', 'Jay')       # Note the parentheses
print(names[0] == 'Zach')
print(len(names) == 2)
print(names)

try:
  names[0] = 'Richard'
except TypeError as e:
  print(e)

empty = tuple()               # Empty tuple
single = (10,)                # Single-element tuple. Comma matters!
print(single)
```

    True
    True
    ('Zach', 'Jay')
    'tuple' object does not support item assignment
    (10,)


### Tuple Methods
- count():   Returns the number of times a specified value occurs in a tuple
- index():   Searches the tuple for a specified value and returns the position of where it was found


```python
a = (2, 3, 10, -1, 10, 10)
print(a.count(10))
print(a.index(-1))
print(a.index(10))
```

    3
    3
    2


### Set
* A set is a collection that is unordered and unindexed. In Python, sets are written with curly brackets.
  - Please note that if you create a set with a curly bracket, it should be non-empty; otherwise, it is treated as a dictionary.
* It is suitable for creating a unique collection of objects.
* We can do set mathematical operations with this data structure such as _unioin_, _difference_, _intersection_, etc.


```python
empty_set = set()                   # Empty set
names = {'Zach', 'Jay', 'Zach'}     # Note the curly brackets and duplicates
print(len(names) == 2)

for name in names:
  print(name)

try:
  print(names[0])
except TypeError as e:
  print(e)

try:
  names[0] = 'Richard'
except TypeError as e:
  print(e)

names_2 = {'Jay', 'Richard'}

print("========= set difference ============")
print(names - names_2)
print(names.difference(names_2))

print("========= set union ============")
print(names | names_2)
print(names.union(names_2))

print("========= set intersection ============")
print(names & names_2)
print(names.intersection(names_2))
```

    True
    Zach
    Jay
    'set' object is not subscriptable
    'set' object does not support item assignment
    ========= set difference ============
    {'Zach'}
    {'Zach'}
    ========= set union ============
    {'Zach', 'Richard', 'Jay'}
    {'Zach', 'Richard', 'Jay'}
    ========= set intersection ============
    {'Jay'}
    {'Jay'}


### Set Methods
- add():	                    Adds an element to the set
- clear():	                Removes all the elements from the set
- copy():	                Returns a copy of the set
- difference():	            Returns a set containing the difference between two or more sets
- difference_update():	    Removes the items in this set that are also included in another, specified set
- discard():	                Remove the specified item
- intersection():	        Returns a set, that is the intersection of two or more sets
- isdisjoint():	            Returns whether two sets have an intersection or not
- issubset():	            Returns whether another set contains this set or not
- issuperset():	            Returns whether this set contains another set or not
- pop():	                    Removes an element from the set
- remove():	                Removes the specified element
- symmetric_difference():	Returns a set with the symmetric differences of two sets
- union():	                Return a set containing the union of sets
- update():	                Update the set with another set, or any other iterable


```python
a = {"Ali", 4, "John", 6, 8, -1}
print("========= add method ============")
a.add("Jeff")
print(a)

print("========= clear method ============")
a.clear()
print(a)
a = {"Ali", 4, "John", 6, 8, -1}
b = {6, "Ali", 100, 3.6}

print("======= difference method ========")
print(a.difference(b))

print("========= discard method ==========")
a.discard("John")
print(a)

print("======= intersection method ========")
print(a.intersection(b))
print(a.isdisjoint(b))
c = {4, 8}

print("========= issubset method =========")
print(c.issubset(a))

print("======= issuperset method ========")
print(a.issuperset(b))

print("========= remove method ==========")
b.remove(3.6)
print(b)

print("======= symmetric_difference method ========")
print(a.symmetric_difference(b))
print(a.union(b))
a = {"Ali", 4, "John", 6, 8, -1}

print("========= update method ==========")
a.update(b)
print(a)
```

    ========= add method ============
    {'John', 'Jeff', 4, 6, 8, 'Ali', -1}
    ========= clear method ============
    set()
    ======= difference method ========
    {8, 'John', 4, -1}
    ========= discard method ==========
    {4, 6, 8, 'Ali', -1}
    ======= intersection method ========
    {'Ali', 6}
    False
    ========= issubset method =========
    True
    ======= issuperset method ========
    False
    ========= remove method ==========
    {100, 'Ali', 6}
    ======= symmetric_difference method ========
    {4, 100, 8, -1}
    {4, 100, 6, 8, 'Ali', -1}
    ========= update method ==========
    {'John', 4, 100, 6, 8, 'Ali', -1}


## Dictionary (key-value pairs)
* A dictionary is a collection that is unordered, changeable, and indexed.
* In Python, dictionaries are written with curly brackets, and they have keys and values (This is used for creating hash tables).


```python
phonebook = dict()                       # Empty dictionary 
phonebook = {'Zach': '12-37'}            # Dictionary with one item
phonebook['Jay'] = '34-23'               # Add another item
print(phonebook)
print('Zach' in phonebook)
print('Kevin' in phonebook)
print(phonebook['Jay'])
print("=================================")

for name, number in phonebook.items():
  print(name, number)
print("=================================")

del phonebook['Zach']                    # Delete an item
print(phonebook)
print("=================================")

for name, number in phonebook.items():
  print(name, number)
```

    {'Zach': '12-37', 'Jay': '34-23'}
    True
    False
    34-23
    =================================
    Zach 12-37
    Jay 34-23
    =================================
    {'Jay': '34-23'}
    =================================
    Jay 34-23


### Dictionary Methods
- get():	            Returns the value of the specified key
- items():	        Returns a list containing a tuple for each key-value pair
- keys():	        Returns a list containing the dictionary's keys
- values():	        Returns a list of all the values in the dictionary
- update():	        Updates the dictionary with the specified key-value pairs
- clear():	        Removes all the elements from the dictionary


```python
a = {"Ali": 33, "John": 25, "Jeff": 58}

print(a.get("Ali"))
print(a.items())
print(a.keys())
print(a.values())

b = {"Chris": 14}
a.update(b)
print(a)

a.clear()
print(a)
```

    33
    dict_items([('Ali', 33), ('John', 25), ('Jeff', 58)])
    dict_keys(['Ali', 'John', 'Jeff'])
    dict_values([33, 25, 58])
    {'Ali': 33, 'John': 25, 'Jeff': 58, 'Chris': 14}
    {}


## Functions
### The syntax for defining functions is as follows:


```python
def f(a, b):
    """Doing something with inputs a and b.
      
    """
    return someting

def Myfunction():
    print('Hello World!')

Myfunction()                          # call the function

print('Outside function')

# function with two arguments
def add_numbers(num1, num2):
    sum = num1 + num2
    print('Sum: ',sum)

# Return value in a function with two arguments
def add_numbers(num1, num2):
    sum = num1 + num2
    return sum

result = add_numbers(num1, num2)      # call the function
print(result)

def test_function(a. b):              # some placeholder (doinf nothing) for a function
  pass
```

### *args and **kwargs
*  These two are mostly used in function definitions.
*  *args (non-keyworded) and **kwargs (keyworded) arguments allow one to pass an unspecified number of arguments to a function.
*  it is not required to write *args or **kwargs. Only the * (asterisk) is necessary.
*  Order of different types of arguments in a function
    - formal args --- > *args --- > **kwargs 


```python
def test(normal_arg, *argv):
    print("first normal arg:", normal_arg)
    for arg in argv:
        print("another arg through *argv:", arg)
    
test(25, 'John', 'Zach', 'Marry')
```

    first normal arg: 25
    another arg through *argv: John
    another arg through *argv: Zach
    another arg through *argv: Marry



```python
def test(**kwargs):
  for key, value in kwargs.items():
      print(f"{key}: {value}")

test(name="John", age=23)
```

    name: John
    age: 23


#### Using *args and **kwargs to call a function


```python
def test(arg1, arg2, arg3):
    print("arg1:", arg1)
    print("arg2:", arg2)
    print("arg3:", arg3)

print("========== with *args ==========")
args = ("five", 4, 1)
test(*args)

print("========= with **kwargs ========")
kwargs = {"arg3": 1, "arg2": "two", "arg1": 15}
test(**kwargs)
```

    ========== with *args ==========
    arg1: five
    arg2: 4
    arg3: 1
    ========= with **kwargs ========
    arg1: 15
    arg2: two
    arg3: 1


### Lambda Function
* A lambda function is a small anonymous function.  
* A lambda function can take any number of arguments, but can only have one expression.
    ```lambda argument: manipulate(argument)```


```python
# These two operations are identical
def add(a, b):
  return a+b

add_lambda = lambda x, y: x + y

print("======== regular function =========")
print(add(3, 5))
print("====== using lambda function =======")
print(add_lambda(3, 5))

print("====================================")
add = lambda a, b, c : a + b + c
print(add(5, 6, 2))

# Sorting a List of tuples
a = [(1, 2), (4, 1), (9, 10), (13, -3)]
a.sort(key=lambda x: x[1])
print(a)
```

    ======== regular function =========
    8
    ====== using lambda function =======
    8
    ====================================
    13
    [(13, -3), (4, 1), (1, 2), (9, 10)]


### Some built-in functions:
- map():         It applies a function to all the items in an input_list
  ``` python map(function_to_apply, list_of_inputs)```
- filter:        It creates a list of elements for which a function returns true
- reduce:        It applies a rolling computation to sequential pairs of values in a list
- All these functions return an iterator that yields the output of the desired function. To get the whole results, you should wrap it with the _list()_ method.

#### map()


```python
# passing all the list elements to a function one-by-one and then collecting the output. For instance:
items = [1, 2, 3, 4, 5]
squared = []

for i in items:
  squared.append(i**2)

# map allows us to implement this in a much simpler and nicer way. Here you go:
items = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, items))
print(squared)
```

    [1, 4, 9, 16, 25]


#### filter()


```python
number_list = range(-5, 5)
less_than_zero = list(filter(lambda x: x < 0, number_list))
print(less_than_zero)
```

    [-5, -4, -3, -2, -1]


#### reduce()


```python
# multiplication without reduce
product = 1
list = [1, 2, 3, 4]
for num in list:
  product = product * num

# multiplication with reduce
from functools import reduce
product = reduce((lambda x, y: x * y), [1, 2, 3, 4])
print(product)
```

    24


#### I/O file operation
* Open a file
* Read or write (perform operation)
* Close the file
    - "r" Open a file for reading (default)
    - "w" Open a file for writing. Creates a new file if it does not exist or overwrite the file if it exists.
    - "a" Open a file for appending at the end of the file without overwriting it. Creates a new file if it does not exist.
    - "b" Open in binary mode


```python
# Wrting in a file
with open("test.txt", "w") as file:
    file.write("Hello My Dear !!!")

# Reading a file
with open("test.txt", "r") as file:
    a = file.read()
print(a)
```

    Hello My Dear !!!


* **Pickle file**
* Python's Pickle module is a format used to serialize and deserialize data types. 
* ickle objects cannot be loaded using any other programming language.
* Pickle can serialize almost every used built-in Python data type:
    - list, dictionary, numpy array, pandas dataframe, machine learning. models
* serialization formats like JSON does not support tuples and datetime objects. 
* Pickle also retains the exact state of the object, while JSON does not do it.
* Pickle is slower and produces larger serialized values than JSON.
* Pickle is unsafe because it can execute malicious Python callables to construct objects.


```python
import pickle

names = ["Abby", "John", "Maggy"]
print("==== before loading pickle file ====")
print(names)

with open('names.pkl', 'wb') as f:   # open a text file
    pickle.dump(names, f)            # serialize the list
    
with open('names.pkl', 'rb') as f:   # open a text file
    array = pickle.load(f)           # serialize the list

print("==== after loading pickle file ====")
print(array)
```

    ==== before loading pickle file ====
    ['Abby', 'John', 'Maggy']
    ==== after loading pickle file ====
    ['Abby', 'John', 'Maggy']


* dumps() and loads() functions are used to serialize and and deserialize byte objects. 

## Classes (Object-oriented Programming)
* The super() built-in returns a proxy object, a substitute object that can call methods of the base class via delegation. Indirection call or ability to reference base object with super().


```python
class Animal(object):
  def __init__(self, species, age): # Constructor `a = Animal(‘bird’, 10)`
      self.species = species # Refer to instance with `self`
      self.age = age # All instance variables are public

  def isPerson(self): # Invoked with `a.isPerson()`
      return self.species == 'Homo Sapiens'

  def growup(self):
      self.age += 1

class Dog(Animal): # Inherits Animal’s methods
  def __init__(self, age):
      super().__init__(self.__class__.__name__, age)

  def growup(self): # Override for dog years
      self.age += 7

mydog = Dog(5)
print(mydog.species, mydog.age)
print(mydog.isPerson())
mydog.growup()
print(mydog.age)
```

    Dog 5
    False
    12


## List, set and disctionary comprehensions


```python
print("===== list comprehension =====")
print([a**2 for a in range(6)])

print("===== list comprehension with condition =====")
print([a**2 for a in range(6) if a%2 != 0])

persons = [("John", 23), ("Abby", 18), ("Frank", 39)]

print("===== set comprehension =====")
print({item[0] for item in persons})

print("===== dictionary comprehension =====")
print({name: age for name, age in persons})
```

    ===== list comprehension =====
    [0, 1, 4, 9, 16, 25]
    ===== list comprehension with condition =====
    [1, 9, 25]
    ===== set comprehension =====
    {'Frank', 'Abby', 'John'}
    ===== dictionary comprehension =====
    {'John': 23, 'Abby': 18, 'Frank': 39}


## Iteration, Iterators, Iterables, and Generators

#### Iteration
* It is the process of taking an item from something e.g a list. When we use a loop to loop over something which is called iteration. 

#### Iterable
* An iterable is any object in Python which has an **\_iter\_()** or a **\_getitem\_()** method.
* It returns an iterator or can take indexes (an object that can generate an iterator).
* It produce items on demand.
* lists, tuples, dictionaries, and sets are built-in iterables.

####  Iterator
* An object that allows you to iterate over collections of data such list.
* it should implement a **\_next\_()** method and **\_iter\_()** methods that returns an item in every call.
* Iterators take responsibility for two main actions:
    - Returning the data from a stream or container one item at a time
    - Keeping track of the current and visited items
* Every Iterator is an interable but the other way around is not always true. 


#### Generators
* Generators are iterators, but you can only iterate over them once. 
* Generators do'not store all the values in memory, they generate the values on the fly.
* You use them by iterating over them, either with a ‘for’ loop or by passing them to any function or construct that iterates.
* There are two ways generators:
    - Using functions using _yield_ instead of _return_
    - Uisng Generator expression

#### Implementing a generator


```python
class Myiterator():
    def __init__(self, sequence):
        self.sequence = sequence
        self.next = 0
        
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.next < len(self.sequence):
            temp = self.sequence[self.next]
            self.next += 1
            return temp
        else:
            raise StopIteration

a = [1, 2, 3, 4]
for item in Myiterator(a):
    print(item)
```

    1
    2
    3
    4


#### Generator using function


```python
def generator_function():
    for i in range(4):
        yield i

for item in generator_function():
    print(item)
```

    0
    1
    2
    3


#### Generator expression \\(\Longrightarrow\\) (expression for item in iterable)


```python
My_Generator = (i ** 2 for i in range(4))

for i in My_Generator:
    print(i)
```

    0
    1
    4
    9


#### Pipelining Generators
* Multiple generators can be used to pipeline a series of operations.


```python
def fibonacci_numbers(nums):
    x, y = 0, 1
    for _ in range(nums):
        x, y = y, x+y
        yield x

def square(nums):
    for num in nums:
        yield num**2

print("Finding the sum of squares of numbers in the Fibonacci series")
print(sum(square(fibonacci_numbers(10))))
```

    Finding the sum of squares of numbers in the Fibonacci series
    4895


## Packages and Modules (file.py)
#### Using import
 - Importing Modules
  - Modules refer to a file containing Python statements and definitions.
  - A file containing Python code is called a module. For example: My_Module.py is a module where its module name would be My_Module.


```python
import os, time                          # Import ‘os’ and ‘time’ modules
import numpy as np                       # Import numpy module and changing its name to _np_ for less writing
from yaml.loader import SafeLoader       # Import a function called "Safeloader" from module "yaml"
```

## Decorators

* A Python decorator is a function that takes in a function and returns it by adding some functionality ([link](https://www.programiz.com/python-programming/decorator)).
* Python uses **@** to define a decorator. 


```python
def zero_devision(func):
    # define the inner function 
    def inner(a, b):
        # add some additional functionality to the decorated function
        if b==0:
            print("Cannot devide by zero !!!")
            return
        
        # return the inner function
        return func(a, b)
    return inner

@zero_devision
def devide(a, b):
    return a/b

print("==== deviding by non-zero =====")
print(devide(4,3))

print("\n==== deviding by zero =====")
devide(4,0)
```

    ==== deviding by non-zero =====
    1.3333333333333333
    
    ==== deviding by zero =====
    Cannot devide by zero !!!


## Type Hints


```python

```

## Numpy
* Numpy is the core library for scientific computing in Python. 
* It provides a high-performance multidimensional array object and tools for working with these arrays.
* Optimized library for matrix and vector computation.  
* Makes use of C/C++ subroutines and memory-efficient data structures.  
  - Lots of computation can be efficiently represented as np.ndarray.
* This is the data type that you will use to represent matrix/vector computations.
  - Constructor function is np.array()  


### Arrays
* A numpy array is a grid of values, all the same type, and is indexed by a tuple of nonnegative integers.
* The number of dimensions is the rank of the array
  - The shape of an array is a tuple of integers giving the size of the array along each dimension.
* One can initialize numpy arrays from nested Python lists.
* The first dimension (axis = 0) denotes the columns
* The second dimension (axis = 1) denotes the rows.


```python
import numpy as np

x = np.array([1,2,3])              # Create a rank 1 array
y = np.array([[3,4,5]])            # Create a rank 2 array
z = np.array([[1], [2], [3]])      # Create a rank 2 array
t = np.array([[6,7],[8,9]])        # Create a rank 2 array

print("x:", x)
print("y:", y)
print("z:", z)

print("===== shape of arrays =====")
print(x.shape)
print(y.shape)
print(z.shape)
print(t.shape)
```

    x: [1 2 3]
    y: [[3 4 5]]
    z: [[1]
     [2]
     [3]]
    ===== shape of arrays =====
    (3,)
    (1, 3)
    (3, 1)
    (2, 2)


#### Numpy has many functions to create different arrays:
* Two functions for creating 1-D arrays:
  - ```python np.linspace(stat, stop, # samples) ``` \\(~\Longrightarrow\\) Returns evenly spaced numbers over a specified interval.
  - ```python np.arange(stat, stop, step) ``` \\(~\Longrightarrow\\) Return evenly spaced values within a given interval.
    - step: spacing between values


```python
a = np.linspace(-5, 5, 100)
print("==== a shape linspace ====")
print(a.shape)
b = np.arange(-5, 5, 0.1)
print("==== a shape arange ====")
print(b.shape)
```

    ==== a shape linspace ====
    (100,)
    ==== a shape arange ====
    (100,)


* Functions for creating 2-D arrays (matrix)


```python
print("==== an array of all zeros ====")
a = np.zeros((2,2))   
print(a)    

print("==== an array of all ones ====")
b = np.ones((1,2))   
print(b)             

print("==== an array of constant values ====")
c = np.full((2,2), 7)  
print(c)             

print("==== a 2x2 identity matrix ====")
d = np.eye(2)        
print(d)             

print("==== an array with random entries ====")
e = np.random.random((2,2)) 
print(e) 
```

    ==== an array of all zeros ====
    [[0. 0.]
     [0. 0.]]
    ==== an array of all ones ====
    [[1. 1.]]
    ==== an array of constant values ====
    [[7 7]
     [7 7]]
    ==== a 2x2 identity matrix ====
    [[1. 0.]
     [0. 1.]]
    ==== an array with random entries ====
    [[0.96780339 0.1153085 ]
     [0.78520961 0.12827767]]


### Data Types
* Numpy arrays are a collection of elements of the same type.
* Numpy provides a large set of numeric datatypes that can be used to construct arrays.
* Numpy tries to guess a datatype when an array is created.
    - Functions that construct arrays usually include an optional argument to explicitly specify the data type.


```python
x = np.array([3, 4])                    # numpy choose the datatype
print(x.dtype)         
x = np.array([1.3, 2.0])                # numpy choose the datatype
print(x.dtype)             
x = np.array([4, 2], dtype=np.int64)    # User datatype
print(x.dtype)
```

    int64
    float64
    int64


### Math Operations
* max()/min() operation along one axis:
  - Argument "axis" controls the direction of operation
  - For example, "axis = 1" means taking max() operation row-wise (recall axis = 1 denotes rows)
    - This results in taking max() between entries of a row (different columns)
    - If the original array has two dimensions (i.e., a \\(m\times n\\) matrix), then the max() output is a rank 1 array with dimension written as (m,)
  - "keepdims = True" argument forces the output to have the exact shape (dimension) before applying the operation.
    - "keepdims" is the argument for many other math operations such as np.min, np.sum(), np.mean(), etx


```python
x = np.array([[1, 2, 3],[4, 5, 6]])
print("x shape:", x.shape)

print("=======================")
xm1 = np.max(x, axis = 1)
print(xm1, xm1.shape)

print("=======================")
xm2 = np.max(x, axis = 1, keepdims = True)
print(xm2, xm2.shape)

print("=======================")
y = np.array([[[1, 2, 3],[4, 5, 6]], [[-1, 3, 2],[0, 1, 2]]])
print("y shape:", y.shape)

print("=======================")
ym1 = np.max(y, axis = 1)
print(ym1, ym1.shape)

print("=======================")
ym2 = np.max(y, axis = 1, keepdims=True)
print(ym2, ym2.shape)
```

    x shape: (2, 3)
    =======================
    [3 6] (2,)
    =======================
    [[3]
     [6]] (2, 1)
    =======================
    y shape: (2, 2, 3)
    =======================
    [[4 5 6]
     [0 3 2]] (2, 3)
    =======================
    [[[4 5 6]]
    
     [[0 3 2]]] (2, 1, 3)


* Transpose and reshape operation:
  - "-1" in one axis means everything left. For example, ``` python x.reshape((1,-1) ``` reshapes the \\(x\\) dimension such that the first dimension has one and the second dimension has 3 elements.
  - Taking the transpose of a rank 1 array does nothing (printing the same thing).


```python
x = np.array([1, 2, 3])
print(x.shape)
print(x.T.shape)
print(x.reshape((1, -1)).shape)
print(x.reshape((-1, 1)).shape)
xx = np.array([[1, 2, 3]])
print(xx.shape)
```

* Infix operators (i.e. +, -, *, **, /) are element-wise.  
* Matrix Operations: np.dot, np.linalg.norm, .T, +, -, *, ...
* Matrix multiplication is done with np.dot(x, W) or x.dot(W). Transpose a matrix with x.T
* Note: Shapes \\((N,) != (1, N) != (N,1)\\)


```python
x = np.array([[4,2],[1,9]], dtype=np.float64)
y = np.array([[2,1],[3,9]], dtype=np.float64)

print("==== Elementwise sum ====")
print(x + y)
print(np.add(x, y))

print("==== Elementwise difference ====")
print(x - y)
print(np.subtract(x, y))

print("==== Elementwise product ====")
print(x * y)
print(np.multiply(x, y))

print("==== Elementwise division ====")
print(x / y)
print(np.divide(x, y))

print("==== Elementwise square root ====")
print(np.sqrt(x))

print("==== Elementwise power ====")
print(np.power(x, 2))
print(x**2)

x = np.array([[3,1],[2,-2]])
y = np.array([[3,4],[5,9]])

v = np.array([9,0])
w = np.array([3, 8])

print("==== Inner product of vectors ====")
print(v.dot(w))
print(np.dot(v, w))

# The result is the rank 1 array
print("==== Matrix/vector product ====")
print(x.dot(v))
print(np.dot(x, v))

# The result is the rank 2 array)
print("==== Matrix/matrix product  ====")
print(x.dot(y))
print(np.dot(x, y))

x = np.array([[1,2],[3,4]])
print("==== Compute sum of all elements ====")
print(np.sum(x))  
print("==== Compute the sum of each column ====")
print(np.sum(x, axis=0))
print("==== Compute sum of each row ====")
print(np.sum(x, axis=1))

x = np.array([[4,1],[3,2]])
print("==== Frobenius norm ====")
print(np.linalg.norm(x))
print("==== nuclear norm ====")
print(np.linalg.norm(x, 'nuc'))
print("==== 2 norm ====")
print(np.linalg.norm(x, 2))
```

    ==== Elementwise sum ====
    [[ 6.  3.]
     [ 4. 18.]]
    [[ 6.  3.]
     [ 4. 18.]]
    ==== Elementwise difference ====
    [[ 2.  1.]
     [-2.  0.]]
    [[ 2.  1.]
     [-2.  0.]]
    ==== Elementwise product ====
    [[ 8.  2.]
     [ 3. 81.]]
    [[ 8.  2.]
     [ 3. 81.]]
    ==== Elementwise division ====
    [[2.         2.        ]
     [0.33333333 1.        ]]
    [[2.         2.        ]
     [0.33333333 1.        ]]
    ==== Elementwise square root ====
    [[2.         1.41421356]
     [1.         3.        ]]
    ==== Elementwise power ====
    [[16.  4.]
     [ 1. 81.]]
    [[16.  4.]
     [ 1. 81.]]
    ==== Inner product of vectors ====
    27
    27
    ==== Matrix/vector product ====
    [27 18]
    [27 18]
    ==== Matrix/matrix product  ====
    [[ 14  21]
     [ -4 -10]]
    [[ 14  21]
     [ -4 -10]]
    ==== Compute sum of all elements ====
    10
    ==== Compute the sum of each column ====
    [4 6]
    ==== Compute sum of each row ====
    [3 7]
    ==== Frobenius norm ====
    5.477225575051661
    ==== nuclear norm ====
    6.324555320336758
    ==== 2 norm ====
    5.398345637668169


### Indexing
* Similar to Python lists, numpy arrays can be indexed by slicing or itegers.
* Different ways for indexing:
  - Single element indexing
  - Slicing indexing
  - Integer indexing
  - Mixed indexing

* **Single Element Indexing**
  * Single element indexing works exactly like that for other standard Python sequences.
  * Modifying an array with single element indexing will also modify the original array.
  * It is 0-based, and accepts negative indices for indexing from the end of the array.
      - It is not necessary to separate each dimension’s index into its own set of square brackets.
      - Separating each dimension’s index into its own set of square brackets is recommended.


```python
x = np.arange(10)
print("x=", x)
print(x[2])
print(x[-2])

print("==== changing the shape of x  to (2, 5) ====")
x.shape = (2, 5)
print("x=", x)
print(x[1, 3] == x[1][3])

print("==== The first row of x and its shape")
print(x[0], x[0].shape)
```

    x= [0 1 2 3 4 5 6 7 8 9]
    2
    8
    ==== changing the shape of x  to (2, 5) ====
    x= [[0 1 2 3 4]
     [5 6 7 8 9]]
    True
    ==== The first row of x and its shape
    [0 1 2 3 4] (5,)


* **Slicing Indexing**
    * We need to specify a slice for each dimension of the array.
        - Slice has the format as ```start:stop:step```
    * Modifying a slice of an array will also modify the original array.
    * Mixing integer indexing with slices results in an array of lower rank.
    * Indexing using slices results in an array of the same rank as the original array.
    * For slicing method, **do not use separate square brackets for each dimension**.


```python
a = np.array([[5,6,1,3], [2,-6,0,9], [19,-10,1,2]])
print("a=", a, a.shape)

print("===== A slice of array a =====")
b = a[:2, 1:3]
print(b.shape)
print("a[:2, 1:3] != a[:2][1:3]")

print("===== Entry (0, 1) of array a =====")
print(a[0, 1])

print("===== Modifying entry (0, 0) of b will change the Entry (0, 1) of array a =====")
b[0, 0] = 77     
print(a[0, 1])

print("===== Rank 1 view of the second row of array a ======")
row_r1 = a[1, :]
print(row_r1, row_r1.shape)

print("===== Rank 2 view of the second row of array a ======")
row_r2 = a[1:2, :]  # Rank 2 view of the second row of a
print(row_r2, row_r2.shape)

print("==== slice is selected only for the first dimension")
t = a[1:2]
print(t, t.shape)

print("===== Rank 1 view of the second column of array a ======")
col_r1 = a[:, 1]
print(col_r1, col_r1.shape)

print("===== Rank 2 view of the second column of array a ======")
col_r2 = a[:, 1:2]
print(col_r2, col_r2.shape)
```

    a= [[  5   6   1   3]
     [  2  -6   0   9]
     [ 19 -10   1   2]] (3, 4)
    ===== A slice of array a =====
    (2, 2)
    a[:2, 1:3] != a[:2][1:3]
    ===== Entry (0, 1) of array a =====
    6
    ===== Modifying entry (0, 0) of b will change the Entry (0, 1) of array a =====
    77
    ===== Rank 1 view of the second row of array a ======
    [ 2 -6  0  9] (4,)
    ===== Rank 2 view of the second row of array a ======
    [[ 2 -6  0  9]] (1, 4)
    ==== slice is selected only for the first dimension
    [[ 2 -6  0  9]] (1, 4)
    ===== Rank 1 view of the second column of array a ======
    [ 77  -6 -10] (3,)
    ===== Rank 2 view of the second column of array a ======
    [[ 77]
     [ -6]
     [-10]] (3, 1)


* **Integer Indexing**
    * Integer array indexing allows selection of arbitrary items in the array based on their N-dimensional index.
    * Each integer array represents a number of indices into that dimension.
    * Mixing integer indexing with slices results in an array of lower rank.
    * Indexing using slices results in an array of the same rank as the original array.


```python
x = np.arange(10, 1, -1)
print("x=", x)
print("==== A subset of x with specified location =====")
print("subset = ", x[np.array([3, 3, 1, 8])])
print("==== Integer indexing with constructing index array using np.arrange()  ====")
print("subset = ", x[[3, 3, 1, 8]])

print("==== A new 3 by 2 array =====")
b = np.array([[1,2], [3, 4], [5, 6]])
print("b=", b, b.shape)

print("==== A subset of b with specified locations =====")
t = b[[0, 1, 2], [0, 1, 0]]
print("t =", t, t.shape) 

print("==== t array with another approach =====")
print(np.array([b[0, 0], b[1, 1], b[2, 0]]))

print("==== Integer indexing with constructing index array using np.arrange() ====")
print(b[[0, 0], [1, 1]])

print("==== A new 4 by 3 array =====")
c = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
print("c =", c)

print("==== Creating an array of indices =====")
i = np.array([0, 2, 0, 1])
print("i = ", i)

print("==== Select one element from each row of c using the indices in i ====")
print(c[np.arange(4), i]) 

print("==== Mutating one element from each row of c using the indices in i ====")
c[np.arange(4), i] += 10
print("c = ", c)
```

    x= [10  9  8  7  6  5  4  3  2]
    ==== A subset of x with specified location =====
    subset =  [7 7 9 2]
    ==== Integer indexing with constructing index array using np.arrange()  ====
    subset =  [7 7 9 2]
    ==== A new 3 by 2 array =====
    b= [[1 2]
     [3 4]
     [5 6]] (3, 2)
    ==== A subset of b with specified locations =====
    t = [1 4 5] (3,)
    ==== t array with another approach =====
    [1 4 5]
    ==== Integer indexing with constructing index array using np.arrange() ====
    [2 2]
    ==== A new 4 by 3 array =====
    c = [[ 1  2  3]
     [ 4  5  6]
     [ 7  8  9]
     [10 11 12]]
    ==== Creating an array of indices =====
    i =  [0 2 0 1]
    ==== Select one element from each row of c using the indices in i ====
    [ 1  6  7 11]
    ==== Mutating one element from each row of c using the indices in i ====
    c =  [[11  2  3]
     [ 4  5 16]
     [17  8  9]
     [10 21 12]]


### Broadcasting
https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
* Broadcasting is a super useful mechanism that allows numpy to work with arrays of different shapes.
    - For instance, we have a smaller array and a larger array, and we want to use the smaller array multiple times to perform some operation on the larger array.
* When operating on two arrays, NumPy compares their shapes element-wise. 
* It starts with the trailing dimensions, and works its way forward. 
* Two dimensions are compatible when  
    - They are equal
    - one of them is 1


```python
x = np.ones((3, 4)) # Random (3, 4) matrix
y = 2*np.ones((3, 1)) # Random (3, 1) matrix

print("x = ", x, x.shape)

print("y = ", y, y.shape)
print("======= x+y =====")
print(x+y)

print("=================")
z = -2*np.ones((1, 4)) # Random (3,) vector
print("z = ", z)

print("====== shape of x + y =======")
print((x + y).shape) # Adds y to each column of x

print("====== shape of x * y =======")
print((x * z).shape) # Multiplies z element-wise with each row of x

print("====== shape of x + y.T =======")
print((y + y.T).shape) 

print("====== shape of y + y.T =======")
print((y + y.T).shape)
```

    x =  [[1. 1. 1. 1.]
     [1. 1. 1. 1.]
     [1. 1. 1. 1.]] (3, 4)
    y =  [[2.]
     [2.]
     [2.]] (3, 1)
    ======= x+y =====
    [[3. 3. 3. 3.]
     [3. 3. 3. 3.]
     [3. 3. 3. 3.]]
    =================
    z =  [[-2. -2. -2. -2.]]
    ====== shape of x + y =======
    (3, 4)
    ====== shape of x * y =======
    (3, 4)
    ====== shape of x + y.T =======
    (3, 3)
    ====== shape of y + y.T =======
    (3, 3)


* Broadcasting two arrays together follows these rules ([link](https://cs231n.github.io/python-numpy-tutorial/#array-math)):

    - If the arrays do not have the same rank, prepend the shape of the lower rank array with 1s until both shapes have the same length.
    - The two arrays are said to be compatible in a dimension if they have the same size in the dimension, or if one of the arrays has size 1 in that dimension.
    - The arrays can be broadcast together if they are compatible in all dimensions.
    - After broadcasting, each array behaves as if it had shape equal to the elementwise maximum of shapes of the two input arrays.
    - In any dimension where one array had size 1 and the other array had size greater than 1, the first array behaves as if it were copied along that dimension

#### Avoid explicit for-loops over indices/axes at all costs.
* For-loops will dramatically slow down your code (~10-100x)

* Sqaurring each elemt of a mtrix using for loops and power operation


```python
import time
s = time.time()
x = np.random.rand(1000,1000)
for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        x[i,j] **= 2
print("Using for loop ===== ", f"{time.time()-s:.4f} second")

s = time.time()
x **= 2
print("Using power operation ===== ", f"{time.time()-s:.4f} second")
```

    Using for loop =====  0.4877 second
    Using power operation =====  0.0012 second


* Adding a constant to a matrix using for loops and broadcasting


```python
s = time.time()
for i in range(100, 1000):
    for j in range(x.shape[1]):
        x[i, j] += 5
print("Using for loop ===== ", f"{time.time()-s:.4f} second")

s = time.time()
x[np.arange(100,1000), :] += 5
print("Using broadcasting ===== ", f"{time.time()-s:.4f} second")
```

    Using for loop =====  0.4557 second
    Using broadcasting =====  0.0021 second


## Pandas


```python
import pandas as pd

pd.Series({"Ali": 3, "Abby": 4})
```




    Ali     3
    Abby    4
    dtype: int64



## Matplotlib


```python
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
# %matplotlib
```


```python
x = np.linspace(-5, 5, 100)

y1 = 1/(1+np.exp(-x))
y2 = np.log(1+np.exp(-x))

x.shape, y1.shape, y2.shape
```




    ((100,), (100,), (100,))




```python
plt.plot(x, y1, 'r-', x, y2, 'b--', linewidth=2)
plt.grid()
plt.legend(['Sigmoid Function', 'Exp Function'])
```
![results](/assets/images/output_114_1.png)



```python

```
