---
title: "Module 6 --- Python, Numpy, Pandas, Visualization"
classes: wide
---

## Why Python?
+ Python is a widely used, general-purpose programming language.
+ Easy to start working with.
+ Scientific computation functionality similar to Matlab and Octave.
+ Used by major deep learning frameworks such as PyTorch and TensorFlow.

```python
print('Hello World')
```
## Anaconda
- Anaconda is a popular Python environment/package manager  
- Install from ([here](https://www.anaconda.com/download/))
- We suggest using Python >= 3.8

## IDEs:
- IDLE
  - IDLE comes with Python installation by default.
- Jupyter Notebook and JupyterLab ([link](https://docs.jupyter.org/en/latest/))
  -  Jupyter Notebook and JupyterLab are two web-based notebooks of the bigger project called Project Jupyter documentation. A notebook is a shareable document that combines computer code, plain language descriptions, data, rich visualizations like 3D models, charts, graphs and figures, and interactive controls.
  -  One you installed Anaconda, you can install and launch Jupyter Notebook or JupyterLab, and start imidiately wrtie and run Python codes.
- PyCharm
  - PyCharm is a widely used IDE ([link](https://www.jetbrains.com/pycharm/)).
- Spyder 
  - Spyder already exists in Anaconda once it is installed. Spider is very similar to Matlab ([link](https://www.spyder-ide.org/)).

## Arithmatics
``` python
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
<details markdown=1><summary markdown="span">Results</summary>

- 13
- 7
- 1000
- 3.3333333333333335
- 3.3333333333333335
- 3
- 103

</details>

## Logical
``` python
print(True and False)            # & same as and
print(True or False)             # | same as or
print(not (True or False))
```
<details markdown=1><summary markdown="span">Results</summary>
  
- False
- True
- False

</details>

## Relational
``` python
print(3 == 3) 
print(1 == 5)
print(12 != 30)
print(-2.5 < 3)
print(1 <= 40)
print(0 > 10)
print(0 >= 10)
```
<details markdown=1><summary markdown="span">Results</summary>
  
- True
- False
- True
- True
- True
- False
- False

</details>

## Bitwise
``` python
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
<details markdown=1><summary markdown="span">Results</summary>

- 1010
- 0111
- ============ Bitwise AND operation ============
- 2 0010
- ============ Bitwise OR operation =============
- 15 1111
- ============ Bitwise XOR operation ============
- 13 1101
- ============ Bitwise Left Shift ================
- 26 11010
- ============ Bitwise Right Shift ===============
- 6 0110

</details>

## Membership
``` python
print('hell' in 'hello')
print(3 in range(5), 7 in range(5))
print('a' in dict(zip('abc', range(3))))
```
<details markdown=1><summary markdown="span">Results</summary>

- True
- True False
- True

</details>

## Identity
``` python
x = [2,3]
y = [2,3]
print(x == y, x is y)
print(id(x), id(y))
x = 'hello'
y = 'hello'
print(x == y, x is y)
print(id(x), id(y))
```
<details markdown=1><summary markdown="span">Results</summary>

- True False
- 140536971848256 140536971833280
- True True
- 140536995546928 140536995546928

</details>

## Assignment
``` python
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
<details markdown=1><summary markdown="span">Results</summary>

- 10
- 12
- 24
- 29
- 26

</details>

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
### There are cannot be used a a keyword as a variable name, function name or any other identifier.
``` python
False	await	else	import	pass
None	break	except	in	raise
True	class	finally	is	return
and	for	lambda	try     continue	
as	def	from	while   nonlocal	
assert	del	global	not	with
async	elif	if	or	yield
```

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
  - Two methods %s and %r actually convert any python object to a string
``` python
s = 'DEAR'
print("Place another string with a mod and s: %s" %(s))
print("Floating point numbers: %1.2f" %(13.144567))
print("Floating point numbers: %2.4f" %(13.144567))
print("Here is a number: %s. Here is a string: %s" %(123.1,'hi')) 
print("Here is a number: %r. Here is a string: %r" %(123.1,'hi'))
```
    <details markdown=1><summary markdown="span">Results</summary>
    
    - Place another string with a mod and s: DEAR
    - Floating point numbers: 13.14
    - Floating point numbers: 13.1446
    - Here is a number: 123.1. Here is a string: hi
    - Here is a number: 123.1. Here is a string: 'hi'
    
    </details>

#### Using _format()_ method
* This approach is more cleaner to write
``` python
print("This is a string with an {var}".format(var='DEAR'))
print("One: {var1}, Two: {var1}, Three: {var1}".format(var1='HI !!!'))                                   # Multiple times
print("Object 1: {var1}, Object 2: {var2}, Object 3: {var1}".format(var1=1,var2='two',var3=12.3))        # Several Objects
```
  <details markdown=1><summary markdown="span">Results</summary>
  
  - This is a string with an DEAR
  - One: HI !!!, Two: HI !!!, Three: HI !!!
  - Object 1: 1, Object 2: two, Object 3: 1
  
  </details>

#### Using _f-String_
* f-string is the modern way, and the shorteset and best approach to format a string:
* Syntax \\(~\Longrightarrow\\) ``` python f"This is an f-string {var_name} and {var_name}." ```

``` python
language = "Python"
school = "freeCodeCamp"
print(f"I'm learning {language} from {school}.")
num1 = 83.98765
num2 = 9.876543218765
print(f"The product of {num1} and {num2} is {num1 * num2}.")
print(f"The product of {num1:.2f} and {num2:.4f} is {num1 * num2:.2f}.")
```
  <details markdown=1><summary markdown="span">Results</summary>
  
  - I'm learning Python from freeCodeCamp.
  - The product of 83.98765 and 9.876543218765 is 829.5076550675084.
  - The product of 83.99 and 9.8765 is 829.51.
  
  </details>

## Code blocks are created using indents.
### Indents can be 2 or 4 spaces but should be consistent throughout the file.
``` python
def fib(n):
    # Indent level 1: function body
    if n <= 1:
        # Indent level 2: if statement body
        return 1
    else:
        # Indent level 2: else statement body
        return fib(n-1)+fib(n-2)
```

## Loops
### For loops (If you want an index \\(\Longrightarrow\\) using enumerate()!)
```python
for i, name in enumerate(['Zack','Jay','Richard']):
    print('Hi ' + '! {0}: {1:.4f}'.format(name, i))
```
<details markdown=1><summary markdown="span">Results</summary>

  - Hi ! Zack: 0.0000  
  - Hi ! Jay: 1.0000
  - Hi ! Richard: 2.0000
    
</details>

### While Loops
``` python
while True:
  print('We are stuck in a loop...')
  break           # Break out of the while loop
```
<details markdown=1><summary markdown="span">Results</summary>

  - We are stuck in a loop...
    
</details>

### What about for (i=0; i<10; i++)? \\(\Longrightarrow\\) using range():
``` python
for i in range(5):
    print('Line' + str(I))
```
<details markdown=1><summary markdown="span">Results</summary>

  - Line0
  - Line1
  - Line2
  - Line3
  - Line4
    
</details>

### Looping over a list, unpacking tuples:
``` python
for x, y in [(1,10), (2,20), (3,30)]:
    print(x, y)
```
<details markdown=1><summary markdown="span">Results</summary>

  - 1 10
  - 2 20
  - 3 30

</details>

### _Enumerate_ is a built-in function of Python, allowing us to loop over something and have an automatic counter

``` python
for i, item in enumerate(["Ali", "John", "Zach"]):
    print(i, item)
```
<details markdown=1><summary markdown="span">Results</summary>

- 0 Ali
- 1 John
- 2 Zach

</details>

## Conditions
### Similar to other languages, for wrting conditions we can use _if_, _elif_, _else_, or in _while argumnet_ as shown below:
``` python
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
<details markdown=1><summary markdown="span">Results</summary>

- yeah!
- yeah!
- A
- C
- B
- Are you in the right class?
- 4
- 3
- 2
- 1
- =======================
- 1
- 3
- =======================
- 1
- =======================
- 1
- 3
- 
</details>

## Errors and Error Handling
### We can handle errors nicely using _try_ and _exception_ built in:
``` python
try:
    1 / 0
except ZeroDivisionError as e:
    print(e)
```
<details markdown=1><summary markdown="span">Results</summary>

- division by zero

</details>

### List of possible exceptions ([link](https://www.programiz.com/python-programming/exceptions)): 
``` python
AssertionError	      Raised when an assert statement fails.
AttributeError	      Raised when attribute assignment or reference fails.
EOFError	      Raised when the input() function hits end-of-file condition.
FloatingPointError    Raised when a floating point operation fails.
GeneratorExit	      Raise when close() method a generator is called.
ImportError	      Raised when the imported module is not found.
IndexError	      Raised when the index of a sequence is out of range.
KeyError	      Raised when a key is not found in a dictionary.
KeyboardInterrupt	Raised when the user hits the interrupt key (Ctrl+C or Delete).
MemoryError	      Raised when an operation runs out of memory.
NameError	      Raised when a variable is not found in local or global scope.
NotImplementedError	Raised by abstract methods.
OSError	              Raised when system operation causes system related error.
OverflowError	      Raised when the result of an arithmetic operation is too large to be represented.
ReferenceError	      Raised when a weak reference proxy is used to access a garbage collected referent.
RuntimeError	      Raised when an error does not fall under any other category.
StopIteration	      Raised by next() function to indicate that there is no further item to be returned by iterator.
SyntaxError	      Raised by parser when syntax error is encountered.
IndentationError	   Raised when there is incorrect indentation.
TabError	      Raised when indentation consists of inconsistent tabs and spaces.
SystemError	      Raised when interpreter detects internal error.
SystemExit	      Raised by sys.exit() function.
TypeError	      Raised when a function or operation is applied to an object of incorrect type.
UnboundLocalError	 Raised when a reference is made to a local variable in a function or method, but no value has been bound to that variable.
UnicodeError	      Raised when a Unicode-related encoding or decoding error occurs.
UnicodeEncodeError    Raised when a Unicode-related error occurs during encoding.
UnicodeDecodeError    Raised when a Unicode-related error occurs during decoding.
UnicodeTranslateError	 Raised when a Unicode-related error occurs during translating.
ValueError	      Raised when a function gets an argument of correct type but improper value.
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
* List is a collection which is ordered and changeable (Mutable). Allows duplicate members.

  ``` python
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
  <details markdown=1><summary markdown="span">Results</summary>
  
  - True
  - True
  - ['Zach', 'Jay', 'Richard']
  - ['Zach', 'Jay', 'Richard', 'Abi', 'Kevin']
  - [1, ['hi', 'bye'], -0.12, None]
  
  </details>

#### List slicing (indexing)
  * **x[start:stop:step]**
    - start - starting integer where the slicing of the object starts
    - stop - integer until which the slicing takes place. The slicing stops at index stop - 1.
    - step - integer value which determines the increment between each index for slicing

    ``` python
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
    <details markdown=1><summary markdown="span">Results</summary>
    
    - [0, 1, 2]
    - [5, 6]
    - [0, 1, 2, 3, 4, 5, 6]
    - 6
    - [4, 5, 6]
    - [3, 4]
    - [0, 2, 4]
    - [6, 5, 4, 3, 2, 1, 0]
    
    </details>

### List Methods:
- sort(): Sorts the list in ascending order.
  - We can also use sorted() built-in function.
- append(): Adds a single element to a list.
- extend(): Adds multiple elements to a list.
- index(): Returns the first appearance of the specified value.
- max(): It returns an item from the list with max value.
- min(): It returns an item from the list with min value.
- len(): It gives the total length of the list.
- pop(): It remove and return item at index (default last).
- remove(): It remove first occurrence of value

  ``` python
  a = [5, 1, 7, -1, 4, 10, 3]
  rint("======= using sorted function =======")
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
  <details markdown=1><summary markdown="span">Results</summary>
  
  - ======= using sorted function =======
  - [-1, 1, 3, 4, 5, 7, 10]
  - ======== using sort() method ========
  - [-1, 1, 3, 4, 5, 7, 10]
  - [-1, 1, 3, 4, 5, 7, 10, 100]
  - ======== extend method ========
  - [-1, 1, 3, 4, 5, 7, 10, 100, 20, 20, 40]
  - 3
  - 100
  - -1
  - 11
  - ======== Pop method ===========
  - 10
  - [5, 1, 7, -1, 4, 3, 100, 20, 20, 40]
  - ======== Remove method ========
  - [5, 1, 7, -1, 3, 100, 20, 20, 40]
  
  </details>

### Tuple
* A tuple is a collection which is ordered and unchangeable (Immutable).
  - tupels cannot be modifies. This makes the a good choice for _key_ choice in hashtables or dictionaries.

  ``` python
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
  <details markdown=1><summary markdown="span">Results</summary>
  
  - True
  - True
  - ('Zach', 'Jay')
  - 'tuple' object does not support item assignment
  - (10,)
  
  </details>

### Tuple Methods
- count():   Returns the number of times a specified value occurs in a tuple
- index():   Searches the tuple for a specified value and returns the position of where it was found

  ``` python
  a = (2, 3, 10, -1, 10, 10)
  print(a.count(10))
  print(a.index(-1))
  print(a.index(10))
  ```
  <details markdown=1><summary markdown="span">Results</summary>
  
  - 3
  - 3
  - 2
  
  </details>

### Set
* A set is a collection which is unordered and unindexed. In Python sets are written with curly brackets.
  - Please note that if you create a set with a curly brackets, it should be non-empty; otherwise it is treated as a dictionary.
* It is suitable for creating unique collection of objects.
* We can do set mathematical operations with this data structure such as _unioin_, _difference_, _intersection_, etc.

  ``` python
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
  <details markdown=1><summary markdown="span">Results</summary>
  
  - True
  - Jay
  - Zach
  - 'set' object is not subscriptable
  - 'set' object does not support item assignment
  - ========= set difference ============
  - {'Zach'}
  - {'Zach'}
  - ========= set union ============
  - {'Jay', 'Richard', 'Zach'}
  - {'Jay', 'Richard', 'Zach'}
  - ========= set intersection ============
  - {'Jay'}
  - {'Jay'}
  
  </details>

### Set Methods
- add():	                    Adds an element to the set
- clear():	                Removes all the elements from the set
- copy():	                Returns a copy of the set
- difference():	            Returns a set containing the difference between two or more sets
- difference_update():	    Removes the items in this set that are also included in another, specified set
- discard():	                Remove the specified item
- intersection():	        Returns a set, that is the intersection of two or more sets
- isdisjoint():	            Returns whether two sets have a intersection or not
- issubset():	            Returns whether another set contains this set or not
- issuperset():	            Returns whether this set contains another set or not
- pop():	                    Removes an element from the set
- remove():	                Removes the specified element
- symmetric_difference():	Returns a set with the symmetric differences of two sets
- union():	                Return a set containing the union of sets
- update():	                Update the set with another set, or any other iterable

  ``` python
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
  <details markdown=1><summary markdown="span">Results</summary>
  
  - ========= add method ============
  - {4, 6, 8, 'John', 'Ali', 'Jeff', -1}
  - ========= clear method ============
  - set()
  - ======= difference method ========
  - {8, 4, 'John', -1}
  - ========= discard method ==========
  - {4, 6, 8, 'Ali', -1}
  - ======= intersection method ========
  - {'Ali', 6}
  - False
  - ========= issubset method =========
  - True
  - ======= issuperset method ========
  - False
  - ========= remove method ==========
  - {'Ali', 100, 6}
  - ======= symmetric_difference method ========
  - {100, 4, 8, -1}
  - {'Ali', 4, 100, 6, 8, -1}
  - ========= update method ==========
  - {4, 100, 6, 8, 'John', 'Ali', -1}
  
  
  </details>

### Dictionary (key-value pairs)
* A dictionary is a collection which is unordered, changeable and indexed.
* In Python, dictionaries are written with curly brackets, and they have keys and values (This is used for creating hash tables).

  ``` python
  
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
  <details markdown=1><summary markdown="span">Results</summary>
    
  - {'Zach': '12-37', 'Jay': '34-23'}
  - True
  - False
  - 34-23
  - =================================
  - Zach 12-37
  - Jay 34-23
  - =================================
  - {'Jay': '34-23'}
  - =================================
  - Jay 34-23
  
  </details>
  
  ### Dictionary Methods
  - get():	            Returns the value of the specified key
  - items():	        Returns a list containing a tuple for each key value pair
  - keys():	        Returns a list containing the dictionary's keys
  - values():	        Returns a list of all the values in the dictionary
  - update():	        Updates the dictionary with the specified key-value pairs
  - clear():	        Removes all the elements from the dictionary
  
    ``` python
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
    <details markdown=1><summary markdown="span">Results</summary>
    
    - 33
    - dict_items([('Ali', 33), ('John', 25), ('Jeff', 58)])
    - dict_keys(['Ali', 'John', 'Jeff'])
    - dict_values([33, 25, 58])
    - {'Ali': 33, 'John': 25, 'Jeff': 58, 'Chris': 14}
    - {}
    
    </details>

## Functions
### The syntax for defining functions is as follows:
``` python
def f(a, b):
    """Doing something with inputs a and b.
      
    """
    return someting
```
``` python
def Myfunction():
    print('Hello World!')

Myfunction()                          # call the function

print('Outside function')
```
``` python
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

    ``` python
    def test(normal_arg, *argv):
        print("first normal arg:", normal_arg)
        for arg in argv:
            print("another arg through *argv:", arg)
    
    test(25, 'John', 'Zach', 'Marry')
    ```
    <details markdown=1><summary markdown="span">Results</summary>
    
    - first normal arg: 25
    - another arg through *argv: John
    - another arg through *argv: Zach
    - another arg through *argv: Marry
    
    </details>
  
    ``` python
    def test(**kwargs):
      for key, value in kwargs.items():
          print(f"{key}: {value}")
  
    test(name="John", age=23)
    ```
    <details markdown=1><summary markdown="span">Results</summary>
      
    - name: John
    - age: 23
    
    </details>


* Using *args and **kwargs to call a function
  
  ``` python
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
  <details markdown=1><summary markdown="span">Results</summary>
    
  - ========== with *args ==========
  - arg1: five
  - arg2: 4
  - arg3: 1
  - ========= with **kwargs ========
  - arg1: 15
  - arg2: two
  - arg3: 1
  
  </details>

### Lambda Function
* A lambda function is a small anonymous function.  
* A lambda function can take any number of arguments, but can only have one expression.
    ``` lambda argument: manipulate(argument)```

  ``` python
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
  <details markdown=1><summary markdown="span">Results</summary>
  
  - ======== regular function =========
  - 8
  - ====== using lambda function =======
  - 8
  - ====================================
  - 13
  - [(13, -3), (4, 1), (1, 2), (9, 10)]
  
  </details>

### Some built-in functions:
- map():         It applies a function to all the items in an input_list
  ``` python map(function_to_apply, list_of_inputs)```
- filter:        It creates a list of elements for which a function returns true
- reduce:        It applies a rolling computation to sequential pairs of values in a list
- All these functions return an iterator that yield the output of the desired funciton. To get the whole results, you sould wrap it with the _list()_ method.

* **map()**

  ``` python
  # passing all the list elements to a function one-by-one and then collect the output. For instance:
  items = [1, 2, 3, 4, 5]
  squared = []
  for i in items:
      squared.append(i**2)
  
  # map allows us to implement this in a much simpler and nicer way. Here you go:
  items = [1, 2, 3, 4, 5]
  squared = list(map(lambda x: x**2, items))
  print(squared)
  ```
  <details markdown=1><summary markdown="span">Results</summary>
    
    - [1, 4, 9, 16, 25]
  
  </details>

* **filter()**

  ``` python
  number_list = range(-5, 5)
  less_than_zero = list(filter(lambda x: x < 0, number_list))
  print(less_than_zero)
  ```
  <details markdown=1><summary markdown="span">Results</summary>
    
    - [-5, -4, -3, -2, -1]
  
  </details>

* **reduce()**

  ``` python
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
  <details markdown=1><summary markdown="span">Results</summary>
    
    - 24
  
  </details>


## Iterators, Iterables, Comprehensions

## Classes (Object-oriented Programming)
* The super() builtin returns a proxy object, a substitute object that can call methods of the base class via delegation. Indirection call or ability to reference base object with super().

  ``` python
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
  ```
  ``` python
  mydog = Dog(5)
  print(mydog.species, mydog.age)
  print(mydog.isPerson())
  mydog.growup()
  print(mydog.age)
  ```
  <details markdown=1><summary markdown="span">Results</summary>
  
  - Dog 5
  - False
  - 12
  
  </details>

## Packages and namespace
- Modules (file)
- Package (hierarchical modules)
- Namespace and naming conflicts
- **Using import**
  - Importing Modules
  - Modules refer to a file containing Python statements and definitions.
  - A file containing Python code is called a module. For example: My_Module.py is a module where its module name would be My_Module.
    
      ``` python
      import os, time                          # Import ‘os’ and ‘time’ modules
      import numpy as np                       # Import numpy modlue and changing its name to _np_ for less wrting
      from yaml.loader import SafeLoader       # Import a function called "Safeloader" from module "yaml"
      ```

## Decorators

## Type Hints

## Numpy
* Numpy is the core library for scientific computing in Python. 
* It provides a high-performance multidimensional array object, and tools for working with these arrays.
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

  ``` python
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
  <details markdown=1><summary markdown="span">Results</summary>
    
    - x: [1 2 3]
    - y: [[3 4 5]]
    - z: [[1]
    -     [2]
    -     [3]]
    - ===== shape of arrays =====
    - (3,)
    - (1, 3)
    - (3, 1)
    - (2, 2)
    
  </details>

* Numpy has many functions to create different arrays:

  ``` python
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
  <details markdown=1><summary markdown="span">Results</summary>

  - ==== an array of all zeros ====
  - [[0. 0.]
  -   [0. 0.]]
  - ==== an array of all ones ====
  - [[1. 1.]]
  - ==== an array of constant values ====
  - [[7 7]
  -   [7 7]]
  - ==== a 2x2 identity matrix ====
  - [[1. 0.]
  -   [0. 1.]]
  - ==== an array with random entries ====
  - [[0.33147697 0.76320062]
  - [0.08826422 0.47941476]]

  </details>

### Data Types
* Numpy array are a collection of elements with the same type.
* Numpy provides a large set of numeric datatypes that can be used to construct arrays.
* Numpy tries to guess a datatype when an array is created.
    - Functions that construct arrays usually include an optional argument to explicitly specify the datatyp
  ``` python
  x = np.array([3, 4])                    # numpy choose the datatype
  print(x.dtype)         
  x = np.array([1.3, 2.0])                # numpy choose the datatype
  print(x.dtype)             
  x = np.array([4, 2], dtype=np.int64)    # User datatype
  print(x.dtype)
  ```
  <details markdown=1><summary markdown="span">Results</summary>

  - int64
  - float64
  - int64

  </details>
  
### Math Operations
* max()/min() operation along one axis:
  - Argument "axis" controls the direction of operation
  - For example, "axis = 1" means taking max() operaiton row-wise (recall axis = 1 denotes rows)
    - This results in taking max() between entries of a row (different columns)
    - If the original array has two dimension (i.e., a \\(m\times n\\) matrix), then the max() output is a rank 1 array with dimension written as (m,)
  - "keepdims = True" argument force the output has the exact shape (dimension) before applying the operation.
    - "keepdims" is the argument for many other math operations such as np.min, np.sum(), np.mean(), etx

  ``` python
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
  <details markdown=1><summary markdown="span">Results</summary>
    
  - x shape: (2, 3)
  - =======================
  - [3 6] (2,)
  - =======================
  - [[3]
  -    [6]] (2, 1)
  - =======================
  - y shape: (2, 2, 3)
  - =======================
  - [[4 5 6]
  -    [0 3 2]] (2, 3)
  - =======================
  -  [[[4 5 6]]
  -    [[0 3 2]]] (2, 1, 3)
  
  </details>

* Transpose and reshape operation:
  - "-1" in one axis means everything left. For example, ``` python x.reshape((1,-1) ``` reshapes the \\(x\\) dimension such that the first dimension has one and the second dimension has 3 elements.

  ``` python
  x = np.array([1, 2, 3])
  print(x.shape)
  print(x.T.shape)
  print(x.reshape((1, -1)).shape)
  print(x.reshape((-1, 1)).shape)
  xx = np.array([[1, 2, 3]])
  print(xx.shape)
  ```
  <details markdown=1><summary markdown="span">Results</summary>
    
  - (3,)
  - (3,)
  - (1, 3)
  - (3, 1)
  - (1, 3)

  </details>

* Matrix Operations: np.dot, np.linalg.norm, .T, +, -, *, ...
* Infix operators (i.e. +, -, *, **, /) are element-wise.  
* Matrix multiplication is done with np.dot(x, W) or x.dot(W). Transpose a matrix with x.T 
* Note: Shapes (N,) != (1, N) != (N,1)
