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
  - IDLE comes with Python installation by default
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

## Built-in Values
```python
True, False          # Usual true/false values
None                 # Represents the absence of something
x = None             # Variables can be None
array = [1,2,None]   # Lists can contain None

def func():
    return None      # Functions can return None

if [1,2] != [3,4]:   # Can check for equality
    print('Error')
int
float
str
```
## Code blocks are created using indents.
* **Indents can be 2 or 4 spaces but should be consistent throughout the file.**
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

  - 10
  - 20
  - 30

</details>

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

### Tuple
* A tuple is a collection which is ordered and unchangeable (Immutable).
  - tupels cannot be modifies. This makes the a good choice for _key_ choice in hashtables or dictionaries.

``` python
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

### Set
  * A set is a collection which is unordered and unindexed. In Python sets are written with curly brackets.
  * It is suitable for creating unique collection of objects.
  * We can do set mathematical operations with this data structure such as _unioin_, _difference_, _intersection_, etc.
``` python
names = {'Zach', 'Jay', 'Zach'} # Note the curly brackets and duplicates
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
  {'Zach'}
  {'Zach'}
- ========= set union ============
  {'Jay', 'Richard', 'Zach'}
  {'Jay', 'Richard', 'Zach'}
- ========= set intersection ============
  {'Jay'}
  {'Jay'}

</details>
