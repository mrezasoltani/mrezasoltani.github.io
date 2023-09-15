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
### Anaconda
- Anaconda is a popular Python environment/package manager  
- Install from ([here](https://www.anaconda.com/download/))
- We suggest using Python >= 3.8

### IDEs:
- IDLE
  - IDLE comes with Python installation by default
- PyCharm
  - PyCharm is a widely used IDE ([link](https://www.jetbrains.com/pycharm/)).
- Spyder 
  - Spyder already exists in Anaconda once it is installed. Spider is very similar to Matlab ([link](https://www.spyder-ide.org/)).

### Arithmatics
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
<details>
<summary>
  Results
</summary>

- 13
- 7
- 1000
- 3.3333333333333335
- 3.3333333333333335
- 3
- 103

</details>

### Built-in Values
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
### Code blocks are created using indents.
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

### Loops
### For loops (If you want an index, use enumerate()!)
```python
for i, name in enumerate(['Zack','Jay','Richard']):
    print('Hi ' + '! {0}: {1:.4f}'.format(name, i))
```
<details>
  <summary>Results</summary>

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
<details>
  <summary>Results</summary>

  - We are stuck in a loop...
    
</details>

### What about for (i=0; i<10; i++)? Use range():
``` python
for i in range(5):
    print('Line' + str(I))
```
<details>
  <summary>Results</summary>

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
<details>
  <summary>Results</summary>

  - 10
  - 20
  - 30

</details>
