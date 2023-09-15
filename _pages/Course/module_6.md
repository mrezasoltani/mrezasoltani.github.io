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
- Install from [here](https://www.anaconda.com/download/)
- We suggest using Python >= 3.8

### IDEs:
- IDLE
  - IDLE comes with Python installation by default
- PyCharm
  - PyCharm is a widely used IDE ([link](https://www.jetbrains.com/pycharm/)).
- Spyder 
  - Spyder already exists in Anaconda once it is installed. Spider is very similar to Matlab [link](https://www.spyder-ide.org/).

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
  <summary>Results</summary>
  \\(13\\)
  
  7
  
  1000
  
  3.3333333333333335
  
  3.3333333333333335
  
  3
  
  103
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
```
