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

## Logical


```python
print(True and False)            # & same as and
print(True or False)             # | same as or
print(not (True or False))
```

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

## Bitwise


```python
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

## Membership


```python
print('hell' in 'hello')
print(3 in range(5), 7 in range(5))
print('a' in dict(zip('abc', range(3))))
```

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


```python

```


```python

```


```python

```


```python

```

### The super() builtin returns a proxy object, a substitute object that can call methods of the base class via delegation. Indirection call or ability to reference base object with super().

## Importing Modules
### Modules refer to a file containing Python statements and definitions.
### A file containing Python code is called a module. For example: My_Module.py is a module where its module name would be My_Module.


```python
# Import ‘os’ and ‘time’ modules
import os, time
# Import under an alias
import numpy as np
x = np.random.rand(5, 4)
print(x)
a = x.std(axis=1)
b = np.expand_dims(a, 1)
c = np.expand_dims(a, 0)
print(a.shape, b.shape, c.shape)
print(a[1], b[1], c[0,1])
```

    [[0.22810576 0.71698872 0.81095739 0.74619174]
     [0.29773628 0.249962   0.86396911 0.86312541]
     [0.52630913 0.05478592 0.80820333 0.58132644]
     [0.26844949 0.30717331 0.09197187 0.23920073]
     [0.89514996 0.26937607 0.24454529 0.91652552]]
    (5,) (5, 1) (1, 5)
    0.2953326175104657 [0.29533262] 0.2953326175104657


## Numpy

* Optimized library for matrix and vector computation.  
* Makes use of C/C++ subroutines and memory-efficient data structures.  
(Lots of computation can be efficiently represented as np.ndarray.)
* This is the data type that you will use to represent matrix/vector computations.  
Constructor function is np.array()  

### np.ndarry


```python
x = np.array([1,2,3])
y = np.array([[3,4,5]])
z = np.array([[1], [2], [3]])
# t = np.array([[6,7],[8,9]])
# print(x,y,z)
print(x.shape)
print(y.shape)
print(z.shape)
# print(t.shape)
```

    (3,)
    (1, 3)
    (3, 1)



```python
x = np.array([[1,2,3],[4,5,6]])
print(x.shape)
print(np.max(x, axis = 1))
np.max(x, axis = 1).shape, np.max(x, axis = 1, keepdims = True).shape
```

    (2, 3)
    [3 6]





    ((2,), (2, 1))



* Matrix Operations: np.dot, np.linalg.norm, .T, +, -, *, ...  
* Infix operators (i.e. +, -, *, **, /) are element-wise.  
* Matrix multiplication is done with np.dot(x, W) or x.dot(W). Transpose a matrix with x.T 
* Note: Shapes (N,) != (1, N) != (N,1)


```python
x = np.array([1,2,3])
print(x.shape)
print(x.T.shape)
print(x.reshape((1,-1)).shape)
print(x.reshape((-1,1)).shape)
xx = np.array([[1,2,3]])
print(xx.shape)
```

    (3,)
    (3,)
    (1, 3)
    (3, 1)
    (1, 3)


### Indexing
https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html


```python
x = np.random.random((3, 4)) # Random (3,4) matrix
print(x)
print('******************')
print(x[np.array([0, 2]), :]) # Selects the 1st and 3rd rows
print('******************')
print(x[1, 1:3]) # Selects 1st row as 1-D vector and 1st through 2nd elements
print('******************')
print(x > 0.5)
print('******************')
print([x > 0.5])
print('******************')
print(x[x > 0.5]) # Boolean indexing
```

    [[0.57366881 0.3041909  0.80756713 0.59008399]
     [0.058083   0.57994224 0.2707391  0.67437963]
     [0.51523371 0.15897214 0.3877647  0.12326957]]
    ******************
    [[0.57366881 0.3041909  0.80756713 0.59008399]
     [0.51523371 0.15897214 0.3877647  0.12326957]]
    ******************
    [0.57994224 0.2707391 ]
    ******************
    [[ True False  True  True]
     [False  True False  True]
     [ True False False False]]
    ******************
    [array([[ True, False,  True,  True],
           [False,  True, False,  True],
           [ True, False, False, False]])]
    ******************
    [0.57366881 0.80756713 0.59008399 0.57994224 0.67437963 0.51523371]


### Broadcasting
https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html  
When operating on two arrays, NumPy compares their shapes element-wise. It starts with the trailing dimensions, and works its way forward. Two dimensions are compatible when  
* they are equal, or
* one of them is 1


```python
x = np.ones((3, 4)) # Random (3, 4) matrix
y = 2*np.ones((3, 1)) # Random (3, 1) matrix
print(x)
print("*****")
print(y)
print("*****")
print(x+y)
z = -2*np.ones((1, 4)) # Random (3,) vector
print("*****")
print(z)
print("*****")
print((x + y).shape) # Adds y to each column of x
print("*****")
print((x * z)) # Multiplies z element-wise with each row of x
print("*****")
print((y + y.T).shape) # Can give unexpected results!
print("*****")
print(y + y.T)
```

    [[1. 1. 1. 1.]
     [1. 1. 1. 1.]
     [1. 1. 1. 1.]]
    *****
    [[2.]
     [2.]
     [2.]]
    *****
    [[3. 3. 3. 3.]
     [3. 3. 3. 3.]
     [3. 3. 3. 3.]]
    *****
    [[-2. -2. -2. -2.]]
    *****
    (3, 4)
    *****
    [[-2. -2. -2. -2.]
     [-2. -2. -2. -2.]
     [-2. -2. -2. -2.]]
    *****
    (3, 3)
    *****
    [[4. 4. 4.]
     [4. 4. 4.]
     [4. 4. 4.]]



```python
import numpy as np

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
print("==== 2 norm (max singular value) ====")
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
    ==== 2 norm (max singular value) ====
    5.398345637668169


## Avoid explicit for-loops over indices/axes at all costs.
## For-loops will dramatically slow down your code (~10-100x)


```python
import time
s = time.time()
x = np.random.rand(1000,1000)
for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        x[i,j] **= 2
print(time.time()-s)

s = time.time()
x **= 2
print(time.time()-s)
```

    0.4328644275665283
    0.12763500213623047



```python
s = time.time()
for i in range(100, 1000):
    for j in range(x.shape[1]):
        x[i, j] += 5
print(time.time()-s)

s = time.time()
x[np.arange(100,1000), :] += 5
print(time.time()-s)
```

    0.37599611282348633
    0.020943164825439453


## Tricks

### List Comprehension


```python
# Format: [func(x) for x in some_list]
# Following are equivalent:
squares = []
for i in range(10):
    squares.append(i**2)
print(squares)
squares = [i**2 for i in range(10)]
print(squares)
# Can be conditional:
odds = [i**2 for i in range(10) if i%2 == 1]
```

    [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
    [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]


### Lambda
A lambda function is a small anonymous function.  
A lambda function can take any number of arguments, but can only have one expression.


```python
# These two operations are identical
def add(a,b,c):
    return a + b + c
print(add(5, 6, 2))

add = lambda a, b, c : a + b + c
print(add(5, 6, 2))
```

    13
    13



```python
def multiplier(input, option):
    if option =='double':
        return 2*input
    elif option =='triple':
        return 3*input
print(multiplier(1,'double'))

# To handle multiple expressions
multiplierDict = { 'double' : lambda x : 2*x,  'triple' : lambda x : 3*x}
def multiplier(input, option):
    return multiplierDict[option](input)

print(multiplier(1,'double'))
```

    2
    2


### Convenient Syntax


```python
# Multiple assignment / unpacking iterables
x, y, z = ['Tensorflow', 'PyTorch', 'Chainer']
age, name, pets = 20, 'Joy', ['cat']

# Returning multiple items from a function
def some_func():
    return 10, 1
ten, one = some_func()

# Joining and splitting list of strings with a delimiter
x = ['1', '2', '3']
print(x)
x = ','.join(x)
print('x:', x)
y = x.split(',')
print('y:', y)

# String literals with both single and double quotes
message = 'I like "single" quotes.'
reply = "I prefer 'double' quotes."
print(message)
print(reply)
```

    ['1', '2', '3']
    x: 1,2,3
    y: ['1', '2', '3']
    I like "single" quotes.
    I prefer 'double' quotes.


## Matplotlib for plotting 

https://matplotlib.org/

## Import pyplot module from matplotlib package


```python
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
```


```python
x = np.linspace(-5, 5, 100)
```


```python
y1 = 1/(1+np.exp(-x))
y2 = np.log(1+np.exp(-x))
```


```python
np.log(10)
```




    2.302585092994046




```python
x.shape, y1.shape, y2.shape
```




    ((100,), (100,), (100,))




```python
plt.plot(x, y1, 'r-', x, y2, 'b--', linewidth=2)
plt.grid()
plt.legend(['Sigmoid Function', 'Exp Function'])
```




    <matplotlib.legend.Legend at 0x115349b10>




    
![png](output_54_1.png)
    

