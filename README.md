{ "cells": \[ { "cell\_type": "markdown", "metadata": {}, "source": \[
"\# Parallelising Python code\n", "\n", "It is sometimes stated that
parallel code is difficult in Python. However, for most scientific
applications, we can achieve the level of parallelism without much
effort. In this notebook I will show some simple ways to get parallel
code execution in Python.\n", "\n", "1. With NumPy\n", "2. With Joblib
and multiprocessing\n", "3. With Numba\n", "4. With Cython\n", "\n",
"These methods range in complexity from easiest to most difficult." \]
}, { "cell\_type": "markdown", "metadata": {}, "source": \[ "\#\#
Parallel code with NumPy\n", "\n", "By default, NumPy will dispatch the
computations to an efficient BLAS (basic linear algebra subproblem) and
LAPACK (Linear Algebra PACKage) implementation. BLAS and LAPACK routines
are very efficient linear algebra routines that are implemented by
groups of people that are experts getting as much speed as humanly
possible out of the CPU, and we cannot compete with those for linear
algebra computations.\n", "\n", "A benefit and downside with NumPy is
that it will likely parallelise the code for you without your knowledge.
Try to compute the matrix product between two large matrices and look at
your CPU load. It will likely use all of your CPU hardware threads
(hardware threads are essentially cores)." \] }, { "cell\_type":
"markdown", "metadata": {}, "source": \[ "\#\# Moving the parallelism to
the outer loop\n", "\n", "Often, when we program, we have nested loops.
Like below" \] }, { "cell\_type": "code", "execution\_count": 1,
"metadata": {}, "outputs": \[ { "name": "stdout", "output\_type":
"stream", "text": \[ "\[0, 0, 0, 0, 0, 0, 0, 0, 0, 0\]\n", "\[0, 1, 2,
3, 4, 5, 6, 7, 8, 9\]\n", "\[0, 2, 4, 6, 8, 10, 12, 14, 16, 18\]\n",
"\[0, 3, 6, 9, 12, 15, 18, 21, 24, 27\]\n", "\[0, 4, 8, 12, 16, 20, 24,
28, 32, 36\]\n", "\[0, 5, 10, 15, 20, 25, 30, 35, 40, 45\]\n", "\[0, 6,
12, 18, 24, 30, 36, 42, 48, 54\]\n", "\[0, 7, 14, 21, 28, 35, 42, 49,
56, 63\]\n", "\[0, 8, 16, 24, 32, 40, 48, 56, 64, 72\]\n", "\[0, 9, 18,
27, 36, 45, 54, 63, 72, 81\]\n" \] } \], "source": \[ "x = \[\[\] for \_
in range(10)\]\n", "for i in range(10):\n", \" for j in range(10):\n","
x\[i\].append(i\*j)\n","\n","for row in x:\n"," print(row) \" \] }, {
"cell\_type": "markdown", "metadata": {}, "source": \[ "Here, we have
nested loop, and there are two ways to make this parallel, either by
doing multiple iterations of the outer loop (`for i in range(10)`) at
the same time or by doing multiple iterations of the inner loop
(`for j in range(10)`) at the same time. \n", "\n", "Generally, we
prefer to have the parallel code on the outer loop, as that is the most
work per iteration, which means that there is less likelihood for our
cores to stay idle. If there are more hardware threads available than
there are iterations on the outer loop, we may split it up and have some
of the parallelism on the inner loop as well. However, it is important
to make sure that we don't try to do more things in parallel than we
have hardware threads available, as otherwise, much time will be spent
switching between tasks rather than actually performing the
computations.\n", "\n", "\#\# Disabling parallel code execution in NumPy
routines\n", "Unfortunately, we have a loop where we use a NumPy
function, then that function likely runs in parallel using all the
available hardware threads. To avoid this from happening, we have to set
some envionment variables *before* importing NumPy. Specifically, we
need to set\n", "\n",
"`\n",     "OMP_NUM_THREADS=#THREADS\n",     "OPENBLAS_NUM_THREADS=#THREADS\n",     "MKL_NUM_THREADS=#THREADS\n",     "VECLIB_MAXIMUM_THREADS=#THREADS\n",     "NUMEXPR_NUM_THREADS=#THREADS\n",     "`\n",
"\n", "The first variable sets the number of OpenMP threads to
`#THREADS`. OpenMP is used by many software packages that implement
parallel code. The next three variables sets the number of threads for
NumPy for various different BLAS backends. Finally, the last line sets
the number of threads for a useful package called numexpr, which can
optimise operations on the form `a*x + b*x - c`, which with pure numpy
would entail four separate loops, but with numexpr is compiled to a
single parallel loop.\n", "\n", "We can either set these variables
directly from Python, but then we MUST do it before any library has
imported NumPy. Or, alternatively, we can set it as global environment
variables. On Linux, you can add these lines to your [`~/.profile`
file](https://www.quora.com/What-is-profile-file-in-Linux):\n", "\n",
"`\n",     "OPENBLAS_NUM_THREADS=1\n",     "MKL_NUM_THREADS=1\n",     "VECLIB_MAXIMUM_THREADS=1\n",     "NUMEXPR_NUM_THREADS=1\n",     "`\n",
"\n", "Notice how we did not set the number of OpenMP threads to 1 in
the `~/.profile` file, as that would likely disable parallelism for most
programs that use OpenMP for parallel code execution.\n", "\n", "**Note
that if we set `OMP_NUM_THREADS` to 1, then parallelism with Numba and
Cython will not work.**" \] }, { "cell\_type": "code",
"execution\_count": 2, "metadata": {}, "outputs": \[\], "source": \[
"import os\n", "\n", "def set\_threads(\n", \" num\_threads,\n","
set\_blas\_threads=True,\n"," set\_numexpr\_threads=True,\n","
set\_openmp\_threads=False\n","):\n"," num\_threads =
str(num\_threads)\n"," if not num\_threads.isdigit():\n"," raise
ValueError(\"Number of threads must be an integer.\")\n"," if
set\_blas\_threads:\n"," os.environ\[\"OPENBLAS\_NUM\_THREADS\"\] =
num\_threads\n"," os.environ\[\"MKL\_NUM\_THREADS\"\] =
num\_threads\n"," os.environ\[\"VECLIB\_MAXIMUM\_THREADS\"\] =
num\_threads\n"," if set\_numexpr\_threads:\n","
os.environ\[\"NUMEXPR\_NUM\_THREADS\"\] = num\_threads\n"," if
set\_openmp\_threads:\n"," os.environ\[\"OMP\_NUM\_THREADS\"\] =
num\_threads\n","\n","set\_threads(1)\" \] }, { "cell\_type":
"markdown", "metadata": {}, "source": \[ "Now, we can import numpy to
our code and it will only run on only one core." \] }, { "cell\_type":
"code", "execution\_count": 3, "metadata": {}, "outputs": \[\],
"source": \[ "import numpy as np" \] }, { "cell\_type": "markdown",
"metadata": {}, "source": \[ "\#\# Parallel code with Joblib and
multiprocessing\n", "\n", "Python does not support parallel threading.
This means that each Python process can only do one thing at a time. The
reason for this lies with the way Python code is run on your computer.
Countless hours has been spent trying to remove this limitation, but all
sucessfull attempts severly impaired the speed of the language (the most
well known attempt is Larry Hasting's
[gilectomy](https://github.com/larryhastings/gilectomy)). \n", "\n",
"Since we cannot run code in parallel within a single process, we need
to start new processes for each task we wish to compute in parallel and
send the relevant information to these processes. This leads to a lot of
overhead, and if we hope to have any performance gain, then we should
parallelise substantial tasks if we wish to do it with multiple
processes." \] }, { "cell\_type": "markdown", "metadata": {}, "source":
\[ "\#\#\# The best approach: Joblib\n", "\n", "The best approach to
multiprocessing in Python is through the Joblib library. It overcomes
some of the shortcomings of multiprocessing (that you may not realise is
a problem until you encounter them) at the cost of an extra dependency
in your code. Below, we see an example of parallel code with joblib" \]
}, { "cell\_type": "code", "execution\_count": 4, "metadata": {},
"outputs": \[ { "name": "stdout", "output\_type": "stream", "text": \[
"\[2, 3, 4, 5, 6, 7, 8, 9, 10, 11\]\n" \] } \], "source": \[ "from
joblib import Parallel, delayed\n", "\n", "def f(x):\n", \" return x +
2\n","\n","numbers1 = Parallel(n\_jobs=2)(delayed(f)(x) for x in
range(10))\n","\n","print(numbers1)\" \] }, { "cell\_type": "markdown",
"metadata": {}, "source": \[ "Here we see how Joblib can help us
parallelise simple for loops. We wrap what we wish to compute in a
function and use it in a list comprehension. The `n_jobs` argument
specifies how many processes to spawn. If it is a positive number (1, 2,
4, etc.) then it is the number of processes to spawn and if it is a
negative number then joblib will spawn (n\_cpu\_threads + 1 -
n\_processes). Thus `n_jobs=-1` will spawn as many processes as there
are CPU threads available, `n_jobs=-2` will spawn n-1 CPU threads, etc.
\n", "\n", "I recommend setting `n_jobs=-2` so you have one CPU thread
free to surf the web while you run hard-core experiments on your
computer." \] }, { "cell\_type": "code", "execution\_count": 5,
"metadata": {}, "outputs": \[ { "name": "stdout", "output\_type":
"stream", "text": \[ "\[2, 3, 4, 5, 6, 7, 8, 9, 10, 11\]\n" \] } \],
"source": \[ "numbers1 = Parallel(n\_jobs=-2)(delayed(f)(x) for x in
range(10))\n", "\n", "print(numbers1)" \] }, { "cell\_type": "markdown",
"metadata": {}, "source": \[ "If we cannot wrap all the logic within a
single function, but need to have two separate parallel loops, then we
should use the `Parallel` object in a slightly different fashion. If we
do the following:" \] }, { "cell\_type": "code", "execution\_count": 6,
"metadata": {}, "outputs": \[ { "name": "stdout", "output\_type":
"stream", "text": \[ "\[2, 3, 4, 5, 6, 7, 8, 9, 10, 11\]\n", "\[2, 3, 4,
5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21\]\n" \] }
\], "source": \[ "from joblib import Parallel, delayed\n", "\n", "def
f(x):\n", \" return x + 2\n","\n","numbers1 =
Parallel(n\_jobs=2)(delayed(f)(x) for x in range(10))\n","numbers2 =
Parallel(n\_jobs=2)(delayed(f)(x) for x in
range(20))\n","\n","\n","print(numbers1)\n","print(numbers2)\" \] }, {
"cell\_type": "markdown", "metadata": {}, "source": \[ "Then we will
first create two new Python processes, compute the parallel list
comprehension, close these two processess before spawning two new Python
processes and computing the second parallel list comprehension. This is
obviously not ideal, and we can reuse the pool of processes with a
context manager:" \] }, { "cell\_type": "code", "execution\_count": 7,
"metadata": {}, "outputs": \[ { "name": "stdout", "output\_type":
"stream", "text": \[ "\[2, 3, 4, 5, 6, 7, 8, 9, 10, 11\]\n", "\[2, 3, 4,
5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21\]\n" \] }
\], "source": \[ "with Parallel(n\_jobs=2) as parallel:\n", \" numbers1
= parallel(delayed(f)(x) for x in range(10))\n"," numbers2 =
parallel(delayed(f)(x) for x in
range(20))\n","\n","print(numbers1)\n","print(numbers2)\" \] }, {
"cell\_type": "markdown", "metadata": {}, "source": \[ "Here, the same
processes are used for both list comprehensions!" \] }, { "cell\_type":
"markdown", "metadata": {}, "source": \[ "\#\# Async operations with
multiprocessing\n", "An alternative to using Joblib for multiprocessing
in Python is to use the builtin `multiprocessing` module.\n", "This
module is not as user friendly as joblib, and may break with weird error
messages." \] }, { "cell\_type": "code", "execution\_count": 8,
"metadata": {}, "outputs": \[ { "name": "stdout", "output\_type":
"stream", "text": \[ "\[2, 3, 4, 5, 6, 7, 8, 9, 10, 11\]\n" \] } \],
"source": \[ "import multiprocessing\n", "\n", "def add\_2(x):\n", \"
return x + 2\n","\n","with multiprocessing.Pool(4) as p:\n","
print(p.map(add\_2, range(10)))\" \] }, { "cell\_type": "markdown",
"metadata": {}, "source": \[ "Here, we see that multiprocessing also
requires us to wrap the code we wish to run in parallel in a function.
\n", "\n", "However, one particular of multiprocessing is that it
requires all inputs to be picklable. That means that we cannot use
output a factory function and you may also have problems with using
multiprocessing with instance methods. Below is an example that fails."
\] }, { "cell\_type": "code", "execution\_count": 9, "metadata": {},
"outputs": \[ { "name": "stdout", "output\_type": "stream", "text": \[
"4\n" \] }, { "ename": "AttributeError", "evalue": "Can't pickle local
object 'add.<locals>.add\_x'", "output\_type": "error", "traceback": \[
\"\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mAttributeError\u001b[0m                            Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-9-6e36bedeabae>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m     10\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     11\u001b[0m \u001b[0;32mwith\u001b[0m \u001b[0mmultiprocessing\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mPool\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;36m4\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0mp\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 12\u001b[0;31m     \u001b[0mp\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mmap\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0madd
