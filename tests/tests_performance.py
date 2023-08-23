from collections import namedtuple

STest = namedtuple("TEST", "a b c")
a = STest(a=1, b=2, c=3)


class Test:
    a = 1
    b = 2
    c = 3


b = Test()

c = {"a": 1, "b": 2, "c": 3}

d = (1, 2, 3)
e = [1, 2, 3]
f = (1, 2, 3)
g = [1, 2, 3]
key = 2


from timeit import timeit
import jax

def f(x, key):
    return x[key]
pjit = jax.jit(f, static_argnums=1)

pjit(c, "b")
code = """
for i in range(10000):
    pjit(c, "b")
"""
print(timeit(code, globals=globals(), number=5))

pjit(d, 1)
code = """
for i in range(10000):
    pjit(d, 2)
"""
print(timeit(code, globals=globals(), number=5))
