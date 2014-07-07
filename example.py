"""
The Problem With Massive Datasets:

A programmer faces many challenges when making the transition into
the realm of Big Data.  Algorithms previously used to great effect
no longer work, or quickly reach asymptotic worst cases.  Even more
fundamental is the problem that often times, the data set we'd like
to analyze won't even fit into memory.  Python fits nicely into the
Big Data space, in that the language is great for fast prototyping
and has wonderful 3rd party libraries such as numpy and pandas for
hardcore number crunching; however, through the effective use of
iterators, generators, and coroutines, we can get a lot more out of
our pure python implementations.

In the code that follows, we explore some new ways to use generators
and coroutines to implement Big Data ready tools -- Python 3.4 style!
"""

from collections import namedtuple, defaultdict
from itertools import chain, tee, groupby, filterfalse
from functools import wraps
import csv
from operator import itemgetter
from asyncio import coroutine


# Define Tree structure, useful for quick prototyping of traning data
def yesno(F, yes, no):
    control_flow = lambda I: \
        map(lambda x: yes(x) if F(x) else no(x), I)
    return control_flow


is_traversable = lambda v: hasattr(v, 'traverse')


class Tree(defaultdict):
    def __init__(self):
        super().__init__(Tree)

    def traverse(self):
        control_flow = yesno(is_traversable,
                             lambda X: X.traverse(), iter)
        yield from control_flow(self.values())

    def __iter__(self):
        yield from filterfalse(is_traversable, self.values())

    def __str__(self):
        return ''.join(self)


# Pure python big data tools
class LazyAggregator(dict):
    """
    Lazy, key accessible aggregator that defers data insertion
    to the last possible moment. This data structure is roughly
    equivalent to the following idiom:

    d = {}
    for x in iterator:
        key, value = key_value_maker(x)
        d.setdefault(key, []).append(value)

    The difference lies in the use of

    >>> d = {}; d.setdefault(key, []).append(value)
    vs.
    >>> d = LazyAggregator(); d.append(key, value)

    Namely, we are chaining iterables as opposed to growing
    a list.
    """
    def __missing__(self, key):
        # this generator-closure is what is responsible
        # for the defered lookup behavior.
        @coroutine
        def coroutine_factory():
            while True:
                value = (yield)
                super(LazyAggregator, self).__setitem__(
                                      key, chain(self[key], value))
        # returns a coroutine into which values can be sent
        return self.setdefault(key, coroutine_factory())

    def append(self, key, value):
        """
        Appends a single value at the specified key
        """
        # forces a `__getitem__`, which in turn calls `__missing__`
        # the first time we try to insert a value
        self[key].send((value,))

    def append_group(self, _groupby):
        """
        Takes a `groupby`-like iterator which yields
        key/iterable pairs.
        """
        # forces a `__getitem__`, which in turn calls `__missing__`
        # the first time we try to insert a value
        def do_append(key, group):
            self[key].send(group)
        appender = yield from starmap(do_append, _groupby)

    def __setitem__(self, key, value):
        # prevents misuse
        raise NotImplementedError(
                    'Can only append, not overwrite values')


def map_reduce(data, mapper, reducer=None):
    """
    Simple map/reduce with micro threads.

    `data` is a LazyAggregator

    `mapper` argument is a callable which takes an item of data
    and returns a key/value pair or `None`.

    `reducer` argument is a callable which takes an iterable of
    elements and returns a single value

    """
    grouped = LazyAggregator()
    reduced = LazyAggregator() if reducer else None

    def do_map():
        for item in data:
            yield mapper(item)

    def do_reduce():
        # user `iteritems` to grab a fresh iterator
        for key, group in grouped.iteritems():
            yield key, reducer(group)

    M = do_map()
    R = do_reduce() if reducer else None

    for x in M:
        if x:
            key, value = x
            grouped.append(key, value)
            if R:
                reduced.append(*next(R))

    return reduced or grouped


# useful tool for abstractifying big data algs with pure python
class DataTable:
    """
    Exposes a just-in-time iterable that lazily streams data from a
    source, say a csv file, rather than read everything into memory.

    Given the following csv data in a file `contacts.csv`,

    Dan,Cohn,5/28/1983
    Jim,Phoo,5/28/1983
    Tim,Berr,2/15/1973
    ...

    We can use a DataTable to easily find everyone who shares a birthday.

    >>> from csv import reader
    >>> src = reader(open('contacts.csv', 'r'))
    >>> Person = namedtuple('Person', 'firstname lastname birthday')
    >>> contacts = DataTable(Person, src)
    >>> birthdays = contacts.aggregate('birthday')
    >>> for person in birthdays('5/28/1983'):
    ...     print(person.firstname)
    Dan
    Jim
    # ...
    >>> for person in birthdays('2/15/1973'):
    ...     print(person.firstname)
    Tim
    # ...
    """
    def __init__(self, schema, source):
        self.source = source  # source of tuple data
        self.schema = schema  # namedtuple type that wraps the data

    def aggregate(self, field, buffer_size=1):
        """
        Implements a cooperative JIT iteration protocol

        `aggregate` takes one argument, the field which should be
        used for indexing the data set.  Return value is a callable
        that yields all elements in the table that have a matching
        index value.
        """
        aggregator = LazyAggregator()
        source, _  = tee(self.source)  # discard second copy

        def _advance(sz):
            """
            Raises a StopIteration when `source` is exhausted
            """
            def yield_key_group():
                while sz > 0:
                    row = self.schema(*next(source))
                    key = getattr(row, field)
                    yield key, row
                    sz -= 1
                raise StopIteration

            # its ok that the data isn't pre-sorted. while it would make
            # things more efficient, it would also require reading a bunch
            # of data into memory, so we accept this inefficiency.
            aggregator.append_group(
                    groupby(yield_key_group(), itemgetter(0)))

        def _iterate(key):
            iterator, _ = tee(aggregator[key])  # discard second copy
            while True:
                try:
                    yield from iterator         # python 3.2+ only
                except StopIteration:
                    _advance(buffer_size)       # implicitly re-raises StopIteration

        return _iterate


# Example, populating a tree with a values from a probability distribution.
if __name__ == '__main__':
    import random

    def populate(T, N):
        for i in range(1, N):
            num = bin(random.choice(range(i)))
            hsh = list(num.split('b')[-1])
            key = ''.join(hsh)
            t = T[key]
            while len(hsh) > 1:
                x = hsh.pop(0)
                if x not in t:
                    t = t[x]
                else:
                    break
            else:
                t[hsh.pop(0)] = i

    T = Tree()
    populate(T, 100000)

    # do data mining and inferential modeling here...