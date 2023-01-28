"""A :class:`SetExpr` represents either a named set, or a set computed
from various unary or binary operations applied to other set
expressions.

A `SetExpr` is a completely abstract representation of such an
expression, but can be evaluated by providing a way to look up a set's
value from its name.

It is used by setvis to filter based on matching intersections (see
:func:`setvis.membership.Membership.matching_intersections`).

A `SetExpr` is either:

- A named set, :class:`Set`
- A :class:`Union` (set union), :class:`Intersect` (set intersection),
  :class:`Diff` (set difference) or :class:`SymDiff` (symmetric set
  difference) instance, formed from two `SetExpr`
- A :class:`Complement` instance, formed from one `SetExpr`.

A set expression can be built using boolean set notation:

- ``|`` Union
- ``&`` Intersect
- ``-`` Diff
- ``^`` SymDiff
- ``~`` Complement

For example:

>>> (Set('A') & Set('B')) | ~Set('C')
Union(a=Intersect(a=Set(name='A'), b=Set(name='B')), b=Complement(a=Set(name='C')))

For a `SetExpr` to be evaluated with :func:`SetExpr.evaluate()`: the
`lookup_name` argument to ``evaluate`` must be a function (or other
callable) that returns values supporting boolean set operations.  The
required methods are:

- ``__or__`` (``|``) set union
- ``__and__`` (``&``) set intersection
- ``__xor__`` (``^``) symmetric difference
- ``__invert__`` (``~``) set complement

Note that ``__sub__`` (``-``) for set difference is not needed, and
``__invert__`` is required only for evaluating ``Complement`` (the
builtin python ``set`` does not support this operation).


Examples
--------

>>> Set('A')
Set(name='A')
>>> U = Set('A') | Set('B')
>>> U
Union(a=Set(name='A'), b=Set(name='B'))
>>> I = Set('A') & Set('B')
>>> I
Intersect(a=Set(name='A'), b=Set(name='B'))


Associate named sets with values and evaluate the expression. In this
case, the values have type python ``set``, which support ``|`` and
``&`` operators for union and intersection.

>>> set_dict1 = {'A': {1,2,3}, 'B': {1,3,4}}}
>>> U.evaluate(lookup_name=set_dict1.get)
{1,2,3,4}
>>> I.evaluate(lookup_name=set_dict1.get)
{1,3}

The same expressions can be evaluated in a different context.  Here,
set membership is represented as a bit field, and the various set
operations are computed via bitwise operators on integers (``|`` and
``&``).

>>> set_dict2 = {'A': 0b1110, 'B': 0b1011}
>>> bin(U.evaluate(lookup_name=set_dict2.get))
'0b1111'
>>> bin(I.evaluate(lookup_name=set_dict2.get))
'0b1010'

"""


from abc import ABC, abstractmethod
from dataclasses import dataclass


class SetExpr(ABC):
    """An abstract expression involving named sets

    See the docstring on this module (:mod:`setvis.setexpression`)
    for details.
    """

    def __or__(self, b):
        """Union of this set with another"""
        return Union(self, b)

    def __and__(self, b):
        """Intersection of this set with another"""
        return Intersect(self, b)

    def __sub__(self, b):
        """Subtract another set from this set"""
        return Diff(self, b)

    def __xor__(self, b):
        """For the symmetric difference of this set with another"""
        return SymDiff(self, b)

    def __invert__(self):
        """Complement of this set"""
        return Complement(self)

    @abstractmethod
    def evaluate(self, lookup_name):
        """Evaluate this set expression, resolving names with `lookup_name`

        Parameters
        ----------
        lookup_name: Callable
            `lookup_name(name)` should return a set-like value (the
            value of the set with the given `name`).

            This set-like value should support the following methods:

            - ``self.__or__(other)`` returning the set union
            - ``self.__and__(other)`` returning the set intersection
            - ``self.__xor__(other)`` returning the symmetric difference
            - ``self.__invert__()`` returning the set complement
        """


@dataclass
class Set(SetExpr):
    """A set with the given name"""

    name: str

    def evaluate(self, lookup_name):
        """See :func:`SetExpr.evaluate`"""
        return lookup_name(self.name)


@dataclass
class UnaryOp(SetExpr):
    """A generic unary operation on sets"""

    a: SetExpr


@dataclass
class BinaryOp(SetExpr):
    """A generic binary operation on sets"""

    a: SetExpr
    b: SetExpr


class Union(BinaryOp):
    """Set union (a ∪ b)"""

    def evaluate(self, lookup_name):
        """See :func:`SetExpr.evaluate`"""
        return self.a.evaluate(lookup_name) | self.b.evaluate(lookup_name)


class Intersect(BinaryOp):
    """Set intersection (a ∩ b)"""

    def evaluate(self, lookup_name):
        """See :func:`SetExpr.evaluate`"""
        return self.a.evaluate(lookup_name) & self.b.evaluate(lookup_name)


class Diff(BinaryOp):
    """Set difference (a - b)"""

    def evaluate(self, lookup_name):
        """See :func:`SetExpr.evaluate`"""
        # to avoid calling '__invert__' (not supported by builtin set)
        # or '__sub__' (incorrect result in bitwise context)
        a_val = self.a.evaluate(lookup_name)
        return a_val & (a_val ^ self.b.evaluate(lookup_name))


class SymDiff(BinaryOp):
    """Set symmetric difference (a ∆ b)"""

    def evaluate(self, lookup_name):
        """See :func:`SetExpr.evaluate`"""
        return self.a.evaluate(lookup_name) ^ self.b.evaluate(lookup_name)


class Complement(UnaryOp):
    """Set complement (aᶜ)"""

    def evaluate(self, lookup_name):
        """See :func:`SetExpr.evaluate`"""
        return ~self.a.evaluate(lookup_name)
