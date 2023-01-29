import pytest

import numpy as np

from setvis.setexpression import SetExpr, Set, Union, Intersect, Diff, SymDiff, Complement


## Some mappings from names to sets (with different representations)

def lookup_table_set():
    return {'A': {1,3}, 'B': {1,2,4}}


def lookup_table_np():
    return {
        'A': np.array([True, False, True, False]),
        'B': np.array([True, True, False, True]),
        'C': np.array([False, False, False, True])
    }


## Some SetExprs

def expr_union():
    return Set('A') | Set('B')


def expr_diff():
    return Set('A') - Set('B')


def expr_comp():
    return ~Set('A')


def expr1():
    return Set('A') & (Set('B') | ~Set('C'))


def test_expr():
    """Test building a SetExpr with set operations"""

    expr = expr1()

    assert isinstance(expr, SetExpr)
    assert isinstance(expr, Intersect)
    assert expr.a.name == 'A'
    assert isinstance(expr.b, Union)
    assert expr.b.a.name == 'B'
    assert isinstance(expr.b.b, Complement)


@pytest.mark.parametrize(
    "names, expr, expected",
    [
        (lookup_table_set(), expr_union(), {1,2,3,4}),
        (lookup_table_set(), expr_diff(), {3}),

        (lookup_table_np(), expr_union(), np.array([True, True, True, True])),
        (lookup_table_np(), expr_diff(), np.array([False, False, True, False])),
        (lookup_table_np(), expr_comp(), np.array([False, True, False, True])),
        (lookup_table_np(), expr1(), np.array([True, False, True, False])),
    ]
)
def test_eval(names, expr, expected):
    # array_equal for deep equality in arrays and behaves like == for plain values
    assert np.array_equal(
        expr.evaluate(lookup_name=names.get),
        expected
    )



