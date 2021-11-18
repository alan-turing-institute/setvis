import pydantic
from typing import Optional, List, Dict, Sequence, Union, Callable, Any
from .membership import Membership, selection_to_series


class Selection(pydantic.BaseModel, frozen=True, extra="forbid"):
    """A collection to hold data describing a selection of items in a
    ``Membership`` object

    This class is used as a convenient container for lists of records
    identifiers, intersection identifiers and column names to be
    considered by a method -- it is up to the particular method how
    these are interpreted, but generally as a "selection" of items to
    include (or exclude) from processing.

    Since columns, record IDs, and intersection IDs are specific to a
    ``Membership`` object, a Selection should always be used to refer
    to a selection of items from the same object, although no
    reference is kept to it.

    Specifying ``None`` for any of the attributes is intended to be
    equivalent to specifying a list of all values of that attribute in
    the related ``Membership`` object.

    Attributes
    ----------
    columns : Optional[List]
        The included column names (may be any value returned by
        ``Membership.columns()``, which will generally be the same as
        in the underlying data source)
    records : Optional[List]
        The included record IDs (may be any value in
        ``Membership.columns()["_record_id"]``)
    intersections : Optional[List]
        The included intersection IDs (may be any value in
        ``Membership.intersections().index``)

    """

    columns: Optional[List] = []
    records: Optional[List] = []
    intersections: Optional[List] = []


class SubSelection(pydantic.BaseModel, extra="forbid"):
    """A selection relative to a named parent selection

    It is used internally by ``SelectionHistory``, and should not be
    needed in user code.

    This is intended to represent a selection that is the same as the
    one named `parent` (extrinsic to this class), but with the
    selection `exclude` removed by ``drop_selection``.

    Selections specify the items that they exclude, rather than
    include, so that nested selections can be stored efficiently.

    """

    parent: Optional[str] = None
    exclude: Selection = Selection()


def iterated_apply(f: Callable, x: Any):
    """Generator of repeated function applications

    The function `f` must be callable with a single argument.  This is
    repeatedly applied to its own output, ``yield``\ ing the result
    each time until `f` returns ``None``. The initial result is
    `x`. This generates the sequence

        x, f(x), f(f(x)), ... , f( ... f(x) ...)

    which either does not terminate, or terminates when one more
    application of `f` would be ``None``.

    """
    while x is not None:
        yield x
        x = f(x)


def drop_selection(m: Membership, exclude: Selection) -> Membership:
    """Remove records and columns from a ``Membership`` object

    Returns a new membership object the same as `m` but with:

    - records with the given record IDs (`exclude.records`) removed
    - all records with the given intersection IDs (`exclude.intersections`) removed
    - columns with the given column names (`exclude.columns`) dropped

    Parameters
    ----------
    m : Membership
        The initial ``Membership`` object
    exclude : Selection
        The items to exclude

    Returns
    -------
    Membership
        A membership object, as described
    """

    # It is important that the intersections are dropped before the
    # columns, since the intersection IDs are reset after a column
    # selection
    return (
        m.drop_records(exclude.records)
        .drop_intersections(exclude.intersections)
        .drop_columns(exclude.columns)
    )


def _parse_selections(d):
    """Helper function for SelectionHistory constructor"""
    return pydantic.parse_obj_as(Dict[str, SubSelection], d)


class SelectionHistory:
    """The SelectionHistory class"""

    def __init__(
        self,
        membership: Membership,
        selections: Optional[Dict[str, SubSelection]] = None,
    ):
        self._membership = membership
        self._selections = (
            _parse_selections(selections) if selections is not None else {}
        )

    def new_selection(self, name, base_selection=None, force_reset=False):
        keep_current = (
            not force_reset
            and name in self._selections
            and base_selection == self.parent(name)
        )
        if not keep_current:
            # (re)set the active selection to the entire base_selection
            self._selections[name] = SubSelection(parent=base_selection)

    def __getitem__(self, name):
        return self._selections[name].exclude

    # Can't straightforwardly provide an interface specifying what
    # should be included in the selection (rather than excluded from
    # it), since this would mean calculating the parent's selection --
    # this is left to the caller
    def __setitem__(self, name, exclude: Optional[Selection]):
        self._selections[name].exclude = exclude

    def parent(self, name: str):
        return self._selections[name].parent

    def ancestors(self, name):
        return list(iterated_apply(self.parent, name))

    def membership(self, name: Optional[str] = None):
        m = self._membership
        for ancestor in reversed(self.ancestors(name)):
            m = drop_selection(m, self._selections[ancestor].exclude)
        return m

    def selected_records(
        self,
        name: Optional[str] = None,
        base_selection: Optional[str] = None,
        sort=True,
    ):
        base_records = self.membership(base_selection).records()
        selected_records = self.membership(name).records()
        return selection_to_series(base_records, selected_records, sort)

    def dict(self):
        return {k: v.dict() for k, v in self._selections.items()}
