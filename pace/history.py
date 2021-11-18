import pydantic
from typing import Optional, List, Dict, Sequence, Union
from .membership import Membership, selection_to_series


class Selection(pydantic.BaseModel, frozen=True, extra="forbid"):
    """The Selection class"""
    columns: List = []
    records: List = []
    intersections: List = []


# Selections specify the items that they exclude, so that a nested
# selection can be stored efficiently
class SubSelection(pydantic.BaseModel, extra="forbid"):
    """The SubSelection class"""
    parent: Optional[str] = None
    exclude: Selection = Selection()


def iterated_apply(f, x):
    while x is not None:
        yield x
        x = f(x)


def drop_selection(m: Membership, exclude: Selection):
    """The drop_selection function"""
    # It's important that the pattern selection is made before the
    # column selection, since the pattern indices are reset after a
    # column selection
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
