from typing import Optional, List, Sequence, Dict
from dataclasses import dataclass, field
from .missingness import Missingness


@dataclass
class Selection:
    columns: List = field(default_factory=list)
    records: Sequence[int] = field(default_factory=list)
    patterns: Sequence[int] = field(default_factory=list)


# Selections specify the items that they exclude, so that a nested
# selection can be stored efficiently
@dataclass
class SubSelection:
    parent: Optional[str] = None
    exclude: Selection = Selection()


def iterated_apply(f, x):
    while x is not None:
        yield x
        x = f(x)


def drop_selection(m: Missingness, exclude: Selection):
    # It's important that the pattern selection is made before the
    # column selection, since the pattern indices are reset after a
    # column selection
    return (
        m.drop_records(exclude.records)
        .drop_patterns(exclude.patterns)
        .drop_columns(exclude.columns)
    )


class SelectionHistory:
    def __init__(self, missingness: Missingness, journal_file=None):
        # attempt to open the journal file for reading if it exists,
        # otherwise create it
        if journal_file:
            pass

        self._missingness = missingness
        self._selections: Dict[str, SubSelection] = {}

    def new_selection(self, name, base_selection=None):
        self._selections[name] = SubSelection(base_selection)

    # Can't straightforwardly provide an interface specifying what
    # should be included in the selection (rather than excluded from
    # it), since this would mean calculating the parent's selection --
    # this is left to the caller
    def update_active(self, name, exclude: Optional[Selection]):
        self._selections[name].exclude = exclude

    def parent(self, name: str):
        return self._selections[name].parent

    def ancestors(self, name):
        return list(iterated_apply(self.parent, name))

    def missingness(self, name: Optional[str] = None):
        m = self._missingness
        for ancestor in reversed(self.ancestors(name)):
            m = drop_selection(m, self._selections[ancestor].exclude)
        return m

    def save(self):
        raise NotImplementedError()
