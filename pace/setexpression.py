from abc import ABC, abstractmethod
from dataclasses import dataclass


class SetExpr(ABC):
    def __or__(self, b):
        return Union(self, b)

    def __and__(self, b):
        return Intersect(self, b)

    def __invert__(self):
        return Complement(self)

    @abstractmethod
    def run_with(self, f):
        pass


@dataclass
class Set(SetExpr):
    name: str
    def run_with(self, f):
        return f(self.name)


@dataclass
class UnaryOp(SetExpr):
    a: SetExpr


@dataclass
class BinaryOp(SetExpr):
    a: SetExpr
    b: SetExpr

    
@dataclass
class Union(BinaryOp):
    def run_with(self, f):
        return self.a.run_with(f) | self.b.run_with(f)


@dataclass
class Intersect(BinaryOp):
    def run_with(self, f):
        return self.a.run_with(f) & self.b.run_with(f)


@dataclass
class Complement(UnaryOp):
    def run_with(self, f):
        return ~self.a.run_with(f)

