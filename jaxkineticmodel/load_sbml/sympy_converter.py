import types
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum, auto, unique
from typing import Callable, Optional

import libsbml
import sympy


# Full list of LibSBML AST node types:
@unique
class LibSBMLASTNode(Enum):
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        return libsbml.__dict__['AST_' + name]  # noqa

    CONSTANT_E = auto()
    CONSTANT_FALSE = auto()
    CONSTANT_PI = auto()
    CONSTANT_TRUE = auto()
    DIVIDE = auto()
    FUNCTION = auto()
    FUNCTION_ABS = auto()
    FUNCTION_ARCCOS = auto()
    FUNCTION_ARCCOSH = auto()
    FUNCTION_ARCCOT = auto()
    FUNCTION_ARCCOTH = auto()
    FUNCTION_ARCCSC = auto()
    FUNCTION_ARCCSCH = auto()
    FUNCTION_ARCSEC = auto()
    FUNCTION_ARCSECH = auto()
    FUNCTION_ARCSIN = auto()
    FUNCTION_ARCSINH = auto()
    FUNCTION_ARCTAN = auto()
    FUNCTION_ARCTANH = auto()
    FUNCTION_CEILING = auto()
    FUNCTION_COS = auto()
    FUNCTION_COSH = auto()
    FUNCTION_COT = auto()
    FUNCTION_COTH = auto()
    FUNCTION_CSC = auto()
    FUNCTION_CSCH = auto()
    FUNCTION_DELAY = auto()
    FUNCTION_EXP = auto()
    FUNCTION_FACTORIAL = auto()
    FUNCTION_FLOOR = auto()
    FUNCTION_LN = auto()
    FUNCTION_LOG = auto()
    FUNCTION_MAX = auto()
    FUNCTION_MIN = auto()
    FUNCTION_PIECEWISE = auto()
    FUNCTION_POWER = auto()
    FUNCTION_QUOTIENT = auto()
    FUNCTION_RATE_OF = auto()
    FUNCTION_REM = auto()
    FUNCTION_ROOT = auto()
    FUNCTION_SEC = auto()
    FUNCTION_SECH = auto()
    FUNCTION_SIN = auto()
    FUNCTION_SINH = auto()
    FUNCTION_TAN = auto()
    FUNCTION_TANH = auto()
    INTEGER = auto()
    LAMBDA = auto()
    LOGICAL_AND = auto()
    LOGICAL_IMPLIES = auto()
    LOGICAL_NOT = auto()
    LOGICAL_OR = auto()
    LOGICAL_XOR = auto()
    MINUS = auto()
    NAME = auto()
    NAME_AVOGADRO = auto()
    NAME_TIME = auto()
    # ORIGINATES_IN_PACKAGE = auto()
    PLUS = auto()
    POWER = auto()
    RATIONAL = auto()
    REAL = auto()
    REAL_E = auto()
    RELATIONAL_EQ = auto()
    RELATIONAL_GEQ = auto()
    RELATIONAL_GT = auto()
    RELATIONAL_LEQ = auto()
    RELATIONAL_LT = auto()
    RELATIONAL_NEQ = auto()
    TIMES = auto()
    UNKNOWN = auto()


LIBSBML_TIME_NAME = "time"


@dataclass
class Mapping:
    # If a mapping exactly one of sympy_op and libsbml_op set,
    # SympyConverter should have a custom method for that op.
    # Otherwise, it must have both sympy_op and libsbml_op set and
    # should _not_ have such a method.
    sympy_op: Optional[type[sympy.Basic]]
    libsbml_op: Optional[LibSBMLASTNode]
    arg_count: Optional[int]


# TODO: Rewrite the below as documentation for this class/module
# Here, we would really like to use sympy's MathML utilities.  However, we run into several issues that
#  make them unsuitable for now (i.e. requiring a lot of work):
#  - sympy takes presentation issues into account when producing content MathML, e.g. parsing A_Km as a variable
#  A with subscript Km and producing <mml:msub> tags for it, which libsbml can't handle.
#  - sympy also seems to always produce (xml entities for) e.g. Greek letters in the MathML.
#  - libsbml sometimes sets <cn type="integer"> while sympy can only create bare <cn>.
#  A final small issue is that MathML string must be preceded by an <?xml?> preamble and surrounded by a <math>
#  tag.
#  The sympy implementation seems to store the produced XML DOM in MathMLPrinterBase.dom, which would allow for
#  traversing it and fixing some of these issues.  But this seems like a lot more trouble than it's worth.

MAPPINGS = [
    Mapping(sympy.Add, LibSBMLASTNode.PLUS, None),
    Mapping(sympy.Mul, LibSBMLASTNode.TIMES, None),
    Mapping(sympy.Pow, LibSBMLASTNode.POWER, 2),
    Mapping(sympy.sin, LibSBMLASTNode.FUNCTION_SIN, 1),
    Mapping(sympy.Symbol, None, 0),
    Mapping(sympy.Integer, None, 0),
    Mapping(sympy.Float, None, 0),
    Mapping(None, LibSBMLASTNode.NAME_TIME, 0),
]


class SympyConverter:
    time_variable_name: str

    SYMPY2LIBSBML: dict[type[sympy.Basic], Mapping] = {}
    LIBSBML2SYMPY: dict[LibSBMLASTNode, Mapping] = {}
    for mp in MAPPINGS:
        SYMPY2LIBSBML[mp.sympy_op] = mp
        LIBSBML2SYMPY[mp.libsbml_op] = mp

    def __init__(self, time_variable_name='t'):
        self.time_variable_name = time_variable_name

    def sympy2libsbml(self, expression: sympy.Basic) -> libsbml.ASTNode:
        children = [self.sympy2libsbml(child) for child in expression.args]

        result = libsbml.ASTNode()
        for sympy_op in expression.__class__.__mro__:
            custom_method = getattr(self,
                                    'convert_sympy_' + sympy_op.__name__,
                                    None)
            mp = self.SYMPY2LIBSBML.get(sympy_op, None)
            if mp is not None or custom_method is not None:
                break
        else:
            raise NotImplementedError(f"can't deal yet with expression type {type(expression)}")

        assert mp is not None
        if mp.arg_count is not None and len(children) != mp.arg_count:
            raise ValueError(f'Unexpected number of arguments for '
                             f'{mp.sympy_op}: expected {mp.arg_count}, got '
                             f'{len(children)}')

        if custom_method is not None:
            assert mp.libsbml_op is None
            result = custom_method(expression, result, children)
        else:
            result.setType(mp.libsbml_op.value)
            for child in children:
                result.addChild(child)

        if not result.isWellFormedASTNode():
            raise RuntimeError('Failed to build a well-formed '
                               'LibSBML AST node')
        return result

    def libsbml2sympy(self, node: libsbml.ASTNode) -> sympy.Basic:
        raise NotImplementedError('work in progress')
        if not node.isWellFormedASTNode():
            raise ValueError('Got invalid libSBML AST node')

        children = []
        for idx in range(node.getNumChildren()):
            child = node.getChild(idx)
            children.append(self.libsbml2sympy(child))

        libsbml_op = LibSBMLASTNode(node.getType())
        m = self.LIBSBML2SYMPY.get(libsbml_op, None)
        if m is None:
            raise NotImplementedError(f"can't deal yet with libsbml ASTNode "
                                      f"type {node.getType()}")
        if m.arg_count is not None and len(children) != m.arg_count:
            raise ValueError(f'Unexpected number of arguments for '
                             f'{m.sympy_op}: expected {m.arg_count}, got '
                             f'{len(children)}')

        result = m.sympy_op(*children)

        return result

    def convert_sympy_Integer(self, number, result, children) -> libsbml.ASTNode:
        assert isinstance(number, sympy.Integer) and len(children) == 0
        result.setType(LibSBMLASTNode.INTEGER.value)
        result.setValue(int(number))
        return result

    def convert_sympy_Float(self, number, result, children) -> libsbml.ASTNode:
        assert isinstance(number, sympy.Float) and len(children) == 0
        result.setType(LibSBMLASTNode.REAL.value)
        result.setValue(float(number))
        return result

    def convert_sympy_Symbol(self, symbol, result, children) -> libsbml.ASTNode:
        assert isinstance(symbol, sympy.Symbol) and len(children) == 0
        if symbol.name == self.time_variable_name:
            result.setType(LibSBMLASTNode.NAME_TIME.value)
            result.setName(LIBSBML_TIME_NAME)
        else:
            result.setType(LibSBMLASTNode.NAME.value)
            result.setName(symbol.name)
        return result
