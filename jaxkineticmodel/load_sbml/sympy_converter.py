from dataclasses import dataclass
from logging import Logger
from typing import Optional
from jaxkineticmodel.utils import get_logger

import libsbml
import sympy

logger = get_logger(__name__)

LIBSBML_TIME_NAME = "time"


@dataclass
class Mapping:
    # If a mapping has exactly one of sympy_op and libsbml_op set,
    # then SympyConverter should have a custom method for that op.
    # Otherwise, it must have both sympy_op and libsbml_op set and
    # should _not_ have such a method.
    sympy_op: Optional[type[sympy.Basic]]
    libsbml_op: Optional[int]
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

#here I can add extra mappings ! #PAUL
MAPPINGS = [
    Mapping(sympy.Add, libsbml.AST_PLUS, None),
    Mapping(sympy.Mul, libsbml.AST_TIMES, None),
    Mapping(None, libsbml.AST_DIVIDE, 2), #new
    Mapping(None, libsbml.AST_FUNCTION, None),#new
    Mapping(None, libsbml.AST_FUNCTION_DELAY,2), #new
    Mapping(None, libsbml.AST_MINUS, None), #new
    Mapping(None, libsbml.AST_REAL_E, 0), #new
    Mapping(sympy.exp,libsbml.AST_FUNCTION_EXP,1),
    Mapping(None, libsbml.AST_FUNCTION_PIECEWISE, None),
    Mapping(sympy.root, libsbml.AST_FUNCTION_ROOT, 2),
    Mapping(None, libsbml.AST_LAMBDA, None), #new
    Mapping(sympy.Piecewise, None, None),
    Mapping(sympy.Pow, libsbml.AST_POWER, 2),
    Mapping(sympy.Pow, libsbml.AST_FUNCTION_POWER, 2), #new
    Mapping(sympy.log,libsbml.AST_FUNCTION_LOG, None),
    Mapping(sympy.Lt, libsbml.AST_RELATIONAL_LT, 2),
    Mapping(sympy.Le, libsbml.AST_RELATIONAL_LEQ, 2),
    Mapping(sympy.Gt, libsbml.AST_RELATIONAL_GT, 2),
    Mapping(sympy.Ge, libsbml.AST_RELATIONAL_GEQ, 2),
    Mapping(sympy.Eq, libsbml.AST_RELATIONAL_EQ, 2),

    Mapping(sympy.And,libsbml.AST_LOGICAL_AND, None),
    Mapping(sympy.Or,libsbml.AST_LOGICAL_OR, None),
    Mapping(sympy.Xor,libsbml.AST_LOGICAL_XOR, None),

    Mapping(sympy.Ne, libsbml.AST_RELATIONAL_NEQ, 2),

    Mapping(sympy.Abs, libsbml.AST_FUNCTION_ABS, 1),
    Mapping(sympy.sin, libsbml.AST_FUNCTION_SIN, 1),
    Mapping(sympy.cos, libsbml.AST_FUNCTION_COS, 1), #new
    Mapping(sympy.ln, libsbml.AST_FUNCTION_LN,1), #new
    Mapping(sympy.Min, libsbml.AST_FUNCTION_MIN, None),
    Mapping(sympy.Max, libsbml.AST_FUNCTION_MAX, None),  # new
    Mapping(sympy.logic.boolalg.BooleanTrue, libsbml.AST_CONSTANT_TRUE, 0),
    Mapping(sympy.logic.boolalg.BooleanFalse, libsbml.AST_CONSTANT_FALSE, 0),
    Mapping(sympy.core.numbers.Exp1, libsbml.AST_CONSTANT_E, 0),
    Mapping(sympy.core.numbers.Pi, libsbml.AST_CONSTANT_PI, 0),
    Mapping(sympy.core.numbers.NaN, None, 0),
    Mapping(sympy.Symbol, None, 0),
    Mapping(sympy.Integer, None, 0),
    Mapping(sympy.Float, None, 0),
    Mapping(sympy.Rational, None, 0),
    Mapping(None, libsbml.AST_NAME, 0),
    Mapping(None, libsbml.AST_NAME_TIME, 0),
    Mapping(None, libsbml.AST_INTEGER, 0),
    Mapping(None, libsbml.AST_REAL, 0),
    Mapping(None, libsbml.AST_RATIONAL, 0),
    Mapping(None, libsbml.AST_NAME_AVOGADRO, 0),

]

AST_NODE_TYPE_NAMES = {
    t: n[4:] for n, t in libsbml.__dict__.items() if n.startswith("AST_")  # noqa
}


class Converter:
    time_variable_name: str
    precision: Optional[float]

    def __init__(self, time_variable_name='t', precision=1e-6):
        """
        Create a converter.
        :param time_variable_name: the name of the time variable in sympy. Defaults to 't'.
        :param precision: the desired precision for recognising constants
        such as Avogadro's number.  Defaults to 1e-6.  Set to None to
        disable recognising constants.
        """
        self.time_variable_name = time_variable_name
        self.precision = precision


class SympyConverter(Converter):
    SYMPY2LIBSBML: dict[type[sympy.Basic], Mapping] = {
        mp.sympy_op: mp for mp in MAPPINGS
    }

    avogadro_number: float = libsbml.ASTNode(libsbml.AST_NAME_AVOGADRO).getValue()
    avogadro_lb: Optional[float] = None
    avogadro_ub: Optional[float] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.precision:
            # The SBML standard (Level 3 Version 2, section 3.4.6) stipulates that the value of Avogadro's constant is
            # taken to be equivalent to the value one officially determined in 2006.  However, the current official
            # value (also in sympy.physics.units) is very slightly lower.  This deals with this tiny variation.
            self.avogadro_lb = self.avogadro_number * (1 - self.precision)
            self.avogadro_ub = self.avogadro_number * (1 + self.precision)

    def sympy2libsbml(self, expression: sympy.Basic) -> libsbml.ASTNode:
        result = libsbml.ASTNode()
        for sympy_op in expression.__class__.__mro__:
            custom_method = getattr(self,
                                    f'convert_sympy_{sympy_op.__name__}',
                                    None)
            mp = self.SYMPY2LIBSBML.get(sympy_op, None)
            if mp is not None or custom_method is not None:
                break
        else:
            raise NotImplementedError(f"can't deal yet with expression type {type(expression)}")

        assert mp is not None
        if mp.arg_count is not None and len(expression.args) != mp.arg_count:
            raise ValueError(f'Unexpected number of arguments for '
                             f'{mp.sympy_op}: expected {mp.arg_count}, got '
                             f'{len(expression.args)}')

        if custom_method is not None:
            assert mp.libsbml_op is None
            result = custom_method(expression, result)
        else:
            result.setType(mp.libsbml_op)
            for child in expression.args:
                # Recursively convert child nodes
                result.addChild(self.sympy2libsbml(child))

        if not result.isWellFormedASTNode():
            raise RuntimeError('Failed to build a well-formed '
                               'LibSBML AST node')
        return result

    def convert_sympy_Integer(self, number, result) -> libsbml.ASTNode:
        assert isinstance(number, sympy.Integer) and len(number.args) == 0
        value = int(number)
        if self.precision and self.avogadro_lb <= value <= self.avogadro_ub:
            if value != self.avogadro_number:
                logger.warning("Assumed approximation of Avogadro's number")
            result.setType(libsbml.AST_NAME_AVOGADRO)
        else:
            result.setType(libsbml.AST_INTEGER)
            result.setValue(value)
        return result

    def convert_sympy_Float(self, number, result) -> libsbml.ASTNode:
        assert isinstance(number, sympy.Float) and len(number.args) == 0
        value = float(number)
        if self.precision and self.avogadro_lb <= value <= self.avogadro_ub:
            if value != self.avogadro_number:
                logger.warning("Assumed approximation of Avogadro's number")
            result.setType(libsbml.AST_NAME_AVOGADRO)
        else:
            result.setType(libsbml.AST_REAL)
            result.setValue(value)
        return result

    def convert_sympy_Rational(self, number, result) -> libsbml.ASTNode:
        assert isinstance(number, sympy.Rational) and len(number.args) == 0
        result.setType(libsbml.AST_RATIONAL)
        result.setValue(number.p, number.q)
        return result

    def convert_sympy_NaN(self, nan, result) -> libsbml.ASTNode:
        assert isinstance(nan, sympy.core.numbers.NaN) and len(nan.args) == 0
        result.setType(libsbml.AST_REAL)
        result.setValue(float('nan'))
        return result

    def convert_sympy_Symbol(self, symbol, result) -> libsbml.ASTNode:
        assert isinstance(symbol, sympy.Symbol) and len(symbol.args) == 0
        if symbol.name == self.time_variable_name:
            result.setType(libsbml.AST_NAME_TIME)
            result.setName(LIBSBML_TIME_NAME)
        else:
            result.setType(libsbml.AST_NAME)
            result.setName(symbol.name)
        return result

    def convert_sympy_Piecewise(self, expr, result) -> libsbml.ASTNode:
        assert isinstance(expr, sympy.Piecewise)
        result.setType(libsbml.AST_FUNCTION_PIECEWISE)
        # For sympy piecewise functions, the conditions don't have to be
        # mutually exclusive; they are evaluated left-to-right and the
        # first one that matches is applied.
        # However, for libsbml, no order is assumed, and the entire
        # expression is considered to be undefined if multiple conditions
        # evaluate to true (but values differ).
        # Fortunately, sympy offers functionality to rewrite a piecewise
        # expression to make the conditions mutually exclusive.
        piecewise = sympy.functions.piecewise_exclusive(expr)
        for (value, condition) in piecewise.args:
            result.addChild(self.sympy2libsbml(value))
            result.addChild(self.sympy2libsbml(condition))

        return result


class LibSBMLConverter(Converter):
    LIBSBML2SYMPY: dict[int, Mapping] = {
        mp.libsbml_op: mp for mp in MAPPINGS
    }

    def libsbml2sympy(self, node: libsbml.ASTNode) -> sympy.Basic:
        node = libsbml.ASTNode(node)    # Work around a bug in libsbml
        if not node.isWellFormedASTNode():
            raise ValueError('Got invalid libSBML AST node')

        children = []
        for idx in range(node.getNumChildren()):
            child = node.getChild(idx)
            children.append(self.libsbml2sympy(child))

        libsbml_op = node.getType()
        libsbml_op_name = AST_NODE_TYPE_NAMES.get(libsbml_op, "(unknown)")

        m = self.LIBSBML2SYMPY.get(libsbml_op, None)
        if m is None:
            raise NotImplementedError(f"can't deal yet with libsbml ASTNode "
                                      f"type {libsbml_op_name}")
        if m.arg_count is not None and len(children) != m.arg_count:
            raise ValueError(f'Unexpected number of children for '
                             f'{libsbml_op_name}: expected {m.arg_count}, '
                             f'got {len(children)}')

        custom_method = getattr(self,
                                f'convert_libsbml_{libsbml_op_name}',
                                None)

        if custom_method is not None:
            assert m.sympy_op is None
            result = custom_method(node, children)
        else:
            result = m.sympy_op(*children)

        return result

    def convert_libsbml_NAME(self, node, children) -> sympy.Basic:
        assert node.getType() == libsbml.AST_NAME
        assert len(children) == 0
        return sympy.Symbol(node.getName())

    def convert_libsbml_NAME_TIME(self, node, children) -> sympy.Basic:
        assert node.getType() == libsbml.AST_NAME_TIME
        assert len(children) == 0
        return sympy.Symbol(self.time_variable_name)

    def convert_libsbml_INTEGER(self, node, children) -> sympy.Basic:
        assert node.getType() == libsbml.AST_INTEGER
        assert len(children) == 0
        return sympy.Integer(node.getValue())

    def convert_libsbml_REAL(self, node, children) -> sympy.Basic:
        assert node.getType() == libsbml.AST_REAL
        assert len(children) == 0
        return sympy.Float(node.getValue())

    def convert_libsbml_DIVIDE(self, node, children) -> sympy.Basic:
        "Division has two children a and b (a/b)"
        assert node.getType() == libsbml.AST_DIVIDE
        assert len(children) == 2
        numerator, denominator = children
        return sympy.Mul(numerator,sympy.Pow(denominator,-1))

    def convert_libsbml_FUNCTION_PIECEWISE(self, node, children) -> sympy.Basic:
        assert node.getType() == libsbml.AST_FUNCTION_PIECEWISE
        if len(children) == 0:
            # According to MathML documentation: "The degenerate case of no
            # piece elements and no otherwise element is treated as
            # undefined for all values of the domain."
            return sympy.S.NaN

        if len(children) % 2 == 1:
            # Handle <otherwise> case.  This can be dealt with in sympy by
            # having the last condition-value-pair always match.
            children.append(sympy.S.true)
        pieces = []
        for idx in range(0, len(children), 2):
            value = children[idx]
            condition = children[idx + 1]
            pieces.append((value, condition))
        return sympy.Piecewise(*pieces)

    def convert_libsbml_MINUS(self, node, children) -> sympy.Basic:
        "MINUS can have one child (symbol is negative) or two children (a-b)"
        assert node.getType() == libsbml.AST_MINUS
        if len(children) == 1:
            a=children[0]
            return -a
        if len(children) == 2:
            a, b = children
            return sympy.Add(a, -b)
        else:
            raise logger.error(f"ERROR: Unexpected number of children for MINUS: {len(children)}")

    def convert_libsbml_REAL_E(self, node, children) -> sympy.Basic:
        assert node.getType() == libsbml.AST_REAL_E
        assert len(children) == 0

        base = sympy.Float(node.getMantissa())  # Extracts the base (mantissa)
        exponent = sympy.Integer(node.getExponent())  # Extracts the exponent

        return sympy.Mul(base, sympy.Pow(10, exponent))  # Represents base * 10^exponent


    def convert_libsbml_FUNCTION(self, node, children) -> sympy.Basic:
        """some functions have no children, deal with this later """
        assert node.getType() == libsbml.AST_FUNCTION

        function_name = node.getName()  # Get function name (e.g., 'f', 'g', etc.)
        if not function_name:
            raise ValueError("FUNCTION node has no associated name")

        sympy_function = sympy.Function(function_name)  # Define function
        return sympy_function(*children)  # Apply function to arguments

    def convert_libsbml_LAMBDA(self,node,children)-> sympy.Basic:
        """Mapping to sp lambda.
        The argument order matters in lambda functions and needs to be retrieved properly """
        assert node.getType() == libsbml.AST_LAMBDA
        assert len(children) >= 1
        lambda_function=sympy.Lambda(tuple(children[:-1]),children[-1]) #last child is the expression
        return lambda_function

    def convert_libsbml_RATIONAL(self,node,children)-> sympy.Basic:
        """Convert rational to sympy float. Mapping directly to sympy.float didnt work."""
        assert node.getType() == libsbml.AST_RATIONAL
        assert len(children) == 0
        return sympy.Rational(node.getNumerator(), node.getDenominator())

    def convert_libsbml_NAME_AVOGADRO(self,node,children)-> sympy.Basic:
        assert node.getType() == libsbml.AST_NAME_AVOGADRO
        return sympy.Float(node.getValue())

    def convert_libsbml_FUNCTION_DELAY(self,node,children)-> sympy.Basic:
        assert node.getType() == libsbml.AST_FUNCTION_DELAY
        assert len(children)==2
        expression= children[0] - children[1]
        return expression





