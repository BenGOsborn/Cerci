from autograd_expressions import ExpressionSingle, ExpressionDouble, Variable
from autograd_operators import OperatorSin, OperatorCos, OperatorTan, OperatorLog
from math import e as exp

# Singles
def Sin(a):
    return ExpressionSingle(a, OperatorSin)

def Cos(a):
    return ExpressionSingle(a, OperatorCos)

def Tan(a):
    return ExpressionSingle(a, OperatorTan)

# Doubles
def Log(a, b=Variable(exp)):
    return ExpressionDouble(a, b, OperatorLog)