import math

class OperatorAdd:
    def eval(a, b):
        return a + b

    def dda(a, b):
        return 1

    def ddb(a, b):
        return 1

class OperatorSubtract:
    def eval(a, b):
        return a - b

    def dda(a, b):
        return 1

    def ddb(a, b):
        return -1

class OperatorMultiply:
    def eval(a, b):
        return a * b

    def dda(a, b):
        return b

    def ddb(a, b):
        return a

class OperatorDivide:
    def eval(a, b):
        return a / b

    def dda(a, b):
        return 1 / b

    def ddb(a, b):
        return -a / (b ** 2)

class OperatorPower:
    def eval(a, b):
        return a ** b

    def dda(a, b):
        return b * (a ** (b - 1))

    def ddb(a, b):
        return math.log(a) * (a ** b)

class OperatorLog:
    def eval(a, b):
        return math.log(a) / math.log(b)

    def dda(a, b):
        return 1 / (math.log(b) * a)

    def ddb(a, b):
        return -math.log(a) / (b * ( (math.log(b)) ** 2 ) )

class OperatorSin:
    def eval(a):
        return math.sin(a)
    
    def dda(a):
        return math.cos(a)

class OperatorCos:
    def eval(a):
        return math.cos(a)

    def dda(a):
        return -math.sin(a)

class OperatorTan:
    def eval(a):
        return math.tan(a)

    def dda(a):
        return 1 / ((math.cos(a)) ** 2)