"""Built-in calculator / computational engine (safe subset)."""

from __future__ import annotations

import ast
import operator as op


class Calculator:
    def evaluate(self, expr: str):
        node = ast.parse(expr, mode="eval")
        return self._eval(node.body)

    def _eval(self, node):
        operators = {
            ast.Add: op.add,
            ast.Sub: op.sub,
            ast.Mult: op.mul,
            ast.Div: op.truediv,
            ast.Pow: op.pow,
            ast.USub: op.neg,
            ast.Mod: op.mod,
        }
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp):
            return operators[type(node.op)](self._eval(node.left), self._eval(node.right))
        if isinstance(node, ast.UnaryOp):
            return operators[type(node.op)](self._eval(node.operand))
        raise ValueError("Unsupported expression")
