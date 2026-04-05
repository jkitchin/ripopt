#!/usr/bin/env python3
"""
Parse Schittkowski PROB.FOR and generate Rust + Python implementations
of Hock-Schittkowski test problems for validating ripopt against cyipopt.

Uses sympy to compute Hessians (not provided in the Fortran source).
"""

import re
import os
import sys
import textwrap
from dataclasses import dataclass, field
from typing import Optional

import sympy
from sympy import symbols, sqrt, log, exp, sin, cos, tan, asin, acos, atan, Abs, sign, pi, Rational

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class HSProblem:
    """Parsed HS test problem."""
    number: int
    n: int = 0  # number of variables
    nili: int = 0  # number of linear inequality constraints
    ninl: int = 0  # number of nonlinear inequality constraints
    neli: int = 0  # number of linear equality constraints
    nenl: int = 0  # number of nonlinear equality constraints
    x0: list = field(default_factory=list)  # initial point
    xl: dict = field(default_factory=dict)  # lower bounds {index: value}
    xu: dict = field(default_factory=dict)  # upper bounds {index: value}
    lxl: dict = field(default_factory=dict)  # whether lower bound active {index: bool}
    lxu: dict = field(default_factory=dict)  # whether upper bound active {index: bool}
    fex: Optional[float] = None  # known optimal objective
    xex: list = field(default_factory=list)  # known optimal solution
    obj_expr: str = ""  # objective expression (Fortran)
    grad_exprs: list = field(default_factory=list)  # gradient expressions
    con_exprs: list = field(default_factory=list)  # constraint expressions
    jac_exprs: dict = field(default_factory=dict)  # Jacobian {(i,j): expr}
    # Constant Jacobian entries set in MODE=1
    jac_const: dict = field(default_factory=dict)
    raw_mode1: str = ""
    raw_mode2: str = ""
    raw_mode3: str = ""
    raw_mode4: str = ""
    raw_mode5: str = ""

    @property
    def m(self):
        return self.nili + self.ninl + self.neli + self.nenl


# ---------------------------------------------------------------------------
# Fortran parser
# ---------------------------------------------------------------------------

def read_fortran_file(path):
    """Read Fortran file, join continuation lines."""
    with open(path, 'r') as f:
        lines = f.readlines()

    # Join continuation lines (column 6 is non-blank)
    joined = []
    for line in lines:
        if len(line) > 5 and line[5] not in (' ', '\n', '\t') and line[0] != 'C' and joined:
            # continuation line
            joined[-1] = joined[-1].rstrip('\n') + line[6:].rstrip('\n') + '\n'
        else:
            joined.append(line)
    return joined


def split_subroutines(lines):
    """Split into individual subroutines."""
    subs = {}
    current_name = None
    current_lines = []

    for line in lines:
        m = re.match(r'\s+SUBROUTINE\s+TP(\d+)\s*\(MODE\)', line)
        if m:
            if current_name is not None:
                subs[current_name] = current_lines
            current_name = int(m.group(1))
            current_lines = [line]
        elif current_name is not None:
            current_lines.append(line)

    if current_name is not None:
        subs[current_name] = current_lines

    return subs


def split_modes(lines):
    """Split subroutine into MODE blocks based on GOTO labels."""
    text = ''.join(lines)

    # Find GOTO statement to understand label mapping
    goto_match = re.search(r'GOTO\s*\(([^)]+)\)\s*,?\s*MODE', text)
    if not goto_match:
        return {}

    labels = [l.strip() for l in goto_match.group(1).split(',')]

    # Extract blocks between labels
    modes = {}
    all_lines = text.split('\n')

    for mode_idx, label in enumerate(labels):
        mode_num = mode_idx + 1
        # Find line starting with this label
        start_idx = None
        for i, line in enumerate(all_lines):
            # Match label at start of line (Fortran labels are in columns 1-5)
            stripped = line.lstrip()
            if re.match(rf'^{label}\b', stripped):
                start_idx = i
                break

        if start_idx is None:
            continue

        # Collect lines until RETURN or next label
        block_lines = []
        for i in range(start_idx, len(all_lines)):
            line = all_lines[i]
            stripped = line.strip()

            # Skip the label itself on the first line - keep the content after it
            if i == start_idx:
                # Remove label prefix
                content = re.sub(rf'^\s*{label}\s+', '', line)
                if content.strip():
                    block_lines.append(content)
                continue

            if stripped == 'RETURN' or stripped.startswith('RETURN'):
                break

            # Check if we hit another numbered label that's in our label list
            label_match = re.match(r'^\s*(\d+)\s+', line)
            if label_match:
                other_label = label_match.group(1)
                if other_label in labels and other_label != label:
                    break

            block_lines.append(line)

        modes[mode_num] = '\n'.join(block_lines)

    return modes


def fortran_to_python_expr(expr, n_vars):
    """Convert a Fortran expression to Python/sympy compatible expression."""
    s = expr.strip()

    # Remove Fortran line continuations
    s = re.sub(r'\n\s*[&+]', ' ', s)

    # Replace X(i) with x{i-1}
    def replace_x(m):
        idx = int(m.group(1)) - 1
        return f'x{idx}'
    s = re.sub(r'X\((\d+)\)', replace_x, s)

    # Replace Fortran intrinsics
    replacements = [
        (r'\bDSQRT\b', 'sqrt'),
        (r'\bSQRT\b', 'sqrt'),
        (r'\bDABS\b', 'Abs'),
        (r'\bABS\b', 'Abs'),
        (r'\bDLOG\b', 'log'),
        (r'\bALOG\b', 'log'),
        (r'\bLOG\b', 'log'),
        (r'\bDLOG10\b', 'log10'),
        (r'\bDEXP\b', 'exp'),
        (r'\bEXP\b', 'exp'),
        (r'\bDSIN\b', 'sin'),
        (r'\bSIN\b', 'sin'),
        (r'\bDCOS\b', 'cos'),
        (r'\bCOS\b', 'cos'),
        (r'\bDTAN\b', 'tan'),
        (r'\bTAN\b', 'tan'),
        (r'\bDASIN\b', 'asin'),
        (r'\bASIN\b', 'asin'),
        (r'\bDACOS\b', 'acos'),
        (r'\bACOS\b', 'acos'),
        (r'\bDATAN\b', 'atan'),
        (r'\bATAN\b', 'atan'),
        (r'\bDATAN2\b', 'atan2'),
        (r'\bATAN2\b', 'atan2'),
        (r'\bDBLE\b', 'float'),
        (r'\bSIGN\b', 'sign'),
        (r'\bDSIGN\b', 'sign'),
        (r'\bDMAX1\b', 'max'),
        (r'\bMAX\b', 'max'),
        (r'\bDMIN1\b', 'min'),
        (r'\bMIN\b', 'min'),
        (r'\bDFLOAT\b', 'float'),
        (r'\bFLOAT\b', 'float'),
    ]
    for pat, repl in replacements:
        s = re.sub(pat, repl, s, flags=re.IGNORECASE)

    # Replace D0, D+nn, D-nn exponent notation
    s = re.sub(r'(\d+)\.(\d*)D([+-]?\d+)', r'\1.\2e\3', s, flags=re.IGNORECASE)
    s = re.sub(r'(\d+)\.D([+-]?\d+)', r'\1.0e\2', s, flags=re.IGNORECASE)
    s = re.sub(r'(\d+)\.(\d*)D0', r'\1.\2', s, flags=re.IGNORECASE)
    s = re.sub(r'(\d+)\.D0', r'\1.0', s, flags=re.IGNORECASE)
    # Handle standalone .D0 like 0.D0
    s = re.sub(r'(\d)D0\b', r'\1.0', s, flags=re.IGNORECASE)

    # Replace ** with ** (Python already uses this)
    # Replace .TRUE. .FALSE.
    s = s.replace('.TRUE.', 'True').replace('.FALSE.', 'False')

    return s


def expand_do_loops(text):
    """Expand Fortran DO loops into individual statements with loop variable substituted.

    Handles patterns like:
        DO 6 I=1,3
          X(I)=10.D0
          LXL(I)=.TRUE.
        6 XL(I)=0.D0

    Expands to:
        X(1)=10.D0
        X(2)=10.D0
        X(3)=10.D0
        LXL(1)=.TRUE.
        LXL(2)=.TRUE.
        LXL(3)=.TRUE.
        XL(1)=0.D0
        XL(2)=0.D0
        XL(3)=0.D0
    """
    lines = text.split('\n')
    result = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Match DO label VAR=start,end
        m = re.match(r'DO\s+(\d+)\s+(\w+)\s*=\s*(\d+)\s*,\s*(\d+)', line)
        if m:
            label = m.group(1)
            var = m.group(2)
            start = int(m.group(3))
            end = int(m.group(4))
            # Collect body lines until we find the label
            body_lines = []
            label_line = None
            i += 1
            while i < len(lines):
                bline = lines[i].strip()
                if not bline or bline.startswith('C'):
                    i += 1
                    continue
                # Check if this line starts with the label number
                lm = re.match(r'^' + re.escape(label) + r'\s+(.+)', bline)
                if lm:
                    # This is the label line - it's the last statement in the loop
                    label_line = lm.group(1).strip()
                    i += 1
                    break
                body_lines.append(bline)
                i += 1
            # Now expand: substitute var with each index value
            all_body = body_lines + ([label_line] if label_line else [])
            for idx in range(start, end + 1):
                for stmt in all_body:
                    # Replace VAR used as array index: e.g. X(I) -> X(3)
                    expanded = re.sub(r'\b' + var + r'\b', str(idx), stmt)
                    result.append(expanded)
        else:
            result.append(lines[i])
            i += 1
    return '\n'.join(result)


def parse_mode1(text, prob):
    """Parse MODE=1 block to extract problem dimensions, x0, bounds, known solution."""
    # Expand DO loops before parsing
    text = expand_do_loops(text)
    lines = text.split('\n')

    for line in lines:
        line = line.strip()
        if not line or line.startswith('C'):
            continue

        # N=2
        m = re.match(r'N\s*=\s*(\d+)', line)
        if m:
            prob.n = int(m.group(1))
            continue

        # NILI, NINL, NELI, NENL
        for attr in ['NILI', 'NINL', 'NELI', 'NENL']:
            m = re.match(rf'{attr}\s*=\s*(\d+)', line)
            if m:
                setattr(prob, attr.lower(), int(m.group(1)))

        # X(i)=value or X(i)=X(j) cross-reference
        m = re.match(r'X\((\d+)\)\s*=\s*(.+)', line)
        if m:
            idx = int(m.group(1))
            val_str = m.group(2).strip()
            val = None
            # Check for X(j) cross-reference first
            xref = re.match(r'^X\((\d+)\)$', val_str)
            if xref:
                ref_idx = int(xref.group(1))
                if ref_idx - 1 < len(prob.x0):
                    val = prob.x0[ref_idx - 1]
            if val is None:
                try:
                    val = eval_fortran_const(val_str)
                except:
                    pass
            if val is not None:
                while len(prob.x0) < idx:
                    prob.x0.append(0.0)
                if idx <= len(prob.x0):
                    prob.x0[idx-1] = val
                else:
                    prob.x0.append(val)

        # XL(i)=value
        m = re.match(r'XL\((\d+)\)\s*=\s*(.+)', line)
        if m:
            idx = int(m.group(1))
            try:
                prob.xl[idx] = eval_fortran_const(m.group(2).strip())
            except:
                pass

        # XU(i)=value
        m = re.match(r'XU\((\d+)\)\s*=\s*(.+)', line)
        if m:
            idx = int(m.group(1))
            try:
                prob.xu[idx] = eval_fortran_const(m.group(2).strip())
            except:
                pass

        # LXL(i)=.TRUE./.FALSE.
        m = re.match(r'LXL\((\d+)\)\s*=\s*\.(\w+)\.', line)
        if m:
            idx = int(m.group(1))
            prob.lxl[idx] = m.group(2).upper() == 'TRUE'

        # LXU(i)=.TRUE./.FALSE.
        m = re.match(r'LXU\((\d+)\)\s*=\s*\.(\w+)\.', line)
        if m:
            idx = int(m.group(1))
            prob.lxu[idx] = m.group(2).upper() == 'TRUE'

        # FEX=value
        m = re.match(r'FEX\s*=\s*(.+)', line)
        if m:
            val_str = m.group(1).strip()
            try:
                prob.fex = eval_fortran_const(val_str)
            except:
                pass

        # XEX(i)=value
        m = re.match(r'XEX\((\d+)\)\s*=\s*(.+)', line)
        if m:
            idx = int(m.group(1))
            try:
                val = eval_fortran_const(m.group(2).strip())
                while len(prob.xex) < idx:
                    prob.xex.append(0.0)
                if idx <= len(prob.xex):
                    prob.xex[idx-1] = val
                else:
                    prob.xex.append(val)
            except:
                pass

        # Constant Jacobian entries GG(i,j)=value
        m = re.match(r'GG\((\d+)\s*,\s*(\d+)\)\s*=\s*(.+)', line)
        if m:
            ci = int(m.group(1))
            cj = int(m.group(2))
            try:
                val = eval_fortran_const(m.group(3).strip())
                prob.jac_const[(ci, cj)] = val
            except:
                pass


def eval_fortran_const(s):
    """Evaluate a Fortran constant expression to a float."""
    import math
    s = s.strip()
    # Remove trailing comments
    s = s.split('!')[0].strip()
    # Handle Fortran double precision
    s = re.sub(r'(\d+)\.(\d*)D([+-]?\d+)', r'\1.\2e\3', s, flags=re.IGNORECASE)
    s = re.sub(r'(\d+)\.D([+-]?\d+)', r'\1.0e\2', s, flags=re.IGNORECASE)
    s = re.sub(r'(\d+)\.(\d*)D0', r'\1.\2', s, flags=re.IGNORECASE)
    s = re.sub(r'(\d+)\.D0', r'\1.0', s, flags=re.IGNORECASE)
    s = re.sub(r'(\d)D0\b', r'\1.0', s, flags=re.IGNORECASE)
    # Replace Fortran intrinsics for eval
    s = re.sub(r'\bDATAN\b', 'math.atan', s, flags=re.IGNORECASE)
    s = re.sub(r'\bATAN\b', 'math.atan', s, flags=re.IGNORECASE)
    s = re.sub(r'\bDSQRT\b', 'math.sqrt', s, flags=re.IGNORECASE)
    s = re.sub(r'\bSQRT\b', 'math.sqrt', s, flags=re.IGNORECASE)
    s = re.sub(r'\bDLOG\b', 'math.log', s, flags=re.IGNORECASE)
    s = re.sub(r'\bDEXP\b', 'math.exp', s, flags=re.IGNORECASE)
    s = re.sub(r'\bDACOS\b', 'math.acos', s, flags=re.IGNORECASE)
    s = re.sub(r'\bDASIN\b', 'math.asin', s, flags=re.IGNORECASE)
    s = re.sub(r'\bDBLE\b', 'float', s, flags=re.IGNORECASE)
    # Handle -value at end
    s = s.rstrip()
    try:
        return float(eval(s, {"math": math, "float": float}))
    except:
        return float(s)


def parse_mode2(text, prob):
    """Parse MODE=2 to get objective expression."""
    # Look for FX=...
    m = re.search(r'FX\s*=\s*(.+)', text)
    if m:
        prob.obj_expr = m.group(1).strip()


def parse_mode4(text, prob):
    """Parse MODE=4 to get constraint expressions."""
    # Look for G(i)=... or IF (INDEX1(i)) G(i)=...
    for m in re.finditer(r'(?:IF\s*\(INDEX1\(\d+\)\)\s*)?G\((\d+)\)\s*=\s*(.+)', text):
        idx = int(m.group(1))
        expr = m.group(2).strip()
        while len(prob.con_exprs) < idx:
            prob.con_exprs.append("")
        prob.con_exprs[idx-1] = expr


def parse_mode5(text, prob):
    """Parse MODE=5 to get Jacobian expressions."""
    for m in re.finditer(r'GG\((\d+)\s*,\s*(\d+)\)\s*=\s*(.+)', text):
        ci = int(m.group(1))
        cj = int(m.group(2))
        expr = m.group(3).strip()
        prob.jac_exprs[(ci, cj)] = expr


def extract_local_constants(lines):
    """Extract constant variable assignments before the GOTO statement.

    These are lines like:
      V=4.D0*DATAN(1.D0)
      A=DSQRT(3.0D0)
    that define local constants used in the mode blocks.
    """
    constants = {}
    text = ''.join(lines)

    # Find lines before GOTO
    goto_pos = text.find('GOTO')
    if goto_pos < 0:
        return constants

    pre_goto = text[:goto_pos]
    for line in pre_goto.split('\n'):
        line = line.strip()
        if not line or line.startswith('C') or line.startswith('c'):
            continue
        # Skip declarations
        if any(line.startswith(kw) for kw in ['IMPLICIT', 'INTEGER', 'DOUBLEPRECISION',
                                                'DOUBLE', 'LOGICAL', 'PARAMETER',
                                                'COMMON', 'DIMENSION', 'DATA',
                                                'SUBROUTINE', 'REAL', 'CHARACTER']):
            continue
        if line.startswith('/'):  # continuation of COMMON
            continue

        # Match simple assignment: VARNAME=expr (not array)
        m = re.match(r'^([A-Z]\w*)\s*=\s*(.+)', line)
        if m:
            varname = m.group(1)
            expr = m.group(2).strip()
            # Skip if it looks like an array or standard variable
            if varname in ('N', 'NILI', 'NINL', 'NELI', 'NENL', 'FX', 'FEX', 'LEX', 'NEX'):
                continue
            constants[varname] = expr

    return constants


def substitute_constants(expr_str, constants, n):
    """Substitute known constants into an expression string."""
    result = expr_str
    # Sort by length descending to avoid partial matches
    for varname in sorted(constants.keys(), key=len, reverse=True):
        # Only substitute if it's a word boundary match (not part of X(1), GF(1), etc.)
        pattern = rf'\b{varname}\b'
        val_expr = constants[varname]
        # Try to evaluate to a number
        try:
            val = eval_fortran_const(val_expr)
            replacement = repr(val)
        except:
            # It's a more complex expression - use parenthesized version
            replacement = f'({fortran_to_python_expr(val_expr, n)})'
        result = re.sub(pattern, replacement, result)
    return result


def parse_problem(number, lines):
    """Parse a complete TP subroutine."""
    prob = HSProblem(number=number)
    prob.raw_mode1 = ''.join(lines)

    # Extract local constants before GOTO
    local_consts = extract_local_constants(lines)

    modes = split_modes(lines)

    if 1 in modes:
        prob.raw_mode1 = modes[1]
        parse_mode1(modes[1], prob)
    if 2 in modes:
        prob.raw_mode2 = modes[2]
        parse_mode2(modes[2], prob)
    if 4 in modes:
        prob.raw_mode4 = modes[4]
        parse_mode4(modes[4], prob)
    if 5 in modes:
        prob.raw_mode5 = modes[5]
        parse_mode5(modes[5], prob)

    # Substitute local constants into expressions
    if local_consts and prob.n > 0:
        if prob.obj_expr:
            prob.obj_expr = substitute_constants(prob.obj_expr, local_consts, prob.n)
        new_cons = []
        for ce in prob.con_exprs:
            if ce:
                new_cons.append(substitute_constants(ce, local_consts, prob.n))
            else:
                new_cons.append(ce)
        prob.con_exprs = new_cons

        new_jac = {}
        for k, v in prob.jac_exprs.items():
            new_jac[k] = substitute_constants(v, local_consts, prob.n)
        prob.jac_exprs = new_jac

    return prob


# ---------------------------------------------------------------------------
# Sympy-based Hessian computation
# ---------------------------------------------------------------------------

def try_sympify_expr(expr_str, n, prob_number):
    """Try to convert a Fortran expression string to a sympy expression."""
    py_expr = fortran_to_python_expr(expr_str, n)

    # Create symbols
    xs = symbols([f'x{i}' for i in range(n)])

    # Build local namespace
    ns = {}
    for i, x in enumerate(xs):
        ns[f'x{i}'] = x
    ns['sqrt'] = sympy.sqrt
    ns['log'] = sympy.log
    ns['log10'] = lambda x: sympy.log(x, 10)
    ns['exp'] = sympy.exp
    ns['sin'] = sympy.sin
    ns['cos'] = sympy.cos
    ns['tan'] = sympy.tan
    ns['asin'] = sympy.asin
    ns['acos'] = sympy.acos
    ns['atan'] = sympy.atan
    ns['atan2'] = sympy.atan2
    ns['Abs'] = sympy.Abs
    ns['sign'] = sympy.sign
    ns['pi'] = sympy.pi
    ns['max'] = sympy.Max
    ns['min'] = sympy.Min
    ns['float'] = lambda x: x

    try:
        result = eval(py_expr, {"__builtins__": {}}, ns)
        return result, xs
    except Exception as e:
        return None, xs


def compute_hessian_data(prob):
    """
    Compute Hessian of Lagrangian using sympy.
    Returns (success, hess_struct, hess_code_rust, hess_code_python) or (False, ...) on failure.
    """
    n = prob.n
    m = prob.m
    xs = symbols([f'x{i}' for i in range(n)])

    # Parse objective
    obj_sym, _ = try_sympify_expr(prob.obj_expr, n, prob.number)
    if obj_sym is None:
        return False, None, None, None

    # Parse constraints
    con_syms = []
    for i, cexpr in enumerate(prob.con_exprs):
        if not cexpr:
            return False, None, None, None
        csym, _ = try_sympify_expr(cexpr, n, prob.number)
        if csym is None:
            return False, None, None, None
        con_syms.append(csym)

    # Compute objective Hessian
    try:
        obj_hess = sympy.hessian(obj_sym, xs)
    except Exception:
        return False, None, None, None

    # Compute constraint Hessians
    con_hess_list = []
    for csym in con_syms:
        try:
            ch = sympy.hessian(csym, xs)
        except Exception:
            return False, None, None, None
        con_hess_list.append(ch)

    # Extract lower triangle structure and expressions
    hess_entries = []  # [(row, col, obj_expr, [con_expr_0, con_expr_1, ...])]
    for i in range(n):
        for j in range(i+1):
            obj_entry = sympy.simplify(obj_hess[i, j])
            con_entries = []
            for k, ch in enumerate(con_hess_list):
                con_entries.append(sympy.simplify(ch[i, j]))

            # Check if any entry is nonzero
            has_nonzero = (obj_entry != 0) or any(ce != 0 for ce in con_entries)
            if has_nonzero:
                hess_entries.append((i, j, obj_entry, con_entries))

    # If no entries, add a diagonal with zeros (solver needs at least something)
    if not hess_entries:
        for i in range(n):
            hess_entries.append((i, i, sympy.Integer(0), [sympy.Integer(0)] * m))

    return True, hess_entries, obj_sym, con_syms


# ---------------------------------------------------------------------------
# Code generation helpers
# ---------------------------------------------------------------------------

def sympy_to_rust(expr, xs):
    """Convert sympy expression to Rust code string."""
    from sympy.printing import rust_code as _rust_code
    # Expand the expression to avoid sympy rust_code bug where Float coefficients
    # in factored products (e.g. 400.0*x0*(x0**2 - x1)) lose parentheses,
    # producing incorrect code like 400.0*x0*x0.powi(2) - x1 instead of
    # 400.0*x0*(x0.powi(2) - x1). The expanded form prints correctly.
    expr = sympy.expand(expr)
    try:
        code = _rust_code(expr, strict=False)
    except TypeError:
        code = _rust_code(expr)
    # Fix any remaining issues - go in reverse to avoid x1 matching x10
    for i in range(len(xs)-1, -1, -1):
        code = code.replace(f'x{i}', f'x[{i}]')
    # Fix bare integers in arithmetic contexts — convert to f64
    # But don't touch: array indices x[0], grad[0], vals[0], lambda[0]
    # and don't touch powi arguments
    # Strategy: temporarily replace array indices, fix integers, then restore
    def _int_to_float(code_str):
        # Protect array indices: x[N], grad[N], vals[N], lambda[N], g[N], g_l[N], etc.
        protected = {}
        counter = [0]
        def protect(m):
            key = f"__PROT{counter[0]}__"
            counter[0] += 1
            protected[key] = m.group(0)
            return key
        result = re.sub(r'\[\d+\]', protect, code_str)
        # Protect .powi(N) including negative arguments
        result = re.sub(r'\.powi\(-?\d+\)', protect, result)
        # Protect scientific notation like 1.0e-5, 2.5e+10
        result = re.sub(r'\d+\.\d*[eE][+-]?\d+', protect, result)
        result = re.sub(r'\d+[eE][+-]?\d+', protect, result)
        # Fix Rust integer suffixes like 10_i32 -> 10.0_f64
        result = re.sub(r'(\d+)_i32\b', r'\1.0_f64', result)
        # Now convert bare integers to floats
        result = re.sub(r'(?<![.\d\w])(\d+)(?![\.\d\w])', lambda m: m.group(1) + '.0', result)
        # Fix double .0 on numbers that were already float
        result = result.replace('.0.0', '.0')
        # Restore protected tokens
        for key, val in protected.items():
            result = result.replace(key, val)
        return result
    code = _int_to_float(code)
    return code


def sympy_to_python(expr, xs):
    """Convert sympy expression to Python code string."""
    expr = sympy.expand(expr)
    code = str(expr)
    for i in range(len(xs)-1, -1, -1):  # reverse to avoid x1 matching x10
        code = code.replace(f'x{i}', f'x[{i}]')
    # Replace sympy functions with numpy
    code = code.replace('sqrt', 'np.sqrt')
    code = code.replace('log', 'np.log')
    code = code.replace('exp', 'np.exp')
    code = code.replace('sin', 'np.sin')
    code = code.replace('cos', 'np.cos')
    code = code.replace('tan', 'np.tan')
    code = code.replace('Abs', 'np.abs')
    code = code.replace('sign', 'np.sign')
    # Fix double replacements like np.np.
    code = code.replace('np.np.', 'np.')
    code = code.replace('anp.', 'a')  # fix atan -> anp.tan
    code = code.replace('anp.', 'a')
    return code


def fortran_expr_to_rust(expr_str, n):
    """Convert Fortran expression directly to Rust (without sympy)."""
    sym, xs = try_sympify_expr(expr_str, n, 0)
    if sym is not None:
        return sympy_to_rust(sym, xs)
    return None


def fortran_expr_to_python(expr_str, n):
    """Convert Fortran expression directly to Python (without sympy)."""
    sym, xs = try_sympify_expr(expr_str, n, 0)
    if sym is not None:
        return sympy_to_python(sym, xs)
    return None


def get_bounds_rust(prob):
    """Generate Rust code for variable bounds."""
    lines = []
    for i in range(1, prob.n + 1):
        has_lb = prob.lxl.get(i, False)
        has_ub = prob.lxu.get(i, False)
        lb = prob.xl.get(i, 0.0) if has_lb else None
        ub = prob.xu.get(i, 0.0) if has_ub else None

        if lb is not None:
            lines.append(f"        x_l[{i-1}] = {format_rust_float(lb)};")
        else:
            lines.append(f"        x_l[{i-1}] = f64::NEG_INFINITY;")

        if ub is not None:
            lines.append(f"        x_u[{i-1}] = {format_rust_float(ub)};")
        else:
            lines.append(f"        x_u[{i-1}] = f64::INFINITY;")

    return '\n'.join(lines)


def get_constraint_bounds_rust(prob):
    """Generate Rust code for constraint bounds.

    Constraint ordering in Fortran:
    - First NILI linear inequality constraints: G(i) >= 0
    - Then NINL nonlinear inequality constraints: G(i) >= 0
    - Then NELI linear equality constraints: G(i) = 0
    - Then NENL nonlinear equality constraints: G(i) = 0
    """
    lines = []
    idx = 0
    # Inequality constraints (both linear and nonlinear): G(i) >= 0
    for i in range(prob.nili + prob.ninl):
        lines.append(f"        g_l[{idx}] = 0.0;")
        lines.append(f"        g_u[{idx}] = f64::INFINITY;")
        idx += 1
    # Equality constraints: G(i) = 0
    for i in range(prob.neli + prob.nenl):
        lines.append(f"        g_l[{idx}] = 0.0;")
        lines.append(f"        g_u[{idx}] = 0.0;")
        idx += 1

    return '\n'.join(lines)


def format_rust_float(v):
    """Format a float for Rust code."""
    if v == float('inf'):
        return "f64::INFINITY"
    if v == float('-inf'):
        return "f64::NEG_INFINITY"
    if v == 0.0:
        return "0.0"
    if v == int(v) and abs(v) < 1e15:
        return f"{int(v)}.0"
    s = f"{v}"
    if '.' not in s and 'e' not in s and 'E' not in s:
        s += '.0'
    # Ensure _f64 suffix isn't needed (Rust infers)
    return s


def format_python_float(v):
    """Format a float for Python code."""
    if v == float('inf'):
        return "np.inf"
    if v == float('-inf'):
        return "-np.inf"
    return repr(v)


# ---------------------------------------------------------------------------
# Rust code generation
# ---------------------------------------------------------------------------

def generate_rust_problem(prob, hess_entries, obj_sym, con_syms):
    """Generate a Rust struct implementing NlpProblem for one HS problem."""
    n = prob.n
    m = prob.m
    xs = symbols([f'x{i}' for i in range(n)])

    struct_name = f"HsTp{prob.number:03d}"

    # --- Objective ---
    obj_rust = sympy_to_rust(obj_sym, xs)

    # --- Gradient ---
    grad_lines = []
    for i in range(n):
        deriv = sympy.diff(obj_sym, xs[i])
        deriv = sympy.simplify(deriv)
        grad_lines.append(f"        grad[{i}] = {sympy_to_rust(deriv, xs)};")

    # --- Constraints ---
    con_lines = []
    for i, csym in enumerate(con_syms):
        con_lines.append(f"        g[{i}] = {sympy_to_rust(csym, xs)};")

    # --- Jacobian structure and values ---
    jac_rows = []
    jac_cols = []
    jac_val_lines = []
    jac_idx = 0
    for i, csym in enumerate(con_syms):
        for j in range(n):
            deriv = sympy.diff(csym, xs[j])
            deriv = sympy.simplify(deriv)
            if deriv != 0:
                jac_rows.append(str(i))
                jac_cols.append(str(j))
                jac_val_lines.append(f"        vals[{jac_idx}] = {sympy_to_rust(deriv, xs)};")
                jac_idx += 1

    # --- Hessian structure and values ---
    hess_rows = []
    hess_cols = []
    hess_val_lines = []
    for idx, (row, col, obj_entry, con_entries) in enumerate(hess_entries):
        hess_rows.append(str(row))
        hess_cols.append(str(col))

        terms = []
        if obj_entry != 0:
            terms.append(f"obj_factor * ({sympy_to_rust(obj_entry, xs)})")
        for k, ce in enumerate(con_entries):
            if ce != 0:
                terms.append(f"lambda[{k}] * ({sympy_to_rust(ce, xs)})")

        if terms:
            hess_val_lines.append(f"        vals[{idx}] = {' + '.join(terms)};")
        else:
            hess_val_lines.append(f"        vals[{idx}] = 0.0;")

    # --- Initial point ---
    x0_lines = []
    for i in range(n):
        val = prob.x0[i] if i < len(prob.x0) else 0.0
        x0_lines.append(f"        x0[{i}] = {format_rust_float(val)};")

    # --- Build the impl ---
    bounds_code = get_bounds_rust(prob)
    con_bounds_code = get_constraint_bounds_rust(prob)

    code = f"""
pub struct {struct_name};

impl NlpProblem for {struct_name} {{
    fn num_variables(&self) -> usize {{
        {n}
    }}

    fn num_constraints(&self) -> usize {{
        {m}
    }}

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {{
{bounds_code}
    }}

    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {{
{con_bounds_code}
    }}

    fn initial_point(&self, x0: &mut [f64]) {{
{chr(10).join(x0_lines)}
    }}

    fn objective(&self, x: &[f64], _new_x: bool, obj: &mut f64) -> bool {{
        *obj = {obj_rust};
        true
    }}

    fn gradient(&self, x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool {{
{chr(10).join(grad_lines)}
        true
    }}

    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) -> bool {{
{chr(10).join(con_lines) if con_lines else '        let _ = (x, g);'}
        true
    }}

    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {{
        (vec![{', '.join(jac_rows)}], vec![{', '.join(jac_cols)}])
    }}

    fn jacobian_values(&self, x: &[f64], _new_x: bool, vals: &mut [f64]) -> bool {{
{chr(10).join(jac_val_lines) if jac_val_lines else '        let _ = (x, vals);'}
        true
    }}

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {{
        (vec![{', '.join(hess_rows)}], vec![{', '.join(hess_cols)}])
    }}

    fn hessian_values(&self, x: &[f64], _new_x: bool, obj_factor: f64, lambda: &[f64], vals: &mut [f64]) -> bool {{
{chr(10).join(hess_val_lines) if hess_val_lines else '        let _ = (x, obj_factor, lambda, vals);'}
        true
    }}
}}
"""
    return struct_name, code


# ---------------------------------------------------------------------------
# Python code generation
# ---------------------------------------------------------------------------

def generate_python_problem(prob, hess_entries, obj_sym, con_syms):
    """Generate Python cyipopt problem code for one HS problem."""
    n = prob.n
    m = prob.m
    xs = symbols([f'x{i}' for i in range(n)])

    func_prefix = f"_tp{prob.number:03d}"

    # --- Objective ---
    obj_py = sympy_to_python(obj_sym, xs)

    # --- Gradient ---
    grad_entries = []
    for i in range(n):
        deriv = sympy.diff(obj_sym, xs[i])
        deriv = sympy.simplify(deriv)
        grad_entries.append(sympy_to_python(deriv, xs))

    # --- Constraints ---
    con_entries = []
    for csym in con_syms:
        con_entries.append(sympy_to_python(csym, xs))

    # --- Jacobian ---
    jac_rows = []
    jac_cols = []
    jac_val_entries = []
    for i, csym in enumerate(con_syms):
        for j in range(n):
            deriv = sympy.diff(csym, xs[j])
            deriv = sympy.simplify(deriv)
            if deriv != 0:
                jac_rows.append(str(i))
                jac_cols.append(str(j))
                jac_val_entries.append(sympy_to_python(deriv, xs))

    # --- Hessian ---
    hess_rows = []
    hess_cols = []
    hess_val_entries = []
    for idx, (row, col, obj_entry, con_entries_h) in enumerate(hess_entries):
        hess_rows.append(str(row))
        hess_cols.append(str(col))

        terms = []
        if obj_entry != 0:
            terms.append(f"obj_factor * ({sympy_to_python(obj_entry, xs)})")
        for k, ce in enumerate(con_entries_h):
            if ce != 0:
                terms.append(f"lagrange[{k}] * ({sympy_to_python(ce, xs)})")

        if terms:
            hess_val_entries.append(' + '.join(terms))
        else:
            hess_val_entries.append('0.0')

    # --- Bounds ---
    lb_entries = []
    ub_entries = []
    for i in range(1, n+1):
        has_lb = prob.lxl.get(i, False)
        has_ub = prob.lxu.get(i, False)
        lb_entries.append(format_python_float(prob.xl.get(i, 0.0) if has_lb else -1e19))
        ub_entries.append(format_python_float(prob.xu.get(i, 0.0) if has_ub else 1e19))

    # --- Constraint bounds ---
    cl_entries = []
    cu_entries = []
    for i in range(prob.nili + prob.ninl):
        cl_entries.append('0.0')
        cu_entries.append('2.0e19')
    for i in range(prob.neli + prob.nenl):
        cl_entries.append('0.0')
        cu_entries.append('0.0')

    # --- x0 ---
    x0_entries = []
    for i in range(n):
        val = prob.x0[i] if i < len(prob.x0) else 0.0
        x0_entries.append(repr(val))

    # Build code
    grad_str = ', '.join(grad_entries) if grad_entries else ''
    con_str = ', '.join(con_entries) if con_entries else ''
    jac_str = ', '.join(jac_val_entries) if jac_val_entries else ''

    hess_lines = []
    for i, hve in enumerate(hess_val_entries):
        hess_lines.append(f"    h[{i}] = {hve}")

    code = f"""
def {func_prefix}_objective(x):
    return {obj_py}

def {func_prefix}_gradient(x):
    return np.array([{grad_str}])

def {func_prefix}_constraints(x):
    return np.array([{con_str}])

def {func_prefix}_jacobian(x):
    return np.array([{jac_str}])

def {func_prefix}_jacobianstructure():
    return (np.array([{', '.join(jac_rows)}], dtype=int), np.array([{', '.join(jac_cols)}], dtype=int))

def {func_prefix}_hessian(x, lagrange, obj_factor):
    h = np.zeros({len(hess_entries)})
{chr(10).join(hess_lines)}
    return h

def {func_prefix}_hessianstructure():
    return (np.array([{', '.join(hess_rows)}], dtype=int), np.array([{', '.join(hess_cols)}], dtype=int))

def tp{prob.number:03d}_factory(intermediate_cb=None):
    prob = cyipopt.Problem(
        n={n}, m={m},
        problem_obj=_make_problem_obj(
            {func_prefix}_objective, {func_prefix}_gradient, {func_prefix}_constraints,
            {func_prefix}_jacobian, {func_prefix}_jacobianstructure,
            {func_prefix}_hessian, {func_prefix}_hessianstructure,
            intermediate_cb,
        ),
        lb=np.array([{', '.join(lb_entries)}]),
        ub=np.array([{', '.join(ub_entries)}]),
        cl=np.array([{', '.join(cl_entries)}]),
        cu=np.array([{', '.join(cu_entries)}]),
    )
    prob.add_option("mu_strategy", "adaptive")
    prob.add_option("tol", 1e-8)
    return prob, np.array([{', '.join(x0_entries)}])
"""
    return f"tp{prob.number:03d}_factory", code


# ---------------------------------------------------------------------------
# Main generation pipeline
# ---------------------------------------------------------------------------

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fortran_path = os.path.join(script_dir, 'PROB.FOR')
    gen_dir = os.path.join(script_dir, 'generated')
    os.makedirs(gen_dir, exist_ok=True)

    print("Reading PROB.FOR...")
    lines = read_fortran_file(fortran_path)

    print("Splitting subroutines...")
    subs = split_subroutines(lines)
    print(f"  Found {len(subs)} subroutines")

    # Focus on HS problems 1-119 (the classic set)
    # Also include 201+ if they're simple enough
    target_numbers = sorted(subs.keys())

    # Parse all problems
    problems = {}
    for num in target_numbers:
        prob = parse_problem(num, subs[num])
        if prob.n > 0 and prob.obj_expr:
            problems[num] = prob

    print(f"  Parsed {len(problems)} problems with valid objective expressions")

    # Generate code for each problem
    rust_structs = []
    rust_codes = []
    python_factories = []
    python_codes = []
    registry_entries = []
    skipped = []

    for num in sorted(problems.keys()):
        prob = problems[num]

        # Skip problems with too many variables (our solver is dense)
        if prob.n > 20:
            skipped.append((num, f"n={prob.n} too large"))
            continue

        # Try to compute Hessian via sympy
        success, hess_entries, obj_sym, con_syms = compute_hessian_data(prob)

        if not success:
            skipped.append((num, "sympy parse/hessian failed"))
            continue

        # Generate Rust code
        try:
            struct_name, rust_code = generate_rust_problem(prob, hess_entries, obj_sym, con_syms)
            rust_structs.append(struct_name)
            rust_codes.append(rust_code)
        except Exception as e:
            skipped.append((num, f"Rust codegen failed: {e}"))
            continue

        # Generate Python code
        try:
            factory_name, python_code = generate_python_problem(prob, hess_entries, obj_sym, con_syms)
            python_factories.append(factory_name)
            python_codes.append(python_code)
        except Exception as e:
            skipped.append((num, f"Python codegen failed: {e}"))
            # Remove the Rust code we just added
            rust_structs.pop()
            rust_codes.pop()
            continue

        # Registry entry
        fex_str = format_rust_float(prob.fex) if prob.fex is not None else "f64::NAN"
        registry_entries.append(
            f'    HsProblemEntry {{ number: {num}, name: "TP{num:03d}", '
            f'n: {prob.n}, m: {prob.m}, '
            f'nili: {prob.nili}, ninl: {prob.ninl}, neli: {prob.neli}, nenl: {prob.nenl}, '
            f'known_fopt: {fex_str} }},'
        )

        print(f"  TP{num:03d}: n={prob.n}, m={prob.m} -> OK")

    print(f"\nGenerated {len(rust_structs)} problems")
    print(f"Skipped {len(skipped)} problems:")
    for num, reason in skipped:
        print(f"  TP{num}: {reason}")

    # --- Write Rust file ---
    rust_file = os.path.join(gen_dir, 'hs_problems.rs')
    with open(rust_file, 'w') as f:
        f.write("""\
// Auto-generated by hs_suite/generate.py — DO NOT EDIT
// Hock-Schittkowski test problems for ripopt validation

#![allow(unused_variables)]
#![allow(clippy::excessive_precision)]
#![allow(clippy::needless_return)]

use ripopt::NlpProblem;

""")
        for code in rust_codes:
            f.write(code)
            f.write('\n')

        # Write registry
        f.write("""
// ---------------------------------------------------------------------------
// Problem registry
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct HsProblemEntry {
    pub number: usize,
    pub name: &'static str,
    pub n: usize,
    pub m: usize,
    pub nili: usize,
    pub ninl: usize,
    pub neli: usize,
    pub nenl: usize,
    pub known_fopt: f64,
}

pub static HS_PROBLEMS: &[HsProblemEntry] = &[
""")
        for entry in registry_entries:
            f.write(entry + '\n')
        f.write('];\n\n')

        # Write solve_all function
        f.write("""
/// Solve all HS problems and return results as JSON-compatible data.
pub fn solve_all(options: &ripopt::SolverOptions) -> Vec<HsSolveResult> {
    let mut results = Vec::new();
""")
        for i, struct_name in enumerate(rust_structs):
            num = int(struct_name[4:])  # HsTp001 -> 1
            f.write(f"""
    // TP{num:03d}
    {{
        let problem = {struct_name};
        let t0 = std::time::Instant::now();
        let result = ripopt::solve(&problem, options);
        let elapsed = t0.elapsed().as_secs_f64();
        results.push(HsSolveResult {{
            number: {num},
            status: format!("{{:?}}", result.status),
            objective: result.objective,
            x: result.x.clone(),
            constraint_multipliers: result.constraint_multipliers.clone(),
            bound_multipliers_lower: result.bound_multipliers_lower.clone(),
            bound_multipliers_upper: result.bound_multipliers_upper.clone(),
            constraint_values: result.constraint_values.clone(),
            iterations: result.iterations,
            solve_time: elapsed,
            known_fopt: HS_PROBLEMS[{i}].known_fopt,
            n: HS_PROBLEMS[{i}].n,
            m: HS_PROBLEMS[{i}].m,
            final_primal_inf: result.diagnostics.final_primal_inf,
            final_dual_inf: result.diagnostics.final_dual_inf,
            final_dual_inf_scaled: result.diagnostics.final_dual_inf_scaled,
            final_compl: result.diagnostics.final_compl,
            final_mu: result.diagnostics.final_mu,
            final_s_d: result.diagnostics.final_s_d,
        }});
    }}
""")

        f.write("""    results
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct HsSolveResult {
    pub number: usize,
    pub status: String,
    pub objective: f64,
    pub x: Vec<f64>,
    pub constraint_multipliers: Vec<f64>,
    pub bound_multipliers_lower: Vec<f64>,
    pub bound_multipliers_upper: Vec<f64>,
    pub constraint_values: Vec<f64>,
    pub iterations: usize,
    pub solve_time: f64,
    pub known_fopt: f64,
    pub n: usize,
    pub m: usize,
    pub final_primal_inf: f64,
    pub final_dual_inf: f64,
    pub final_dual_inf_scaled: f64,
    pub final_compl: f64,
    pub final_mu: f64,
    pub final_s_d: f64,
}
""")

    print(f"\nWrote {rust_file}")

    # --- Write Python file ---
    python_file = os.path.join(gen_dir, 'hs_cyipopt.py')
    with open(python_file, 'w') as f:
        f.write("""\
#!/usr/bin/env python3
# Auto-generated by hs_suite/generate.py — DO NOT EDIT
# Hock-Schittkowski test problems for cyipopt validation

import numpy as np
import cyipopt


def _make_problem_obj(objective, gradient, constraints, jacobian,
                      jacobianstructure, hessian, hessianstructure,
                      intermediate_cb=None):
    class ProblemDef:
        pass
    p = ProblemDef()
    p.objective = objective
    p.gradient = gradient
    p.constraints = constraints
    p.jacobian = jacobian
    p.jacobianstructure = jacobianstructure
    p.hessian = hessian
    p.hessianstructure = hessianstructure
    if intermediate_cb is not None:
        p.intermediate = intermediate_cb
    else:
        p.intermediate = lambda *a, **kw: True
    return p

""")
        for code in python_codes:
            f.write(code)
            f.write('\n')

        # Write registry
        f.write("""
# ---------------------------------------------------------------------------
# Problem registry
# ---------------------------------------------------------------------------

HS_PROBLEMS = [
""")
        for i, factory_name in enumerate(python_factories):
            num = int(factory_name[2:5])
            prob = problems[num]
            fex = repr(prob.fex) if prob.fex is not None else "float('nan')"
            f.write(f"    {{'number': {num}, 'factory': {factory_name}, "
                    f"'n': {prob.n}, 'm': {prob.m}, 'known_fopt': {fex}}},\n")
        f.write(']\n')

    print(f"Wrote {python_file}")

    # --- Write summary ---
    summary_file = os.path.join(gen_dir, 'generation_summary.txt')
    with open(summary_file, 'w') as f:
        f.write(f"Generated {len(rust_structs)} HS test problems\n\n")
        f.write("Included problems:\n")
        for sn in rust_structs:
            num = int(sn[4:])
            prob = problems[num]
            f.write(f"  TP{num:03d}: n={prob.n}, m={prob.m}, fex={prob.fex}\n")
        f.write(f"\nSkipped {len(skipped)} problems:\n")
        for num, reason in skipped:
            f.write(f"  TP{num}: {reason}\n")

    print(f"Wrote {summary_file}")
    print(f"\nDone! Generated {len(rust_structs)} problems.")


if __name__ == '__main__':
    main()
