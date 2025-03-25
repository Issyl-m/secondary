# Copyright (c) 2025 Andrés Morán (andres.moran.l@uc.cl)
# Licensed under the terms of the MIT License (see ./LICENSE).

from functools import cache
import numpy as np

# from pympler import asizeof # tests
import sys  # tests

# import time # tests
# from sage.all import * # tests

# Constants

TAG_MILNOR_BASIS = 0
TAG_SERRE_CARTAN_BASIS = 1
TAG_MONOMIAL_CONSTANT = 0
TAG_MONOMIAL_BOCKSTEIN = 1
TAG_MONOMIAL_POWER = 2
TAG_MONOMIAL_POWER_MILNOR_BASIS = -1
MULTIPLE_OF_FIXED_P = 2

RET_ERR = -1
RET_ZERO_MOD_P = 1

# Parameters

PARAM_FIXED_PRIME = 3

#####################################################################
#           Secondary Steenrod Algebra classes                      #
#####################################################################


class TensorProduct:
    """Pure tensor of the form (a1 + ... + a_n) ⊗ (b1 + ... + b_m)"""

    def __init__(self, linear_comb_1, linear_comb_2):
        self.lc1 = linear_comb_1
        self.lc2 = linear_comb_2

    def expand(self):
        list_linear_comb = []

        for m1 in self.lc1:
            for m2 in self.lc2:
                list_linear_comb.append(TensorProductBasic(m1, m2))

        return LinearCombination(list_linear_comb)

    def __add__(self, other):
        return self.expand() + other.expand()

    def __rmul__(self, other):
        r = self.copy()
        r.lc1 *= other
        return r

    def __mul__(self, other):
        return self.expand() * other.expand()

    def __str__(self):
        if self.lc1.isZero() or self.lc2.isZero():
            return "0"
        return f"({self.lc1}) ⊗ ({self.lc2})"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.expand().isEqual(other.expand())

    def asLinearCombination(self):
        return LinearCombination([self])

    def isZero(self):
        return self.expand().isEqual(LinearCombination([]))

    def copy(self):
        return TensorProduct(self.lc1.copy(), self.lc2.copy())

    def sameGenerator(self, other):
        return self.lc1.isEqual(other.lc1) and self.lc2.isEqual(other.lc2)


class TensorProductBasic:
    """Pure tensor a ⊗ b"""

    def __init__(self, monomial_1, monomial_2):
        self.m1 = monomial_1
        self.m2 = monomial_2

    def __radd__(self, other):
        return self

    def __add__(self, other):
        if (
            self.m1.monomial == other.m1.monomial
            and self.m2.monomial == other.m2.monomial
        ):

            c = self.m1.c * self.m2.c + other.m1.c * other.m2.c
            m1 = (self.m1 + other.m1).monomials[0]
            m2 = (self.m2 + other.m2).monomials[0]

            m1.c = c
            m2.c = 1

            return LinearCombination([TensorProductBasic(m1, m2)])
        else:
            return LinearCombination([self, other])

    def __rmul__(self, other):
        r = self.copy()
        r.m1.c *= other
        return r

    def __mul__(self, other):
        return TensorProductBasic(
            (-1) ** (deg(self.m2) * deg(other.m1)) * self.m1 * other.m1,
            self.m2 * other.m2,
        )

    def __str__(self):
        if self.m1.isZero() or self.m2.isZero():
            return "0"
        return f"{str(self.m1)} ⊗ {str(self.m2)}"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        c1 = self.m1.c * self.m2.c
        c2 = other.m1.c * other.m2.c

        return (
            (c1.coeffCmp(c1, c2))
            and self.m1.monomial == other.m1.monomial
            and self.m2.monomial == other.m2.monomial
        )

    def asLinearCombination(self):
        return LinearCombination([self])

    def isZero(self):
        return self.m1.coeffCmp(self.m1.c * self.m2.c, 0)

    def copy(self):
        return TensorProductBasic(self.m1.copy(), self.m2.copy())

    def sameGenerator(self, other):
        return (
            self.m1.monomial == other.m1.monomial
            and self.m2.monomial == other.m2.monomial
        )


class LinearCombination:
    """Generic linear combination"""

    def __init__(self, list_monomials):
        self.monomials = list_monomials

    def __radd__(self, other):
        """Compatibility with python preexisting routines"""
        return self

    def __add__(self, other):
        """Sum between two linear combinations"""

        if other.isZero():
            return self

        list_monomials = []

        for s_monomial in self.monomials:
            bool_monomial_found = False

            for o_monomial in other.monomials:
                if s_monomial.sameGenerator(o_monomial):
                    m_sum = s_monomial + o_monomial
                    if not m_sum.isZero():
                        list_monomials += m_sum.monomials

                    bool_monomial_found = True
                    break

            if not bool_monomial_found:
                list_monomials.append(s_monomial)

        for o_monomial in other.monomials:
            bool_monomial_found = False

            for s_monomial in self.monomials:
                if s_monomial.sameGenerator(o_monomial):
                    bool_monomial_found = True
                    break

            if not bool_monomial_found:
                list_monomials.append(o_monomial)

        return LinearCombination(list_monomials)

    def __rmul__(self, other):
        if other == 0:  # TODO: - % p^2 (not important)
            return LinearCombination([])
        else:
            return LinearCombination(
                [
                    other * monomial
                    for monomial in self.monomials
                    if not (other * monomial).isZero()
                ]
            )

    def __mul__(self, other):
        """Multiplication between two linear combinations"""

        list_linear_comb = [
            (s_monomial * o_monomial).asLinearCombination()
            for s_monomial in self.monomials
            for o_monomial in other.monomials
        ]

        list_linear_comb = sum(list_linear_comb)

        if list_linear_comb == 0:
            return LinearCombination([])

        return LinearCombination(
            [
                monomial
                for monomial in list_linear_comb.monomials
                if not monomial.isZero()
            ]
        )

    def __iter__(self):
        for monomial in self.monomials:
            yield monomial

    def __str__(self):
        str_output = ""

        if len(self.monomials) == 0:
            return "0"

        bool_singleton = True
        for monomial in self.monomials:
            if bool_singleton:
                str_output += f"{monomial}"

                bool_singleton = False
            else:
                str_output += f" + {monomial}"

        return str_output

    def asLinearCombination(self):
        return self

    def __repr__(self):
        return self.__str__()

    def copy(self):
        return LinearCombination([m.copy() for m in self.monomials])

    def isEqual(self, other):
        return (self + (-1) * other).isZero()

    def isZero(self):
        if len(self.monomials) == 0:
            return True

        bool_is_zero = True

        for monomial in self.monomials:
            if not monomial.isZero():
                bool_is_zero = False
                break

        return bool_is_zero


class Monomial:
    """Generic Steenrod Algebra/B_0 monomial"""

    def __init__(self, coeff, tuple_pow_operations, prime_power=PARAM_FIXED_PRIME**2):
        self.c = coeff
        self.monomial = tuple_pow_operations
        self.p_pow = prime_power
        self.basis = TAG_SERRE_CARTAN_BASIS

    def __add__(self, other):
        if self.monomial == other.monomial:
            return LinearCombination(
                [Monomial((self.c + other.c) % self.p_pow, self.monomial, self.p_pow)]
            )
        else:
            return LinearCombination([self, other])

    def __rmul__(self, other):
        return Monomial((other * self.c) % self.p_pow, self.monomial, self.p_pow)

    def __mul__(self, other):
        return Monomial(
            (self.c * other.c) % self.p_pow, self.monomial + other.monomial, self.p_pow
        )

    def __hash__(self):
        return hash(str(self))

    def __str__(self):
        if self.isZero():
            return "0"
        if len(self.monomial) == 0:
            return str(self.c % self.p_pow)
        str_operations = ""
        for i in range(0, len(self.monomial) >> 1):
            operation_tag = self.monomial[2 * i]
            operation_exp = self.monomial[2 * i + 1]

            if operation_tag == TAG_MONOMIAL_BOCKSTEIN:
                operation_str = "b"
            else:
                operation_str = "P"

            str_operations += f"{operation_str}^{
                operation_exp} "

        if int(self.c) % self.p_pow == 1:
            return f"{str_operations[:-1]}"
        return f"{self.c % self.p_pow}{str_operations[:-1]}"

    def __eq__(self, other):
        if self.isZero() and other.isZero():
            return True

        return (self.coeffCmp(self.c, other.c)) and (self.monomial == other.monomial)

    def asLinearCombination(self):
        return LinearCombination([self])

    def isZero(self):
        return self.c % self.p_pow == 0

    def copy(self):
        return Monomial(self.c, self.monomial, self.p_pow)

    def sameGenerator(self, other):
        return self.monomial == other.monomial

    @staticmethod
    def coeffCmp(c1, c2, p_pow=PARAM_FIXED_PRIME**2):
        return (c1 - c2) % p_pow == 0

    @staticmethod
    def str2monomial(coeff, str_operations, p_pow=PARAM_FIXED_PRIME**2):
        """INPUT: (c, 'b1 p40 p2 b1')"""
        list_operations = []

        for c in str_operations.split(" "):
            if c[0] == "b":
                list_operations.append(TAG_MONOMIAL_BOCKSTEIN)
            elif c[0] == "p":
                list_operations.append(TAG_MONOMIAL_POWER)

            list_operations.append(int(c[1:]))

        return Monomial(coeff, tuple(list_operations), p_pow)

    @staticmethod
    def str2lc(coeff, str_operations, p_pow=PARAM_FIXED_PRIME**2):
        """INPUT: (c, 'b1 p40 p2 b1')"""
        """OUTPUT: element as LinearCombination object"""

        return Monomial.str2monomial(coeff, str_operations, p_pow).asLinearCombination()


class MonomialMilnorBasis:
    """Generic Steenrod Algebra monomial"""

    def __init__(self, coeff, tuple_pow_operations, prime_power=PARAM_FIXED_PRIME):
        self.c = coeff
        self.monomial = (
            tuple_pow_operations[0],
            np.trim_zeros(tuple_pow_operations[1], "b"),
        )
        self.p_pow = prime_power
        self.basis = TAG_MILNOR_BASIS

    def __add__(self, other):
        if self.monomial == other.monomial:
            return LinearCombination(
                [
                    MonomialMilnorBasis(
                        (self.c + other.c) % self.p_pow, self.monomial, self.p_pow
                    )
                ]
            )
        else:
            return LinearCombination([self, other])

    def __rmul__(self, other):
        return MonomialMilnorBasis(
            (other * self.c) % self.p_pow, self.monomial, self.p_pow
        )

    def __mul__(self, other):
        return milnor_basis_product(self, other)

    def __hash__(self):
        return hash(str(self))

    def __str__(self):
        if self.isZero():
            return "0"
        if milnor_basis_deg(self) == 0:
            return str(self.c % self.p_pow)

        str_operations = ""
        if len(self.monomial[0]) > 0:
            str_operations = f"Q{str(self.monomial[0])}".replace(",)", ")").replace(
                ", ", ","
            )

        if len(self.monomial[1]) > 0:
            str_operations = f"{str_operations}P{str(self.monomial[1])}".replace(
                ",)", ")"
            ).replace(", ", ",")

        if self.c % self.p_pow == 1:
            return f"{str_operations}"
        return f"{self.c % self.p_pow}{str_operations}"

    def __eq__(self, other):
        if self.isZero() and other.isZero():
            return True

        return (self.coeffCmp(self.c, other.c)) and (self.monomial == other.monomial)

    def asLinearCombination(self):
        return LinearCombination([self])

    def isZero(self):
        if self.c % self.p_pow == 0:
            return True
        else:
            if len(tuple(set(self.monomial[0]))) < len(self.monomial[0]):
                return True

        return False

    def copy(self):
        return MonomialMilnorBasis(self.c, self.monomial, self.p_pow)

    def sameGenerator(self, other):
        return self.monomial == other.monomial

    @staticmethod
    def coeffCmp(c1, c2, p_pow=PARAM_FIXED_PRIME):
        return (c1 - c2) % p_pow == 0


#####################################################################
#                           Maps                                    #
#####################################################################


def steenrod_milnor_basis_decompose_positive_deg_partitions(
    m, R_1, prefix=[], depth=-1
):
    r = []

    if depth >= 0:
        for i in range(2):
            r += steenrod_milnor_basis_decompose_positive_deg_partitions(
                m, R_1, prefix=[i == 1] + prefix, depth=depth - 1
            )
    else:
        S = np.array(m.monomial[0])
        len_S = len(S)

        if len(prefix) > 0:
            mask_1 = np.array(prefix)
            mask_2 = np.array([not flag for flag in prefix])

            S_1 = tuple(S[mask_1].tolist())
            S_2 = tuple(S[mask_2].tolist())

            S = S.tolist()
        else:
            S_1 = ()
            S_2 = ()

        len_S_1 = len(S_1)

        R_2 = tuple([m.monomial[1][i] - R_1[i] for i in range(len(R_1))])

        sgn = 1  # (-1) ** (len(np.trim_zeros(S_1, "b")) * len(np.trim_zeros(S_2, "b"))) # WARNING: sign convention
        list_list_permutation_matrix = [[0] * len_S for i in range(len_S)]

        for i in range(len_S):
            if i < len_S_1:
                j = S.index(S_1[i])
            else:
                j = S.index(S_2[i - len_S_1])

            list_list_permutation_matrix[i][j] = 1

        for i in range(len_S):
            if list_list_permutation_matrix[i][i] == 0:
                if i < len_S_1:
                    j = S.index(S_1[i])
                else:
                    j = S.index(S_2[i - len_S_1])

                for k in range(len_S):
                    tmp_entry = list_list_permutation_matrix[i][k]
                    list_list_permutation_matrix[i][k] = list_list_permutation_matrix[
                        j
                    ][k]
                    list_list_permutation_matrix[j][k] = tmp_entry

                sgn *= -1

        r.append(
            TensorProductBasic(
                MonomialMilnorBasis(sgn * m.c, (S_1, tuple(R_1)), PARAM_FIXED_PRIME),
                MonomialMilnorBasis(1, (S_2, R_2), PARAM_FIXED_PRIME),
            )
        )

    return r


def steenrod_milnor_basis_decompose_positive_deg(m, prefix=[], depth=-1):
    r = []

    if depth > 0:
        for i in range(m.monomial[1][depth - 1] + 1):
            r += steenrod_milnor_basis_decompose_positive_deg(
                m, prefix=[i] + prefix, depth=depth - 1
            )
    else:
        r += steenrod_milnor_basis_decompose_positive_deg_partitions(
            m, prefix, depth=len(m.monomial[0]) - 1
        )

    return r


@cache
def steenrod_milnor_basis_decompose(m):
    r = []

    deg_m = milnor_basis_deg(m)
    if deg_m == 0:
        return [
            TensorProductBasic(
                MonomialMilnorBasis(m.c, ((), ()), PARAM_FIXED_PRIME),
                MonomialMilnorBasis(1, ((), ()), PARAM_FIXED_PRIME),
            )
        ]
    elif deg_m == -1:
        return []
    else:
        r = steenrod_milnor_basis_decompose_positive_deg(m, depth=len(m.monomial[0]))

    return r


@cache
def steenrod_decompose(operation, exp, p_pow=PARAM_FIXED_PRIME**2):  # TODO: p = 2
    r = []

    for i in range(0, exp + 1):
        if i == 0:
            m1_operation = tuple([])
        else:
            m1_operation = (operation, i)
        if i == exp:
            m2_operation = tuple([])
        else:
            m2_operation = (operation, exp - i)

        r.append(
            TensorProductBasic(
                Monomial(1, m1_operation, p_pow),
                Monomial(1, m2_operation, p_pow),
            ).asLinearCombination()
        )

    return sum(r)


def reduced_diagonal0(linear_comb):
    """\\tilde{∆} : B_0 ---> B_0 ⊗ B_0"""

    r = []

    for monomial in linear_comb:
        c = monomial.c

        bool_lc_tensors_init = False

        for i in range(len(monomial.monomial) >> 1):
            operation = monomial.monomial[2 * i]
            exp = monomial.monomial[2 * i + 1]

            if i > 0:
                c = 1

            decomposition = c * steenrod_decompose(operation, exp)

            if bool_lc_tensors_init:
                lc_tensors *= decomposition
            else:
                lc_tensors = decomposition
                bool_lc_tensors_init = True

        r.append(lc_tensors)

    sgn_monomial = Monomial(-1, tuple([])).asLinearCombination()
    r.append(
        TensorProduct(sgn_monomial, linear_comb)
        + TensorProduct(linear_comb, sgn_monomial)
    )

    return sum(r)


@cache
def factorial(n):
    r = 1
    if n > 1:
        for i in range(2, n + 1):
            r = r * i
    else:
        r = 1
    return r


@cache
def bin_coeff(p, n, k):
    if (n, k) == (0, 0):
        return 1
    if k > n:
        return 0
    if k < 0:
        return 0
    return (factorial(n) // (factorial(k) * factorial(n - k))) % p


# TODO: p = 2
def mod_p_adem_relation(a, b, type):
    """INPUT: a, b exponents and type: 0, 1 related to the presence of \\beta"""

    list_operations = []

    sum_upper_i = int(a / PARAM_FIXED_PRIME)

    if type == 0:
        list_operations.append(
            Monomial(
                2, (TAG_MONOMIAL_POWER, a, TAG_MONOMIAL_POWER, b)
            ).asLinearCombination()
        )

        for i in range(1, sum_upper_i):
            list_operations.append(
                (-1) ** (a + i)
                * bin_coeff(
                    PARAM_FIXED_PRIME,
                    (PARAM_FIXED_PRIME - 1) * (b - i) - 1,
                    a - PARAM_FIXED_PRIME * i,
                )
                * Monomial(
                    1, (TAG_MONOMIAL_POWER, a + b - i, TAG_MONOMIAL_POWER, i)
                ).asLinearCombination()
            )

        if sum_upper_i > 0:
            list_extra_term = [TAG_MONOMIAL_POWER, sum_upper_i]
        else:
            list_extra_term = []

        i = 0
        list_operations.append(
            (-1) ** (a + i)
            * bin_coeff(
                PARAM_FIXED_PRIME,
                (PARAM_FIXED_PRIME - 1) * (b - i) - 1,
                a - PARAM_FIXED_PRIME * i,
            )
            * Monomial(1, (TAG_MONOMIAL_POWER, a + b)).asLinearCombination()
        )
        i = sum_upper_i
        if sum_upper_i > 0:
            list_operations.append(
                (-1) ** (a + i)
                * bin_coeff(
                    PARAM_FIXED_PRIME,
                    (PARAM_FIXED_PRIME - 1) * (b - i) - 1,
                    a - PARAM_FIXED_PRIME * i,
                )
                * Monomial(
                    1,
                    tuple([TAG_MONOMIAL_POWER, a + b - sum_upper_i] + list_extra_term),
                ).asLinearCombination()
            )
    else:
        list_operations.append(
            Monomial(
                2,
                (
                    TAG_MONOMIAL_POWER,
                    a,
                    TAG_MONOMIAL_BOCKSTEIN,
                    1,
                    TAG_MONOMIAL_POWER,
                    b,
                ),
            ).asLinearCombination()
        )

        for i in range(1, sum_upper_i):
            list_operations.append(
                (-1) ** (a + i)
                * bin_coeff(
                    PARAM_FIXED_PRIME,
                    (PARAM_FIXED_PRIME - 1) * (b - i),
                    a - PARAM_FIXED_PRIME * i,
                )
                * Monomial(
                    1,
                    (
                        TAG_MONOMIAL_BOCKSTEIN,
                        1,
                        TAG_MONOMIAL_POWER,
                        a + b - i,
                        TAG_MONOMIAL_POWER,
                        i,
                    ),
                ).asLinearCombination()
            )

        if sum_upper_i > 0:
            list_extra_term = [TAG_MONOMIAL_POWER, sum_upper_i]
        else:
            list_extra_term = []

        i = 0
        list_operations.append(
            (-1) ** (a + i)
            * bin_coeff(
                PARAM_FIXED_PRIME,
                (PARAM_FIXED_PRIME - 1) * (b - i),
                (a - PARAM_FIXED_PRIME * i),
            )
            * Monomial(
                1, (TAG_MONOMIAL_BOCKSTEIN, 1, TAG_MONOMIAL_POWER, a + b - i)
            ).asLinearCombination()
        )
        i = sum_upper_i
        if sum_upper_i > 0:
            list_operations.append(
                (-1) ** (a + i)
                * bin_coeff(
                    PARAM_FIXED_PRIME,
                    (PARAM_FIXED_PRIME - 1) * (b - i),
                    (a - PARAM_FIXED_PRIME * i),
                )
                * Monomial(
                    1,
                    tuple(
                        [TAG_MONOMIAL_BOCKSTEIN, 1, TAG_MONOMIAL_POWER, a + b - i]
                        + list_extra_term
                    ),
                ).asLinearCombination()
            )

        #######

        sum_upper_i = int((a - 1) / PARAM_FIXED_PRIME)

        for i in range(1, sum_upper_i):
            list_operations.append(
                (-1) ** (a + i - 1)
                * bin_coeff(
                    PARAM_FIXED_PRIME,
                    (PARAM_FIXED_PRIME - 1) * (b - i) - 1,
                    a - PARAM_FIXED_PRIME * i - 1,
                )
                * Monomial(
                    1,
                    (
                        TAG_MONOMIAL_POWER,
                        a + b - i,
                        TAG_MONOMIAL_BOCKSTEIN,
                        1,
                        TAG_MONOMIAL_POWER,
                        i,
                    ),
                ).asLinearCombination()
            )

        if sum_upper_i > 0:
            list_extra_term = [TAG_MONOMIAL_POWER, sum_upper_i]
        else:
            list_extra_term = []

        i = 0
        list_operations.append(
            (-1) ** (a + i - 1)
            * bin_coeff(
                PARAM_FIXED_PRIME,
                (PARAM_FIXED_PRIME - 1) * (b - i) - 1,
                a - PARAM_FIXED_PRIME * i - 1,
            )
            * Monomial(
                1, (TAG_MONOMIAL_POWER, a + b - i, TAG_MONOMIAL_BOCKSTEIN, 1)
            ).asLinearCombination()
        )
        i = sum_upper_i
        if sum_upper_i > 0:
            list_operations.append(
                (-1) ** (a + i - 1)
                * bin_coeff(
                    PARAM_FIXED_PRIME,
                    (PARAM_FIXED_PRIME - 1) * (b - i) - 1,
                    a - PARAM_FIXED_PRIME * i - 1,
                )
                * Monomial(
                    1,
                    tuple(
                        [TAG_MONOMIAL_POWER, a + b - i, TAG_MONOMIAL_BOCKSTEIN, 1]
                        + list_extra_term
                    ),
                ).asLinearCombination()
            )

    r = sum(list_operations)

    for monomial in r:
        monomial.c %= PARAM_FIXED_PRIME

    return r


def extract_R_B_element(monomial):
    m = monomial.monomial

    if len(m) >= 6:
        if (
            m[0] == TAG_MONOMIAL_POWER
            and m[2] == TAG_MONOMIAL_BOCKSTEIN
            and m[4] == TAG_MONOMIAL_POWER
        ):
            if m[1] <= PARAM_FIXED_PRIME * m[5]:
                return (m[1], m[5], 1)
        elif (
            m[0] == TAG_MONOMIAL_BOCKSTEIN
            and m[2] == TAG_MONOMIAL_POWER
            and m[4] == TAG_MONOMIAL_POWER
        ):
            if m[3] < PARAM_FIXED_PRIME * m[5]:
                return (m[3], m[5], 2)
        # TODO: (possibly) unreachable
        elif (
            m[0] == TAG_MONOMIAL_POWER
            and m[2] == TAG_MONOMIAL_POWER
            and m[4] == TAG_MONOMIAL_BOCKSTEIN
        ):
            if m[1] < PARAM_FIXED_PRIME * m[3]:
                return (m[1], m[3], 3)  # WARNING: unimplemented

    if len(m) >= 4:
        if m[0] == TAG_MONOMIAL_POWER and m[2] == TAG_MONOMIAL_POWER:
            if m[1] < PARAM_FIXED_PRIME * m[3]:
                return (m[1], m[3], 0)

    return RET_ERR


def reduced_diagonal0_image_simplify(lc_img, left_or_right):
    clean_elements = []
    pending_elements = []
    output_list = []

    for i in range(0, len(lc_img.monomials)):
        monomial = lc_img.monomials[i]
        if left_or_right == 0:
            current_monomial = monomial.m1
        else:
            current_monomial = monomial.m2

        if (
            monomial.m1.c * monomial.m2.c
        ) % PARAM_FIXED_PRIME == 0 and monomial.m1.c * monomial.m2.c != 0:
            clean_elements.append(monomial)
            # output_list.append([monomial, monomial, left_or_right])
            output_list.append([monomial.asLinearCombination(), MULTIPLE_OF_FIXED_P])
            continue

        r = extract_R_B_element(current_monomial)
        if r == RET_ERR:
            pending_elements.append(monomial.asLinearCombination())

        else:
            bool_first_term = True
            k = 0

            adem_relation_list_tensor = []

            if r[2] == 2:
                relation_type = 0
            else:
                relation_type = r[2]

            for m_rel in mod_p_adem_relation(r[0], r[1], relation_type).monomials:

                if bool_first_term:
                    k = ((PARAM_FIXED_PRIME**2 + 1) >> 1) * current_monomial.c

                m_rel.c = k * m_rel.c

                if r[2] == 2:
                    m_rel_tmp = Monomial(1, (TAG_MONOMIAL_BOCKSTEIN, 1)) * m_rel
                else:
                    m_rel_tmp = m_rel

                if left_or_right == 0:
                    t = TensorProductBasic(m_rel_tmp, monomial.m2)
                else:
                    t = TensorProductBasic(monomial.m1, m_rel_tmp)

                clean_elements.append(t)
                adem_relation_list_tensor.append(t)

                pending_elements.append(-1 * t.asLinearCombination())

                bool_first_term = False

            # output_list.append([LinearCombination(adem_relation_list_tensor), monomial, left_or_right])
            output_list.append(
                [LinearCombination(adem_relation_list_tensor), left_or_right]
            )

    tmp_img = lc_img.copy()
    remaining_elements = tmp_img.copy()
    for el in clean_elements:
        el_lc = el.asLinearCombination()
        tmp_img += -1 * el_lc

        remaining_elements = tmp_img

    return (output_list, remaining_elements)


def rearrange_img_diag0(lc_img):
    r = []

    pending_sum = lc_img.copy()

    while True:
        (
            clean_output,
            pending_sum,
        ) = reduced_diagonal0_image_simplify(pending_sum, 0)

        r += clean_output

        if pending_sum.isZero():
            break

        clean_output, pending_sum = reduced_diagonal0_image_simplify(pending_sum, 1)

        r += clean_output

        if pending_sum.isZero():
            break

    return (r, pending_sum)


def deg(monomial):
    """INPUT: monomial (Milnor/Adem basis)"""
    """OUTPUT: tensor algebra degree"""

    if monomial.basis == TAG_MILNOR_BASIS:
        return milnor_basis_deg(monomial)

    if monomial.isZero():
        return -1

    if len(monomial.monomial) == 0:
        return 0

    r = 0

    for i in range(0, len(monomial.monomial) >> 1):
        op = monomial.monomial[2 * i]
        exp = monomial.monomial[2 * i + 1]

        if op == TAG_MONOMIAL_BOCKSTEIN:
            r += 1
        else:
            r += 2 * exp * (PARAM_FIXED_PRIME - 1)

    return r


def kristensen_derivation(steenrod_operation):
    """INPUT: MonomialMilnorBasis object"""

    if milnor_basis_deg(steenrod_operation) <= 0:
        return steenrod_operation

    if len(steenrod_operation.monomial[0]) > 0:
        sgn = 0
        if min(steenrod_operation.monomial[0]) == 0:
            sgn = 1
        return sgn * MonomialMilnorBasis(
            steenrod_operation.c, ((), steenrod_operation.monomial[1])
        )

    if len(steenrod_operation.monomial[1]) > 0:
        return MonomialMilnorBasis(0, ((), ()))

    return -1


def monomial_to_mod_p(monomial):
    return Monomial(
        monomial.c % PARAM_FIXED_PRIME, monomial.monomial, PARAM_FIXED_PRIME
    )


def monomial_to_milnor_basis(monomial):
    """OUTPUT: LinearCombination object"""

    lc_monomials = MonomialMilnorBasis(1, ((), ())).asLinearCombination()

    for i in range(len(monomial.monomial) >> 1):
        st_op = monomial.monomial[2 * i]
        st_pow = monomial.monomial[2 * i + 1]

        if st_op == TAG_MONOMIAL_BOCKSTEIN:
            list_operations = ((0,), ())
        else:
            list_operations = ((), (st_pow,))

        lc_monomials *= MonomialMilnorBasis(
            monomial.c, list_operations
        ).asLinearCombination()

    return lc_monomials


def A(steenrod_operation, list_adem_relation):  # TODO: monomial_to_mod_p
    if deg(steenrod_operation) <= 0:
        return Monomial(0, tuple([]), PARAM_FIXED_PRIME)

    bool_mod_p_term_found = False
    for monomial in list_adem_relation:
        if monomial.c % PARAM_FIXED_PRIME == 0:
            bool_mod_p_term_found = True

    if bool_mod_p_term_found:
        rel = list_adem_relation[0]
        multiple_of_p = Monomial(
            int(rel.c / PARAM_FIXED_PRIME), rel.monomial, PARAM_FIXED_PRIME
        )
        # mod p > 2
        return -1 * kristensen_derivation(steenrod_operation) * multiple_of_p

    return -1  # TODO: unimplemented, recursion


def A_aux(st_operation_mod_p_2, list_list_relations):
    st_operation = monomial_to_mod_p(st_operation_mod_p_2)
    st_dec = steenrod_decompose(  # TODO: change to Milnor basis implementation
        st_operation.monomial[0], st_operation.monomial[1], PARAM_FIXED_PRIME
    )

    for list_relation in list_list_relations:
        rel = list_relation[0]
        rel_type = list_relation[1]
        rel_deg = deg(rel.monomials[0].m1)

        list_img = []
        for dec_monomial in st_dec.monomials:
            if rel_type == 2 or rel_type == 0:
                rel_adem_part = [t.m1 for t in rel]
                rel_non_adem_part = [t.m2 for t in rel]

                list_img.append(
                    TensorProductBasic(
                        (-1) ** (deg(dec_monomial.m2) * rel_deg)
                        * A(dec_monomial.m1, rel_adem_part),
                        # TODO: convert to mod p (rel_non_adem_part)
                        milnor_basis_product(
                            dec_monomial.m2, rel_non_adem_part
                        ),  # TODO: a * b
                    )
                )

        # TODO: esto es ok, pero falta multiplicar en álgebra de Steenrod
        print(f"@: {list_img}")

        # TODO: otro caso

    return


# Steenrod algebra routines


def milnor_basis_deg(monomial):
    """INPUT: MonomialMilnorBasis()"""
    if monomial.isZero():
        return -1

    if len(monomial.monomial) == 0:
        return 0

    r = 0

    q_tuple = monomial.monomial[0]
    p_tuple = monomial.monomial[1]

    for q_index in q_tuple:
        r += 2 * PARAM_FIXED_PRIME**q_index - 1

    for i in range(len(p_tuple)):
        r += p_tuple[i] * (2 * PARAM_FIXED_PRIME ** (i + 1) - 2)

    return r


def all_Q_j(deg, prefix=[]):
    r = []

    if deg == 0:
        return prefix + [-1]

    upper_bound = int(np.emath.logn(PARAM_FIXED_PRIME, (deg + 1) >> 1))

    if len(prefix) > 0:
        lower_bound = prefix[-1] + 1
    else:
        lower_bound = 0

    if upper_bound < lower_bound:
        return prefix + [-1]
    elif upper_bound == lower_bound:
        if len(prefix) > 0:
            return prefix + [prefix[-1] + 1, -1]
        else:
            return prefix + [-1]

    for i in range(lower_bound, upper_bound + 1):
        r += all_Q_j(deg - (2 * PARAM_FIXED_PRIME**i - 1), prefix=prefix + [i])

    return r


def all_P_r(deg, prefix=[], depth=1):
    r = []

    if deg == 0:
        return prefix + [-1]
    elif deg < 0:
        return -1

    if 2 * PARAM_FIXED_PRIME**depth - 2 > deg:
        return -1

    num_iterations = int(deg / (2 * PARAM_FIXED_PRIME**depth - 2))

    for i in range(0, num_iterations + 1):
        output = all_P_r(
            deg - i * (2 * PARAM_FIXED_PRIME**depth - 2),
            prefix=prefix + [i],
            depth=depth + 1,
        )
        if output != -1:
            r += output

    return r


@cache
def milnor_basis(deg):
    r = []

    q_truncations = set()

    all_p = all_P_r(deg)
    if all_p != -1:
        p_curr = []
        for j in all_p:
            if j != -1:
                p_curr.append(j)
            else:
                r.append((tuple([]), tuple(p_curr)))

                p_curr = []

    q_curr = []
    for i in all_Q_j(deg):
        if i != -1:
            q_curr.append(i)

            if tuple(q_curr) in q_truncations:
                continue

            q_truncations.add(tuple(q_curr))

            q_deg = milnor_basis_deg(
                MonomialMilnorBasis(1, (tuple(q_curr), tuple([])), PARAM_FIXED_PRIME)
            )  # TODO: avoid new object

            if q_deg == deg:
                r.append((tuple(q_curr), tuple([])))
                continue

            all_p = all_P_r(deg - q_deg)
            if all_p == -1:
                continue

            p_curr = []
            for j in all_p:
                if j != -1:
                    p_curr.append(j)
                else:
                    r.append((tuple(q_curr), tuple(p_curr)))

                    p_curr = []

        else:
            q_curr = []

    return r


def milnor_basis_product_diophantine(r_i, depth, prefix=[], acc=0):
    r = []

    if acc > r_i:
        return []

    for i in range(int(r_i / PARAM_FIXED_PRIME**depth) + 1):
        if depth == 0:
            x_1 = r_i - sum(
                [prefix[i] * PARAM_FIXED_PRIME ** (i + 1) for i in range(len(prefix))]
            )
            return [-1, x_1] + prefix
        else:
            r += milnor_basis_product_diophantine(
                r_i,
                depth - 1,
                prefix=[i] + prefix,
                acc=acc + PARAM_FIXED_PRIME**depth * i,
            )

    return r


def milnor_basis_product_retrieve_solutions(
    list_list_sols, depth, solution_len, prefix=[]
):
    r = []

    if depth == -1:
        return [-1] + prefix

    list_sols = list_list_sols[depth]
    for i in range(len(list_sols) // (solution_len + 1)):
        list_new_prefix = list_sols[
            i * solution_len + (i + 1) : (i + 1) * solution_len + (i + 1)
        ]

        r += milnor_basis_product_retrieve_solutions(
            list_list_sols, depth - 1, solution_len, prefix=list_new_prefix + prefix
        )

    return r


# TODO: this should be an "inline" routine
def milnor_basis_pow_product(m1, m2):
    Q = m1.monomial[0]
    R = m1.monomial[1]
    S = m2.monomial[1]

    external_factor = m1.c * m2.c % PARAM_FIXED_PRIME

    M = len(S)
    N = len(R)

    matrix_eq = [[0] * ((N + 1) * M + N) for i in range(M)]

    for i in range(M):
        for k in range(N + 1):
            matrix_eq[i][i + (M + 1) * k] = 1

    list_list_sols = []
    for i in range(N):
        list_sols = milnor_basis_product_diophantine(R[i], M)
        list_list_sols.append(list_sols)

    list_solutions = milnor_basis_product_retrieve_solutions(
        list_list_sols, N - 1, M + 1
    )

    len_list_solutions = len(list_solutions)
    list_complete_solutions = []
    i = 1
    k = 0  # matrix_eq row
    while i <= len_list_solutions:  # TODO: performance (too many entries)
        j = (k + 1) * N * (M + 1) + (k + 1)

        list_y = list_solutions[i:j]

        bool_valid_solution = True
        prefix_complete_solution = []
        for t in range(M):
            dot_prod = 0
            for r in range(len(list_y)):
                dot_prod += list_y[r] * matrix_eq[t][M + r]

            leading_term = S[t] - dot_prod
            if leading_term >= 0:
                prefix_complete_solution.append(leading_term)
            else:
                bool_valid_solution = False
                break

        if bool_valid_solution:
            list_complete_solutions.append(prefix_complete_solution + list_y)

        i = j + 1
        k += 1

    lc_solution = []

    for solution in list_complete_solutions:
        diagonal_factorial_prod = 1
        factorial_prod_t_n = 1
        T = []

        for n in range(1, M + 1):
            s_range = min(n, N)
            t_n = 0

            for j in range(s_range + 1):
                d = solution[n - 1 + j * M]

                t_n += d
                diagonal_factorial_prod *= factorial(d)

            factorial_prod_t_n *= factorial(t_n)
            T.append(t_n)

        for n in range(1, N + 1):
            s_range = min(M, N - n)
            t_n = 0

            for j in range(s_range + 1):
                d = solution[M - 1 + n * (M + 1) + j * M]

                t_n += d
                diagonal_factorial_prod *= factorial(d)

            factorial_prod_t_n *= factorial(t_n)
            T.append(t_n)

        m_coeff = factorial_prod_t_n // diagonal_factorial_prod
        if m_coeff % PARAM_FIXED_PRIME != 0:
            lc_solution.append(
                MonomialMilnorBasis(
                    (external_factor * factorial_prod_t_n // diagonal_factorial_prod)
                    % PARAM_FIXED_PRIME,
                    (Q, tuple(T)),
                    PARAM_FIXED_PRIME,
                )
            )

    return lc_solution


def milnor_basis_product(m1, m2):
    external_factor = m1.c * m2.c % PARAM_FIXED_PRIME
    if external_factor == 0:
        return []

    len_R_1 = len(m1.monomial[1])

    lc_rearranged = [m1]
    lc_rearranged_new = []

    if len_R_1 > 0:  # this part also parses the coefficient part
        if len(m2.monomial[0]) > 0:
            for i in m2.monomial[0]:
                for m in lc_rearranged:
                    bool_flag_continue = True
                    j = i
                    while bool_flag_continue:
                        Q_part = m.monomial[0] + (j,)
                        P_part = list(m.monomial[1])

                        if len(P_part) <= j - i - 1:
                            bool_flag_continue = False

                        if j - i > 0 and bool_flag_continue:
                            P_part[j - i - 1] -= PARAM_FIXED_PRIME**i
                            if P_part[j - i - 1] < 0:
                                bool_flag_continue = False

                        if bool_flag_continue:
                            lc_rearranged_new.append(
                                MonomialMilnorBasis(
                                    external_factor,
                                    (Q_part, tuple(P_part)),
                                    PARAM_FIXED_PRIME,
                                )
                            )

                        j += 1

                lc_rearranged = lc_rearranged_new
                lc_rearranged_new = []
        else:
            lc_rearranged[0] = m2.c * m1
    else:
        for m in lc_rearranged:
            m_monomial_Q = list(m.monomial[0])
            m_monomial_Q += m2.monomial[0]
            lc_rearranged_new.append(
                MonomialMilnorBasis(
                    m.c * m2.c, (tuple(m_monomial_Q), m.monomial[1]), PARAM_FIXED_PRIME
                )
            )

        lc_rearranged = lc_rearranged_new
        lc_rearranged_new = []

    for m in lc_rearranged:
        sgn = 1
        Q_sorted = []
        Q_tmp = list(m.monomial[0])
        for k in range(len(m.monomial[0])):
            min_Q_tmp = min(Q_tmp)
            index_to_remove = Q_tmp.index(min_Q_tmp)

            if len(Q_sorted) > 0:
                if min_Q_tmp == Q_sorted[-1]:
                    sgn = 0
                    break

            Q_sorted.append(min_Q_tmp)
            Q_tmp.pop(index_to_remove)

            sgn *= (-1) ** index_to_remove

        if sgn != 0:
            m.c *= sgn
            m.monomial = list(m.monomial)
            m.monomial[0] = tuple(Q_sorted)
            m.monomial = tuple(m.monomial)
            lc_rearranged_new.append(m)

    lc_rearranged = lc_rearranged_new
    lc_rearranged_new = []

    if len(m2.monomial[1]) > 0:
        for m in lc_rearranged:
            lc_rearranged_new += milnor_basis_pow_product(
                m,
                # coeff previously considered
                MonomialMilnorBasis(1, ((), m2.monomial[1]), PARAM_FIXED_PRIME),
            )

        lc_rearranged = lc_rearranged_new
        lc_rearranged_new = []

    return sum([m.asLinearCombination() for m in lc_rearranged])


# TODO: remaining terms

#####################################################################
#                           Main                                    #
#####################################################################


print("=" * 120)

# r = reduced_diagonal0(mod_p_adem_relation(29, 20, 0))
# r = reduced_diagonal0(mod_p_adem_relation(75, 25, 1)) # ~10 min
r = reduced_diagonal0(mod_p_adem_relation(1, 1, 0))  # 0 sec

print("Diagonal ok.")
print(len(r.monomials))

rearranged = rearrange_img_diag0(r)
print(rearranged[0])
print("=" * 120)
print(rearranged[1])
print("=" * 120)
print("Rearrangement ok.")

## Sketch ##

m = MonomialMilnorBasis(1, (tuple([1, 2]), tuple([1])), PARAM_FIXED_PRIME)
r = steenrod_milnor_basis_decompose_positive_deg(m, depth=len(m.monomial[1]))
print(r)
r = reduced_diagonal0(Monomial.str2lc(1, "b1 p1 b1"))
print(r)

print(monomial_to_milnor_basis(Monomial.str2monomial(1, "b1 p1 b1")))
sys.exit()

r = steenrod_milnor_basis_decompose(Monomial(1, ((0,), ()), PARAM_FIXED_PRIME))
print(r)
# print(A_aux(Monomial(1, ((0,), ()), PARAM_FIXED_PRIME), rearranged[0]))

# print("="*120)
# print("Steenrod algebra tests")
# print("="*120)

# b = milnor_basis(80)
# print(b)

# p = milnor_basis_pow_product(Monomial(1, ((), (100,5))), Monomial(1, ((), (4,5,6))))
# p = milnor_basis_pow_product(Monomial(1, ((), (29,3,3,3,3)), PARAM_FIXED_PRIME), Monomial(1, ((), (10,3,2)), PARAM_FIXED_PRIME))
# p = milnor_basis_pow_product(Monomial(1, ((), (300,3,3,3,3))), Monomial(1, ((), (1000,3,2))))
# p = milnor_basis_pow_product(Monomial(1, ((), (3000,3,3,3,3))), Monomial(2, ((), (1000,3,2))))
# print(len(p))

# p = milnor_basis_product(Monomial(1, ((1, 9), (33, 1)), PARAM_FIXED_PRIME),
#                          Monomial(2, ((3,), (10, 3)), PARAM_FIXED_PRIME))
# p = milnor_basis_product(Monomial(1, (tuple([1,3]), tuple([1])), PARAM_FIXED_PRIME), Monomial(2, (tuple([4]), tuple([1])), PARAM_FIXED_PRIME))
# print(p)
