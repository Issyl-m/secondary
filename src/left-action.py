#Copyright (c) 2025 Andrés Morán (andres.moran.l@uc.cl)
#Licensed under the terms of the MIT License (see ./LICENSE).

from functools import cache

#from pympler import asizeof # tests 
import sys # tests

# Constants

TAG_MONOMIAL_CONSTANT = 0
TAG_MONOMIAL_BOCKSTEIN = 1
TAG_MONOMIAL_POWER = 2
TYPE_B_0 = 0

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
        if self.m1.monomial == other.m1.monomial and self.m2.monomial == other.m2.monomial:

            c = self.m1.c * self.m2.c + other.m1.c * other.m2.c
            m1 = (self.m1 + other.m1).monomials[0]
            m2 = (self.m2 + other.m2).monomials[0]

            m1.c = c
            m2.c = 1

            return LinearCombination(
                [
                    TensorProductBasic(m1, m2)
                ]
            )
        else:
            return LinearCombination([self, other])

    def __rmul__(self, other):
        r = self.copy()
        r.m1.c *= other
        return r

    def __mul__(self, other):
        return TensorProductBasic(self.m1 * other.m1, self.m2 * other.m2)

    def __str__(self):
        if self.m1.isZero() or self.m2.isZero():
            return "0"
        return f"{str(self.m1)} ⊗ {str(self.m2)}"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        c1 = self.m1.c * self.m2.c
        c2 = other.m1.c * other.m2.c

        return (c1.coeffCmp(c1, c2))\
            and self.m1.monomial == other.m1.monomial \
            and self.m2.monomial == other.m2.monomial

    def asLinearCombination(self):
        return LinearCombination([self])

    def isZero(self):
        return (self.m1.coeffCmp(self.m1.c * self.m2.c, 0))

    def copy(self):
        return TensorProductBasic(self.m1.copy(), self.m2.copy())

    def sameGenerator(self, other):
        return self.m1.monomial == other.m1.monomial and self.m2.monomial == other.m2.monomial


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
                    if not (other*monomial).isZero()
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

    def __repr__(self):
        return self.__str__()

    def copy(self):
        return LinearCombination([m.copy() for m in self.monomials])

    def isEqual(self, other):
        return (self + (-1)*other).isZero()

    def isZero(self):
        if len(self.monomials) == 0:
            return True

        bool_is_zero = True

        for monomial in self.monomials:
            if not monomial.isZero():
                bool_is_zero = False
                break

        return bool_is_zero


class Monomial0:
    """B_0 monomial"""

    def __init__(self, coeff, tuple_pow_operations):
        # TODO: Super-class

        self.c = coeff
        self.monomial = tuple_pow_operations

    def __add__(self, other):
        if self.monomial == other.monomial:
            return LinearCombination(
                [
                    Monomial0(
                        (self.c + other.c) % PARAM_FIXED_PRIME**2,
                        self.monomial
                    )
                ]
            )
        else:
            return LinearCombination([self, other])

    def __rmul__(self, other):
        return Monomial0((other * self.c) % PARAM_FIXED_PRIME**2, self.monomial)

    def __mul__(self, other):
        return Monomial0((self.c * other.c) % PARAM_FIXED_PRIME**2, self.monomial + other.monomial)

    def __str__(self):
        if self.isZero():
            return "0"
        if len(self.monomial) == 0:
            return str(self.c % PARAM_FIXED_PRIME**2)
        str_operations = ""
        for i in range(0, len(self.monomial) >> 1):
            operation_tag = self.monomial[2*i]
            operation_exp = self.monomial[2*i + 1]

            if operation_tag == TAG_MONOMIAL_BOCKSTEIN:
                operation_str = "b"
            else:
                operation_str = "P"

            str_operations += f"{operation_str}^{
                operation_exp} "

        if int(self.c) % PARAM_FIXED_PRIME**2 == 1:
            return f"{str_operations[:-1]}"
        return f"{self.c % PARAM_FIXED_PRIME**2}{str_operations[:-1]}"

    def __eq__(self, other):
        if self.isZero() and other.isZero():
            return True

        return (self.coeffCmp(self.c, other.c)) and (self.monomial == other.monomial)

    def asLinearCombination(self):
        return LinearCombination([self])

    def isZero(self):
        return self.c % PARAM_FIXED_PRIME**2 == 0

    def copy(self):
        return Monomial0(self.c, self.monomial)

    def sameGenerator(self, other):
        return self.monomial == other.monomial

    @staticmethod
    def coeffCmp(c1, c2):
        return (c1 - c2) % PARAM_FIXED_PRIME**2 == 0

    @staticmethod
    def str2monomial(coeff, str_operations):
        """INPUT: (c, 'b1 p40 p2 b1')"""
        list_operations = []

        for c in str_operations.split(' '):
            if c[0] == "b":
                list_operations.append(TAG_MONOMIAL_BOCKSTEIN)
            elif c[0] == "p":
                list_operations.append(TAG_MONOMIAL_POWER)

            list_operations.append(int(c[1:]))

        return Monomial0(coeff, tuple(list_operations))

    @staticmethod
    def str2lc(coeff, str_operations):
        """INPUT: (c, 'b1 p40 p2 b1')"""
        """OUTPUT: element as LinearCombination object"""
        list_operations = []

        for c in str_operations.split(' '):
            if c[0] == "b":
                list_operations.append(TAG_MONOMIAL_BOCKSTEIN)
            elif c[0] == "p":
                list_operations.append(TAG_MONOMIAL_POWER)

            list_operations.append(int(c[1:]))

        return Monomial0(coeff, tuple(list_operations)).asLinearCombination()

# TODO: Steenrod Algebra A

#####################################################################
#                           Maps                                    #
#####################################################################


@cache
def steenrod_decompose(operation, exp, type):  # TODO: p = 2
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
        if type == TYPE_B_0:
            r.append(
                TensorProductBasic(
                    Monomial0(1, m1_operation),
                    Monomial0(1, m2_operation),
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
            operation = monomial.monomial[2*i]
            exp = monomial.monomial[2*i + 1]

            if i > 0:
                c = 1

            decomposition = c*steenrod_decompose(operation, exp, TYPE_B_0)

            if bool_lc_tensors_init:
                lc_tensors *= decomposition
            else:
                lc_tensors = decomposition
                bool_lc_tensors_init = True

        r.append(lc_tensors)

    sgn_monomial = Monomial0(-1, tuple([])).asLinearCombination()
    r.append(
        TensorProduct(
            sgn_monomial, linear_comb
        )
        +
        TensorProduct(
            linear_comb, sgn_monomial
        )
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


# TODO: p even
def mod_p_adem_relation(a, b, type):
    """INPUT: a, b exponents and type: 0, 1 related to the presence of \\beta"""

    list_operations = []

    sum_upper_i = int(a/PARAM_FIXED_PRIME)

    if type == 0:
        list_operations.append(
            Monomial0(
                2,
                (TAG_MONOMIAL_POWER, a, TAG_MONOMIAL_POWER, b)
            ).asLinearCombination()
        )

        for i in range(1, sum_upper_i):
            list_operations.append(
                (-1)**(a+i) * bin_coeff(
                    PARAM_FIXED_PRIME,
                    (PARAM_FIXED_PRIME-1)*(b-i)-1,
                    a-PARAM_FIXED_PRIME*i
                ) * Monomial0(
                    1,
                    (TAG_MONOMIAL_POWER, a + b - i, TAG_MONOMIAL_POWER, i)
                ).asLinearCombination()
            )

        if sum_upper_i > 0:
            list_extra_term = [TAG_MONOMIAL_POWER, sum_upper_i]
        else:
            list_extra_term = []

        i = 0
        list_operations.append(
            (-1)**(a+i) * bin_coeff(
                PARAM_FIXED_PRIME,
                (PARAM_FIXED_PRIME-1)*(b-i)-1,
                a-PARAM_FIXED_PRIME*i) * Monomial0(
                    1,
                    (TAG_MONOMIAL_POWER, a + b)
            ).asLinearCombination()
        )
        i = sum_upper_i
        if sum_upper_i > 0:
            list_operations.append(
                (-1)**(a+i) * bin_coeff(
                    PARAM_FIXED_PRIME,
                    (PARAM_FIXED_PRIME-1)*(b-i)-1,
                    a-PARAM_FIXED_PRIME*i
                ) * Monomial0(
                    1,
                    tuple([TAG_MONOMIAL_POWER, a + b -
                          sum_upper_i] + list_extra_term)
                ).asLinearCombination()
            )
    else:
        list_operations.append(
            Monomial0(
                2,
                (TAG_MONOMIAL_POWER, a, TAG_MONOMIAL_BOCKSTEIN,
                 1, TAG_MONOMIAL_POWER, b)
            ).asLinearCombination()
        )

        for i in range(1, sum_upper_i):
            list_operations.append(
                (-1)**(a+i) * bin_coeff(
                    PARAM_FIXED_PRIME,
                    (PARAM_FIXED_PRIME-1)*(b-i),
                    a-PARAM_FIXED_PRIME*i
                ) * Monomial0(
                    1,
                    (TAG_MONOMIAL_BOCKSTEIN, 1, TAG_MONOMIAL_POWER,
                     a + b - i, TAG_MONOMIAL_POWER, i)
                ).asLinearCombination()
            )

        if sum_upper_i > 0:
            list_extra_term = [TAG_MONOMIAL_POWER, sum_upper_i]
        else:
            list_extra_term = []

        i = 0
        list_operations.append(
            (-1)**(a+i) * bin_coeff(
                PARAM_FIXED_PRIME,
                (PARAM_FIXED_PRIME-1)*(b-i),
                (a-PARAM_FIXED_PRIME*i)
            ) * Monomial0(
                1,
                (TAG_MONOMIAL_BOCKSTEIN, 1, TAG_MONOMIAL_POWER, a + b - i)
            ).asLinearCombination()
        )
        i = sum_upper_i
        if sum_upper_i > 0:
            list_operations.append(
                (-1)**(a+i) * bin_coeff(
                    PARAM_FIXED_PRIME,
                    (PARAM_FIXED_PRIME-1)*(b-i),
                    (a-PARAM_FIXED_PRIME*i)
                ) * Monomial0(
                    1,
                    tuple([TAG_MONOMIAL_BOCKSTEIN, 1, TAG_MONOMIAL_POWER,
                           a + b - i] + list_extra_term)
                ).asLinearCombination()
            )

        #######

        sum_upper_i = int((a-1)/PARAM_FIXED_PRIME)

        for i in range(1, sum_upper_i):
            list_operations.append(
                (-1)**(a+i-1) * bin_coeff(
                    PARAM_FIXED_PRIME,
                    (PARAM_FIXED_PRIME-1)*(b-i) - 1,
                    a - PARAM_FIXED_PRIME * i - 1
                ) * Monomial0(
                    1,
                    (TAG_MONOMIAL_POWER, a + b - i,
                     TAG_MONOMIAL_BOCKSTEIN, 1, TAG_MONOMIAL_POWER, i)
                ).asLinearCombination()
            )

        if sum_upper_i > 0:
            list_extra_term = [TAG_MONOMIAL_POWER, sum_upper_i]
        else:
            list_extra_term = []

        i = 0
        list_operations.append(
            (-1)**(a+i-1) * bin_coeff(
                PARAM_FIXED_PRIME,
                (PARAM_FIXED_PRIME-1)*(b-i) - 1,
                a - PARAM_FIXED_PRIME * i - 1
            ) * Monomial0(
                1,
                (TAG_MONOMIAL_POWER, a + b - i, TAG_MONOMIAL_BOCKSTEIN, 1)
            ).asLinearCombination()
        )
        i = sum_upper_i
        if sum_upper_i > 0:
            list_operations.append(
                (-1)**(a+i-1) * bin_coeff(
                    PARAM_FIXED_PRIME,
                    (PARAM_FIXED_PRIME-1)*(b-i) - 1,
                    a - PARAM_FIXED_PRIME * i - 1
                ) * Monomial0(
                    1,
                    tuple([TAG_MONOMIAL_POWER, a + b - i,
                           TAG_MONOMIAL_BOCKSTEIN, 1] + list_extra_term)
                ).asLinearCombination()
            )

    r = sum(list_operations)

    for monomial in r:
        monomial.c %= PARAM_FIXED_PRIME

    return r


def extract_R_B_element(monomial):
    m = monomial.monomial

    if len(m) >= 6:
        if m[0] == TAG_MONOMIAL_POWER and m[2] == TAG_MONOMIAL_BOCKSTEIN and m[4] == TAG_MONOMIAL_POWER:
            if m[1] <= PARAM_FIXED_PRIME*m[5]:
                return (m[1], m[5], 1)
        elif m[0] == TAG_MONOMIAL_BOCKSTEIN and m[2] == TAG_MONOMIAL_POWER and m[4] == TAG_MONOMIAL_POWER:
            if m[3] < PARAM_FIXED_PRIME*m[5]:
                return (m[3], m[5], 2)
        elif m[0] == TAG_MONOMIAL_POWER and m[2] == TAG_MONOMIAL_POWER and m[4] == TAG_MONOMIAL_BOCKSTEIN: # TODO: (possibly) unreachable
            if m[1] < PARAM_FIXED_PRIME*m[3]:
                return (m[1], m[3], 3) # WARNING: unimplemented
    
    if len(m) >= 4:
        if m[0] == TAG_MONOMIAL_POWER and m[2] == TAG_MONOMIAL_POWER:
            if m[1] < PARAM_FIXED_PRIME*m[3]:
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

        if (monomial.m1.c * monomial.m2.c) % PARAM_FIXED_PRIME == 0 and monomial.m1.c * monomial.m2.c != 0:
            clean_elements.append(monomial)
            output_list.append([monomial, monomial, left_or_right])
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

            for m_rel in mod_p_adem_relation(
                r[0], r[1], relation_type
            ).monomials:

                if bool_first_term:
                    k = ((PARAM_FIXED_PRIME**2+1) >> 1) * \
                        current_monomial.c

                m_rel.c = k * m_rel.c

                if r[2] == 2:
                    m_rel_tmp = Monomial0(1, (TAG_MONOMIAL_BOCKSTEIN, 1)) * m_rel
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

            output_list.append([LinearCombination(adem_relation_list_tensor), monomial, left_or_right])

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
        clean_output, pending_sum, = reduced_diagonal0_image_simplify(
            pending_sum, 0
        )

        r += clean_output

        if pending_sum.isZero():
            break

        clean_output, pending_sum = reduced_diagonal0_image_simplify(
            pending_sum, 1
        )

        r += clean_output

        if pending_sum.isZero():
            break

    return (r, pending_sum)

#####################################################################
#                           Main                                    #
#####################################################################


# r = reduced_diagonal0(mod_p_adem_relation(25, 20, 0)) # WARNING: tests: mantener cerca de grado 50
# r = reduced_diagonal0(mod_p_adem_relation(29, 20, 0)) # WARNING: tests: mantener cerca de grado 50. OK <10 min
# r = reduced_diagonal0(mod_p_adem_relation(5, 2, 0)) # WARNING: tests: mantener cerca de grado 50
r = reduced_diagonal0(mod_p_adem_relation(29, 20, 1))

print("Diagonal ok.")
print(len(r.monomials))

rearranged = rearrange_img_diag0(r)
print(rearranged[0])
print("="*120)
print(rearranged[1])
