#Copyright (c) 2025 Andrés Morán (andres.moran.l@uc.cl)
#Licensed under the terms of the MIT License (see ./LICENSE).

from functools import cache
import numpy as np

#from pympler import asizeof # tests 
import sys # tests

# Constants

TAG_MILNOR_BASIS = 0
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


class Monomial:
    """Generic A, B_0 monomial"""

    def __init__(self, coeff, tuple_pow_operations, prime_power=PARAM_FIXED_PRIME**2):
        self.c = coeff
        self.monomial = tuple_pow_operations
        self.p_pow = prime_power

    def __add__(self, other):
        if self.monomial == other.monomial:
            return LinearCombination(
                [
                    Monomial(
                        (self.c + other.c) % self.p_pow,
                        self.monomial,
                        self.p_pow
                    )
                ]
            )
        else:
            return LinearCombination([self, other])

    def __rmul__(self, other):
        return Monomial((other * self.c) % self.p_pow, self.monomial, self.p_pow)

    def __mul__(self, other):
        return Monomial((self.c * other.c) % self.p_pow, self.monomial + other.monomial, self.p_pow)

    def __str__(self): # Just for debugging purposes. TODO: Milnor basis.
        if self.isZero():
            return "0"
        if len(self.monomial) == 0:
            return str(self.c % self.p_pow)
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

        for c in str_operations.split(' '):
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


#####################################################################
#                           Maps                                    #
#####################################################################


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
            operation = monomial.monomial[2*i]
            exp = monomial.monomial[2*i + 1]

            if i > 0:
                c = 1

            decomposition = c*steenrod_decompose(operation, exp)

            if bool_lc_tensors_init:
                lc_tensors *= decomposition
            else:
                lc_tensors = decomposition
                bool_lc_tensors_init = True

        r.append(lc_tensors)

    sgn_monomial = Monomial(-1, tuple([])).asLinearCombination()
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


# TODO: p = 2
def mod_p_adem_relation(a, b, type):
    """INPUT: a, b exponents and type: 0, 1 related to the presence of \\beta"""

    list_operations = []

    sum_upper_i = int(a/PARAM_FIXED_PRIME)

    if type == 0:
        list_operations.append(
            Monomial(
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
                ) * Monomial(
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
                a-PARAM_FIXED_PRIME*i) * Monomial(
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
                ) * Monomial(
                    1,
                    tuple([TAG_MONOMIAL_POWER, a + b -
                          sum_upper_i] + list_extra_term)
                ).asLinearCombination()
            )
    else:
        list_operations.append(
            Monomial(
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
                ) * Monomial(
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
            ) * Monomial(
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
                ) * Monomial(
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
                ) * Monomial(
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
            ) * Monomial(
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
                ) * Monomial(
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

            for m_rel in mod_p_adem_relation(
                r[0], r[1], relation_type
            ).monomials:

                if bool_first_term:
                    k = ((PARAM_FIXED_PRIME**2+1) >> 1) * \
                        current_monomial.c

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
            output_list.append([LinearCombination(adem_relation_list_tensor), left_or_right])

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


def deg(monomial):
    """INPUT: monomial (Adem basis)"""
    """OUTPUT: tensor algebra degree"""
    if monomial.isZero():
        return -1

    if len(monomial.monomial) == 0:
        return 0

    r = 0

    for i in range(0, len(monomial.monomial) >> 1):
        op = monomial.monomial[2*i]
        exp = monomial.monomial[2*i + 1]

        if op == TAG_MONOMIAL_BOCKSTEIN:
            r += 1
        else:
            r += 2*exp*(PARAM_FIXED_PRIME-1)

    return r


def st_mult(m1, m2, basis=TAG_MILNOR_BASIS):
    if basis == TAG_MILNOR_BASIS:
        return milnor_basis_product(m1, m2)

def kristensen_derivation(steenrod_operation):
    """INPUT: \\beta, P^i and constants"""
    if deg(steenrod_operation) <= 0:
        return steenrod_operation

    if steenrod_operation.monomial[0] == TAG_MONOMIAL_BOCKSTEIN:
        return Monomial(steenrod_operation.c, tuple([]), PARAM_FIXED_PRIME)

    if steenrod_operation.monomial[0] == TAG_MONOMIAL_POWER:
        return Monomial(0, tuple([]))

    return -1

def monomial_to_mod_p(monomial):
    return Monomial(
        monomial.c % PARAM_FIXED_PRIME, monomial.monomial, PARAM_FIXED_PRIME
    ) 

def A(steenrod_operation, list_adem_relation): # TODO: monomial_to_mod_p
    if deg(steenrod_operation) <= 0:
        return Monomial(0, tuple([]), PARAM_FIXED_PRIME)

    bool_mod_p_term_found = False
    for monomial in list_adem_relation:
        if monomial.c % PARAM_FIXED_PRIME == 0:
            bool_mod_p_term_found = True

    if bool_mod_p_term_found:
        rel = list_adem_relation[0]
        multiple_of_p = Monomial(int(rel.c / PARAM_FIXED_PRIME), rel.monomial, PARAM_FIXED_PRIME)
        return -1 * kristensen_derivation(steenrod_operation) * multiple_of_p # mod p > 2
        
    return -1 # TODO: unimplemented, recursion

def A_aux(st_operation_mod_p2, list_list_relations):
    st_operation = monomial_to_mod_p(st_operation_mod_p2)
    st_dec = steenrod_decompose(st_operation.monomial[0], st_operation.monomial[1], PARAM_FIXED_PRIME)
    
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
                        (-1)**(deg(dec_monomial.m2) * rel_deg) * A(dec_monomial.m1, rel_adem_part),
                        st_mult(dec_monomial.m2, rel_non_adem_part) # TODO: convert to mod p (rel_non_adem_part)
                    )
                )

        print(f"@: {list_img}") # TODO: esto es ok, pero falta multiplicar en álgebra de Steenrod

        # TODO: otro caso

    return


# Steenrod algebra routines

def milnor_basis_deg(monomial):
    """INPUT: Monomial(1, (tuple([]),tuple([])), PARAM_FIXED_PRIME)"""
    if monomial.isZero():
        return -1
    
    if len(monomial.monomial) == 0:
        return 0

    r = 0

    q_tuple = monomial.monomial[0]
    p_tuple = monomial.monomial[1]

    for q_index in q_tuple:
        r += 2*PARAM_FIXED_PRIME**q_index - 1

    for i in range(len(p_tuple)):
        r += p_tuple[i] * (2*PARAM_FIXED_PRIME**(i+1) - 2)

    return r
    

def all_Q_j(deg, prefix=[]):
    r = []
    
    if deg == 0:
        return prefix + [-1]
        
    upper_bound = int(np.emath.logn(PARAM_FIXED_PRIME, (deg+1) >> 1))

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
        r += all_Q_j(deg - (2*PARAM_FIXED_PRIME**i-1), prefix=prefix + [i])

    return r


def all_P_r(deg, prefix=[], depth=1):
    r = []
    
    if deg == 0:
        return prefix + [-1]
    elif deg < 0:
        return -1

    if 2*PARAM_FIXED_PRIME**depth - 2 > deg:
        return -1

    num_iterations = int(deg / (2*PARAM_FIXED_PRIME**depth - 2))

    for i in range(0, num_iterations + 1):
        output = all_P_r(deg - i*(2*PARAM_FIXED_PRIME**depth - 2), prefix=prefix + [i], depth=depth+1)
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

            q_deg = milnor_basis_deg(Monomial(1, (tuple(q_curr), tuple([])), PARAM_FIXED_PRIME)) # TODO: avoid new object

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


def milnor_basis_product(m1, m2):
    print("UNIMPLEMENTED")
    return [m1, m2]


# TODO: remaining terms

#####################################################################
#                           Main                                    #
#####################################################################

print("="*120)

# r = reduced_diagonal0(mod_p_adem_relation(29, 20, 0))
# r = reduced_diagonal0(mod_p_adem_relation(75, 25, 1)) # ~10 min
r = reduced_diagonal0(mod_p_adem_relation(1, 1, 0)) # 0 sec

print("Diagonal ok.")
print(len(r.monomials))

rearranged = rearrange_img_diag0(r)
print(rearranged[0])
print("="*120)
print(rearranged[1])
print("="*120)
print("Rearrangement ok.")

## Sketch ##

# print(A_aux(Monomial.str2monomial(1, "b1"), rearranged[0]))

print("="*120)
print("Steenrod algebra tests")
print("="*120)

b = milnor_basis(14)
print(b)

