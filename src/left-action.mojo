# Copyright (c) 2025 Andrés Morán (andres.moran.l@uc.cl)
# Licensed under the terms of the MIT License (see ./LICENSE).

from algorithm.functional import vectorize, parallelize
from memory import UnsafePointer

from sys import has_nvidia_gpu_accelerator, exit
from collections import InlineArray, List, Dict, Set
from math import log

from os.path import exists
from os import SEEK_SET, SEEK_END

import time

from gpu.host import Dim
from gpu.id import block_dim, block_idx, thread_idx
from layout import LayoutTensor, Layout
from max.driver import (
    Accelerator,
    Device,
    Tensor,
    accelerator,
    cpu,
)
 
# Constants

alias TAG_MILNOR_BASIS = 0
alias TAG_SERRE_CARTAN_BASIS = 1
alias TAG_MONOMIAL_CONSTANT = 0
alias TAG_MONOMIAL_BOCKSTEIN = 1
alias TAG_MONOMIAL_POWER = 2
alias TAG_MONOMIAL_POWER_MILNOR_BASIS = -1
alias MULTIPLE_OF_FIXED_P = 2
alias TAG_COMPUTING_WITHOUT_BOCKSTEIN = 0
alias TAG_COMPUTING_WITH_BOCKSTEIN = 1
alias TAG_COMPUTATIONS_FINISHED = 2
alias TAG_COMPUTATIONS_NEW = 3

alias RET_ERR = -1
alias RET_ZERO_MOD_P = 1

# Parameters

alias CONSTANT_OUTPUT_FILE = "A-map.bin"
alias CONSTANT_STOP_FILE = "stop"
alias PARAM_FIXED_PRIME = 3
alias MAX_DEG = 100 # 200 # 600

# Global variables

var dict_A_img = Dict[String, List[SIMD[DType.int16, 32]]]() # TODO: improve hash function
var bool_signal_save_and_exit = False # TODO: unused

var dict_monomial_to_coord_pos = Dict[String, Int]()
var list_matrices = List[List[List[Int]]]()
var list_list_milnor_basis = List[List[SIMD[DType.int16, 32]]]()

var f = FileHandle()

# Structs

struct ComputationStatus:
    var curr_basis_deg: Int
    var curr_basis_tuple_idx: Int
    var curr_i: Int
    var curr_j: Int
    var max_basis_deg: Int
    var max_rel_deg: Int
    var current_computation_status: Int

    fn __init__(out self, curr_basis_deg: Int, curr_basis_tuple_idx: Int, curr_i: Int, curr_j: Int, max_basis_deg: Int, max_rel_deg: Int, current_computation_status: Int):
        self.curr_basis_deg = curr_basis_deg
        self.curr_basis_tuple_idx = curr_basis_tuple_idx
        self.curr_i = curr_i
        self.curr_j = curr_j
        self.max_basis_deg = max_basis_deg
        self.max_rel_deg = max_rel_deg
        self.current_computation_status = current_computation_status

    fn __copyinit__(out self, existing: Self):
        self.curr_basis_deg = existing.curr_basis_deg
        self.curr_basis_tuple_idx = existing.curr_basis_tuple_idx
        self.curr_i = existing.curr_i
        self.curr_j = existing.curr_j
        self.max_basis_deg = existing.max_basis_deg
        self.max_rel_deg = existing.max_rel_deg
        self.current_computation_status = existing.current_computation_status

#####################################################################
#                      Adem relation free monomials                 #
#####################################################################

@always_inline
fn check_coeff_zero(m1: SIMD[DType.int16, 32], p: Int) -> Bool: # p
    return (m1[0]) % p == 0

@always_inline
fn coeff_cmp(m1: SIMD[DType.int16, 32], m2: SIMD[DType.int16, 32], p: Int) -> Bool: # p
    return (m1[0] - m2[0]) % p == 0

@always_inline
fn get_monomial(m1: SIMD[DType.int16, 32]) -> SIMD[DType.int16, 32]:
    var monomial = m1.shift_left[1]() # WARNING: performance
    return monomial

@always_inline
fn check_adem_same_ev_gen(m1: SIMD[DType.int16, 32], m2: SIMD[DType.int16, 32]) -> Bool:
    var m = get_monomial(m1) ^ get_monomial(m2)
    return not any(m)

@always_inline
fn check_adem_equal_monomial(m1: SIMD[DType.int16, 32], m2: SIMD[DType.int16, 32], p: Int) -> Bool: # p
    if check_coeff_zero(m1, p) and check_coeff_zero(m2, p):
        return True
    return (coeff_cmp(m1, m2, p)) and (check_adem_same_ev_gen(m1, m2))

@always_inline
fn adem_get_monomial_length(m: SIMD[DType.int16, 32]) -> Int:
    var r = 0

    for i in range(31):
        if m[31 - i] == 0:
            r += 1
        else:
            break

    return 31 - r

@always_inline
fn check_zero_adem_basis_powers(m: SIMD[DType.int16, 32]) -> Bool:
    return check_adem_same_ev_gen(m, SIMD[DType.int16, 32](0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0))

fn str_adem(m: SIMD[DType.int16, 32]) -> String:
    if check_coeff_zero(m, PARAM_FIXED_PRIME**2):
        return String("0")
    if check_zero_adem_basis_powers(m):
        return String(m[0] % PARAM_FIXED_PRIME**2)
      
    var str_operations = String("")
    var operation_str = String("")
    for i in range(0, (len(m) - 1) >> 1): # WARNING: last entry is unused
        var m_monomial = get_monomial(m)
        operation_tag = m_monomial[2 * i]
        operation_exp = m_monomial[2 * i + 1]

        if operation_tag == TAG_MONOMIAL_BOCKSTEIN:
            operation_str = String("b")
        elif operation_tag == TAG_MONOMIAL_POWER:
            operation_str = String("P")
      
        if operation_tag == 0:
            continue
      
        str_operations += String(operation_str) + String("^") + String(operation_exp) + String(" ")

    if m[0] % PARAM_FIXED_PRIME**2 == 1:
        return String(str_operations)[:-1]

    return String(m[0] % PARAM_FIXED_PRIME**2) + String(str_operations)[:-1]

fn str_adem(lc: List[SIMD[DType.int16, 32]]) -> String:
    if len(lc) == 0:
        return String("0")

    var r = String()

    bool_first_entry_found = False
    for m in lc:
        if bool_first_entry_found:
            r += String(" + ")
        
        bool_first_entry_found = True

        r += str_adem(m[])

    return r

fn str_milnor(lc: List[SIMD[DType.int16, 32]]) -> String:
    if len(lc) == 0:
        return String("0")

    var r = String()

    bool_first_entry_found = False
    for m in lc:
        if bool_first_entry_found:
            r += String(" + ")
        
        bool_first_entry_found = True

        r += str_milnor(m[])

    return r

@always_inline
fn adem_free_mult(m1: SIMD[DType.int16, 32], m2: SIMD[DType.int16, 32], p: Int) -> SIMD[DType.int16, 32]:
    var m_out = SIMD[DType.int16, 32](0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
    m_out[0] = m1[0] * m2[0] % p

    var i = 1
    for j in range(0, 31):
        if not m1[1 + j] == 0:
            m_out[i] = m1[1 + j]
            i += 1

    for j in range(0, 31):
        if not m2[1 + j] == 0:
            m_out[i] = m2[1 + j]
            i += 1

    return m_out

@always_inline
fn generic_add(m1: SIMD[DType.int16, 32], m2: SIMD[DType.int16, 32], p: Int) -> List[SIMD[DType.int16, 32]]:
    r = List[SIMD[DType.int16, 32]](m1)

    var m = m1
    if check_adem_equal_monomial(m1, m2, p):
        m[0] = (m[0] + m2[0]) % p
        r[0] = m
        return r

    r.append(m2)

    return r

@always_inline
fn generic_add_check_same_ev_generator(m1: SIMD[DType.int16, 32], m2: SIMD[DType.int16, 32], p: Int) -> SIMD[DType.int16, 32]:
    var m = m1
    m[0] = (m[0] + m2[0]) % p
    return m

@always_inline
fn rmul(c: Int, read m: SIMD[DType.int16, 32], p: Int) -> SIMD[DType.int16, 32]:
    r = m
    
    r[0] = r[0] * c % p

    return r

@always_inline
fn adem_deg(m: SIMD[DType.int16, 32], p: Int) -> Int:
    if check_coeff_zero(m, p):
        return -1
    else:
        var r: Int16

        r = 0

        for i in range((32 - 2) >> 1):
            if (m[1 + 2*i] == TAG_MONOMIAL_BOCKSTEIN):
                r += 1
            else:
                r += (2*PARAM_FIXED_PRIME-2) * m[1 + 2*i + 1]
                
        return Int(r) 

#####################################################################
#                       MonomialMilnorBasis                         #
#####################################################################

@always_inline
fn milnor_set_P(m: SIMD[DType.int16, 32], P_list: List[Int]) -> SIMD[DType.int16, 32]:
    var r = m
    for k in range(min(len(P_list), 16)):
        r[16 + k] = P_list[k]
    return r

@always_inline
fn milnor_set_Q(m: SIMD[DType.int16, 32], Q_list: List[Int]) -> SIMD[DType.int16, 32]:
    var r = m
    for k in range(len(Q_list)):
        r[1 + k] = Q_list[k]
    return r

@always_inline
fn milnor_get_Q(read m: SIMD[DType.int16, 32]) -> SIMD[DType.int16, 32]:
    return m.shift_left[1]() & SIMD[DType.int16, 32](0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)   

@always_inline
fn milnor_get_P(read m: SIMD[DType.int16, 32]) -> SIMD[DType.int16, 32]:
    return m.shift_left[16]() & SIMD[DType.int16, 32](0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)

@always_inline
fn milnor_get_P_length(read m: SIMD[DType.int16, 32]) -> Int:
    var r = 0

    @parameter
    for i in range(16):
        if m[31 - i] <= 1: # Trailing zeros
            r += 1
        else:
            break

    return 16 - r

@always_inline
fn milnor_get_Q_length(read m: SIMD[DType.int16, 32]) -> Int:
    var r = 0

    @parameter
    for i in range(15):
        if m[15 - i] <= 0:
            r += 1
        else:
            break

    return 15 - r

@always_inline
fn check_milnor_same_ev_gen(read m1: SIMD[DType.int16, 32], read m2: SIMD[DType.int16, 32]) -> Bool:
    var m1_Q_len = milnor_get_Q_length(m1)
    var m2_Q_len = milnor_get_Q_length(m2)

    if m1_Q_len != m2_Q_len:
        return False

    var m1_P_len = milnor_get_P_length(m1)
    var m2_P_len = milnor_get_P_length(m2)

    if m1_P_len != m2_P_len:
        return False

    var m1_Q = milnor_get_Q(m1)
    var m2_Q = milnor_get_Q(m2)

    var bool_equal_terms = True
    for i in range(m1_Q_len):
        if m1_Q[i] != m2_Q[i]:
            bool_equal_terms = False
            break

    if bool_equal_terms == False:
        return False

    var m1_P = milnor_get_P(m1)
    var m2_P = milnor_get_P(m2)

    bool_equal_terms = True
    for i in range(m1_P_len):
        if m1_P[i] != m2_P[i]:
            bool_equal_terms = False
            break

    return bool_equal_terms

@always_inline
fn check_milnor_basis_zero(read m: SIMD[DType.int16, 32], read p: Int) -> Bool:
    if check_coeff_zero(m, p):
        return True

    var m_Q = milnor_get_Q(m)
    var bool_zero_found = False

    @parameter
    for i in range(15):
        if m_Q[i] == 0:
            continue

        var c = 0

        xor_m_Q = m_Q ^ m_Q[i]
        
        @parameter
        for j in range(15):
            if xor_m_Q[j] == 0:
                c += 1
            if c == 2:
                bool_zero_found = True
                break

        if bool_zero_found:
            break

    return bool_zero_found

fn str_milnor(read m: SIMD[DType.int16, 32]) -> String:
    if milnor_basis_deg(m) <= 0:
        return String(m[0] % PARAM_FIXED_PRIME)

    var str_operations = String("")

    var m_Q = milnor_get_Q(m)
    var m_P = milnor_get_P(m)

    var m_Q_parsed = List[Int]()
    @parameter
    for t in range(15):
        if m_Q[t] < 1:
            m_Q_parsed.append(0)
        else:
            m_Q_parsed.append(Int(m_Q[t]) - 1)
    m_Q_parsed = List[Int](m_Q_parsed[0]) + trim_zeros(m_Q_parsed[1:])

    var m_P_parsed = List[Int]()
    @parameter
    for t in range(16):
        if m_P[t] < 1:
            m_P_parsed.append(0)
        else:
            m_P_parsed.append(Int(m_P[t]) - 1)
    m_P_parsed = trim_zeros(m_P_parsed)

    if any(m_P) == False and any(m_Q) == False:
        return String(m[0] % PARAM_FIXED_PRIME)    

    if any(m_Q > 0):
        var str_op = String("")
        
        for i in range(len(m_Q_parsed)):
            if m_Q_parsed[i] < 0:
                continue

            str_op += String(m_Q_parsed[i]) + ","

        str_operations = String("Q(") + String(str_op) + String(")")
    
    if any(m_P > 1):
        var str_op = String("")
        
        for i in range(len(m_P_parsed)):
            if m_P_parsed[i] == -1:
                continue

            str_op += String(m_P_parsed[i]) + ","

        str_operations += String("P(") + String(str_op) + String(")")

    if m[0] % PARAM_FIXED_PRIME == 1:
        return str_operations.replace(",)", ")")
    
    return String(m[0]) + str_operations.replace(",)", ")")

#TODO: cache/memoization
fn monomial_to_milnor_basis(read monomial: SIMD[DType.int16, 32]) -> List[SIMD[DType.int16, 32]]:
    var lc_monomials = List[SIMD[DType.int16, 32]](SIMD[DType.int16, 32](1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0))

    for i in range(adem_get_monomial_length(monomial) >> 1):
        var st_op = monomial[1 + 2 * i]
        var st_pow = monomial[1 + 2 * i + 1]

        var list_operations = SIMD[DType.int16, 32]()
    
        if st_op == Int16(TAG_MONOMIAL_BOCKSTEIN):
            list_operations[1] = 1 # Q(0)
        else:
            list_operations[16] = st_pow + 1

        list_operations[0] = 1 # coefficient

        lc_monomials = milnor_basis_product(lc_monomials, list_operations) # List[obj] * obj

    return rmul(Int(monomial[0]), lc_monomials, PARAM_FIXED_PRIME)

fn milnor_basis_deg(read monomial: SIMD[DType.int16, 32]) -> Int:
    """INPUT: MonomialMilnorBasis()."""
    if check_milnor_basis_zero(monomial, PARAM_FIXED_PRIME):
        return -1

    if check_zero_adem_basis_powers(monomial):
        return 0

    var r: Int16

    r = 0

    var q_tuple = milnor_get_Q(monomial)
    var p_tuple = milnor_get_P(monomial)
   
    @parameter
    for i in range(15):
        var q_tuple_i = q_tuple[i]
        if q_tuple_i == 0:
            continue

        q_index = q_tuple_i - 1 # NEW INDEXING
        r += 2 * PARAM_FIXED_PRIME**q_index - 1

    @parameter
    for i in range(16):
        var p_tuple_i = p_tuple[i]
        if p_tuple_i == 0:
            continue

        r += (p_tuple[i] - 1) * (2 * PARAM_FIXED_PRIME ** (i + 1) - 2)

    return Int(r)

fn all_Q_subsets(deg: Int, Q_sample: List[Int], mask: List[Int]=List[Int]()) -> List[Int]:
    var r = List[Int]()

    if len(Q_sample) == len(mask):
        r.append(-1)
        var deg_sum = 0
        for idx in range(len(mask)):
            if mask[idx]:
                r.append(Q_sample[idx])
                deg_sum += 2 * PARAM_FIXED_PRIME ** Q_sample[idx] - 1

        if deg_sum <= deg:
            return r
        else: 
            return List[Int](-1)

    for i in range(2):
        r += all_Q_subsets(deg, Q_sample, mask=mask + List[Int](i == 1))

    return r

@always_inline
fn all_Q_j(deg: Int) -> List[Int]:
    var upper_bound = log_p((deg + 1) >> 1, PARAM_FIXED_PRIME)
    var Q_sample = List[Int]()

    for i in range(upper_bound + 1):
        Q_sample.append(Int(i))

    return all_Q_subsets(deg, Q_sample)

fn all_P_r(deg: Int, prefix: List[Int]=List[Int](), depth: Int=1) -> List[Int]:
    var r = List[Int]()

    if deg == 0:
        return prefix + List[Int](-1)
    elif deg < 0:
        return List[Int](-2) 

    if 2 * PARAM_FIXED_PRIME**depth - 2 > deg:
        return List[Int](-2)

    var num_iterations = deg // (2 * PARAM_FIXED_PRIME**depth - 2)
    
    for i in range(num_iterations + 1):
        var output = all_P_r(
            deg - i * (2 * PARAM_FIXED_PRIME**depth - 2),
            prefix=prefix + List[Int](i),
            depth=depth + 1,
        )
        
        if len(output) > 0:
            if output[0] == -2:
                continue
       
        r += output
    
    return r

#TODO: cache/memoize
fn milnor_basis(deg: Int) -> List[SIMD[DType.int16, 32]]:
    if len(list_list_milnor_basis) > deg:
        return list_list_milnor_basis[deg]

    var r = List[SIMD[DType.int16, 32]]()

    var q_truncations = List[List[Int]]()

    var all_p = all_P_r(deg)
    if len(all_p) > 0:
        if all_p[0] != -2:
            var p_curr = List[Int]()
            for _j in all_p:
                var j = _j[]
                if j != -1:
                    p_curr.append(j)
                else:
                    var m = SIMD[DType.int16, 32](1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
                    var len_p_curr = len(p_curr)
                    for idx in range(16):
                        if idx == len_p_curr:
                            break
                        
                        m[16 + idx] = p_curr[idx] + 1

                    r.append(m)

                    p_curr = List[Int]()
    
    var q_curr = List[Int]()
    for _i in all_Q_j(deg):
        var i = _i[]
        if i != -1:
            q_curr.append(i)
    
            var bool_q_curr_in_q_truncations = False

            for q_truncation in q_truncations:
                if q_truncation[] == q_curr:
                    bool_q_curr_in_q_truncations = True
                    break
            
            if bool_q_curr_in_q_truncations:
                continue

            q_truncations.append(q_curr)

            var m = SIMD[DType.int16, 32](1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
            var len_q_curr = len(q_curr)
            for idx in range(15):
                if idx == len_q_curr:
                    break
                
                m[1 + idx] = q_curr[idx] + 1

            var q_deg = milnor_basis_deg(m)

            if q_deg == deg:
                r.append(m)
                continue

            var all_p = all_P_r(deg - q_deg)
            if len(all_p) > 0:
                if all_p[0] == -2:
                    continue

            p_curr = List[Int]()
            for _j in all_p:
                j = _j[]  
                if j != -1:
                    p_curr.append(j)
                else:
                    var m = SIMD[DType.int16, 32](1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
                    var len_q_curr = len(q_curr)
                    for idx in range(15):
                        if idx == len_q_curr:
                            break
                        
                        m[1 + idx] = q_curr[idx] + 1

                    var len_p_curr = len(p_curr)
                    for idx in range(16):
                        if idx == len_p_curr:
                            break
                        
                        m[16 + idx] = p_curr[idx] + 1

                    r.append(m)

                    p_curr = List[Int]()

        else:
            q_curr = List[Int]()

    list_list_milnor_basis.append(r)

    return r

#####################################################################
#                        Milnor product                             # 
#####################################################################

fn milnor_basis_product_diophantine(r_i: Int, depth: Int, prefix: List[Int]=List[Int](), acc: Int=0) -> List[Int]:
    var r = List[Int]()

    if acc > r_i:
        return List[Int]()

    for i in range((r_i // PARAM_FIXED_PRIME**depth) + 1):
        if depth == 0:
            var x1 = r_i 
        
            var len_prefix = len(prefix)
            for idx in range(len_prefix):
                x1 -= prefix[idx] * PARAM_FIXED_PRIME ** (idx + 1)

            return List[Int](-1, x1) + prefix
        else:
            r += milnor_basis_product_diophantine(
                r_i,
                depth - 1,
                prefix=List[Int](i) + prefix,
                acc=acc + PARAM_FIXED_PRIME**depth * i,
            )

    return r

fn milnor_basis_product_retrieve_solutions(list_list_sols: List[List[Int]], depth: Int, solution_len: Int, prefix: List[Int]=List[Int]()) -> List[Int]:
    var r = List[Int]()

    if depth == -1:
        return List[Int](-1) + prefix

    var list_sols = list_list_sols[depth]
    for i in range(len(list_sols) // (solution_len + 1)):
        var list_new_prefix = list_sols[
            i * solution_len + (i + 1) : (i + 1) * solution_len + (i + 1)
        ]

        list_new_prefix += prefix

        r += milnor_basis_product_retrieve_solutions(
            list_list_sols, depth - 1, solution_len, prefix=list_new_prefix
        )

    return r

#TODO: cache/memoization
@always_inline
fn milnor_basis_pow_product(read m1: SIMD[DType.int16, 32], read m2: SIMD[DType.int16, 32]) -> List[SIMD[DType.int16, 32]]:
    var Q = milnor_get_Q(m1)

    var R_SIMD = milnor_get_P(m1)
    var S_SIMD = milnor_get_P(m2)
    var R = List[Int16]()
    var S = List[Int16]()
    for i in range(16):
        var d = R_SIMD[i] - 1
        if d >= 0:
            R.append(Int(d))
        d = S_SIMD[i] - 1
        if d >= 0:
            S.append(Int(d))

    R = trim_zeros(R)
    S = trim_zeros(S)
    
    var external_factor = m1[0] * m2[0] % PARAM_FIXED_PRIME

    var M = len(S)
    var N = len(R)

    var matrix_eq = List[List[Int]](fill=List[Int](fill=0, length=((N + 1) * M + N)), length=M)

    for i in range(M):
        for k in range(N + 1):
            matrix_eq[i][i + (M + 1) * k] = 1

    var list_list_sols = List[List[Int]]()
    for i in range(N):
        var list_sols = milnor_basis_product_diophantine(Int(R[i]), M)
        list_list_sols.append(list_sols)

    var list_solutions = milnor_basis_product_retrieve_solutions(
        list_list_sols, N - 1, M + 1
    )
    var len_list_solutions = len(list_solutions)
    var list_complete_solutions = List[List[Int]]()
    var i = 1
    var k = 0  # matrix_eq row
    while i <= len_list_solutions:  # TODO: performance (too many entries for VERY high degrees)
        var j = (k + 1) * N * (M + 1) + (k + 1)

        var list_y = list_solutions[i:j]

        var bool_valid_solution = True
        var prefix_complete_solution = List[Int]()
        for t in range(M):
            var dot_prod = 0
            for r in range(len(list_y)):
                dot_prod += list_y[r] * matrix_eq[t][M + r]

            var leading_term = S[t] - dot_prod
            if leading_term >= 0:
                prefix_complete_solution.append(Int(leading_term))
            else:
                bool_valid_solution = False
                break

        if bool_valid_solution:
            list_complete_solutions.append(prefix_complete_solution + list_y)

        i = j + 1
        k += 1

    var lc_solution = List[SIMD[DType.int16, 32]]()
    for _solution in list_complete_solutions:
        var solution = _solution[]
        
        var list_diagonal_factorial_prod = List[Int]()

        var multinomial_coeff_prod = 1
        var T = List[Int]()

        for n in range(1, M + 1):
            var s_range = min(n, N)
            var t_n = 0

            list_diagonal_factorial_prod = List[Int]()

            for j in range(s_range + 1):
                var d = solution[n - 1 + j * M]

                t_n += d
                list_diagonal_factorial_prod.append(d)

            multinomial_coeff_prod = Int(multinomial_coeff_prod * mod_p_multinomial_coeff(PARAM_FIXED_PRIME, t_n, list_diagonal_factorial_prod) % PARAM_FIXED_PRIME)
            T.append(t_n + 1) # INDEXING

        for n in range(1, N + 1):
            var s_range = min(M, N - n)
            var t_n = 0

            list_diagonal_factorial_prod = List[Int]()

            for j in range(s_range + 1):
                var d = solution[M - 1 + n * (M + 1) + j * M]

                t_n += d
                list_diagonal_factorial_prod.append(d)

            multinomial_coeff_prod = Int(multinomial_coeff_prod * mod_p_multinomial_coeff(PARAM_FIXED_PRIME, t_n, list_diagonal_factorial_prod) % PARAM_FIXED_PRIME)
            T.append(t_n + 1) # INDEXING

        var m_coeff = multinomial_coeff_prod
        
        if m_coeff % PARAM_FIXED_PRIME != 0:
            var output_m_c = Int(external_factor) * m_coeff % PARAM_FIXED_PRIME
            var output_m = Q.shift_right[1]()

            output_m[0] = output_m_c
            output_m = milnor_set_P(output_m, T)
             
            lc_solution.append(output_m)
    
    return lc_solution

#TODO: cache/memoization
fn milnor_basis_product(read m1: SIMD[DType.int16, 32], read m2: SIMD[DType.int16, 32]) -> List[SIMD[DType.int16, 32]]:
    var external_factor = m1[0] * m2[0] % PARAM_FIXED_PRIME
    if external_factor == 0:
        return List[SIMD[DType.int16, 32]]()

    var len_R_1 = milnor_get_P_length(m1)

    var lc_rearranged = List[SIMD[DType.int16, 32]](m1)
    var lc_rearranged_new = List[SIMD[DType.int16, 32]]()

    if len_R_1 > 0:  # this part also parses the coefficient part
        var m2_Q_len = milnor_get_Q_length(m2)
        if m2_Q_len > 0:
            for idx in range(m2_Q_len):
                var i = milnor_get_Q(m2)[idx] - 1
                for _m in lc_rearranged:
                    var m = _m[]
                    var simd_index_shift = SIMD[DType.int16, 32](1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
                    var m_monomial_P = milnor_get_P(m) - simd_index_shift
                    var m_monomial_Q = milnor_get_Q(m) - simd_index_shift
                    m_monomial_P[15] -= 1

                    var bool_flag_continue = True
                    var j = i

                    while bool_flag_continue:
                        var Q_part = m_monomial_Q
                        var len_Q_part = milnor_get_Q_length(m)
                        Q_part[len_Q_part] = j
                        var P_part = m_monomial_P

                        if Int16(milnor_get_P_length((P_part + simd_index_shift).shift_right[16]())) <= j - i - 1: # WARNING: shifted back
                            bool_flag_continue = False

                        if j - i > 0 and bool_flag_continue:
                            var P_part_entry_val = Int(P_part[Int(j - i - 1)])
                            P_part_entry_val -= Int(PARAM_FIXED_PRIME**i)
                            P_part[Int(j - i - 1)] = Int16(P_part_entry_val) # Possible integer overflow fixed
                            if P_part_entry_val < 0:
                                j += 1
                                continue

                        if bool_flag_continue:
                            var m_out = Q_part.shift_right[1]() + P_part.shift_right[16]()
                            m_out[0] = external_factor
                            m_out += SIMD[DType.int16, 32](0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1)
                            lc_rearranged_new.append(m_out)

                        j += 1

                lc_rearranged = lc_rearranged_new
                lc_rearranged_new = List[SIMD[DType.int16, 32]]()
        else:
            lc_rearranged[0][0] = m2[0] * m1[0] # lc_rearranged[0] = m1
    else:
        for _m in lc_rearranged:
            var m = _m[]
            var simd_index_shift = SIMD[DType.int16, 32](1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
            var m_monomial_Q = milnor_get_Q(m) - simd_index_shift

            var m2_Q = milnor_get_Q(m2) - simd_index_shift
            var m2_Q_len = milnor_get_Q_length(m2)
            var m_Q_len = milnor_get_Q_length(m)

            for r in range(m2_Q_len):
                m_monomial_Q[m_Q_len + r] = m2_Q[r]

            var m_out = m_monomial_Q.shift_right[1]()
            m_out[0] = m[0] * m2[0]
            
            var simd_index_shift_Q = SIMD[DType.int16, 32](0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
            m_out += milnor_get_P(m).shift_right[16]()
            m_out += simd_index_shift_Q
            lc_rearranged_new.append(m_out)

        lc_rearranged = lc_rearranged_new
        lc_rearranged_new = List[SIMD[DType.int16, 32]]()
    for _m in lc_rearranged:
        var m = _m[]
        var sgn = 1
        var Q_sorted = List[Int]()
        var m_Q = milnor_get_Q(m) - SIMD[DType.int16, 32](1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
        var len_m_Q = milnor_get_Q_length(m)
        var Q_tmp = List[Int](length=len_m_Q, fill=0)

        for t in range(len_m_Q):
            Q_tmp[t] = Int(m_Q[t])

        for k in range(len_m_Q):
            var min_Q_tmp = 2**32 - 1
            for v in Q_tmp:
                min_Q_tmp = min(Int(v[]), Int(min_Q_tmp))
            var index_to_remove = Int(0)
            try:
                index_to_remove = Q_tmp.index(min_Q_tmp)
            except e:
                index_to_remove = Int(0) # unreachable

            if len(Q_sorted) > 0:
                if min_Q_tmp == Q_sorted[-1]:
                    sgn = 0
                    break

            Q_sorted.append(min_Q_tmp)
            _ = Q_tmp.pop(index_to_remove)

            sgn *= (-1) ** index_to_remove

        if sgn != 0:
            m[0] *= Int16(sgn)
            var len_Q_sorted = len(Q_sorted)
            for t in range(15):
                if t < len_Q_sorted:
                    m[1 + t] = Int16(Q_sorted[t]) + 1
                else:
                    m[1 + t] = 0

            lc_rearranged_new.append(m)

    lc_rearranged = lc_rearranged_new
    lc_rearranged_new = List[SIMD[DType.int16, 32]]()
    
    if milnor_get_P_length(m2) > 0:
        for _m in lc_rearranged:
            var m = _m[]
            var m2_P = milnor_get_P(m2).shift_right[16]()
            m2_P[0] = 1

            for el in milnor_basis_pow_product(
                m,
                # coeff previously considered
                m2_P,
            ):
                lc_rearranged_new.append(el[])            

        lc_rearranged = lc_rearranged_new
        lc_rearranged_new = List[SIMD[DType.int16, 32]]()

    return lc_simplify_milnor(lc_rearranged, PARAM_FIXED_PRIME)

@always_inline
fn milnor_basis_product(read lc: List[SIMD[DType.int16, 32]], read m: SIMD[DType.int16, 32]) -> List[SIMD[DType.int16, 32]]:
    var r = List[SIMD[DType.int16, 32]]()

    for _lc_m in lc:
        var lc_m = _lc_m[]
        for el in milnor_basis_product(lc_m, m):
            r.append(el[])
 
    return lc_simplify_milnor(r, PARAM_FIXED_PRIME)

@always_inline
fn milnor_basis_product(read m: SIMD[DType.int16, 32], read lc: List[SIMD[DType.int16, 32]]) -> List[SIMD[DType.int16, 32]]:
    var r = List[SIMD[DType.int16, 32]]()
    for _lc_m in lc:
        var lc_m = _lc_m[]
        for el in milnor_basis_product(m, lc_m):
            r.append(el[])

    return lc_simplify_milnor(r, PARAM_FIXED_PRIME)

#####################################################################
#                       LinearCombination                           #
#####################################################################

@always_inline
fn lc_milnor_basis_check_zero(lc: List[SIMD[DType.int16, 32]], p: Int) -> Bool:
    if len(lc) == 0:
        return True

    var bool_is_zero = True

    for m in lc:
        if not check_milnor_basis_zero(m[], p):
            bool_is_zero = False
            break
        else:
            continue

    return bool_is_zero

@always_inline
fn lc_check_zero(lc: List[SIMD[DType.int16, 32]], p: Int) -> Bool:
    if len(lc) == 0:
        return True

    var bool_is_zero = True

    for m in lc:
        if not check_coeff_zero(m[], p):
            bool_is_zero = False
            break
        else:
            continue

    return bool_is_zero

fn generic_add(lc1: List[SIMD[DType.int16, 32]], lc2: List[SIMD[DType.int16, 32]], p: Int) -> List[SIMD[DType.int16, 32]]:
    if lc_check_zero(lc1, p):
        return lc2

    if lc_check_zero(lc2, p):
        return lc1

    var list_monomials = List[SIMD[DType.int16, 32]](length=len(lc1) + len(lc2), fill=SIMD[DType.int16, 32](0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0))

    var i = 0
    for _m1 in lc1:
        var m1 = _m1[]
        var bool_monomial_found = False

        for _m2 in lc2:
            var m2 = _m2[]

            if check_adem_equal_monomial(m1, m2, p):
                var m_sum = generic_add(m1, m2, p)
                if not lc_check_zero(m_sum, p):
                    for k in range(len(m_sum)):
                        list_monomials[i] = m_sum[k]
                        i += 1

                bool_monomial_found = True
                break

        if not bool_monomial_found:
            list_monomials[i] = m1
            i += 1

    for _m2 in lc2:
        m2 = _m2[]
        var bool_monomial_found = False

        for _m1 in lc1:
            m1 = _m1[]

            if check_adem_equal_monomial(m1, m2, p):
                bool_monomial_found = True
                break

        if not bool_monomial_found:
            list_monomials[i] = m2
            i += 1
    
    return list_monomials

@always_inline
fn rmul(c: Int, read lc: List[SIMD[DType.int16, 32]], p: Int) -> List[SIMD[DType.int16, 32]]:
    if c == 0:
        return List[SIMD[DType.int16, 32]](SIMD[DType.int16, 32](0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0))
    else:
        var len_lc = len(lc)
        var r = lc.copy() # WARNING: memory usage

        for i in range(len_lc):
            r[i][0] = c*r[i][0] % p
        
        return r

@always_inline
fn lc_simplify_adem(lc: List[SIMD[DType.int16, 32]], p: Int) -> List[SIMD[DType.int16, 32]]:
    var lc_output = List[SIMD[DType.int16, 32]]()

    var set_visited_ptr = Set[Int]()

    var i1 = 0
    for _m1 in lc:
        var c: Int16
        var monomial = SIMD[DType.int16, 32](0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)

        if i1 in set_visited_ptr:
            i1 += 1
            continue

        c = 0
        var i2 = 0
        for _m2 in lc:
            bool_first_found = False

            var monomial1 = get_monomial(_m1[])
            var m = monomial1 ^ get_monomial(_m2[])
            if not any(m):
                if not bool_first_found:
                    monomial = monomial1.shift_right[1]()
                    bool_first_found = True

                c += _m2[][0] % p

                set_visited_ptr.add(i2)

            i2 += 1
               
        if c % p == 0:
            i1 += 1
            continue
        else:
            monomial[0] = c
            lc_output.append(monomial)

        i1 += 1

    return lc_output

@always_inline
fn lc_simplify_milnor(lc: List[SIMD[DType.int16, 32]], p: Int) -> List[SIMD[DType.int16, 32]]: # WARNING: O(n^2): use quicksort instead
    var lc_output = List[SIMD[DType.int16, 32]]()
    var set_visited_ptr = Set[Int]()

    var i1 = 0 
    for _m1 in lc:
        var c: Int16
        var monomial = SIMD[DType.int16, 32](0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)

        if i1 in set_visited_ptr:
            i1 += 1
            continue

        c = 0

        var i2 = 0
        for _m2 in lc:
            bool_first_found = False

            var monomial1 = get_monomial(_m1[])

            if check_milnor_same_ev_gen(_m1[], _m2[]):
                if not bool_first_found:
                    monomial = monomial1.shift_right[1]()
                    bool_first_found = True

                c += _m2[][0] % p
               
                set_visited_ptr.add(i2)

            i2 += 1
        if c % p == 0:
            i1 += 1
            continue
        else:
            monomial[0] = c
            lc_output.append(monomial)

        i1 += 1
  
    return lc_output

@always_inline
fn adem_free_mult(lc1: List[SIMD[DType.int16, 32]], lc2: List[SIMD[DType.int16, 32]], p: Int) -> List[SIMD[DType.int16, 32]]:
    var lc_output = List[SIMD[DType.int16, 32]]()

    for _m1 in lc1:
        for _m2 in lc2:
            var m = adem_free_mult(_m1[], _m2[], p)

            if not check_coeff_zero(m, p):
                lc_output.append(m)

    return lc_simplify_adem(lc_output, p)

@always_inline
fn adem_free_mult(m1: SIMD[DType.int16, 32], lc2: List[SIMD[DType.int16, 32]], p: Int) -> List[SIMD[DType.int16, 32]]:
    var lc_output = List[SIMD[DType.int16, 32]]()

    for _m2 in lc2:
        var m = adem_free_mult(m1, _m2[], p)

        if not check_coeff_zero(m, p):
            lc_output.append(m)

    return lc_simplify_adem(lc_output, p)

#####################################################################
#                       Tensor products                             #
#####################################################################

@always_inline
fn tp_c(t: SIMD[DType.int16, 64], p: Int) -> Int:
    return Int(t[0] * t[32]) % p

@always_inline
fn tp_m1(t: SIMD[DType.int16, 64]) -> SIMD[DType.int16, 32]:
    return t.slice[32]()

@always_inline
fn tp_m2(t: SIMD[DType.int16, 64]) -> SIMD[DType.int16, 32]:
    return t.shift_left[32]().slice[32]()

@always_inline
fn generic_add(t1: SIMD[DType.int16, 64], t2: SIMD[DType.int16, 64], p: Int) -> List[SIMD[DType.int16, 64]]:
    var t1_m1 = tp_m1(t1)
    var t1_m2 = tp_m2(t1)
    var t2_m1 = tp_m1(t2)
    var t2_m2 = tp_m2(t2)

    var t1_c = tp_c(t1, p)
    var t2_c = tp_c(t2, p)

    if check_adem_same_ev_gen(t1_m1, t2_m1) and check_adem_same_ev_gen(t1_m2, t2_m2):
        var c = t1_c + t2_c

        var m = SIMD[DType.int16, 64]()

        var m1 = generic_add(t1_m1, t2_m1, p)
        var m2 = generic_add(t1_m2, t2_m2, p)

        m[0] = c
        m[32] = 1

        return List[SIMD[DType.int16, 64]](m)

    elif t1_c == 0 and not t2_c == 0:
        return List[SIMD[DType.int16, 64]](t2)
    elif not t1_c == 0 and t2_c == 0:
        return List[SIMD[DType.int16, 64]](t1)
    else:
        return List[SIMD[DType.int16, 64]](t1, t2)

@always_inline
fn rmul(c: Int, read t: SIMD[DType.int16, 64], p: Int) -> SIMD[DType.int16, 64]:
    var r = t
    r[0] = r[0] * c % p
    return r

@always_inline
fn rmul(c: Int, lc: List[SIMD[DType.int16, 64]], p: Int) -> List[SIMD[DType.int16, 64]]:
    var r = List[SIMD[DType.int16, 64]](length=len(lc), fill=SIMD[DType.int16, 64]())
    
    var i = 0
    for t in lc:
        r[i] = t[]
        r[i][0] = r[i][0] * c % p
        i += 1

    return r.copy()

@always_inline
fn tp_adem_free_mult(t1: SIMD[DType.int16, 64], t2: SIMD[DType.int16, 64], p: Int) -> SIMD[DType.int16, 64]:
    var t1_m1 = tp_m1(t1)
    var t1_m2 = tp_m2(t1)
    var t2_m1 = tp_m1(t2)
    var t2_m2 = tp_m2(t2)

    var t1_c = tp_c(t1, p)
    var t2_c = tp_c(t2, p)

    var t_m1 = adem_free_mult(t1_m1, t2_m1, p)
    var t_m2 = adem_free_mult(t1_m2, t2_m2, p)

    var r = t_m1.join(t_m2)

    r[0] *= (-1)**(adem_deg(t1_m2, p) * adem_deg(t2_m1, p)) % p 

    return r

@always_inline
fn tp_lc_adem_free_mult(lc1: List[SIMD[DType.int16, 64]], lc2: List[SIMD[DType.int16, 64]], p: Int) -> List[SIMD[DType.int16, 64]]:
    var r = List[SIMD[DType.int16, 64]]()

    for _m1 in lc1:
        for _m2 in lc2:
            m = tp_adem_free_mult(_m1[], _m2[], p)

            if not check_tensor_zero(m, p):
                r.append(m)

    return r

@always_inline
fn tp_lc_simplify_milnor(mut lc: List[SIMD[DType.int16, 64]], p: Int) -> List[SIMD[DType.int16, 64]]:
    if len(lc) == 0:
        return lc
    if len(lc) == 1:
        var m_out = lc[0]
        if m_out[0] % p == 0:
            return List[SIMD[DType.int16, 64]]()
        else:
            return List[SIMD[DType.int16, 64]](m_out)

    var lc_output = List[SIMD[DType.int16, 64]]()

    quicksort_hoare(lc)

    var c = Int16(0)
    for i in range(len(lc)):
        var m = lc[i]
        var m_prev: SIMD[DType.int16, 64]

        if i > 0:
            m_prev = lc[i - 1]

            if not lexicographic_order_64_eq(m, m_prev):
                c = c % p
                if c != 0:
                    var m_out = m_prev
                    m_out[0] = c
                    m_out[32] = 1
                    lc_output.append(m_out)

                c = m[0]*m[32]
            else:
                c += m[0]*m[32]
                
            if i == len(lc) - 1:
                c = c % p
                if c != 0:
                    var m_out = m
                    m_out[0] = c
                    m_out[32] = 1
                    lc_output.append(m_out)
        else:
            c = m[0]*m[32]

    return lc_output

@always_inline
fn tp_lc_simplify_adem(lc: List[SIMD[DType.int16, 64]], p: Int) -> List[SIMD[DType.int16, 64]]:
    var lc_output = List[SIMD[DType.int16, 64]]() # WARNING: O(n^2): use quicksort instead
    var set_visited_ptr = Set[Int]()

    var i1 = 0
    for _t1 in lc:
        var c: Int16
        var t1_monomial1 = SIMD[DType.int16, 32](0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
        var t1_monomial2 = SIMD[DType.int16, 32](0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
    
        if i1 in set_visited_ptr:
            i1 += 1
            continue

        c = 0

        var i2 = 0
        for _t2 in lc:
            var t1 = _t1[]
            var t2 = _t2[]

            var bool_first_found = False

            var t1_m1 = tp_m1(t1)
            var t1_m2 = tp_m2(t1)
            var t2_m1 = tp_m1(t2)
            var t2_m2 = tp_m2(t2)
            
            var t1_c = tp_c(t1, p)
            var t2_c = tp_c(t2, p)

            var t1_m1_monomial = get_monomial(t1_m1)
            var t1_m2_monomial = get_monomial(t1_m2)
            var t2_m1_monomial = get_monomial(t2_m1)
            var t2_m2_monomial = get_monomial(t2_m2)

            var simd_bool_check_array_1 = (t1_m1_monomial ^ t2_m1_monomial)
            var simd_bool_check_array_2 = (t1_m2_monomial ^ t2_m2_monomial)
            
            if not any(simd_bool_check_array_1) and not any(simd_bool_check_array_2):
                if not bool_first_found:
                    t1_monomial1 = t1_m1_monomial.shift_right[1]()
                    t1_monomial2 = t1_m2_monomial.shift_right[1]()
                    bool_first_found = True

                c += t2_c % p

                set_visited_ptr.add(i2)

            i2 += 1

        if c % p != 0:
            t1_monomial1[0] = c
            t1_monomial2[0] = 1
            lc_output.append(t1_monomial1.join(t1_monomial2))

        i1 += 1
    
    return lc_output

fn str_tensor_adem(t: SIMD[DType.int16, 64]) -> String:
    var t_m1 = tp_m1(t)
    var t_m2 = tp_m2(t)
    
    var t_c = tp_c(t, PARAM_FIXED_PRIME**2)
    
    if check_coeff_zero(t_m1, PARAM_FIXED_PRIME**2) or check_coeff_zero(t_m2, PARAM_FIXED_PRIME**2):
        return String("0")

    return str_adem(t_m1) + " ⊗ " + str_adem(t_m2)

fn str_tensor_adem(lc: List[SIMD[DType.int16, 64]]) -> String:
    var r = String()

    bool_first_entry_found = False
    for t in lc:
        if bool_first_entry_found:
            r += String(" + ")
        
        bool_first_entry_found = True

        r += str_tensor_adem(t[])

    return r

fn str_tensor_milnor(t: SIMD[DType.int16, 64]) -> String:
    var t_m1 = tp_m1(t)
    var t_m2 = tp_m2(t)
    
    var t_c = tp_c(t, PARAM_FIXED_PRIME)
    
    if check_milnor_basis_zero(t_m1, PARAM_FIXED_PRIME) or check_milnor_basis_zero(t_m2, PARAM_FIXED_PRIME):
        return String("0")

    if milnor_basis_deg(t_m1) == 0:
        return String(t_m1[0]) + " ⊗ " + str_milnor(t_m2)
    elif milnor_basis_deg(t_m2) == 0:
        return str_milnor(t_m1) + " ⊗ " + String(t_m2[0])

    return str_milnor(t_m1) + " ⊗ " + str_milnor(t_m2)

fn str_tensor_milnor(lc: List[SIMD[DType.int16, 64]]) -> String: # TODO: repeated code, check for macro-like alternatives
    var r = String()

    bool_first_entry_found = False
    for t in lc:
        if bool_first_entry_found:
            r += String(" + ")
        
        bool_first_entry_found = True

        r += str_tensor_milnor(t[])

    return r

@always_inline
fn check_adem_equal_tensor(t1: SIMD[DType.int16, 64], t2: SIMD[DType.int16, 64], p: Int) -> Bool:
    var t1_m1 = tp_m1(t1)
    var t1_m2 = tp_m2(t1)
    var t2_m1 = tp_m1(t2)
    var t2_m2 = tp_m2(t2)
    
    var t1_c = tp_c(t1, p)
    var t2_c = tp_c(t2, p)

    return (coeff_cmp(t1_c, t2_c, p)) and (check_adem_same_ev_gen(t1_m1, t2_m1) and check_adem_same_ev_gen(t1_m2, t2_m2))

@always_inline
fn check_tensor_zero(t: SIMD[DType.int16, 64], p: Int) -> Bool:
    return (tp_c(t, p) % p == 0)

@always_inline
fn check_adem_same_ev_gen(t1: SIMD[DType.int16, 64], t2: SIMD[DType.int16, 64]) -> Bool:
    var t1_m1 = tp_m1(t1)
    var t1_m2 = tp_m2(t1)
    var t2_m1 = tp_m1(t2)
    var t2_m2 = tp_m2(t2)

    return all(t1_m1 == t2_m1) and all(t1_m2 == t2_m2)

@always_inline
fn check_milnor_same_ev_gen(t1: SIMD[DType.int16, 64], t2: SIMD[DType.int16, 64]) -> Bool:
    var t1_m1 = tp_m1(t1)
    var t1_m2 = tp_m2(t1)
    var t2_m1 = tp_m1(t2)
    var t2_m2 = tp_m2(t2)
    
    return check_milnor_same_ev_gen(t1_m1, t2_m1) and check_milnor_same_ev_gen(t1_m2, t2_m2)

@always_inline
fn tp_expand(t_lc1: List[SIMD[DType.int16, 32]], t_lc2: List[SIMD[DType.int16, 32]]) -> List[SIMD[DType.int16, 64]]:
    var r = List[SIMD[DType.int16, 64]]()

    for m1 in t_lc1:
        for m2 in t_lc2:
            r.append(m1[].join(m2[]))

    return r

@always_inline
fn tp_expand(t_m1: SIMD[DType.int16, 32], t_lc2: List[SIMD[DType.int16, 32]]) -> List[SIMD[DType.int16, 64]]:
    var r = List[SIMD[DType.int16, 64]]()

    for m2 in t_lc2:
        r.append(t_m1.join(m2[]))

    return r

@always_inline
fn tp_expand(t_lc1: List[SIMD[DType.int16, 32]], t_m2: SIMD[DType.int16, 32]) -> List[SIMD[DType.int16, 64]]:
    var r = List[SIMD[DType.int16, 64]]()

    for m1 in t_lc1:
        r.append(m1[].join(t_m2))

    return r

@always_inline
fn lc_check_zero(lc: List[SIMD[DType.int16, 64]], p: Int) -> Bool:
    if len(lc) == 0:
        return True

    var bool_is_zero = True

    for m in lc:
        if not check_tensor_zero(m[], p):
            bool_is_zero = False
            break
        else:
            continue

    return bool_is_zero

@always_inline
fn generic_add(lc1: List[SIMD[DType.int16, 64]], lc2: List[SIMD[DType.int16, 64]], p: Int) -> List[SIMD[DType.int16, 64]]:
    if lc_check_zero(lc1, p):
        return lc2

    if lc_check_zero(lc2, p):
        return lc1

    var list_monomials = List[SIMD[DType.int16, 64]](length=len(lc1) + len(lc2), fill=SIMD[DType.int16, 64]())

    var i = 0
    for _m1 in lc1:
        var m1 = _m1[]
        var bool_monomial_found = False

        for _m2 in lc2:
            var m2 = _m2[]

            if check_adem_same_ev_gen(m1, m2):
                var m_sum = generic_add(m1, m2, p)
                if not lc_check_zero(m_sum, p):
                    for k in range(len(m_sum)):
                        list_monomials[i] = m_sum[k]
                        i += 1

                bool_monomial_found = True
                break

        if not bool_monomial_found:
            list_monomials[i] = m1
            i += 1

    for _m2 in lc2:
        m2 = _m2[]
        var bool_monomial_found = False

        for _m1 in lc1:
            m1 = _m1[]

            if check_adem_same_ev_gen(m1, m2):
                bool_monomial_found = True
                break

        if not bool_monomial_found:
            list_monomials[i] = m2
            i += 1
    
    return list_monomials

fn str_tp_free_adem(t_lc1: List[SIMD[DType.int16, 32]], t_lc2: List[SIMD[DType.int16, 32]]) -> String:
    if lc_check_zero(t_lc1, PARAM_FIXED_PRIME**2) or lc_check_zero(t_lc2, PARAM_FIXED_PRIME**2):
        return String("0")

    return str_adem(t_lc1) + String(" ⊗ ") + str_adem(t_lc2)

fn str_tp_milnor(t_lc1: List[SIMD[DType.int16, 32]], t_lc2: List[SIMD[DType.int16, 32]]) -> String:
    if lc_milnor_basis_check_zero(t_lc1, PARAM_FIXED_PRIME) or lc_milnor_basis_check_zero(t_lc2, PARAM_FIXED_PRIME):
        return String("0")

    return str_milnor(t_lc1) + String(" ⊗ ") + str_milnor(t_lc2)

# TODO: cache/memoization
@always_inline
fn milnor_basis_tensor_product(deg: Int) -> List[Tuple[List[SIMD[DType.int16, 32]], List[SIMD[DType.int16, 32]]]]:
    var r = List[Tuple[List[SIMD[DType.int16, 32]], List[SIMD[DType.int16, 32]]]]()

    for i in range(deg + 1):
        var j = deg - i

        var milnor_basis_1 = milnor_basis(i)
        var milnor_basis_2 = milnor_basis(j)

        if len(milnor_basis_1) > 0 and len(milnor_basis_2) > 0:
            r.append(Tuple[List[SIMD[DType.int16, 32]], List[SIMD[DType.int16, 32]]](milnor_basis_1, milnor_basis_2))

    return r

@always_inline
fn tp_milnor_basis_check_zero(m: SIMD[DType.int16, 64]) -> Bool: # WARNING: hardcoded: p = PARAM_FIXED_PRIME
    if check_milnor_basis_zero(tp_m1(m), PARAM_FIXED_PRIME) or check_milnor_basis_zero(tp_m2(m), PARAM_FIXED_PRIME) or (m[0]*m[32] % PARAM_FIXED_PRIME) == 0:
        return True
    return False

@always_inline
fn milnor_basis_tensor_prod_deg(monomial: SIMD[DType.int16, 64]) -> Int:
    if tp_milnor_basis_check_zero(monomial):
        return -1

    return milnor_basis_deg(tp_m1(monomial)) + milnor_basis_deg(tp_m2(monomial))

#####################################################################
#                               I/O                                 #
#####################################################################

@always_inline
fn bytes2int32(byte_arr: List[SIMD[DType.uint8, 1]]) -> Int32:
    var r = Int32(0)

    for i in range(0, 4):
        r += Int32(byte_arr[i]) * 256**(3 - i) # WARNING: little endian convention

    return r

fn milnor_basis_monomial_to_byte_arr(monomial: SIMD[DType.int16, 32]) raises -> List[InlineArray[SIMD[DType.uint8, 1], DType.int32.sizeof()]]:
    var chunk_byte_arr = List[InlineArray[SIMD[DType.uint8, 1], DType.int32.sizeof()]]()

    chunk_byte_arr.append(Int32(monomial[0]).as_bytes[big_endian=True]())  # alignment

    var monomial_P = milnor_get_P(monomial)
    var monomial_Q = milnor_get_Q(monomial)
    var monomial_P_length = milnor_get_P_length(monomial)
    var monomial_Q_length = milnor_get_Q_length(monomial)

    chunk_byte_arr.append(Int32(monomial_Q_length).as_bytes[big_endian=True]())
    for idx in range(15):
        if idx >= monomial_Q_length:
            break
        var d = Int32(monomial_Q[idx] - 1)
        chunk_byte_arr.append(Int32(d).as_bytes[big_endian=True]())

    chunk_byte_arr.append(Int32(monomial_P_length).as_bytes[big_endian=True]())
    for idx in range(16):
        if idx >= monomial_P_length:
            break
        var d = Int32(monomial_P[idx] - 1)
        chunk_byte_arr.append(Int32(d).as_bytes[big_endian=True]())

    return chunk_byte_arr

fn save_A_img_to_file_write_file_header(cs: ComputationStatus):
    try:
        chunk_hdr_byte_arr = List[InlineArray[SIMD[DType.uint8, 1], DType.int32.sizeof()]]()

        # _ = f.seek(0, SEEK_SET) # TODO: unimplemented in Mojo

        chunk_hdr_byte_arr.append(Int32(cs.curr_basis_deg).as_bytes[big_endian=True]())
        chunk_hdr_byte_arr.append(Int32(cs.curr_basis_tuple_idx).as_bytes[big_endian=True]())
        chunk_hdr_byte_arr.append(Int32(cs.curr_i).as_bytes[big_endian=True]())
        chunk_hdr_byte_arr.append(Int32(cs.curr_j).as_bytes[big_endian=True]())
        chunk_hdr_byte_arr.append(Int32(cs.max_basis_deg).as_bytes[big_endian=True]())
        chunk_hdr_byte_arr.append(Int32(cs.max_rel_deg).as_bytes[big_endian=True]())
        chunk_hdr_byte_arr.append(Int32(cs.current_computation_status).as_bytes[big_endian=True]())

        for _bytes in chunk_hdr_byte_arr:
            f.write_bytes(_bytes[])
    except:
        pass

fn save_A_img_to_file(basis_element: SIMD[DType.int16, 32], list_adem_rel: List[SIMD[DType.int16, 32]], lc_A_img: List[SIMD[DType.int16, 32]], cs: ComputationStatus):
    try:
        var chunk_byte_arr = milnor_basis_monomial_to_byte_arr(basis_element)

        chunk_byte_arr.append(Int32(list_adem_rel[0][0]).as_bytes[big_endian=True]())
        var adem_rel_info = extract_R_B_element(list_adem_rel[0])
        chunk_byte_arr.append(Int32(adem_rel_info[0]).as_bytes[big_endian=True]())
        chunk_byte_arr.append(Int32(adem_rel_info[1]).as_bytes[big_endian=True]())
        chunk_byte_arr.append(Int32(adem_rel_info[2]).as_bytes[big_endian=True]())

        var len_lc_A_img_monomials = len(lc_A_img)
        chunk_byte_arr.append(Int32(len_lc_A_img_monomials).as_bytes[big_endian=True]())
        for _m in lc_A_img:
            chunk_byte_arr += milnor_basis_monomial_to_byte_arr(_m[])

        var b_buff_size = Int32(len(chunk_byte_arr)).as_bytes[big_endian=True]()

        # _ = f.seek(0, SEEK_END) # TODO: not working, unimplemented in Mojo

        f.write_bytes(b_buff_size)

        for _bytes in chunk_byte_arr:
            f.write_bytes(_bytes[])

        # if bool_signal_save_and_exit:
        if exists(CONSTANT_STOP_FILE): # TODO: wait for new Mojo I/O features
            save_A_img_to_file_write_file_header(cs)

            exit(0)
    except:
        pass

fn milnor_basis_read_monomial_from_file(f: FileHandle) raises -> SIMD[DType.int16, 32]:
    var m_c = Int16(bytes2int32(f.read_bytes(4)))
    var m = SIMD[DType.int16, 32](m_c,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)

    var len_Q = bytes2int32(f.read_bytes(4))
    for i in range(len_Q):
        m[1 + Int(i)] = Int16(bytes2int32(f.read_bytes(4))) + 1

    var len_P = bytes2int32(f.read_bytes(4))
    for i in range(len_P):
        m[16 + Int(i)] = Int16(bytes2int32(f.read_bytes(4))) + 1

    return m

fn load_A_img_data() -> ComputationStatus:
    try:
        if not exists(CONSTANT_OUTPUT_FILE):
            return ComputationStatus(0, 0, 0, 0, 0, 0, TAG_COMPUTATIONS_NEW)

        with open(CONSTANT_OUTPUT_FILE, "r") as f:
            var basis_deg = Int(bytes2int32(f.read_bytes(4)))
            var basis_tuple_idx = Int(bytes2int32(f.read_bytes(4)))
            var curr_i = Int(bytes2int32(f.read_bytes(4)))
            var curr_j = Int(bytes2int32(f.read_bytes(4)))
            var max_basis_deg = Int(bytes2int32(f.read_bytes(4)))
            var max_rel_deg = Int(bytes2int32(f.read_bytes(4)))
            var computations_status = Int(bytes2int32(f.read_bytes(4)))

            var bool_keep_reading = True
            while bool_keep_reading:
                var len_chunk = bytes2int32(f.read_bytes(4))
                if len_chunk > 0:  # WARNING: last chunk should end with 0x00000000
                    var m = milnor_basis_read_monomial_from_file(f)

                    var adem_rel_c = Int16(bytes2int32(f.read_bytes(4))) 
                    var adem_rel_i = Int16(bytes2int32(f.read_bytes(4)))
                    var adem_rel_j = Int16(bytes2int32(f.read_bytes(4)))
                    var adem_rel_k = Int16(bytes2int32(f.read_bytes(4)))
                    
                    var number_of_monomials = Int16(bytes2int32(f.read_bytes(4)))

                    var lc_monomials = List[SIMD[DType.int16, 32]]()
                    for i in range(number_of_monomials):
                        lc_monomials.append(milnor_basis_read_monomial_from_file(f))

                    var adem_rel = (
                        rmul(
                            Int(adem_rel_c)
                            * mod_p_inv(-1 % PARAM_FIXED_PRIME, PARAM_FIXED_PRIME**2),
                            mod_p_adem_relation(
                                Int(adem_rel_i), Int(adem_rel_j), Int(adem_rel_k)
                            ),
                            PARAM_FIXED_PRIME**2
                        )
                    )

                    dict_A_img[hash_A_input(m, adem_rel)] = lc_monomials
                    
                else:
                    bool_keep_reading = False

            return ComputationStatus(
                basis_deg,
                basis_tuple_idx,
                curr_i,
                curr_j,
                max_basis_deg,
                max_rel_deg,
                computations_status,
            )
    except:
        pass    

    return ComputationStatus(0,0,0,0,0,0,TAG_COMPUTATIONS_NEW)

fn print_A_img_data():
    print("#" * 120)
    print("[*] dict_A_img:")
    print("#" * 120)
    print("{")
    for _key in dict_A_img.keys():
        try:
            print(
                String("  ") + String(_key[]) + String(": ") + String(str_milnor(dict_A_img[_key[]]))
            )
        except:
            pass
    print("}")
    print("#" * 120)

#####################################################################
#                           Math utils                              #
#####################################################################

@always_inline
fn lexicographic_order_cmp_monomial(read m1: SIMD[DType.int16, 32], read m2: SIMD[DType.int16, 32]) -> Int:
    var m1_Q_len = milnor_get_Q_length(m1)
    var m2_Q_len = milnor_get_Q_length(m2)

    for i in range(min(m1_Q_len, m2_Q_len)):
        if m1[1 + i] == m2[1 + i]:
            continue
        if m1[1 + i] > m2[1 + i]:
            return 1
        if m1[1 + i] < m2[1 + i]:
            return 2

    if m1_Q_len > m2_Q_len:
        return 1
    if m1_Q_len < m2_Q_len:
        return 2

    var m1_P_len = milnor_get_P_length(m1)
    var m2_P_len = milnor_get_P_length(m2)

    for i in range(min(m1_P_len, m2_P_len)):
        if m1[16 + i] == m2[16 + i]:
            continue
        if m1[16 + i] > m2[16 + i]:
            return 1
        if m1[16 + i] < m2[16 + i]:
            return 2

    if m1_P_len > m2_P_len:
        return 1
    if m1_P_len < m2_P_len:
        return 2

    return 0

@always_inline
fn lexicographic_order_64_geq(read t1: SIMD[DType.int16, 64], read t2: SIMD[DType.int16, 64]) -> Bool:
    var t1_m1 = tp_m1(t1) # TODO: memory consumption
    var t1_m2 = tp_m2(t1)
    var t2_m1 = tp_m1(t2)
    var t2_m2 = tp_m2(t2)
    
    var cmp_1 = lexicographic_order_cmp_monomial(t1_m1, t2_m1)
    if cmp_1 == 1:
        return True
    elif cmp_1 == 2:
        return False

    var cmp_2 = lexicographic_order_cmp_monomial(t1_m2, t2_m2)
    if cmp_2 == 1:
        return True
    elif cmp_2 == 2:
        return False

    return True

@always_inline
fn lexicographic_order_64_eq(read t1: SIMD[DType.int16, 64], read t2: SIMD[DType.int16, 64]) -> Bool:
    var t1_m1 = tp_m1(t1) # TODO: memory consumption
    var t1_m2 = tp_m2(t1)
    var t2_m1 = tp_m1(t2)
    var t2_m2 = tp_m2(t2)
    
    var cmp_1 = lexicographic_order_cmp_monomial(t1_m1, t2_m1)

    if cmp_1 != 0:
        return False

    var cmp_2 = lexicographic_order_cmp_monomial(t1_m2, t2_m2)

    if cmp_2 != 0:
        return False

    return True

@always_inline
fn lexicographic_order_64_leq(read t1: SIMD[DType.int16, 64], read t2: SIMD[DType.int16, 64]) -> Bool:
    var t1_m1 = tp_m1(t1) # TODO: memory consumption
    var t1_m2 = tp_m2(t1)
    var t2_m1 = tp_m1(t2)
    var t2_m2 = tp_m2(t2)
    
    var cmp_1 = lexicographic_order_cmp_monomial(t1_m1, t2_m1)
    if cmp_1 == 2:
        return True
    elif cmp_1 == 1:
        return False

    var cmp_2 = lexicographic_order_cmp_monomial(t1_m2, t2_m2)
    if cmp_2 == 2:
        return True
    elif cmp_2 == 1:
        return False

    return True

fn quicksort_hoare(mut list_elements: List[SIMD[DType.int16, 64]], offset_start: Int=0, offset_end: Int=-1):
    var len_list_elements: Int
    var offset_end_effective: Int

    if offset_end == -1:
        len_list_elements = len(list_elements)
        offset_end_effective = len_list_elements - 1
    else:
        offset_end_effective = offset_end

    var i = offset_start - 1
    var j = offset_end_effective + 1
    while True:
        while i < offset_end_effective:
            i += 1
            if lexicographic_order_64_geq(list_elements[i], list_elements[offset_start]):
                break

        while j > offset_start:
            j -= 1
            if lexicographic_order_64_leq(list_elements[j], list_elements[offset_start]):
                break

        if i > j:
            break

        var list_elements_i = list_elements[i]
        list_elements[i] = list_elements[j]
        list_elements[j] = list_elements_i

    if j > offset_start:
        quicksort_hoare(list_elements, offset_start=offset_start, offset_end=j)
    if j + 1 < offset_end_effective:
        quicksort_hoare(list_elements, offset_start=j + 1, offset_end=offset_end_effective)

    return

fn hash_A_input(read m: SIMD[DType.int16, 32], read list_adem_relation: List[SIMD[DType.int16, 32]]) -> String:
    return str_milnor(m) + "@" + str_adem(list_adem_relation) # TODO: better hasheable object implementation

fn trim_zeros(read input_tuple: List[Int16]) -> List[Int16]: # Printing purposes only
    var len_tuple = len(input_tuple)
    if len_tuple == 0:
        return List[Int16]()

    var last_index = 0
    for i in range(len_tuple):
        if input_tuple[len_tuple - 1 - i] != 0:
            last_index = len_tuple - i
            break

    return input_tuple[:last_index]

fn trim_zeros(read input_tuple: List[Int]) -> List[Int]: # Printing purposes only
    var len_tuple = len(input_tuple)
    if len_tuple == 0:
        return List[Int]()

    var last_index = 0
    for i in range(len_tuple):
        if input_tuple[len_tuple - 1 - i] != 0:
            last_index = len_tuple - i
            break

    return input_tuple[:last_index]

@always_inline
fn log_p(n: Int, p: Int) -> Int64:
    return Int64(log(SIMD[DType.float64, 1](n)) // log(SIMD[DType.float64, 1](p)))

# TODO: memoization/cache for large values
@always_inline 
fn factorial(n: Int64) -> Int64:
    var r = 1

    if n > 1:
        for i in range(2, n + 1):
            r *= i

    return r

@always_inline
fn convert_to_mod_p_basis(p: Int, n: Int) -> List[Int]:
    var list_output = List[Int]()

    var max_p_pow = log_p(n, p)

    var n_r = n

    for j in range(max_p_pow + 1):
        var i = max_p_pow - j
        var p_pow = p**i
        
        var n_c_i = n_r // p_pow
        n_r -= Int(n_c_i*p_pow)

        list_output.append(Int(n_c_i))
         
    return list_output

@always_inline
fn mod_p_multinomial_coeff_reduced(p: Int, n: Int, list_lower: List[Int]) -> Int64:
    var r = Int64(1)

    var s = 0
    for k in list_lower:
        s += k[]
        if s >= p:
            return 0

    var n_factorial = factorial(n)

    for m in list_lower:
        r *= factorial(m[])

    return (n_factorial // r) % p

# TODO: memoization/cache for large values
@always_inline
fn mod_p_multinomial_coeff(p: Int, n: Int, list_lower: List[Int]) -> Int64:
    var r = Int64(1)

    var n_p_basis = convert_to_mod_p_basis(p, n)
    var list_lower_p_basis = List[List[Int]]()

    var len_n_p_basis = len(n_p_basis)

    for k in list_lower:
        list_lower_p_basis.append(convert_to_mod_p_basis(p, k[]))

    for i in range(len_n_p_basis):
        var n_i = n_p_basis[len_n_p_basis - 1 - i]
        var list_expansion_fixed_power = List[Int]()

        for entry in list_lower_p_basis:
            var len_entry = len(entry[])
            var s = entry[][len_entry - 1 - i]

            if len_entry <= i:
                s = 0

            list_expansion_fixed_power.append(s)
        
        r = r * mod_p_multinomial_coeff_reduced(p, n_i, list_expansion_fixed_power) % p

    var n_fact = factorial(n)
    for el in list_lower:
        n_fact //= factorial(el[])

    return r

# TODO: memoization/cache for large values
fn bin_coeff(p: Int, n: Int, k: Int) -> Int64:
    """Lucas theorem implementation."""
    if k > n or k < 0:
        return 0
    if k == 0:
        return 1

    var r: Int64

    var max_p_pow = log_p(n, p)

    var n_r = n
    var k_r = k

    r = 1

    for j in range(max_p_pow + 1):
        var i = max_p_pow - j
        var p_pow = p**i
        
        var n_c_i = n_r // p_pow
        n_r -= Int(n_c_i*p_pow)
        
        var k_c_i = k_r // p_pow
        k_r -= Int(k_c_i*p_pow)

        r *= raw_bin_coeff(p, n_c_i, k_c_i) % p

    return r

# TODO: memoization/cache for large values
fn raw_bin_coeff(p: Int64, n: Int64, k: Int64) -> Int64:
    if n == 0 and k == 0:
        return 1
    if k > n:
        return 0
    if k < 0:
        return 0
    return (factorial(n) // (factorial(k) * factorial(n - k))) % p

#####################################################################
#                 Math utils (just for testing purposes)            #
#####################################################################

# TODO: memoization/cache for large values
fn mod_p_inv(n: Int, prime: Int) -> Int:  # TODO: OK for (very) small primes
    for i in range(1, prime):
        if n * i % prime == 1:
            return i
    return 1

#####################################################################
#               Steenrod algebra relevant routines                  #
#####################################################################

# TODO: cache
fn mod_p_adem_relation(a: Int, b: Int, type: Int) -> List[SIMD[DType.int16, 32]]:
    """INPUT: a, b exponents and type: 0, 1 related to the presence of \\beta."""

    var list_operations = List[SIMD[DType.int16, 32]]()
    var sum_upper_i = a // PARAM_FIXED_PRIME

    var c: Int64

    if type == 0:
        list_operations.append(
            SIMD[DType.int16, 32]((PARAM_FIXED_PRIME-1),TAG_MONOMIAL_POWER,a,TAG_MONOMIAL_POWER,b,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
        )

        for i in range(1, sum_upper_i):
            c = (-1) ** (a + i) * bin_coeff(
                PARAM_FIXED_PRIME,
                (PARAM_FIXED_PRIME - 1) * (b - i) - 1,
                a - PARAM_FIXED_PRIME * i,
            ) % PARAM_FIXED_PRIME
            if c != 0:
                list_operations.append(
                    SIMD[DType.int16, 32](Int16(c),TAG_MONOMIAL_POWER,a+b-i,TAG_MONOMIAL_POWER,i,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
                )

        var list_extra_term = Tuple[Int, Int](0,0)

        if sum_upper_i > 0:
            list_extra_term = Tuple[Int, Int](TAG_MONOMIAL_POWER, sum_upper_i)

        var i = 0
        c = (-1) ** (a + i) * bin_coeff(
            PARAM_FIXED_PRIME,
            (PARAM_FIXED_PRIME - 1) * (b - i) - 1,
            a - PARAM_FIXED_PRIME * i,
        ) % PARAM_FIXED_PRIME
        if c != 0:
            list_operations.append( 
                SIMD[DType.int16, 32](Int16(c),TAG_MONOMIAL_POWER,a+b,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
            )

        i = sum_upper_i
        if sum_upper_i > 0:
            c = (-1) ** (a + i) * bin_coeff(
                PARAM_FIXED_PRIME,
                (PARAM_FIXED_PRIME - 1) * (b - i) - 1,
                a - PARAM_FIXED_PRIME * i,
            ) % PARAM_FIXED_PRIME
            if c != 0:
                list_operations.append(
                    SIMD[DType.int16, 32](Int(c),TAG_MONOMIAL_POWER,a+b-sum_upper_i,list_extra_term[0],list_extra_term[1],0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
                )
    else:
        list_operations.append(
            SIMD[DType.int16, 32](PARAM_FIXED_PRIME - 1,TAG_MONOMIAL_POWER,a,TAG_MONOMIAL_BOCKSTEIN,1,TAG_MONOMIAL_POWER,b,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
        )

        for i in range(1, sum_upper_i):
            c = (-1) ** (a + i) * bin_coeff(
                PARAM_FIXED_PRIME,
                (PARAM_FIXED_PRIME - 1) * (b - i),
                a - PARAM_FIXED_PRIME * i, 
            ) % PARAM_FIXED_PRIME
            if c != 0:
                list_operations.append( 
                    SIMD[DType.int16, 32](Int(c),TAG_MONOMIAL_BOCKSTEIN,1,TAG_MONOMIAL_POWER,a+b-i,TAG_MONOMIAL_POWER,i,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
                )

        if sum_upper_i > 0:
            list_extra_term = Tuple[Int, Int](TAG_MONOMIAL_POWER, sum_upper_i)
        else:
            list_extra_term = Tuple[Int, Int](0,0)

        var i = 0
        var c = (-1) ** (a + i) * bin_coeff(
            PARAM_FIXED_PRIME,
            (PARAM_FIXED_PRIME - 1) * (b - i),
            (a - PARAM_FIXED_PRIME * i),

        ) % PARAM_FIXED_PRIME 
        if c != 0:
            list_operations.append(
                SIMD[DType.int16, 32](Int(c),TAG_MONOMIAL_BOCKSTEIN,1,TAG_MONOMIAL_POWER,a+b-i,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
            )
        i = sum_upper_i
        if sum_upper_i > 0:
            c = (-1) ** (a + i) * bin_coeff(
                PARAM_FIXED_PRIME,
                (PARAM_FIXED_PRIME - 1) * (b - i),
                (a - PARAM_FIXED_PRIME * i),
            ) % PARAM_FIXED_PRIME
            if c != 0:
                list_operations.append(
                    SIMD[DType.int16, 32](Int(c),TAG_MONOMIAL_BOCKSTEIN,1,TAG_MONOMIAL_POWER,a+b-i,list_extra_term[0],list_extra_term[1],0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
                )

        #######

        sum_upper_i = (a - 1) // PARAM_FIXED_PRIME

        for i in range(1, sum_upper_i):
            c = (-1) ** (a + i - 1) * bin_coeff(
                PARAM_FIXED_PRIME,
                (PARAM_FIXED_PRIME - 1) * (b - i) - 1,
                a - PARAM_FIXED_PRIME * i - 1,
            ) % PARAM_FIXED_PRIME
            if c != 0:
                list_operations.append(
                    SIMD[DType.int16, 32](Int(c),TAG_MONOMIAL_POWER,a+b-i,TAG_MONOMIAL_BOCKSTEIN,1,TAG_MONOMIAL_POWER,i,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
                )

        if sum_upper_i > 0:
            list_extra_term = Tuple[Int, Int](TAG_MONOMIAL_POWER, sum_upper_i)
        else:
            list_extra_term = Tuple[Int, Int](0,0)

        i = 0
        c = (-1) ** (a + i - 1) * bin_coeff(
            PARAM_FIXED_PRIME,
            (PARAM_FIXED_PRIME - 1) * (b - i) - 1,
            a - PARAM_FIXED_PRIME * i - 1,
        ) % PARAM_FIXED_PRIME
        if c != 0:
            list_operations.append(
                SIMD[DType.int16, 32](Int(c),TAG_MONOMIAL_POWER,a+b-i,TAG_MONOMIAL_BOCKSTEIN,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
            )

        i = sum_upper_i
        if sum_upper_i > 0:
            c = (-1) ** (a + i - 1) * bin_coeff(
                PARAM_FIXED_PRIME,
                (PARAM_FIXED_PRIME - 1) * (b - i) - 1,
                a - PARAM_FIXED_PRIME * i - 1,
            ) % PARAM_FIXED_PRIME
            if c != 0:
                list_operations.append(
                    SIMD[DType.int16, 32](Int(c),TAG_MONOMIAL_POWER,a+b-i,TAG_MONOMIAL_BOCKSTEIN,1,list_extra_term[0],list_extra_term[1],0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)    
                )

    return list_operations

# TODO: memoization
fn mod_p_steenrod_decompose(operation: Int, exp: Int) -> List[SIMD[DType.int16, 64]]:
    var r = List[SIMD[DType.int16, 64]]()
    
    var m1_operation = List[Int16](length=2, fill=0)
    var m2_operation = List[Int16](length=2, fill=0)

    for i in range(0, exp + 1):
        if i == 0:
            m1_operation[0] = 0
            m1_operation[1] = 0
        else:
            m1_operation[0] = operation
            m1_operation[1] = i
        if i == exp:
            m2_operation[0] = 0
            m2_operation[1] = 0
        else:
            m2_operation[0] = operation
            m2_operation[1] = exp - i

        r.append(
            SIMD[DType.int16, 32](1,m1_operation[0],m1_operation[1],0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0).join(
                SIMD[DType.int16, 32](1,m2_operation[0],m2_operation[1],0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
            )
        )

    return r

#TODO: memoization
fn reduced_diagonal0(lc: List[SIMD[DType.int16, 32]]) -> List[SIMD[DType.int16, 64]]:
    """\\tilde{∆} : B_0 ---> B_0 ⊗ B_0."""

    var r = List[SIMD[DType.int16, 64]]()
    var lc_tensors = List[SIMD[DType.int16, 64]]()

    for _m in lc:
        var m = _m[]
        var c = m[0]

        var bool_lc_tensors_init = False
        for i in range(15):
            var operation = m[1 + 2 * i]
            var exp = m[1 + 2 * i + 1]

            if operation == 0 or exp == 0:
                continue

            if i > 0:
                c = 1
            
            var decomposition = rmul(Int(c), mod_p_steenrod_decompose(Int(operation), Int(exp)), PARAM_FIXED_PRIME**2)

            if bool_lc_tensors_init:
                lc_tensors = tp_lc_adem_free_mult(lc_tensors, decomposition, PARAM_FIXED_PRIME**2)
            else:
                lc_tensors = decomposition
                bool_lc_tensors_init = True
 
        r += lc_tensors

    var sgn_m = List[SIMD[DType.int16, 32]](SIMD[DType.int16, 32]())

    sgn_m[0][0] = -1 % PARAM_FIXED_PRIME**2

    r += tp_expand(sgn_m, lc) + tp_expand(lc, sgn_m)

    return tp_lc_simplify_adem(r, PARAM_FIXED_PRIME**2)

@always_inline
fn extract_R_B_element(monomial: SIMD[DType.int16, 32]) -> Tuple[Int16, Int16, Int16]:
    var m = get_monomial(monomial)
    var r = Tuple[Int16, Int16, Int16](0, 0, -1)

    if (
        m[0] == TAG_MONOMIAL_POWER
        and m[2] == TAG_MONOMIAL_BOCKSTEIN
        and m[4] == TAG_MONOMIAL_POWER
    ):
        if m[1] <= PARAM_FIXED_PRIME * m[5]:
            r[0] = m[1]
            r[1] = m[5]
            r[2] = 1
            return r
    elif (
        m[0] == TAG_MONOMIAL_BOCKSTEIN
        and m[2] == TAG_MONOMIAL_POWER
        and m[4] == TAG_MONOMIAL_POWER
    ):
        if m[3] < PARAM_FIXED_PRIME * m[5]:
            r[0] = m[3]
            r[1] = m[5]
            r[2] = 2
            return r

    if m[0] == TAG_MONOMIAL_POWER and m[2] == TAG_MONOMIAL_POWER:
        if m[1] < PARAM_FIXED_PRIME * m[3]:
            r[0] = m[1]
            r[1] = m[3]
            r[2] = 0
            return r

    return r

@always_inline
fn reduced_diagonal0_image_simplify(lc_img: List[SIMD[DType.int16, 64]], left_or_right: Int) -> Tuple[
    List[
        Tuple[
            List[SIMD[DType.int16, 64]],
            Int
        ]
    ],
    List[SIMD[DType.int16, 64]]
]:
    var clean_elements = List[SIMD[DType.int16, 64]]()
    var output_list = List[Tuple[ List[SIMD[DType.int16, 64]], Int ]]()

    var current_monomial = SIMD[DType.int16, 32]()

    for i in range(len(lc_img)):
        var monomial = lc_img[i]
        if left_or_right == 0:
            current_monomial = tp_m1(monomial)
        else:
            current_monomial = tp_m2(monomial)

        var monomial_c = tp_c(monomial, PARAM_FIXED_PRIME**2)

        if (monomial_c) % PARAM_FIXED_PRIME == 0 and monomial_c != 0:
            clean_elements.append(monomial)
            output_list.append(
                Tuple[ List[SIMD[DType.int16, 64]], Int ](List[SIMD[DType.int16, 64]](monomial), MULTIPLE_OF_FIXED_P)
            )
            continue

        var r = extract_R_B_element(current_monomial)
        if r[2] != RET_ERR:
            var bool_first_term = True
            var k = 0

            var relation_type: Int16
            var adem_relation_list_tensor = List[SIMD[DType.int16, 64]]()

            if r[2] == 2:
                relation_type = 0
            else:
                relation_type = r[2]

            for _m_rel in mod_p_adem_relation(Int(r[0]), Int(r[1]), Int(relation_type)):
                var m_rel = _m_rel[]

                if bool_first_term:
                    k = (
                        mod_p_inv(-1 % PARAM_FIXED_PRIME, PARAM_FIXED_PRIME**2)
                        * Int(current_monomial[0])
                    )

                m_rel[0] *= Int16(k)

                var m_rel_tmp: SIMD[DType.int16, 32]
                var t: SIMD[DType.int16, 64]

                if r[2] == 2:
                    m_rel_tmp = adem_free_mult(SIMD[DType.int16, 32](1,TAG_MONOMIAL_BOCKSTEIN,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0), m_rel, PARAM_FIXED_PRIME**2)
                else:
                    m_rel_tmp = m_rel

                if left_or_right == 0:
                    t = m_rel_tmp.join(tp_m2(monomial))
                else:
                    t = tp_m1(monomial).join(m_rel_tmp)

                clean_elements.append(t)
                adem_relation_list_tensor.append(t)

                bool_first_term = False

            output_list.append(
                Tuple[List[SIMD[DType.int16, 64]], Int](adem_relation_list_tensor, left_or_right)
            )

    var tmp_img = lc_img
    for el in clean_elements:
        tmp_img.append(rmul(-1, el[], PARAM_FIXED_PRIME**2))

    var remaining_elements = tp_lc_simplify_adem(tmp_img, PARAM_FIXED_PRIME**2)

    return Tuple[List[Tuple[List[SIMD[DType.int16, 64]], Int]], List[SIMD[DType.int16, 64]]](output_list, remaining_elements)

fn rearrange_img_diag0(lc_img: List[SIMD[DType.int16, 64]]) -> Tuple[List[Tuple[List[SIMD[DType.int16, 64]], Int]], List[SIMD[DType.int16, 64]]]:
    var r = List[Tuple[List[SIMD[DType.int16, 64]], Int]]() 
    var clean_output: List[Tuple[List[SIMD[DType.int16, 64]], Int]]
    var pending_sum = lc_img

    while True:
        clean_output, pending_sum = reduced_diagonal0_image_simplify(pending_sum, 0)

        r += clean_output

        if lc_check_zero(pending_sum, PARAM_FIXED_PRIME**2):
            break

        clean_output, pending_sum = reduced_diagonal0_image_simplify(pending_sum, 1)

        r += clean_output

        if lc_check_zero(pending_sum, PARAM_FIXED_PRIME**2):
            break

    return Tuple[List[Tuple[List[SIMD[DType.int16, 64]], Int]], List[SIMD[DType.int16, 64]]](r, pending_sum)

#TODO: cache/memoization
# x11 better performance than python original implementation
fn reduced_diagonal0_rearranged_img(linear_comb: List[SIMD[DType.int16, 32]]) -> Tuple[List[Tuple[List[SIMD[DType.int16, 64]], Int]], List[SIMD[DType.int16, 64]]]:
    var reduced_diagonal = reduced_diagonal0(linear_comb)
    return rearrange_img_diag0(reduced_diagonal)

#####################################################################
#                   Steenrod Algebra coproduct/diagonal             #
#####################################################################

fn milnor_basis_reduced_coproduct(read m: SIMD[DType.int16, 32]) -> List[SIMD[DType.int16, 64]]:
    var r = milnor_basis_coproduct(m)

    var m_as_lc = List[SIMD[DType.int16, 32]](m)
    var minus_1_as_lc = List[SIMD[DType.int16, 32]](SIMD[DType.int16, 32](PARAM_FIXED_PRIME - 1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0))

    r += tp_expand(m_as_lc, minus_1_as_lc)

    r += tp_expand(minus_1_as_lc, m_as_lc)

    return tp_lc_simplify_milnor(r, PARAM_FIXED_PRIME)

fn milnor_basis_coproduct_positive_deg_partitions(m: SIMD[DType.int16, 32], R_1: List[Int], prefix: List[Bool]=List[Bool](), depth: Int=-1) -> List[SIMD[DType.int16, 64]]:
    var r = List[SIMD[DType.int16, 64]]()
    if depth >= 0:
        for i in range(2):
            var new_prefix = List[Bool](i == 1) + prefix
            r += milnor_basis_coproduct_positive_deg_partitions(
                m, R_1, prefix=new_prefix, depth=depth - 1
            )
    else:
        var S = milnor_get_Q(m)
        var len_S = milnor_get_Q_length(m)

        var S_1 = List[Int]()
        var S_2 = List[Int]()

        if len_S > 0:
            for idx in range(len_S):
                if prefix[idx]:
                    S_1.append(Int(S[idx]))
                else:
                    S_2.append(Int(S[idx]))

        var len_S_1 = len(S_1)

        var m_P = milnor_get_P(m)
        var len_R_1 = len(R_1)
        var R_2 = List[Int](fill=0, length=len_R_1)
        for idx in range(len_R_1):
            R_2[idx] = Int(m_P[idx] - R_1[idx]) + 1

        var sgn = 1
        var list_list_permutation_matrix = List[List[Int]](fill=List[Int](fill=0, length=len_S), length=len_S)

        for i in range(len_S):
            var j = 0
            
            if i < len_S_1:
                for idx in range(len_S):
                    if S[idx] == S_1[i]:
                        j = idx
                        break
            else:
                for idx in range(len_S):
                    if S[idx] == S_2[i - len_S_1]:
                        j = idx
                        break

            list_list_permutation_matrix[j][i] = 1

        for j in range(len_S):
            if list_list_permutation_matrix[j][j] == 1:
                continue
            for i in range(1, len_S - j):  # usual indexing
                if list_list_permutation_matrix[j + i][j] == 0:
                    continue
                else:
                    sgn *= -1

                    for k in range(len_S - j):
                        xchg_val = list_list_permutation_matrix[j][j + k]

                        list_list_permutation_matrix[j][j + k] = (
                            list_list_permutation_matrix[j + i][j + k]
                        )
                        list_list_permutation_matrix[j + i][j + k] = xchg_val

                    break

        var t_m1 = SIMD[DType.int16, 32]()
        var t_m2 = SIMD[DType.int16, 32]()

        t_m1[0] = sgn * m[0] % PARAM_FIXED_PRIME
        t_m2[0] = 1

        t_m1 = milnor_set_Q(t_m1, S_1)
        t_m1 = milnor_set_P(t_m1, R_1)

        t_m2 = milnor_set_Q(t_m2, S_2)
        t_m2 = milnor_set_P(t_m2, R_2)
        
        r.append(t_m1.join(t_m2))

    return r

fn milnor_basis_coproduct_positive_deg(m: SIMD[DType.int16, 32], prefix: List[Int]=List[Int](), depth: Int=-1) -> List[SIMD[DType.int16, 64]]:
    var r = List[SIMD[DType.int16, 64]]()

    if depth > 0:
        for i in range(1, milnor_get_P(m)[depth - 1] + 1): # NEW INDEXING
            r += milnor_basis_coproduct_positive_deg(
                m, prefix=List[Int](Int(i)) + prefix, depth=depth - 1
            )
    else:
        r += milnor_basis_coproduct_positive_deg_partitions(
            m, prefix, depth=milnor_get_Q_length(m) - 1
        )

    return r

fn milnor_basis_coproduct(read m: SIMD[DType.int16, 32]) -> List[SIMD[DType.int16, 64]]:
    var r = List[SIMD[DType.int16, 64]]()

    var deg_m = milnor_basis_deg(m)
    if deg_m == 0:
        var ret_m = SIMD[DType.int16, 64]()
        ret_m[0] = m[0]
        ret_m[32] = 1
        return List[SIMD[DType.int16, 64]](ret_m)
    elif deg_m == -1:
        return List[SIMD[DType.int16, 64]]()
    else:
        r = milnor_basis_coproduct_positive_deg(m, depth=milnor_get_P_length(m))

    return r

#####################################################################
#                       Linear algebra routines                     #
#####################################################################

@always_inline
fn mod_p_rref_parallel_cpu(list_list_m_arg: List[List[Int]], n_rows: Int, n_cols: Int, n_last_cols_to_ignore: Int) -> Tuple[List[List[Int]], List[Tuple[Int, Int, Int]]]:
    
    if n_rows == 0 or n_cols == 0:
        return Tuple[List[List[Int]], List[Tuple[Int, Int, Int]]](List[List[Int]](List[Int]()), List[Tuple[Int, Int, Int]](Tuple[Int, Int, Int](0,0,0)))

    var list_pivots = List[Tuple[Int, Int, Int]]()
    var list_list_m = UnsafePointer[Int16].alloc(len(list_list_m_arg) * len(list_list_m_arg[0]))
    var row_leading_coeff = Int16(0)

    @parameter
    fn copy_matrix_to_pointer_array(i_row: Int):
        for j in range(len(list_list_m_arg[0])):
            list_list_m.store(len(list_list_m_arg[0])*i_row + j, list_list_m_arg[i_row][j])

    parallelize[copy_matrix_to_pointer_array](len(list_list_m_arg))

    var row_top = 0
    for i in range(n_cols - n_last_cols_to_ignore):
        var row_pivot = -1

        for j in range(n_rows):
            var bool_is_pivot = Bool()

            if list_list_m.load(j*len(list_list_m_arg[0]) + i) % PARAM_FIXED_PRIME != 0:
                bool_is_pivot = True
                for r in range(i):
                    if list_list_m.load(j*len(list_list_m_arg[0]) + r) % PARAM_FIXED_PRIME != 0:
                        bool_is_pivot = False
                        break

                if not bool_is_pivot:
                    continue

                row_pivot = j
                break

        if row_pivot >= 0:
            for k in range(n_cols):
                var original_entry_val = list_list_m.load(row_pivot * len(list_list_m_arg[0]) + k)
                list_list_m.store(row_pivot * len(list_list_m_arg[0]) + k, list_list_m.load(row_top * len(list_list_m_arg[0]) + k))
                list_list_m.store(row_top * len(list_list_m_arg[0]) + k, original_entry_val)

            var pivot_val = list_list_m.load(row_top * len(list_list_m_arg[0]) + i)
            var pivot_inv = mod_p_inv(Int(pivot_val), PARAM_FIXED_PRIME)
            var pivot_new_row = row_top
            row_top += 1

            list_pivots.append(Tuple[Int, Int, Int](pivot_new_row, i, pivot_inv))

            for s in range(n_rows - pivot_new_row - 1):
                var bool_leading_coeff_found = False

                for r in range(n_cols - i):
                    if not bool_leading_coeff_found:
                        row_leading_coeff = list_list_m.load((pivot_new_row + 1 + s) * len(list_list_m_arg[0]) + i)
                        bool_leading_coeff_found = True

                    var val_to_substr = (
                        row_leading_coeff
                        * pivot_inv
                        * list_list_m.load(pivot_new_row*len(list_list_m_arg[0]) + i + r)
                    ) % PARAM_FIXED_PRIME

                    var curr_val = list_list_m.load((pivot_new_row + 1 + s)*len(list_list_m_arg[0]) + i + r)
                    list_list_m.store((pivot_new_row + 1 + s)*len(list_list_m_arg[0]) + i + r, curr_val - val_to_substr)

    for tuple_pivot in list_pivots:
        var pivot_row = tuple_pivot[][0]
        var pivot_col = tuple_pivot[][1]
        var pivot_inv = tuple_pivot[][2]

        for j in range(pivot_row):
            row_leading_coeff = list_list_m.load(j * len(list_list_m_arg[0]) + pivot_col)

            for i in range(n_cols - pivot_col):
                var val_to_substr = (
                    row_leading_coeff
                    * pivot_inv
                    * list_list_m.load(pivot_row * len(list_list_m_arg[0]) + pivot_col + i)
                ) % PARAM_FIXED_PRIME

                var curr_val = list_list_m.load(j * len(list_list_m_arg[0]) + pivot_col + i)
                list_list_m.store(j * len(list_list_m_arg[0]) + pivot_col + i, curr_val - val_to_substr)

    var list_list_m_out = List[List[Int]](fill=List[Int](fill=0, length=len(list_list_m_arg[0])), length=len(list_list_m_arg))

    @parameter
    fn copy_matrix_to_list(i_row: Int):
        for j in range(len(list_list_m_arg[0])):
            list_list_m_out[i_row][j] = Int(list_list_m.load(len(list_list_m_arg[0])*i_row + j))

    parallelize[copy_matrix_to_list](len(list_list_m_arg))

    return Tuple[List[List[Int]], List[Tuple[Int, Int, Int]]](list_list_m_out, list_pivots)

@always_inline
fn mod_p_rref(list_list_m_arg: List[List[Int]], n_rows: Int, n_cols: Int, n_last_cols_to_ignore: Int) -> Tuple[List[List[Int]], List[Tuple[Int, Int, Int]]]:
    
    if n_rows == 0 or n_cols == 0:
        return Tuple[List[List[Int]], List[Tuple[Int, Int, Int]]](List[List[Int]](List[Int]()), List[Tuple[Int, Int, Int]](Tuple[Int, Int, Int](0,0,0)))

    var list_pivots = List[Tuple[Int, Int, Int]]()
    var list_list_m = list_list_m_arg
    var row_leading_coeff = 0

    var row_top = 0
    for i in range(n_cols - n_last_cols_to_ignore):
        var row_pivot = -1

        for j in range(n_rows):
            var bool_is_pivot = Bool()

            if list_list_m[j][i] % PARAM_FIXED_PRIME != 0:
                bool_is_pivot = True
                for r in range(i):
                    if list_list_m[j][r] % PARAM_FIXED_PRIME != 0:
                        bool_is_pivot = False
                        break

                if not bool_is_pivot:
                    continue

                row_pivot = j
                break

        if row_pivot >= 0:
            for k in range(n_cols):
                var original_entry_val = list_list_m[row_pivot][k]
                list_list_m[row_pivot][k] = list_list_m[row_top][k]
                list_list_m[row_top][k] = original_entry_val

            var pivot_val = list_list_m[row_top][i]
            var pivot_inv = mod_p_inv(pivot_val, PARAM_FIXED_PRIME)
            var pivot_new_row = row_top
            row_top += 1

            list_pivots.append(Tuple[Int, Int, Int](pivot_new_row, i, pivot_inv))

            for s in range(n_rows - pivot_new_row - 1):
                var bool_leading_coeff_found = False

                for r in range(n_cols - i):
                    if not bool_leading_coeff_found:
                        row_leading_coeff = list_list_m[pivot_new_row + 1 + s][i]
                        bool_leading_coeff_found = True

                    list_list_m[pivot_new_row + 1 + s][i + r] -= (
                        row_leading_coeff
                        * pivot_inv
                        * list_list_m[pivot_new_row][i + r]
                    ) % PARAM_FIXED_PRIME

    for tuple_pivot in list_pivots:
        var pivot_row = tuple_pivot[][0]
        var pivot_col = tuple_pivot[][1]
        var pivot_inv = tuple_pivot[][2]

        for j in range(pivot_row):
            row_leading_coeff = list_list_m[j][pivot_col]

            for i in range(n_cols - pivot_col):
                list_list_m[j][pivot_col + i] -= (
                    row_leading_coeff
                    * pivot_inv
                    * list_list_m[pivot_row][pivot_col + i]
                ) % PARAM_FIXED_PRIME

    return Tuple[List[List[Int]], List[Tuple[Int, Int, Int]]](list_list_m, list_pivots)

fn find_reduced_coproduct_preimg(read list_list_m: List[List[Int]], read list_v: List[Int], degree: Int) -> List[SIMD[DType.int16, 32]]:
    var list_monomials = List[SIMD[DType.int16, 32]]()
    var len_list_v = len(list_v)

    if len_list_v == 0:
        return list_monomials

    var n_cols_system_to_read = len(list_list_m[0]) + 1
    var list_list_augmented_system = List[List[Int]](length=len_list_v, fill=List[Int](fill=0, length=n_cols_system_to_read))

    for i in range(len_list_v):
        for j in range(n_cols_system_to_read):
            if j == n_cols_system_to_read - 1:
                list_list_augmented_system[i][j] = list_v[i]
            else:
                list_list_augmented_system[i][j] = list_list_m[i][j]

    var tuple_rref_output = mod_p_rref( # mod_p_rref_parallel_cpu(
        list_list_augmented_system,
        len_list_v, # n_rows
        n_cols_system_to_read,
        1
    )

    var rref = tuple_rref_output[0] 
    var list_pivots = tuple_rref_output[1]

    var len_list_vect_sol = len(rref[0]) - 1
    var list_vect_sol = List[Int](fill=0, length=len_list_vect_sol)

    for _tuple_pivot in list_pivots:
        var tuple_pivot = _tuple_pivot[]
        var pivot_row = tuple_pivot[0]
        var pivot_col = tuple_pivot[1]
        var pivot_inv = tuple_pivot[2]
        list_vect_sol[pivot_col] = rref[pivot_row][-1]

    for i in range(len_list_vect_sol):
        var monomial_basis = milnor_basis(degree)[i]
        var c = list_vect_sol[i]
        if c % PARAM_FIXED_PRIME != 0:
            monomial_basis[0] = c
            list_monomials.append(monomial_basis)

    return lc_simplify_milnor(list_monomials, PARAM_FIXED_PRIME)

fn milnor_basis_tensor_prod_lc_to_vector(lc_input: List[SIMD[DType.int16, 64]]) -> List[Int]:
    if len(lc_input) == 0:
        return List[Int]()

    var deg = milnor_basis_tensor_prod_deg(lc_input[0])
    var list_compressed_milnor_basis = milnor_basis_tensor_product(deg)
    var len_milnor_basis_tensor_prod_at_deg = 0
    var milnor_basis_tensor_prod_at_deg = List[Tuple[SIMD[DType.int16, 32], SIMD[DType.int16, 32]]](
        fill=Tuple[SIMD[DType.int16, 32], SIMD[DType.int16, 32]](
            SIMD[DType.int16, 32](),
            SIMD[DType.int16, 32]()
        ), 
        length=len_milnor_basis_tensor_prod_at_deg
    )

    for _basis_tuple in list_compressed_milnor_basis:
        for _m1 in _basis_tuple[][0]:
            for _m2 in _basis_tuple[][1]:
                var m1 = _m1[]
                var m2 = _m2[]
                milnor_basis_tensor_prod_at_deg.append(Tuple[SIMD[DType.int16, 32], SIMD[DType.int16, 32]](m1, m2))

    len_milnor_basis_tensor_prod_at_deg = len(milnor_basis_tensor_prod_at_deg)

    var r = List[Int](length=len_milnor_basis_tensor_prod_at_deg, fill=0)

    for _monomial in lc_input:
        var monomial = _monomial[]
        var monomial_normalized = monomial

        monomial_normalized[0] = 1
        monomial_normalized[32] = 1

        var idx_coord = -1

        try:
            idx_coord = dict_monomial_to_coord_pos[String(str_tensor_milnor(monomial_normalized))]

            r[idx_coord] = Int(monomial[0]) * Int(monomial[32]) 
        except:
            for j in range(len_milnor_basis_tensor_prod_at_deg):

                var bool_same_gen_1 = check_milnor_same_ev_gen(tp_m1(monomial), milnor_basis_tensor_prod_at_deg[j][0])
                var bool_same_gen_2 = check_milnor_same_ev_gen(tp_m2(monomial), milnor_basis_tensor_prod_at_deg[j][1])

                if bool_same_gen_1 and bool_same_gen_2:
                    r[j] = Int(monomial[0]) * Int(monomial[32])
                    dict_monomial_to_coord_pos[String(str_tensor_milnor(monomial_normalized))] = j
                    break
    
    return r

fn milnor_basis_reduced_coproduct_as_matrix(deg: Int) -> List[List[Int]]:
    """Milnor basis coproduct matrix at a given degree."""
    
    if len(list_matrices) > deg:
        return list_matrices[deg]

    var milnor_basis_at_deg = milnor_basis(deg)
    var list_compressed_milnor_basis = milnor_basis_tensor_product(deg)
    var len_milnor_basis_tensor_prod_at_deg = 0
    var milnor_basis_tensor_prod_at_deg = List[Tuple[SIMD[DType.int16, 32], SIMD[DType.int16, 32]]](
        fill=Tuple[SIMD[DType.int16, 32], SIMD[DType.int16, 32]](
            SIMD[DType.int16, 32](),
            SIMD[DType.int16, 32]()
        ), 
        length=len_milnor_basis_tensor_prod_at_deg
    )

    for _basis_tuple in list_compressed_milnor_basis:
        for _m1 in _basis_tuple[][0]:
            for _m2 in _basis_tuple[][1]:
                var m1 = _m1[]
                var m2 = _m2[]
                milnor_basis_tensor_prod_at_deg.append(Tuple[SIMD[DType.int16, 32], SIMD[DType.int16, 32]](m1, m2))

    len_milnor_basis_tensor_prod_at_deg = len(milnor_basis_tensor_prod_at_deg)
    
    var len_milnor_basis_at_deg = len(milnor_basis_at_deg)

    var list_list_matr = List[List[Int]](fill=List[Int](fill=0, length=len_milnor_basis_at_deg), length=len_milnor_basis_tensor_prod_at_deg)
    
    # WARNING: the following statements are a significant bottleneck without cache
    
    for i in range(len_milnor_basis_at_deg):
        var lc_basis_img = milnor_basis_reduced_coproduct(milnor_basis_at_deg[i])
        for j in range(len_milnor_basis_tensor_prod_at_deg):
            for _monomial in lc_basis_img:
                var monomial = _monomial[]

                if (
                    check_milnor_same_ev_gen(tp_m1(monomial), milnor_basis_tensor_prod_at_deg[j][0])
                    and 
                    check_milnor_same_ev_gen(tp_m2(monomial), milnor_basis_tensor_prod_at_deg[j][1])
                ):
                    list_list_matr[j][i] = Int(monomial[0]) * Int(monomial[32])
                    break
    
    list_matrices.append(list_list_matr)
    return list_list_matr

fn main():
    pass
