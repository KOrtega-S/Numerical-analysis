# *********************************************************************************
# Implementation of the Simplex Method for Linear Programming
# Course: Numerical Analysis
# Professor: Jose Hernan Ortiz Ocampo
# Student: Karina Ortega Sánchez
# Semester: 2026-2
# **********************************************************************************
import numpy as np

print("=======================================")
print("        SIMPLEX METHOD")
print("=======================================\n")


# ============================================================
# 1. IMPORT DATA
# ============================================================

def import_data():

    while True:

        n = int(input("Number of variables: "))
        m = int(input("Number of constraints: "))

        matrix_a = []

        print("\nEnter the constraint coefficients (matrix A):")

        for i in range(m):

            while True:

                try:
                    row = list(
                        map(
                            float,
                            input(f"Row {i + 1}: ").split(",")
                        )
                    )

                    if len(row) == n:
                        matrix_a.append(row)
                        break

                    print("Incorrect number of values.")

                except ValueError:
                    print("Please enter numerical values separated by commas.")

        while True:

            try:
                vector_b = list(
                    map(
                        float,
                        input("\nRight-hand side vector b: ").split(",")
                    )
                )

                if len(vector_b) == m:
                    break

                print("Must have", m, "values.")

            except ValueError:
                print("Please enter numerical values separated by commas.")

        while True:

            try:
                vector_c = list(
                    map(
                        float,
                        input("Objective coefficients vector c: ").split(",")
                    )
                )

                if len(vector_c) == n:
                    break

                print("Must have", n, "values.")

            except ValueError:
                print("Please enter numerical values separated by commas.")

        A = np.array(matrix_a, dtype=float)
        b = np.array(vector_b, dtype=float)
        c = np.array(vector_c, dtype=float)

        print("\nA =")
        print(A)

        print("\nb =", b)
        print("c =", c)

        confirm = input(
            "\nAre the data correct? (Y/N): "
        ).strip().upper()

        if confirm == "Y":
            return A, b, c


# ============================================================
# 2. VARIABLE NAME
# ============================================================

def variable_name(index, n):

    if index < n:
        return f"x{index + 1}"

    return f"s{index - n + 1}"


# ============================================================
# 3. SIMPLEX REVISED - REDUCED COSTS
# ============================================================

def simplex_cost(A, b, c, show_iterations=True):

    m, n = A.shape

    if n != len(c):
        raise ValueError(
            "The number of variables in A must match "
            "the number of objective coefficients in c."
        )

    A_extended = np.hstack(
        (
            A,
            np.eye(m)
        )
    )

    c_extended = np.hstack(
        (
            c,
            np.zeros(m)
        )
    )

    # Initial basis = slack variables
    basis = list(range(n, n + m))

    non_basis = list(range(n))

    iteration = 0

    while True:

        iteration += 1

        B = A_extended[:, basis]

        B_inv = np.linalg.inv(B)

        xB = B_inv @ b

        if np.any(xB < -1e-8):

            print("\nThe current basis is infeasible.")
            return None

        cB = c_extended[basis]

        y = cB @ B_inv

        reduced_costs = []

        for j in non_basis:

            a_j = A_extended[:, j]

            rc = c_extended[j] - y @ a_j

            reduced_costs.append(rc)

        reduced_costs = np.array(reduced_costs)

        if show_iterations:

            print(
                f"\n--- Iteration {iteration} ---"
            )

            print(
                "Basis:",
                [
                    variable_name(i, n)
                    for i in basis
                ]
            )

            print(
                "Non-basis:",
                [
                    variable_name(i, n)
                    for i in non_basis
                ]
            )

            print("\nx_B =", xB)

            print("\nReduced costs:")

            for j, rc in zip(
                non_basis,
                reduced_costs
            ):

                print(
                    f"{variable_name(j, n)} = {rc:.6f}"
                )

        if np.all(reduced_costs <= 1e-9):
            break


        entering_position = np.argmax(
            reduced_costs
        )

        entering_variable = (
            non_basis[entering_position]
        )

        direction = (
            B_inv @ A_extended[:, entering_variable]
        )

        ratios = np.full(m, np.inf)

        for i in range(m):

            if direction[i] > 1e-9:

                ratios[i] = (
                    xB[i] / direction[i]
                )

        if np.all(np.isinf(ratios)):

            print("\nProblem is unbounded.")
            return None

        leaving_position = np.argmin(ratios)

        leaving_variable = (
            basis[leaving_position]
        )

        if show_iterations:

            print(
                "\nEntering:",
                variable_name(
                    entering_variable,
                    n
                )
            )

            print(
                "Leaving:",
                variable_name(
                    leaving_variable,
                    n
                )
            )

            print(
                "Pivot:",
                direction[leaving_position]
            )

        non_basis[entering_position] = (
            leaving_variable
        )

        basis[leaving_position] = (
            entering_variable
        )

    # ========================================================
    # OPTIMAL SOLUTION
    # ========================================================

    solution_extended = np.zeros(
        n + m
    )

    solution_extended[basis] = xB

    solution = solution_extended[:n]

    Z = c @ solution

    # --------------------------------------------------------
    # Shadow prices
    # --------------------------------------------------------

    cB = c_extended[basis]

    shadow_prices = cB @ B_inv

    if show_iterations:

        print("\n=======================================")
        print("          OPTIMAL SOLUTION")
        print("=======================================")

        for i, value in enumerate(solution):

            print(
                f"x{i + 1} = {value:.6f}"
            )

        print(
            f"\nOptimal objective value = {Z:.6f}"
        )

        print("\nOptimal basis:")

        print(
            [
                variable_name(i, n)
                for i in basis
            ]
        )

        print("\nShadow prices:")

        for i, value in enumerate(shadow_prices):

            print(
                f"y{i + 1} = {value:.6f}"
            )

    # --------------------------------------------------------
    # Return complete solution information
    # --------------------------------------------------------

    return {
        "A": A.copy(),
        "b": b.copy(),
        "c": c.copy(),

        "x": solution.copy(),

        "Z": Z,

        "basis": basis.copy(),

        "non_basis": non_basis.copy(),

        "B": B.copy(),

        "B_inv": B_inv.copy(),

        "xB": xB.copy(),

        "shadow_prices": shadow_prices.copy(),

        "reduced_costs": reduced_costs.copy(),

        "A_extended": A_extended.copy(),

        "c_extended": c_extended.copy()
    }


# ============================================================
# 4. SHOW CURRENT SOLUTION
# ============================================================

def show_current_solution(result):

    if result is None:

        print(
            "\nNo solution available."
        )

        return

    n = len(result["c"])

    print("\n=======================================")
    print("          CURRENT SOLUTION")
    print("=======================================")

    print("\nOptimal solution:")

    for i, value in enumerate(result["x"]):

        print(
            f"x{i + 1} = {value:.6f}"
        )

    print(
        f"\nOptimal objective value = "
        f"{result['Z']:.6f}"
    )

    print("\nOptimal basis:")

    print(
        [
            variable_name(i, n)
            for i in result["basis"]
        ]
    )

    print("\nShadow prices:")

    for i, value in enumerate(
        result["shadow_prices"]
    ):

        print(
            f"y{i + 1} = {value:.6f}"
        )


# ============================================================
# 5. SENSITIVITY ANALYSIS - B
# ============================================================

def sensitivity_b(result):

    if result is None:

        print("\nNo solution available.")
        return

    A = result["A"]
    b = result["b"]

    B_inv = result["B_inv"]
    xB = result["xB"]

    m, n = A.shape

    print("\n=======================================")
    print("     SENSITIVITY ANALYSIS - B")
    print("=======================================")

    print("\nThe current basis is:")

    print(
        [
            variable_name(i, n)
            for i in result["basis"]
        ]
    )

    print("\nB^-1 =")
    print(B_inv)

    print("\n sensitivity:")

    for k in range(m):

        column = B_inv[:, k]

        delta_min = -np.inf
        delta_max = np.inf

        for i in range(m):

            value = column[i]

            if value > 1e-10:

                upper = (
                    xB[i] / value
                )

                delta_max = min(
                    delta_max,
                    upper
                )

            elif value < -1e-10:

                lower = (
                    xB[i] / value
                )

                delta_min = max(
                    delta_min,
                    lower
                )

        current = b[k]

        # ----------------------------------------------------
        # Absolute allowable range
        # ----------------------------------------------------

        lower_value = (
            current + delta_min
            if not np.isinf(delta_min)
            else -np.inf
        )

        upper_value = (
            current + delta_max
            if not np.isinf(delta_max)
            else np.inf
        )

        # ----------------------------------------------------
        # Shadow price
        # ----------------------------------------------------

        shadow_price = (
            result["shadow_prices"][k]
        )

        print(
            f"\nConstraint {k + 1}"
        )

        print(
            f"Current b{k + 1} = "
            f"{current:.6f}"
        )

        print(
            f"Shadow price = "
            f"{shadow_price:.6f}"
        )

        if np.isinf(delta_min):
            print(
                "Allowable decrease = Infinity"
            )
        else:
            print(
                f"Allowable decrease = "
                f"{abs(delta_min):.6f}"
            )

        if np.isinf(delta_max):
            print(
                "Allowable increase = Infinity"
            )
        else:
            print(
                f"Allowable increase = "
                f"{delta_max:.6f}"
            )

        print(
            "Allowable range for b"
            f"{k + 1}: "
            f"[{lower_value:.6f}, "
            f"{upper_value:.6f}]"
        )


# ============================================================
# 6. SENSITIVITY ANALYSIS - OBJECTIVE
# ============================================================

def sensitivity_objective(result):

    if result is None:

        print("\nNo solution available.")
        return

    A = result["A"]
    c = result["c"]

    basis = result["basis"]
    non_basis = result["non_basis"]

    B_inv = result["B_inv"]

    m, n = A.shape

    A_extended = result["A_extended"]
    c_extended = result["c_extended"]

    cB = c_extended[basis]

    y = cB @ B_inv

    print("\n=======================================")
    print(" SENSITIVITY ANALYSIS - OBJECTIVE (c)")
    print("=======================================")

    print("\nShadow prices:")

    for i, value in enumerate(y):

        print(
            f"y{i + 1} = {value:.6f}"
        )

    for j in range(n):

        if j not in basis:

            a_j = A[:, j]

            current_c = c[j]

            maximum_c = y @ a_j

            allowable_increase = (
                maximum_c - current_c
            )

            # For a maximization problem:
            #
            # c_j <= y^T a_j

            print(
                f"\nx{j + 1}:"
            )

            print(
                f"Current c{j + 1} = "
                f"{current_c:.6f}"
            )

            # Safely extract reduced cost and print
            reduced_cost_j = result['reduced_costs'][
                non_basis.index(j)
            ]
            print(f"Reduced cost = {reduced_cost_j:.6f}")

            print(
                "Allowable decrease = Infinity"
            )

            print(
                f"Allowable increase = "
                f"{max(0, allowable_increase):.6f}"
            )

            print(
                f"Allowable range for c{j + 1}: "
                f"(-Infinity, "
                f"{maximum_c:.6f}]"
            )

    # ========================================================
    # BASIC VARIABLES
    # ========================================================

    print("\n---------------------------------------")
    print("BASIC VARIABLES")
    print("---------------------------------------")

    # Current reduced costs for all nonbasic variables
    reduced_dict = {}

    for position, j in enumerate(non_basis):

        reduced_dict[j] = (
            result["reduced_costs"][position]
        )

    for position, j in enumerate(basis):

        # Slack variables do not have
        # objective coefficients entered by user

        if j >= n:
            continue

        current_c = c[j]

        delta_min = -np.inf
        delta_max = np.inf

        row = B_inv[position, :]

        for k in non_basis:

            a_k = A_extended[:, k]

            q = row @ a_k

            r_k = reduced_dict[k]

            # r_new = r_k - delta*q
            #
            # For MAX:
            # r_new <= 0

            if q > 1e-10:

                lower = r_k / q

                delta_min = max(
                    delta_min,
                    lower
                )

            elif q < -1e-10:

                upper = r_k / q

                delta_max = min(
                    delta_max,
                    upper
                )

        lower_value = (
            current_c + delta_min
            if not np.isinf(delta_min)
            else -np.inf
        )

        upper_value = (
            current_c + delta_max
            if not np.isinf(delta_max)
            else np.inf
        )

        print(
            f"\nx{j + 1}:"
        )

        print(
            f"Current c{j + 1} = "
            f"{current_c:.6f}"
        )

        if np.isinf(delta_min):
            print(
                "Allowable decrease = Infinity"
            )
        else:
            print(
                f"Allowable decrease = "
                f"{abs(delta_min):.6f}"
            )

        if np.isinf(delta_max):
            print(
                "Allowable increase = Infinity"
            )
        else:
            print(
                f"Allowable increase = "
                f"{delta_max:.6f}"
            )

        print(
            f"Allowable range for c{j + 1}: "
            f"[{lower_value:.6f}, "
            f"{upper_value:.6f}]"
        )


# ============================================================
# 7. PERTURB b
# ============================================================

def perturb(result):

    if result is None:

        print("\nSolve a problem first.")
        return result

    b_new = result["b"].copy()

    m = len(b_new)

    print("\n=======================================")
    print("             PERTURB (b)")
    print("=======================================")

    print("\nCurrent b:")

    print(b_new)

    print("\nSelect the constraint to perturb:")

    for i in range(m):

        print(
            f"{i + 1}. b{i + 1} "
            f"(current = {b_new[i]:.6f})"
        )

    print(
        f"{m + 1}. Cancel"
    )

    try:

        option = int(
            input("\nSelect an option: ")
        )

    except ValueError:

        print("\nInvalid option.")
        return result

    if option == m + 1:
        return result

    if option < 1 or option > m:

        print("\nInvalid option.")
        return result

    k = option - 1

    try:

        new_value = float(
            input(
                f"\nEnter new value for b{k + 1}: "
            )
        )

    except ValueError:

        print("\nInvalid value.")
        return result

    old_value = b_new[k]

    b_new[k] = new_value

    print("\n---------------------------------------")
    print("PERTURBATION")
    print("---------------------------------------")

    print(
        f"b{k + 1}: "
        f"{old_value:.6f} -> "
        f"{new_value:.6f}"
    )

    print(
        f"Delta b{k + 1} = "
        f"{new_value - old_value:.6f}"
    )

    # --------------------------------------------------------
    # Solve perturbed problem
    # --------------------------------------------------------

    new_result = simplex_cost(
        result["A"],
        b_new,
        result["c"],
        show_iterations=False
    )

    if new_result is None:

        print(
            "\nThe perturbed problem could not be solved."
        )

        return result

    # --------------------------------------------------------
    # Compare bases
    # --------------------------------------------------------

    original_basis = result["basis"]

    new_basis = new_result["basis"]

    same_basis = (
        original_basis == new_basis
    )

    print("\n=======================================")
    print("       PERTURBED SOLUTION")
    print("=======================================")

    print("\nOriginal solution:")

    print(
        result["x"]
    )

    print(
        f"Original Z = "
        f"{result['Z']:.6f}"
    )

    print("\nPerturbed solution:")

    print(
        new_result["x"]
    )

    print(
        f"Perturbed Z = "
        f"{new_result['Z']:.6f}"
    )

    print(
        f"\nChange in Z = "
        f"{new_result['Z'] - result['Z']:.6f}"
    )

    print("\nOriginal basis:")

    print(
        [
            variable_name(i, len(result["c"]))
            for i in original_basis
        ]
    )

    print("\nNew basis:")

    print(
        [
            variable_name(i, len(result["c"]))
            for i in new_basis
        ]
    )

    if same_basis:

        print(
            "\nThe optimal basis DID NOT change."
        )

    else:

        print(
            "\nThe optimal basis CHANGED."
        )

    # --------------------------------------------------------
    # Save perturbed solution
    # --------------------------------------------------------

    return new_result


# ============================================================
# 8. PERTURB OBJECTIVE c
# ============================================================

def perturb_objective(result):

    if result is None:

        print("\nSolve a problem first.")
        return result

    c_new = result["c"].copy()

    n = len(c_new)

    print("\n=======================================")
    print("       PERTURB OBJECTIVE (c)")
    print("=======================================")

    print("\nCurrent c:")

    print(c_new)

    print("\nSelect the objective coefficient:")

    for i in range(n):

        print(
            f"{i + 1}. c{i + 1} "
            f"(current = {c_new[i]:.6f})"
        )

    print(
        f"{n + 1}. Cancel"
    )

    try:

        option = int(
            input("\nSelect an option: ")
        )

    except ValueError:

        print("\nInvalid option.")
        return result

    if option == n + 1:
        return result

    if option < 1 or option > n:

        print("\nInvalid option.")
        return result

    k = option - 1

    try:

        new_value = float(
            input(
                f"\nEnter new value for c{k + 1}: "
            )
        )

    except ValueError:

        print("\nInvalid value.")
        return result

    old_value = c_new[k]

    c_new[k] = new_value

    print("\n---------------------------------------")
    print("PERTURBATION")
    print("---------------------------------------")

    print(
        f"c{k + 1}: "
        f"{old_value:.6f} -> "
        f"{new_value:.6f}"
    )

    print(
        f"Delta c{k + 1} = "
        f"{new_value - old_value:.6f}"
    )

    # --------------------------------------------------------
    # Solve perturbed problem
    # --------------------------------------------------------

    new_result = simplex_cost(
        result["A"],
        result["b"],
        c_new,
        show_iterations=False
    )

    if new_result is None:

        print(
            "\nThe perturbed problem could not be solved."
        )

        return result

    # --------------------------------------------------------
    # Compare bases
    # --------------------------------------------------------

    original_basis = result["basis"]

    new_basis = new_result["basis"]

    same_basis = (
        original_basis == new_basis
    )

    print("\n=======================================")
    print("       PERTURBED SOLUTION")
    print("=======================================")

    print("\nOriginal solution:")

    print(
        result["x"]
    )

    print(
        f"Original Z = "
        f"{result['Z']:.6f}"
    )

    print("\nPerturbed solution:")

    print(
        new_result["x"]
    )

    print(
        f"Perturbed Z = "
        f"{new_result['Z']:.6f}"
    )

    print(
        f"\nChange in Z = "
        f"{new_result['Z'] - result['Z']:.6f}"
    )

    print("\nOriginal basis:")

    print(
        [
            variable_name(i, len(result["c"]))
            for i in original_basis
        ]
    )

    print("\nNew basis:")

    print(
        [
            variable_name(i, len(result["c"]))
            for i in new_basis
        ]
    )

    if same_basis:

        print(
            "\nThe optimal basis DID NOT change."
        )

    else:

        print(
            "\nThe optimal basis CHANGED."
        )

    return new_result


# ============================================================
# 9. COMPLETE SENSITIVITY SUMMARY
# ============================================================

def sensitivity_summary(result):

    if result is None:

        print("\nNo solution available.")
        return

    print("\n=======================================")
    print("      COMPLETE SENSITIVITY SUMMARY")
    print("=======================================")

    show_current_solution(result)

    sensitivity_b(result)

    sensitivity_objective(result)


# ============================================================
# 10. COMPARE ORIGINAL VS CURRENT
# ============================================================

def compare_results(original, current):

    if original is None or current is None:

        print(
            "\nThere is no original or current solution "
            "available for comparison."
        )

        return

    n = len(original["c"])

    print("\n=======================================")
    print("       ORIGINAL vs CURRENT")
    print("=======================================")

    print("\n---------- b ----------")

    print(
        "Original:",
        original["b"]
    )

    print(
        "Current :",
        current["b"]
    )

    print(
        "Change  :",
        current["b"] - original["b"]
    )

    print("\n---------- Objective c ----------")

    print(
        "Original:",
        original["c"]
    )

    print(
        "Current :",
        current["c"]
    )

    print(
        "Change  :",
        current["c"] - original["c"]
    )

    print("\n---------- Solution ----------")

    print(
        "Original x:",
        original["x"]
    )

    print(
        "Current  x:",
        current["x"]
    )

    print(
        "Change   :",
        current["x"] - original["x"]
    )

    print("\n---------- Objective value ----------")

    print(
        f"Original Z = "
        f"{original['Z']:.6f}"
    )

    print(
        f"Current Z  = "
        f"{current['Z']:.6f}"
    )

    print(
        f"Delta Z    = "
        f"{current['Z'] - original['Z']:.6f}"
    )

    original_basis = original["basis"]

    current_basis = current["basis"]

    print("\n---------- Basis ----------")

    print(
        "Original:",
        [
            variable_name(i, n)
            for i in original_basis
        ]
    )

    print(
        "Current :",
        [
            variable_name(i, n)
            for i in current_basis
        ]
    )

    if original_basis == current_basis:

        print(
            "\nThe optimal basis remained unchanged."
        )

    else:

        print(
            "\nThe optimal basis changed."
        )


# ============================================================
# 11. MAIN MENU
# ============================================================

def main():

    # Current solution
    current_result = None

    # Original solution
    original_result = None

    while True:

        print("\n")
        print("=======================================")
        print("          SIMPLEX METHOD")
        print("=======================================")

        print(
            "\n1. Solve class example"
        )

        print(
            "2. Solve second example"
        )

        print(
            "3. Enter a new problem"
        )

        print(
            "4. Show current solution"
        )

        print(
            "5. Perturb (b)"
        )

        print(
            "6. Perturb Objective coefficients (c)"
        )

        print(
            "7. Sensitivity analysis (b)"
        )

        print(
            "8. Sensitivity analysis - Objective (c)"
        )

        print(
            "9. Show complete sensitivity summary"
        )

        print(
            "10. Compare original vs current"
        )

        print(
            "11. Restore original problem"
        )

        print(
            "12. Exit"
        )

        option = input(
            "\nSelect an option (1-12): "
        ).strip()

        # ====================================================
        # CLASS EXAMPLE
        # ====================================================

        if option == "1":

            A = np.array(
                [
                    [3, 1],
                    [1, 1],
                    [1, 3]
                ],
                dtype=float
            )

            b = np.array(
                [300, 140, 300],
                dtype=float
            )

            c = np.array(
                [2000, 1000],
                dtype=float
            )

            print("\nA =")
            print(A)

            print("\nb =", b)
            print("c =", c)

            result = simplex_cost(
                A,
                b,
                c
            )

            if result is not None:

                original_result = result
                current_result = result

        # ====================================================
        # SECOND EXAMPLE
        # ====================================================

        elif option == "2":

            A = np.array(
                [
                    [2, 1, 3],
                    [1, 2, 1],
                    [3, 1, 2]
                ],
                dtype=float
            )

            b = np.array(
                [300, 240, 360],
                dtype=float
            )

            c = np.array(
                [2500, 2000, 2200],
                dtype=float
            )

            print("\nA =")
            print(A)

            print("\nb =", b)
            print("c =", c)

            result = simplex_cost(
                A,
                b,
                c
            )

            if result is not None:

                original_result = result
                current_result = result

        # ====================================================
        # USER PROBLEM
        # ====================================================

        elif option == "3":

            A, b, c = import_data()

            result = simplex_cost(
                A,
                b,
                c
            )

            if result is not None:

                original_result = result
                current_result = result

        # ====================================================
        # CURRENT SOLUTION
        # ====================================================

        elif option == "4":

            show_current_solution(
                current_result
            )

        # ====================================================
        # PERTURB
        # ===================================================

        elif option == "5":

            if current_result is None:

                print(
                    "\nSolve a problem first."
                )

            else:

                current_result = perturb(
                    current_result
                )

        # ====================================================
        # PERTURB OBJECTIVE
        # ====================================================

        elif option == "6":

            if current_result is None:

                print(
                    "\nSolve a problem first."
                )

            else:

                current_result = perturb_objective(
                    current_result
                )

        # ====================================================
        # B SENSITIVITY
        # ====================================================

        elif option == "7":

            sensitivity_b(
                current_result
            )

        # ====================================================
        # OBJECTIVE SENSITIVITY
        # ====================================================

        elif option == "8":

            sensitivity_objective(
                current_result
            )

        # ====================================================
        # COMPLETE SENSITIVITY
        # ====================================================

        elif option == "9":

            sensitivity_summary(
                current_result
            )

        # ====================================================
        # COMPARE
        # ====================================================

        elif option == "10":

            compare_results(
                original_result,
                current_result
            )

        # ====================================================
        # RESTORE
        # ====================================================

        elif option == "11":

            if original_result is None:

                print(
                    "\nThere is no original problem stored."
                )

            else:

                current_result = simplex_cost(
                    original_result["A"],
                    original_result["b"],
                    original_result["c"],
                    show_iterations=False
                )

                print(
                    "\nOriginal problem restored."
                )

                show_current_solution(
                    current_result
                )

        # ====================================================
        # EXIT
        # ====================================================

        elif option == "12":

            print(
                "\nProgram finished."
            )

            break

        else:

            print(
                "\nInvalid option."
            )


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    main()
