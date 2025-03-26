# Copyright 2018 D-Wave Systems Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import pandas as pd
import numpy as np
from qdeepsdk import QDeepHybridSolver
from circuit_fault_diagnosis.circuits import three_bit_multiplier
from circuit_fault_diagnosis.gates import GATES

_PY2 = sys.version_info.major == 2
if _PY2:
    input = raw_input


def sanitised_input(description, variable, range_):
    start = range_[0]
    stop = range_[-1]

    while True:
        ui = input("Input {:15}({:2} <= {:1} <= {:2}): ".format(description, start, variable, stop))

        try:
            ui = int(ui)
        except ValueError:
            print("Input type must be int")
            continue

        if ui not in range_:
            print("Input must be between {} and {}".format(start, stop))
            continue

        return ui


NUM_READS = 1000

if __name__ == '__main__':
    ####################################################################################################
    # get circuit
    ####################################################################################################
    bqm, labels = three_bit_multiplier()

    ####################################################################################################
    # get input from user
    ####################################################################################################

    fixed_variables = {}

    print("Enter the test conditions")

    A = sanitised_input("multiplier", "A", range(2 ** 3))
    fixed_variables.update(zip(('a2', 'a1', 'a0'), "{:03b}".format(A)))

    B = sanitised_input("multiplicand", "B", range(2 ** 3))
    fixed_variables.update(zip(('b2', 'b1', 'b0'), "{:03b}".format(B)))

    P = sanitised_input("product", "P", range(2 ** 6))
    fixed_variables.update(zip(('p5', 'p4', 'p3', 'p2', 'p1', 'p0'), "{:06b}".format(P)))

    print("\nA   =    {:03b}".format(A))
    print("B   =    {:03b}".format(B))
    print("A*B = {:06b}".format(A * B))
    print("P   = {:06b}\n".format(P))

    fixed_variables = {var: 1 if x == '1' else -1 for (var, x) in fixed_variables.items()}

    # fix variables
    for var, value in fixed_variables.items():
        bqm.fix_variable(var, value)
    # 'aux1' becomes disconnected, so needs to be fixed
    bqm.fix_variable('aux1', 1)  # don't care value

    # find embedding and put on system
    print("Running using QPU\n")
    solver = QDeepHybridSolver()
    solver.token = "mtagdfsplb"  # Replace with your actual token

    # Convert BQM to QUBO format
    qubo, offset = bqm.to_qubo()

    # Create a NumPy matrix from the QUBO dictionary
    n = len(bqm.variables)
    matrix = np.zeros((n, n))
    mapping = {var: idx for idx, var in enumerate(bqm.variables)}

    for (i, j), coeff in qubo.items():
        idx_i = mapping[i]
        idx_j = mapping[j]
        matrix[idx_i, idx_j] = coeff

    # Solve the QUBO using QDeepHybridSolver
    result = solver.solve(matrix)

    # Access the best sample from the result (access solution through 'configuration' key)
    best_sample_vector = result['QdeepHybridSolver']['configuration']

    # Map the solution back to a dictionary using the variable mapping
    best_sample = {list(mapping.keys())[i]: best_sample_vector[i] for i in range(n)}

    ####################################################################################################
    # output results
    ####################################################################################################

    # responses are sorted in order of increasing energy, so the first energy is the minimum
    min_energy = result['QdeepHybridSolver']['energy']

    # Process best sample and remove 'aux' variables
    best_sample_cleaned = {key: value for key, value in best_sample.items() if 'aux' not in key}
    best_sample_cleaned.update(fixed_variables)

    best_results = []
    for sample in [best_sample_cleaned]:
        result = {}
        for gate_type, gates in sorted(labels.items()):
            _, configurations = GATES[gate_type]
            for gate_name, gate in sorted(gates.items()):
                result[gate_name] = 'valid' if tuple(sample[var] for var in gate) in configurations else 'fault'
        best_results.append(result)
    best_results = pd.DataFrame(best_results)

    # at this point, our filtered "best results" all have the same number of faults, so just grab the first one
    num_faults = next(best_results.itertuples()).count('fault')

    best_results = best_results.drop_duplicates().reset_index(drop=True)
    num_ground_states = len(best_results)

    print("The minimum fault diagnosis found is {} faulty component(s)".format(num_faults))
    print("{} distinct fault state(s) with this many faults observed".format(num_ground_states))

    # verbose output
    if len(sys.argv) == 2 and sys.argv[1] == '--verbose':
        pd.set_option('display.width', 120)
        print(best_results)
