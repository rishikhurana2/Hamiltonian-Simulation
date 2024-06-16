from qiskit.quantum_info.operators import Operator
from qiskit import *
from qiskit.visualization import plot_histogram
from qiskit.circuit import Gate
import numpy as np
from qiskit.providers.fake_provider import FakeManilaV2
from qiskit.circuit.library import RYGate
import math
from scipy import linalg

#Hamiltonian Simulation
#Given t, epsilon, and a Hamiltonian H

#setting constants
t = 0.5 #time
epsilon = 0.05 #error bound
r = int((t**2)/(epsilon)) + 1 #number of iterations//error-reduction term
n = 3 #number of qubits
X = np.array([ #Paul X gate (NOT)
   [0 + 0j, 1 + 0j],
   [1 + 0j, 0 + 0j]
])

I = np.array([ #Identity gate
   [1 + 0j, 0 + 0j],
   [0 + 0j, 1 + 0j]
])

intial_state = np.array([ #this is |phi>
   1/2, 0, 1/2, 0, 1/2, 0, 1/2, 0
])

#helper function for computing tensor products
def tensor_product(A, B):
   #Input:  matrix A and matrix B (square matricies)
   #Output: tensor product of A and B
   tensor_product = np.zeros((len(A)*len(B), len(A[0])*len(B[0])), dtype=np.complex_)
   for i in range(len(A)):
      for j in range(len(A[0])):
         a = A[i][j]
         a_x_B = a*B
         for m in range(len(B)):
            for n in range(len(B[0])):
               tensor_product[i*len(B) + m][j*len(B[0]) + n] = a_x_B[m][n]
   return tensor_product

#creating X_1
X_0 = tensor_product(tensor_product(X, I), I)
X_1 = tensor_product(tensor_product(I, X), I)
X_2 = tensor_product(tensor_product(I, I), X)

X_sum = X_0 + X_1 + X_2
e_iHt = linalg.expm(-1j * X_sum * t)


final_state = e_iHt.dot(intial_state)
print("Desired Results: ")
for i in range(len(final_state)):
   print("probability of " + str(bin(i))[2:] + ": " + str(abs(final_state[i])**2))
print()

#One iteration of H_sim
H_sim_one = QuantumCircuit(n, name="One Itr. Hamiltonian Sim.")
for i in range(n):
   H_sim_one.h(i)
   H_sim_one.rz((2*t)/r, i)
   H_sim_one.h(i)
H_sim_one = H_sim_one.to_gate()

#Hamiltonian Sim. is just repated the one iteration r times
H_sim = QuantumCircuit(n, n)
H_sim.reset(range(n))

#START : insert code below to prepare initial state
#i.e. no code here would prepare the initial state |000> = [1 0 0 0 0 0 0 0]
# --> This is the quantum-encoding of the initial_state variable
H_sim.h(0)
H_sim.h(1)
#END

for i in range(r):
   H_sim.append(H_sim_one, range(n))
H_sim.measure(range(n), range(n))

#Test-bench:
backend = Aer.get_backend("aer_simulator") #accurate simualtor using no noise
bit_str_dict = {}
iters = 5000
for i in range(iters):
    job = backend.run(H_sim.decompose(reps = 6), shots=1, memory=True)
    output = job.result().get_memory()[0][::-1] #need to reverse do to formatting (big-endian but needs to be little endian)
    bit_str = output
    if (bit_str in bit_str_dict):
      bit_str_dict[bit_str] += 1
    else:
      bit_str_dict[bit_str] = 1

print("Experimental Results: ")
for key in bit_str_dict:
   print("probability of " + str(key) + ": " + str(bit_str_dict[key]/iters))



