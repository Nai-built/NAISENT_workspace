import math
import random

random.seed(0)

def tanh(x):
    return math.tanh(x)

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def vec_add(a, b):
    return [x + y for x, y in zip(a, b)]

def vec_mul(a, b):
    return [x * y for x, y in zip(a, b)]

def mat_vec_mul(M, v):
    return [
        sum(m_ij * v_j for m_ij, v_j in zip(row, v))
        for row in M
    ]

inputs = [
    [3, 5, 6],
    [3, 5, 5],
    [3, 5, 3],
]

INPUT_VECTOR_SIZE = 3
HIDDEN_STATE_SIZE = 8
CONCAT_VECTOR_SIZE = INPUT_VECTOR_SIZE + HIDDEN_STATE_SIZE  # 11


def random_matrix(num_rows, num_cols):
    return [
        [random.uniform(-1, 1) for _ in range(num_cols)]
        for _ in range(num_rows)
    ]


def random_vector(size):
    return [random.uniform(-1, 1) for _ in range(size)]


# Forget gate parameters
forget_gate_weights = random_matrix(HIDDEN_STATE_SIZE, CONCAT_VECTOR_SIZE)
forget_gate_bias = random_vector(HIDDEN_STATE_SIZE)

# Input gate parameters
input_gate_weights = random_matrix(HIDDEN_STATE_SIZE, CONCAT_VECTOR_SIZE)
input_gate_bias = random_vector(HIDDEN_STATE_SIZE)

# Candidate memory parameters
candidate_memory_weights = random_matrix(HIDDEN_STATE_SIZE, CONCAT_VECTOR_SIZE)
candidate_memory_bias = random_vector(HIDDEN_STATE_SIZE)

# Output gate parameters
output_gate_weights = random_matrix(HIDDEN_STATE_SIZE, CONCAT_VECTOR_SIZE)
output_gate_bias = random_vector(HIDDEN_STATE_SIZE)

# Persistent LSTM state
hidden_state = [0.0] * HIDDEN_STATE_SIZE
cell_state = [0.0] * HIDDEN_STATE_SIZE
print("\n=== LSTM layer (stateful) ===")

for step_index, input_vector in enumerate(inputs):

    # Combine previous hidden state with current input
    concatenated_input = hidden_state + input_vector  # length = 11

    # Forget gate
    forget_gate = [
        sigmoid(weighted_sum + bias)
        for weighted_sum, bias in zip(
            mat_vec_mul(forget_gate_weights, concatenated_input),
            forget_gate_bias
        )
    ]

    # Input gate
    input_gate = [
        sigmoid(weighted_sum + bias)
        for weighted_sum, bias in zip(
            mat_vec_mul(input_gate_weights, concatenated_input),
            input_gate_bias
        )
    ]

    # Candidate memory update
    candidate_memory = [
        tanh(weighted_sum + bias)
        for weighted_sum, bias in zip(
            mat_vec_mul(candidate_memory_weights, concatenated_input),
            candidate_memory_bias
        )
    ]

    # Output gate
    output_gate = [
        sigmoid(weighted_sum + bias)
        for weighted_sum, bias in zip(
            mat_vec_mul(output_gate_weights, concatenated_input),
            output_gate_bias
        )
    ]

    # Update cell (long-term memory)
    cell_state = vec_add(
        vec_mul(forget_gate, cell_state),
        vec_mul(input_gate, candidate_memory)
    )

    # Update hidden state (exposed output)
    hidden_state = vec_mul(
        output_gate,
        [tanh(value) for value in cell_state]
    )

    print(
        f"Step {step_index}: hidden_state =",
        [round(v, 3) for v in hidden_state]
    )