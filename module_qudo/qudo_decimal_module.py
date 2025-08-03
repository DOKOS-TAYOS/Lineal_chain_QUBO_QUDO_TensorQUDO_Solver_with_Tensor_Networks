import numpy as np
from decimal import Decimal
from auxiliary_functions import last_variable_determination_QUDO

def tensor_initial_generator_QUDO_decimal(position: int, solution: list, w_matrix: dict, d_vector: list, dimensions: np.ndarray, tau: Decimal) -> list:
    """
    Generate initial tensor for QUDO problem at given position.
    
    Args:
        position (int): Current position in solution vector
        solution (list): Partial solution vector
        w_matrix (dict): Sparse dictionary containing quadratic interactions
        d_vector (list): Linear term vector of length N containing linear costs
        dimensions (np.ndarray): List of dimensions for each variable
        tau (Decimal): Evolution parameter for optimization
        
    Returns:
        list: Initial tensor of shape (dimensions[position]), normalized
    """
    # Initialize tensor with zeros
    tensor = [Decimal('0') for _ in range(dimensions[position])]

    # Calculate tensor elements
    MAX_ATTEMPTS = 5
    attempt = 0
    success = False
    
    while not success and attempt < MAX_ATTEMPTS:
        try:
            for i in range(dimensions[position]):
                # Compute total energy contribution
                exp_value = d_vector[position] * Decimal(str(i))  # Linear term
                
                # Add self-interaction if exists
                if (position, position) in w_matrix:
                    exp_value += w_matrix[(position, position)] * Decimal(str(i**2))
                
                # Add interaction with previous variable if not first position
                if position >= 1 and (position-1, position) in w_matrix:
                    exp_value += w_matrix[(position-1, position)] * Decimal(str(solution[position-1])) * Decimal(str(i))
                    
                # Set diagonal element with evolution factor
                tensor[i] = (-tau * exp_value).exp()
            success = True
            
        except:
            # If overflow detected, scale down tau to avoid numerical issues
            tau = tau / exp_value
            attempt += 1
            
    if not success:
        raise ValueError("Could not compute tensor values without overflow after maximum attempts")
    # Efficient norm calculation for diagonal matrix
    try:
        norm = sum(elem * elem for elem in tensor).sqrt()
    except:
        norm = Decimal('1e+1000')
    
    # Normalize tensor elements in place
    return [elem/norm for elem in tensor]

def tensor_intermediate_generator_QUDO_decimal(position: int, w_matrix: dict, d_vector: list, dimensions: np.ndarray, tau: Decimal) -> list:
    """
    Generate intermediate tensor for QUDO problem at given position.
    
    Args:
        position (int): Current position in solution vector
        w_matrix (dict): Sparse dictionary containing quadratic interactions
        d_vector (list): Linear term vector of length N containing linear costs
        dimensions (np.ndarray): List of dimensions for each variable
        tau (Decimal): Evolution parameter for optimization
        
    Returns:
        list: Normalized intermediate tensor of shape (dimensions[position-1], dimensions[position])
    """
    # Initialize tensor with zeros
    tensor = [[Decimal('0') for _ in range(dimensions[position])] for _ in range(dimensions[position-1])]
    
    success = False
    attempt = 0
    MAX_ATTEMPTS = 10
    
    while not success and attempt < MAX_ATTEMPTS:
        try:
            # Calculate tensor elements
            for j in range(dimensions[position]):
                for i in range(dimensions[position-1]):
                    # Compute total energy contribution
                    exp_value = d_vector[position] * Decimal(str(j))  # Linear term
                    
                    # Add self-interaction if exists
                    if (position, position) in w_matrix:
                        exp_value += w_matrix[(position, position)] * Decimal(str(j**2))
                    
                    # Add interaction with previous if exists
                    if (position-1, position) in w_matrix:
                        exp_value += w_matrix[(position-1, position)] * Decimal(str(i*j))
                        
                    # Set element with evolution factor
                    tensor[i][j] = (-tau * exp_value).exp()
            success = True
            
        except:
            # If overflow detected, scale down tau to avoid numerical issues
            tau = tau / exp_value
            attempt += 1
            
    if not success:
        raise ValueError("Could not compute tensor values without overflow after maximum attempts")

    try:
        norm = sum(elem * elem for row in tensor for elem in row).sqrt()
    except:
        norm = Decimal('1e+1000')
    return [[elem/norm for elem in row] for row in tensor]

def tensor_final_generator_QUDO_decimal(position: int, w_matrix: dict, d_vector: list, dimensions: np.ndarray, tau: Decimal) -> list:
    """
    Generate final tensor for QUDO problem at given position.
    
    Args:
        position (int): Current position in solution vector
        w_matrix (dict): Sparse dictionary containing quadratic interactions
        d_vector (list): Linear term vector of length N containing linear costs
        dimensions (np.ndarray): List of dimensions for each variable
        tau (Decimal): Evolution parameter for optimization
        
    Returns:
        list: Normalized final tensor of shape (dimensions[position-1])
    """
    # Initialize tensor with zeros
    tensor = [Decimal('0') for _ in range(dimensions[position-1])]
    
    success = False
    attempt = 0
    MAX_ATTEMPTS = 10

    while not success and attempt < MAX_ATTEMPTS:
        try:
            # Calculate tensor elements by summing over last variable
            for i in range(dimensions[position-1]):
                tensor[i] = Decimal('0')
                for j in range(dimensions[position]):
                    # Compute total energy contribution
                    exp_value = d_vector[position] * Decimal(str(j))  # Linear term
                    
                    # Add self-interaction if exists
                    if (position, position) in w_matrix:
                        exp_value += w_matrix[(position, position)] * Decimal(str(j**2))
                    
                    # Add interaction with previous if exists
                    if (position-1, position) in w_matrix:
                        exp_value += w_matrix[(position-1, position)] * Decimal(str(i*j))
                        
                    # Add contribution to element with evolution factor
                    tensor[i] += (-tau * exp_value).exp()
            success = True
            
        except:
            # If overflow detected, scale down tau to avoid numerical issues
            tau = tau / exp_value
            attempt += 1
            
    if not success:
        raise ValueError("Could not compute tensor values without overflow after maximum attempts")

    try:
        norm = sum(elem * elem for elem in tensor).sqrt()
    except:
        norm = Decimal('1e+1000')
    
    return [elem/norm for elem in tensor]


def generate_tensor_network_QUDO_decimal(w_matrix: dict, d_vector: list, dimensions: np.ndarray, tau: Decimal) -> list:
    """
    Generates a tensor network for solving a Quadratic Unconstrained Discrete Optimization problem.
    
    Args:
        w_matrix: Dictionary representing quadratic interactions between variables, with tuples (i,j) as keys
        d_vector: Vector representing linear costs
        dimensions: List containing the dimension of each variable
        tau: Imaginary time for optimization
        
    Returns:
        List of tensors representing the tensor network
    """
    n_variables = len(dimensions)
    tensor_list = []
    
    # Create initial measurement tensor
    tensor_list.append(tensor_initial_generator_QUDO_decimal(0, [], w_matrix, d_vector, dimensions, tau))

    # Create intermediate tensors 
    for position in range(1, n_variables-1):
        tensor_list.append(tensor_intermediate_generator_QUDO_decimal(position, w_matrix, d_vector, dimensions, tau))
    
    # Create final tensor
    tensor_list.append(tensor_final_generator_QUDO_decimal(n_variables-1, w_matrix, d_vector, dimensions, tau))

    return tensor_list


def contraction_tensors_decimal(tensor_list: list) -> list:
    """
    Contracts a list of tensors sequentially and returns intermediate results.
    
    Args:
        tensor_list: List of tensors to contract
        
    Returns:
        List of intermediate contracted tensors, normalized at each step
    """
    if not tensor_list:
        return []
        
    # Initialize with last tensor
    vector = tensor_list[-1]
    intermediate_tensors = [vector.copy()]
    
    # Contract tensors from right to left
    for tensor in reversed(tensor_list[1:-1]):
        # Contract tensors
        output_vector = [sum(tensor[i][j] * vector[j] for j in range(len(vector))) 
                        for i in range(len(tensor))]
        
        # Normalize using decimal
        try:
            norm = sum(elem * elem for elem in output_vector).sqrt()
        except:
            norm = Decimal('1e+1000')
        vector = [elem/norm for elem in output_vector]
        
        intermediate_tensors.append(vector)

    # Last one
    output_vector = [tensor_list[0][i] * vector[i] for i in range(len(tensor_list[0]))]
    
    intermediate_tensors.append(output_vector)

    return list(reversed(intermediate_tensors))



def tn_qudo_solver_decimal(w_matrix: dict, d_vector: np.ndarray, dimensions: np.ndarray, tau: float = 1) -> list:
    """
    Solves a Quadratic Unconstrained Discrete Optimization problem using tensor networks.
    
    Args:
        w_matrix: Dictionary representing quadratic interactions between variables
        d_vector: Vector representing linear costs
        dimensions: Array containing the dimension of each variable
        tau: Imaginary time for optimization (default=1)
        
    Returns:
        List containing the optimal values for each variable
    """
    n_variables = len(d_vector)
    solution = [0] * n_variables
    
    # Convert inputs to Decimal for higher precision
    w_matrix_decimal = {k: Decimal(str(inner)) for k,inner in w_matrix.items()}
    d_vector_decimal = [Decimal(str(elem)) for elem in d_vector]
    tau_decimal = Decimal(str(tau))

    # Create and contract the tensor network
    tensor_network = generate_tensor_network_QUDO_decimal(w_matrix_decimal, d_vector_decimal, dimensions, tau_decimal)
    intermediate_tensors = contraction_tensors_decimal(tensor_network)
    
    # Get first variable value from initial tensor contraction
    solution[0] = int(np.argmax([abs(elem) for elem in intermediate_tensors[0]]))
    
    # Remove first two tensors as they're no longer needed
    intermediate_tensors = intermediate_tensors[2:]

    # Iteratively determine remaining variables except the last one
    for position in range(1, n_variables - 1):
        # Calculate proportion for dynamic tau scaling
        proportion = Decimal(str(n_variables / (n_variables - position)))
        scaled_tau = tau_decimal * proportion
        #We rescale the new tensor evolution

        
        # Generate tensor for current position
        initial_tensor = tensor_initial_generator_QUDO_decimal(position, solution, w_matrix_decimal, 
                                                             d_vector_decimal, dimensions, scaled_tau)
        
        # Contract with remaining intermediate tensors, it is diagonal
        output_vector = [initial_tensor[i]* intermediate_tensors[0][i] for i in range(len(initial_tensor))]
        # Remove used intermediate tensor
        intermediate_tensors.pop(0)

        # Select optimal value based on maximum absolute value
        solution[position] = int(np.argmax([abs(elem) for elem in output_vector]))

    # Determine final variable value
    solution[-1] = last_variable_determination_QUDO(solution, w_matrix, d_vector, dimensions[-1])

    return solution












