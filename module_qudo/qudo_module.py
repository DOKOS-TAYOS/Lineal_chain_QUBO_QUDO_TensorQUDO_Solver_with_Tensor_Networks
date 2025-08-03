import numpy as np
from auxiliary_functions import last_variable_determination_QUDO

def tensor_initial_generator_QUDO(position: int, solution: list, w_matrix: dict, d_vector: np.ndarray, dimensions: np.ndarray, tau: float, phase: bool=False) -> np.ndarray:
    """
    Generate initial tensor for QUDO problem at given position.
    
    Args:
        position (int): Current position in solution vector
        solution (list): Partial solution vector
        w_matrix (dict): Dictionary of sparse quadratic terms with (i,j) tuple keys
        d_vector (ndarray): Linear term vector of length N containing linear costs
        dimensions (np.ndarray): List of dimensions for each variable
        tau (float): Evolution parameter for optimization
        phase (bool): Whether to use phase factors in tensor elements
        
    Returns:
        ndarray: Initial tensor of shape (dimensions[position], dimensions[position])
    """
    # Initialize tensor with appropriate dtype
    dtype = complex if phase else float
    tensor = np.zeros((dimensions[position], dimensions[position]), dtype=dtype)
    phase_factor = 1j*2*np.pi/dimensions[position] if phase else 0
    
    # Calculate tensor elements
    for i in range(dimensions[position]):
        # Compute total energy contribution
        exp_value = d_vector[position]*i  # Linear term
        
        # Add self-interaction if exists
        if (position, position) in w_matrix:
            exp_value += w_matrix[(position, position)]*i*i
            
        # Add interaction with previous variable if not first position
        if position > 0 and (position-1, position) in w_matrix:  # Check solution is not empty
            exp_value += w_matrix[(position-1, position)]*solution[position-1]*i
            
        # Set diagonal element with evolution factor
        tensor[i,i] = np.exp(-tau*exp_value) * np.exp(phase_factor*i)
    
    # Normalize tensor    
    norm = np.linalg.norm(tensor)
    return tensor / (norm if norm > 0 else 1.0)

def tensor_intermediate_generator_QUDO(position: int, w_matrix: dict, d_vector: np.ndarray, dimensions: np.ndarray, tau: float, phase: bool=False) -> np.ndarray:
    """
    Generate intermediate tensor for QUDO problem at given position.
    
    Args:
        position (int): Current position in solution vector
        w_matrix (dict): Dictionary of sparse quadratic terms with (i,j) tuple keys
        d_vector (ndarray): Linear term vector of length N containing linear costs
        dimensions (np.ndarray): List of dimensions for each variable
        tau (float): Evolution parameter for optimization
        phase (bool): Whether to use phase factors in tensor elements
        
    Returns:
        ndarray: Intermediate tensor of shape (dimensions[position-1], dimensions[position])
    """
    # Initialize tensor with appropriate dtype
    dtype = complex if phase else float
    tensor = np.zeros((dimensions[position-1], dimensions[position]), dtype=dtype)
    phase_factor = 1j*2*np.pi/dimensions[position] if phase else 0
    
    # Calculate tensor elements
    for i in range(dimensions[position-1]):
        for j in range(dimensions[position]):
            # Compute total energy contribution
            exp_value = d_vector[position]*j  # Linear term
            
            # Add self-interaction if exists
            if (position, position) in w_matrix:
                exp_value += w_matrix[(position, position)]*j*j
                
            # Add interaction with previous if exists
            if (position-1, position) in w_matrix:
                exp_value += w_matrix[(position-1, position)]*i*j
                
            # Set element with evolution factor
            tensor[i,j] = np.exp(-tau*exp_value) * np.exp(phase_factor*j)

    # Normalize tensor
    norm = np.linalg.norm(tensor)
    return tensor / (norm if norm > 0 else 1.0)

def tensor_final_generator_QUDO(position: int, w_matrix: dict, d_vector: np.ndarray, dimensions: np.ndarray, tau: float, phase: bool=False) -> np.ndarray:
    """
    Generate final tensor for QUDO problem at given position.
    
    Args:
        position (int): Current position in solution vector
        w_matrix (dict): Dictionary of sparse quadratic terms with (i,j) tuple keys
        d_vector (ndarray): Linear term vector of length N containing linear costs
        dimensions (np.ndarray): List of dimensions for each variable
        tau (float): Evolution parameter for optimization
        phase (bool): Whether to use phase factors in tensor elements
        
    Returns:
        ndarray: Final tensor of shape (dimensions[position-1])
    """
    # Initialize tensor with appropriate dtype
    dtype = complex if phase else float
    tensor = np.zeros(dimensions[position-1], dtype=dtype)
    phase_factor = 1j*2*np.pi/dimensions[position] if phase else 0
    
    # Calculate tensor elements by summing over last variable
    for i in range(dimensions[position-1]):
        for j in range(dimensions[position]):
            # Compute total energy contribution
            exp_value = d_vector[position]*j  # Linear term
            
            # Add self-interaction if exists
            if (position, position) in w_matrix:
                exp_value += w_matrix[(position, position)]*j*j
                
            # Add interaction with previous if exists
            if (position-1, position) in w_matrix:
                exp_value += w_matrix[(position-1, position)]*i*j
                
            # Add contribution to element with evolution factor
            tensor[i] += np.exp(-tau*exp_value) * np.exp(phase_factor*j)

    # Normalize tensor
    norm = np.linalg.norm(tensor)
    return tensor / (norm if norm > 0 else 1.0)


def generate_tensor_network_QUDO(w_matrix: dict, d_vector: np.ndarray, dimensions: np.ndarray, tau: float, phase: bool = False) -> list[np.ndarray]:
    """
    Generates a tensor network for solving a Quadratic Unconstrained Discrete Optimization problem.
    
    Args:
        w_matrix (dict): Dictionary of sparse quadratic terms with (i,j) tuple keys
        d_vector (np.ndarray): Vector representing linear costs
        dimensions (np.ndarray): Array containing the dimension of each variable
        tau (float): Imaginary time evolution parameter for optimization
        phase (bool): Whether to include phase factors in tensor elements
        
    Returns:
        list[np.ndarray]: List of tensors representing the tensor network, from left to right
    """
    # Input validation
    n_variables = len(d_vector)
    if len(dimensions) != n_variables:
        raise ValueError("Length of dimensions must match number of variables")
    
    tensor_list = []
    
    # Create initial measurement tensor
    tensor_list.append(tensor_initial_generator_QUDO(0, [], w_matrix, d_vector, dimensions, tau, phase))

    # Create intermediate tensors 
    for position in range(1, n_variables-1):
        tensor_list.append(tensor_intermediate_generator_QUDO(position, w_matrix, d_vector, dimensions, tau, phase))
    
    # Create final tensor
    tensor_list.append(tensor_final_generator_QUDO(n_variables-1, w_matrix, d_vector, dimensions, tau, phase))

    return tensor_list

def contraction_tensors(tensor_list: list[np.ndarray]) -> list[np.ndarray]:
    """
    Contracts a list of tensors sequentially and returns intermediate results.
    
    Args:
        tensor_list: List of tensors to contract. Each tensor should be a numpy array.
        
    Returns:
        List of intermediate contracted tensors, normalized at each step.
        The first element is the rightmost tensor, and subsequent elements
        are the results of contracting from right to left.
    """
    if not tensor_list:
        return []
        
    # Initialize with last tensor normalized
    vector = tensor_list[-1].copy()  # Make copy to avoid modifying input
    norm = np.linalg.norm(vector)
    if norm > 0:  # Only normalize if norm is not zero
        vector /= norm
    intermediate_tensors = [vector]
    
    # Contract tensors from right to left
    for tensor in reversed(tensor_list[:-1]):
        # Contract and normalize
        vector = tensor @ vector
        norm = np.linalg.norm(vector)
        if norm > 0:  # Avoid division by zero
            vector /= norm
        intermediate_tensors.append(vector)

    return list(reversed(intermediate_tensors))



def tn_qudo_solver(w_matrix: dict, d_vector: np.ndarray, dimensions: np.ndarray, tau: float = 1, phase: bool = False) -> list[int]:
    """
    Solves a Quadratic Unconstrained Discrete Optimization problem using tensor networks.
    
    Args:
        w_matrix: Dictionary representing sparse quadratic interactions between variables.
                 Keys are tuples (i,j) and values are the interaction strengths.
        d_vector: Vector representing linear costs. Must have length n_variables.
        dimensions: Array containing the dimension of each variable.
                   Must have length n_variables and contain positive integers.
        tau: Imaginary time for optimization (default=1). Controls optimization strength.
        phase: If True, uses phase-based optimization (humbucker method).
             If False, uses standard amplitude-based optimization.
        
    Returns:
        List of integers containing the optimal values for each variable,
        where the i-th value is in range [0, dimensions[i]-1].
        
    Raises:
        ValueError: If input dimensions don't match or are invalid.
    """
    # Validate inputs
    n_variables = len(dimensions)  # Now determine size from dimensions instead of w_matrix
    if len(d_vector) != n_variables:
        raise ValueError("d_vector length must match number of variables")
    if not np.all(dimensions > 0):
        raise ValueError("All dimensions must be positive integers")
        
    solution = [0] * n_variables

    # Create and contract the tensor network
    tensor_network = generate_tensor_network_QUDO(w_matrix, d_vector, dimensions, tau, phase)
    intermediate_tensors = contraction_tensors(tensor_network)

    # Get first variable value and clean up tensors
    solution[0] = int(np.argmax(np.abs(intermediate_tensors[0])))
    intermediate_tensors = intermediate_tensors[2:]  # Remove first two tensors

    # Iteratively determine remaining variables
    for position in range(1, n_variables-1):
        # Scale tau proportionally as we progress through variables
        proportion = n_variables / (n_variables - position)
        scaled_tau = tau * proportion
        
        # Generate and contract tensors for current position
        initial_tensor = tensor_initial_generator_QUDO(
            position, solution, w_matrix, d_vector, dimensions, scaled_tau, phase
        )
        output_vector = initial_tensor @ intermediate_tensors[0]
        intermediate_tensors.pop(0)

        # Select optimal value based on maximum probability amplitude
        solution[position] = int(np.argmax(np.abs(output_vector)))

    # Determine final variable value
    solution[-1] = last_variable_determination_QUDO(solution, w_matrix, d_vector, dimensions[-1])

    return solution
