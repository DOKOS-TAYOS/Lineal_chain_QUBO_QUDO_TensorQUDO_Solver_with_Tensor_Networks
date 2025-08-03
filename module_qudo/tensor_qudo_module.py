import numpy as np
from qudo_module import contraction_tensors
from auxiliary_functions import last_variable_determination_tensor_QUDO

def tensor_initial_generator_tensor_QUDO(position: int, solution: list, w_tensor: dict, dimensions: int, tau: float, phase: bool=False) -> np.ndarray:
    """
    Generate initial tensor for T-QUDO problem at given position.
    
    Args:
        position (int): Current position in solution vector
        solution (list): Partial solution vector
        w_tensor (dict): Dictionary containing quadratic interactions {(i,j,k,l): value}
        dimensions (int): Number of possible values for each variable
        tau (float): Evolution parameter for optimization
        phase (bool): Whether to use phase factors in tensor elements
        
    Returns:
        ndarray: Initial tensor of shape (dimensions[position], dimensions[position])
    """
    # Initialize tensor with appropriate dtype
    dtype = complex if phase else float
    tensor = np.zeros((dimensions, dimensions), dtype=dtype)
    phase_factor = 1j*2*np.pi/dimensions if phase else 0
    
    # Calculate tensor elements
    for i in range(dimensions):
        # Compute total energy contribution
        exp_value = 0.0
        if (position, position, i, i) in w_tensor:
            exp_value = w_tensor[(position, position, i, i)]  # Self-interaction
        
        # Add interaction with previous variable if not first position
        if position > 0:  # Check solution is not empty
            if (position-1, position, solution[position-1], i) in w_tensor:
                exp_value += w_tensor[(position-1, position, solution[position-1], i)]
            
        # Set diagonal element with evolution factor
        tensor[i,i] = np.exp(-tau*exp_value) * np.exp(phase_factor*i)
    
    # Normalize tensor    
    norm = np.linalg.norm(tensor)
    return tensor / (norm if norm > 0 else 1.0)

def tensor_intermediate_generator_tensor_QUDO(position: int, w_tensor: dict, dimensions: int, tau: float, phase: bool=False) -> np.ndarray:
    """
    Generate intermediate tensor for T-QUDO problem at given position.
    
    Args:
        position (int): Current position in solution vector
        w_tensor (dict): Dictionary containing quadratic interactions {(i,j,k,l): value}
        dimensions (int): Number of possible values for each variable
        tau (float): Evolution parameter for optimization
        phase (bool): Whether to use phase factors in tensor elements
        
    Returns:
        ndarray: Intermediate tensor of shape (dimensions[position-1], dimensions[position])
    """
    # Initialize tensor with appropriate dtype
    dtype = complex if phase else float
    tensor = np.zeros((dimensions, dimensions), dtype=dtype)
    phase_factor = 1j*2*np.pi/dimensions if phase else 0
    
    # Calculate tensor elements
    for i in range(dimensions):
        for j in range(dimensions):
            # Compute total energy contribution
            exp_value = 0.0
            if (position, position, j, j) in w_tensor:
                exp_value += w_tensor[(position, position, j, j)]  # Self-interaction
            if (position-1, position, i, j) in w_tensor:
                exp_value += w_tensor[(position-1, position, i, j)]  # Interaction with previous
                
            # Set element with evolution factor
            tensor[i,j] = np.exp(-tau*exp_value) * np.exp(phase_factor*j)

    # Normalize tensor
    norm = np.linalg.norm(tensor)
    return tensor / (norm if norm > 0 else 1.0)

def tensor_final_generator_tensor_QUDO(position: int, w_tensor: dict, dimensions: int, tau: float, phase: bool=False) -> np.ndarray:
    """
    Generate final tensor for T-QUDO problem at given position.
    
    Args:
        position (int): Current position in solution vector
        w_tensor (dict): Dictionary containing quadratic interactions {(i,j,k,l): value}
        dimensions (int): Number of possible values for each variable
        tau (float): Evolution parameter for optimization
        phase (bool): Whether to use phase factors in tensor elements
        
    Returns:
        ndarray: Final tensor of shape (dimensions[position-1])
    """
    # Initialize tensor with appropriate dtype
    dtype = complex if phase else float
    tensor = np.zeros(dimensions, dtype=dtype)
    phase_factor = 1j*2*np.pi/dimensions if phase else 0
    
    # Calculate tensor elements by summing over last variable
    for i in range(dimensions):
        for j in range(dimensions):
            # Compute total energy contribution
            exp_value = 0.0
            if (position, position, j, j) in w_tensor:
                exp_value += w_tensor[(position, position, j, j)]  # Self-interaction
            if (position-1, position, i, j) in w_tensor:
                exp_value += w_tensor[(position-1, position, i, j)]  # Interaction with previous
                
            # Add contribution to element with evolution factor
            tensor[i] += np.exp(-tau*exp_value) * np.exp(phase_factor*j)

    # Normalize tensor
    norm = np.linalg.norm(tensor)
    return tensor / (norm if norm > 0 else 1.0)


def generate_tensor_network_tensor_QUDO(w_tensor: dict, n_variables: int, dimensions: int, tau: float, phase: bool = False) -> list[np.ndarray]:
    """
    Generates a tensor network for solving a Tensor Quadratic Unconstrained Discrete Optimization problem.
    
    Args:
        w_tensor (dict): Dictionary containing quadratic interactions {(i,j,k,l): value}
        n_variables (int): Number of variables in the problem
        dimensions (int): Number of possible values for each variable
        tau (float): Imaginary time evolution parameter for optimization
        phase (bool): Whether to include phase factors in tensor elements
        
    Returns:
        list[np.ndarray]: List of tensors representing the tensor network, from left to right
    """
    tensor_list = []
    
    # Create initial measurement tensor
    tensor_list.append(tensor_initial_generator_tensor_QUDO(0, [], w_tensor, dimensions, tau, phase))

    # Create intermediate tensors 
    for position in range(1, n_variables-1):
        tensor_list.append(tensor_intermediate_generator_tensor_QUDO(position, w_tensor, dimensions, tau, phase))
    
    # Create final tensor
    tensor_list.append(tensor_final_generator_tensor_QUDO(n_variables-1, w_tensor, dimensions, tau, phase))

    return tensor_list


def tn_qudo_solver_tensor(w_tensor: dict, n_variables: int, dimensions: int, tau: float = 1, phase: bool = False) -> list[int]:
    """
    Solves a Tensor Quadratic Unconstrained Discrete Optimization problem using tensor networks.
    
    Args:
        w_tensor (dict): Dictionary containing quadratic interactions {(i,j,k,l): value}
        n_variables (int): Number of variables in the problem
        dimensions (int): Number of possible values for each variable
        tau (float): Imaginary time for optimization (default=1). Controls optimization strength.
        phase (bool): If True, uses phase-based optimization (humbucker method).
                     If False, uses standard amplitude-based optimization.
        
    Returns:
        List of integers containing the optimal values for each variable.
        
    Raises:
        ValueError: If input dimensions don't match or are invalid.
    """
    solution = [0] * n_variables

    # Create and contract the tensor network
    tensor_network = generate_tensor_network_tensor_QUDO(w_tensor, n_variables, dimensions, tau, phase)
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
        initial_tensor = tensor_initial_generator_tensor_QUDO(
            position, solution, w_tensor, dimensions, scaled_tau, phase
        )
        output_vector = initial_tensor @ intermediate_tensors[0]
        intermediate_tensors.pop(0)

        # Select optimal value based on maximum probability amplitude
        solution[position] = int(np.argmax(np.abs(output_vector)))

    # Determine final variable value
    solution[-1] = last_variable_determination_tensor_QUDO(solution, w_tensor, dimensions)

    return solution