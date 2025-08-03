import numpy as np


def cost_QUDO(w_matrix: dict, d_vector: np.ndarray, solution: list) -> float:
    '''
    Calculate the cost of a solution for a Quadratic Unconstrained Discrete Optimization (QUDO) problem.
    
    The cost function is defined as:
    C(x) = sum(W[i,j] * x[i] * x[j]) + sum(d[i] * x[i])
    where W is a symmetric weight dictionary and d is a vector of linear coefficients.
    
    Args:
        w_matrix: Weight dictionary with (i,j) tuple keys and float values containing quadratic terms
        d_vector: Vector of shape (N,) containing linear coefficients
        solution: Solution vector of shape (N,) containing integer values
    
    Returns:
        float: Cost value for the given solution
    '''
    NEIGHBORS = 2
    n_variables = len(solution)
    
    # Add linear terms - O(n) operation
    _cost = np.dot(d_vector, solution)
    
    # Add quadratic terms - O(n) operation since we only check neighbors
    for i in range(n_variables):
        # Only need to check next 2 neighbors since w_matrix is symmetric
        for j in range(i, min(i+NEIGHBORS, n_variables)):
            if (i,j) in w_matrix:
                _cost += w_matrix[(i,j)] * solution[i] * solution[j]
    return _cost


def cost_tensor_QUDO(w_tensor: dict, solution: list) -> float:
    '''
    Calculate the cost of a solution for a Tensor Quadratic Unconstrained Discrete Optimization (T-QUDO) problem.
    
    The cost function is defined as:
    C(x) = sum(W[i,j,x[i],x[j]]) 
    where W is a dictionary containing the cost coefficients for each combination of variable values.
    
    Args:
        w_tensor: Dictionary with (i,j,xi,xj) tuple keys and float values containing cost coefficients
                 w_tensor[(i,j,xi,xj)] gives cost coefficient for variables i,j taking values xi,xj
        solution: List of N integers representing the solution vector, where solution[i] is the value
                 assigned to variable i.
    
    Returns:
        float: Total cost value for the given solution
    '''
    _cost = 0
    NEIGHBORS = 2
    n_variables = len(solution)
    
    # Only need to check next 2 neighbors since problem has nearest-neighbor interactions
    for i in range(n_variables):
        for j in range(i, min(i+NEIGHBORS, n_variables)):
            key = (i,j,solution[i],solution[j])
            if key in w_tensor:
                _cost += w_tensor[key]
    return _cost
        
        
def last_variable_determination_QUDO(solution: list, w_matrix: dict,
                                   d_vector: np.ndarray, dimension: int) -> int:
    '''
    Determine optimal value for last variable given values of all others in a QUDO problem.
    
    For a partial solution with N-1 variables set, this function evaluates all possible values
    for the Nth variable and returns the value that minimizes the total cost function.
    
    Args:
        solution: Current partial solution with N-1 variables set
        w_matrix: Weight dictionary with (i,j) tuple keys and float values containing quadratic terms
        d_vector: Vector of shape (N,) containing linear coefficients
        dimension: The number of possible values for the last variable
        
    Returns:
        int: Optimal value (0 to dimension-1) for last variable that minimizes total cost
    '''
    _cost_list = np.zeros(dimension)
    for _valor in range(dimension):
        _cost_list[_valor] = cost_QUDO(w_matrix, d_vector, solution[:-1] + [_valor])
    return int(np.argmin(_cost_list))



def last_variable_determination_tensor_QUDO(solution: list, w_tensor: dict, dimension: int) -> int:
    '''
    Determine optimal value for last variable given values of all others in a Tensor QUDO problem.
    
    For a partial solution with N-1 variables set, this function evaluates all possible values
    for the Nth variable and returns the value that minimizes the total cost function.
    The cost is calculated using the tensor representation where costs are stored directly for
    each combination of variable values.
    
    Args:
        solution: List of N-1 integers representing current partial solution with last variable unset
        w_tensor: Dictionary with (i,j,xi,xj) tuple keys and float values containing cost coefficients
        dimension: The number of possible values for the last variable
        
    Returns:
        int: Optimal value (0 to dimension-1) for last variable that minimizes total cost
    '''
    # Get dimension from the maximum value in the last position of any key
    _cost_list = np.zeros(dimension)
    for _valor in range(dimension):
        _cost_list[_valor] = cost_tensor_QUDO(w_tensor, solution[:-1] + [_valor])
    return int(np.argmin(_cost_list))


def generate_QUDO_instance(n_variables: int, values_range: tuple) -> tuple[dict, np.ndarray]:
    """
    Generates a random sparse QUDO instance with nearest-neighbor interactions.
    
    Args:
        n_variables: Number of variables in the QUDO instance
        values_range: Tuple of (min, max) values for matrix elements
        
    Returns:
        tuple: (w_matrix, d_vector)
            - w_matrix: Dictionary with (i,j) tuple keys containing non-zero quadratic terms
            - d_vector: Vector of linear terms with random values in the specified range
    """
    NEIGHBORS = 2
    w_matrix = {}
    min_val, max_val = values_range
    
    # Generate random linear terms
    d_vector = np.random.uniform(min_val, max_val, n_variables)
    
    # Generate random quadratic terms with nearest-neighbor interactions
    total_squared = 0  # For normalization
    for i in range(n_variables):
        for j in range(i, min(i+NEIGHBORS, n_variables)):
            value = np.random.uniform(min_val, max_val, 1)[0]
            w_matrix[(i,j)] = value
            total_squared += value * value

    # Normalize the terms
    norm_factor = np.sqrt(total_squared)
    for key in w_matrix:
        w_matrix[key] /= norm_factor
    d_vector = d_vector / np.linalg.norm(d_vector)
    
    return w_matrix, d_vector

def generate_tensor_QUDO_instance(n_variables: int, dimensions: int, values_range: tuple) -> dict:
    """
    Generates a random sparse Tensor QUDO instance with nearest-neighbor interactions.
    
    Args:
        n_variables: Number of variables in the QUDO instance
        dimensions: Dimension of each variable's state space
        values_range: Tuple of (min, max) values for tensor elements
        
    Returns:
        dict: Dictionary with (i,j,xi,xj) tuple keys containing non-zero interaction terms
    """
    
    NEIGHBORS = 2
    w_tensor = {}
    min_val, max_val = values_range
    
    # Generate random quadratic terms with nearest-neighbor interactions
    total_squared = 0  # For normalization
    for i in range(n_variables):
        for j in range(i, min(i+NEIGHBORS, n_variables)):
            for xi in range(dimensions):
                for xj in range(dimensions):
                    value = np.random.uniform(min_val, max_val, 1)[0]
                    w_tensor[(i,j,xi,xj)] = value
                    total_squared += value * value

    # Normalize the tensor
    norm_factor = np.sqrt(total_squared)
    for key in w_tensor:
        w_tensor[key] /= norm_factor
    
    return w_tensor


# Solve using different methods and compare results
def print_solution(name, solution, time, cost):
    print(f"\n{name} Results:")
    print(f"Solution time: {time:.6e} seconds")
    print(f"Solution: {','.join(map(str, solution))}")
    print(f"Solution cost: {cost:.6e}")











