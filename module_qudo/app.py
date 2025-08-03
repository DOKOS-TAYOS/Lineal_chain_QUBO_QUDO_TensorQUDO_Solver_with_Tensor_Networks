import streamlit as st
import time
import numpy as np
import pandas as pd
from decimal import Decimal
from qudo_module import tn_qudo_solver
from qudo_decimal_module import tn_qudo_solver_decimal
from tensor_qudo_module import tn_qudo_solver_tensor
from auxiliary_functions import generate_QUDO_instance, generate_tensor_QUDO_instance
import json

st.set_page_config(page_title="QUDO Solver 🧮", layout="wide")

st.title("🧮 Nearest Neighbor Linear Chain QUBO/QUDO/Tensor QUDO Solver")

with st.expander("ℹ️ About QUDO and Usage Instructions", expanded=True):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 📝 What is QUDO?
        
        QUDO (Quadratic Unconstrained Discrete Optimization) is a powerful optimization framework that generalizes the well-known QUBO (Quadratic Unconstrained Binary Optimization) problems. While QUBO is limited to binary variables (0 or 1), QUDO allows variables to take multiple discrete values (0, 1, ..., D-1).

        The optimization problem aims to minimize a cost function that consists of:
        - Quadratic interactions between neighboring variables
        - Square terms for each variable
        - Linear terms for each variable

        Mathematically, we minimize:
        $$C(\\vec{x})=\\sum_{i=0}^{N-2}W_{i,i+1}x_{i}x_{i+1}+\\sum_{i=0}^{N-1}(W_{i,i}x_{i}^2 + d_i x_{i})$$

        where:
        - $N$ is the number of variables
        - $x_i$ are integer variables between 0 and $D_i-1$
        - $W$ is a tridiagonal interaction matrix
        - $d_i$ are linear coefficients
        """)

        st.markdown("""
        ### 🔄 Tensor QUDO Extension
        
        Tensor QUDO generalizes QUDO further by allowing arbitrary interactions between neighboring variables:

        $$C(\\vec{x})=\\sum_{i=0}^{N-2}W_{i,i+1,x_{i},x_{i+1}}+\\sum_{i=0}^{N-1}W_{i,i,x_{i},x_{i}}$$

        This formulation captures more complex relationships between variables while maintaining the nearest-neighbor structure.

        Problemas such as Traveling Salesman Problem can be naturally expressed in this formalism.
        """)

    with col2:
        st.markdown("""
        ### 🚀 How to Use This Solver

        1. **Choose Problem Type**
            - QUDO: Standard problem with float precision
            - QUDO with Decimal: Higher precision for numerical stability
            - Tensor QUDO: Generalized problem with 4-tensor interactions

        2. **Set Parameters**
            - Number of Variables (N): Total variables in your problem
            - Dimension (D): Maximum value each variable can take
            - Imaginary Time (tau): Controls the optimization process
            
        3. **Input Problem Data**
            - For QUDO: Enter the W matrix and d vector
            - For Tensor QUDO: Input the 4-dimensional tensor
            - Use "Generate Random Instance" for testing
            - Upload existing problems via JSON/NPY files

        4. **Solve and Analyze**
            - Click "Solve" to run the optimization
            - View results including:
                - Optimal solution vector
                - Minimum cost achieved
                - Computation time
            - Download complete results for further analysis
        """)

        st.markdown("""
        ### 🔧 Technical Details

        The solver employs a quantum-inspired tensor network algorithm based on imaginary time evolution. This approach provides:
        - Efficient solution for nearest-neighbor interactions
        - Polynomial-time complexity
        - High accuracy for large enough tau value

        For more technical details, see our paper:
        [Polynomial-time Solver of Tridiagonal QUBO, QUDO and Tensor QUDO problems with Tensor Networks](https://arxiv.org/abs/2309.10509)

        Source code has been developed by Alejandro Mata Ali, and it is available at:
        [GitHub Repository](https://github.com/DOKOS-TAYOS/Lineal_chain_QUBO_QUDO_TensorQUDO_Solver_with_Tensor_Networks)
        """)

# Create sidebar for basic parameters
with st.sidebar:
    # Problem type selection
    problem_type = st.selectbox(
        "Select Problem Type 🔄",
        ["QUDO", "QUDO with Decimal", "Tensor QUDO"],
        help="QUDO: Standard problem with float precision\nQUDO with Decimal: Higher precision for numerical stability\nTensor QUDO: Generalized problem with 4-tensor interactions"
    )

    # Number of variables and dimensions
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.n_variables = st.number_input("Number of Variables 🔢", min_value=3, max_value=10000, value=10, step=1)
    with col2:
        st.session_state.dimension = st.number_input("Dimension of Variables 📏", min_value=2, max_value=10, value=2, step=1)

    st.session_state.dimensions = np.full(st.session_state.n_variables, st.session_state.dimension, dtype=int)
    
    # Add tau slider
    if problem_type == "QUDO with Decimal":
        default_tau = 1e3 / (st.session_state.n_variables * st.session_state.dimension)
        st.session_state.tau = st.slider(
            "Imaginary Time (tau) ⏱️", 
            min_value=0.0,
            max_value=1e5 / (st.session_state.n_variables * st.session_state.dimension),
            value=float(default_tau),
            help="Parameter controlling the imaginary time evolution. Default is 100/(N*D) where N is number of variables and D is dimension"
        )

    else:
        default_tau = 1e2 / (st.session_state.n_variables * st.session_state.dimension)
        st.session_state.tau = st.slider(
            "Imaginary Time (tau) ⏱️", 
            min_value=0.0,
            max_value=1e4 / (st.session_state.n_variables * st.session_state.dimension),
            value=float(default_tau),
            help="Parameter controlling the imaginary time evolution. Default is 100/(N*D) where N is number of variables and D is dimension"
        )

# Matrix input in main area
if problem_type in ["QUDO", "QUDO with Decimal"]:
    st.markdown("### 📊 Matrix Input")
    st.markdown("Enter values for the matrix elements. Only main diagonal and upper diagonal are used, other elements are zero.")
    # Initialize matrices
    if 'w_matrix' not in st.session_state:
        st.session_state.w_matrix = {}
    if 'd_vector' not in st.session_state:
        st.session_state.d_vector = np.zeros(st.session_state.n_variables)
        
    if st.session_state.n_variables < 21:
        if 'matrix_data' not in st.session_state:
            st.session_state.matrix_data = np.zeros((st.session_state.n_variables, st.session_state.n_variables))
    
    if st.session_state.d_vector.shape[0] != st.session_state.n_variables:
        st.session_state.w_matrix = {}
        st.session_state.d_vector = np.zeros(st.session_state.n_variables)
        st.session_state.dimensions = np.full(st.session_state.n_variables, st.session_state.dimension, dtype=int)
        if st.session_state.n_variables < 21:
            st.session_state.matrix_data = np.zeros((st.session_state.n_variables, st.session_state.n_variables))

    # Random instance generation button
    if st.button("🎲 Generate Random Instance"):
        # Generate random instance using auxiliary function
        st.session_state.w_matrix, st.session_state.d_vector = generate_QUDO_instance(st.session_state.n_variables, (-1, 1))
        st.success("Random instance generated successfully! 🎲")
        
        if st.session_state.n_variables < 21:
            # Update matrix data with generated values
            for (i,j), val in st.session_state.w_matrix.items():
                st.session_state.matrix_data[i,j] = val

    # Create interactive matrix table if n_variables is small enough
    if st.session_state.n_variables < 21:
        st.markdown("#### 🔄 Interaction Matrix (W)")
        matrix_df = pd.DataFrame(st.session_state.matrix_data)

        # Make the dataframe editable
        edited_df = st.data_editor(matrix_df)

        # Update w_matrix based on edited values
        for i in range(st.session_state.n_variables):
            for j in range(i, min(i+2, st.session_state.n_variables)):
                val = edited_df.iloc[i,j]
                if val != 0:
                    st.session_state.w_matrix[(i,j)] = val

        # Linear terms if not binary
        if st.session_state.dimension != 2:
            st.markdown("#### 📈 Linear Terms (d)")
            d_df = pd.DataFrame(st.session_state.d_vector).T
            edited_d_df = st.data_editor(d_df)
            st.session_state.d_vector = edited_d_df.iloc[0].values

    uploaded_file = st.file_uploader("📤 Upload QUDO instance", type=["json", "npy"])
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.json'):
            data = json.load(uploaded_file)
            st.session_state.w_matrix = {tuple(map(int, k.strip('()').split(','))): float(v)
                      for k,v in data['w_matrix'].items()
                      if (lambda x: x[0] == x[1] or x[1] == x[0] + 1)(tuple(map(int, k.strip('()').split(','))))}
            st.session_state.d_vector = np.array(data['d_vector'])
            st.session_state.n_variables = len(st.session_state.d_vector)
            st.session_state.dimensions = np.full(st.session_state.n_variables, st.session_state.dimension, dtype=int)
            if st.session_state.n_variables < 21:
                st.session_state.matrix_data = np.zeros((st.session_state.n_variables, st.session_state.n_variables))
                for (i,j), val in st.session_state.w_matrix.items():
                    st.session_state.matrix_data[i,j] = val
            
        else:  # .npy file
            st.session_state.matrix = np.load(uploaded_file)
            st.session_state.n_variables = len(st.session_state.matrix)
            st.session_state.dimensions = np.full(st.session_state.n_variables, st.session_state.dimension, dtype=int)
            st.session_state.w_matrix = {}
            if st.session_state.n_variables < 21:
                st.session_state.matrix_data = st.session_state.matrix.copy()
            for i in range(st.session_state.n_variables):
                for j in range(i, min(i+2, st.session_state.n_variables)):
                    if st.session_state.matrix[i,j] != 0:
                        st.session_state.w_matrix[(i,j)] = float(st.session_state.matrix[i,j])
            st.session_state.d_vector = np.zeros(st.session_state.n_variables)
        st.success("✅ File loaded successfully!")


    # Solve button
    if st.button("🚀 Solve"):
        try:
            with st.spinner("⚡ Solving..."):
                start_time = time.perf_counter()
                if problem_type == "QUDO":
                    solution = tn_qudo_solver(st.session_state.w_matrix, st.session_state.d_vector, st.session_state.dimensions, st.session_state.tau)
                    cost = sum(st.session_state.w_matrix[(i,j)] * solution[i] * solution[j] 
                            for i,j in st.session_state.w_matrix.keys()) + np.dot(st.session_state.d_vector, solution)
                            
                elif problem_type == "QUDO with Decimal":
                    # Convert to Decimal
                    w_matrix_decimal = {k: Decimal(str(v)) for k,v in st.session_state.w_matrix.items()}
                    d_vector_decimal = [Decimal(str(x)) for x in st.session_state.d_vector]
                    solution = tn_qudo_solver_decimal(w_matrix_decimal, d_vector_decimal, st.session_state.dimensions, st.session_state.tau)
                    cost = sum(st.session_state.w_matrix[(i,j)] * solution[i] * solution[j] 
                            for i,j in st.session_state.w_matrix.keys()) + np.dot(st.session_state.d_vector, solution)
                computation_time = time.perf_counter() - start_time

            # Display results in an elegant container
            with st.container():
                st.markdown("### 🎯 Optimization Results")
                
                # Create metrics in columns
                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    st.metric(label="Objective Value 💎", value=f"{cost:.6e}")
                with metric_col2:
                    st.metric(label="Computation Time ⏱️", value=f"{computation_time:.2e}s")
                
                # Solution visualization
                if st.session_state.n_variables < 1001:
                    st.markdown("#### 📊 Solution Vector")
                    ROW_AMOUNT = 20
                    num_rows = int(np.ceil(st.session_state.n_variables/ROW_AMOUNT))
                    var_cols = []
                    for row in range(num_rows):
                        var_cols.append(st.columns(len(solution[ROW_AMOUNT*row:ROW_AMOUNT*(row+1)])))

                        # Create boxes for each variable
                        for i, val in enumerate(solution[ROW_AMOUNT*row:ROW_AMOUNT*(row+1)]):
                            with var_cols[row][i]:
                                # Start variable container
                                st.markdown(f"""
                                    <div style='
                                        width: 100%;
                                        text-align: center;
                                        display: flex;
                                        flex-direction: row;
                                        align-items: center;
                                        justify-content: center;
                                    '>
                                        <div style='font-weight: bold; margin-right: 5px;'>x<sub>{ROW_AMOUNT*row+i}</sub></div>
                                        <div style='
                                            display: flex;
                                            flex-direction: row;
                                            gap: 4px;
                                            flex: 1;
                                            justify-content: center;
                                        '>
                                """, unsafe_allow_html=True)
                                
                                # Create boxes for each possible value
                                for d in range(st.session_state.dimension):
                                    background = "#ff0000" if d == val else "#0000ff"
                                    text_color = "white"
                                    st.markdown(f"""
                                        <div style='
                                            background-color: {background};
                                            color: {text_color};
                                            border: 1px solid #ccc;
                                            padding: 8px;
                                            margin: 0;
                                            border-radius: 4px;
                                            font-size: 1.1em;
                                            flex: 1;
                                            text-align: center;
                                        '>{d}</div>
                                    """, unsafe_allow_html=True)
                            
                        #     # Close variable container
                        #     st.markdown("</div></div>", unsafe_allow_html=True)
                        # # Close horizontal container
                        # st.markdown("</div>", unsafe_allow_html=True)
                # Show detailed dataframe
                st.markdown("#### 📋 Detailed Solution")
                solution_df = pd.DataFrame({
                    'Variable': [f'x_{i}' for i in range(len(solution))],
                    'Value': solution
                }).transpose()
                st.dataframe(solution_df, use_container_width=True)

            download_data = {
                "w_matrix": {str(k): float(v) for k,v in st.session_state.w_matrix.items()},
                "d_vector": st.session_state.d_vector.tolist(),
                "solution": solution,
                "cost": float(cost),
                "computation_time": float(computation_time)
            }

            st.download_button(
                "⬇️ Download Results",
                data=json.dumps(download_data, indent=2),
                file_name="qudo_results.json",
                mime="application/json"
            )

        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")


        

elif problem_type == "Tensor QUDO":
    if 'w_tensor' not in st.session_state:
        st.session_state.w_tensor = {}
    
    if st.button("🎲 Generate Random Instance"):
        # Generate random instance using auxiliary function
        st.session_state.w_tensor = generate_tensor_QUDO_instance(st.session_state.n_variables, st.session_state.dimension, (-1, 1))
        st.success("✅ Random instance generated successfully!")
    
    uploaded_file = st.file_uploader("📤 Upload Tensor QUDO instance", type=["json", "npy"])
    
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.json'):
            data = json.load(uploaded_file)
            st.session_state.w_tensor = {tuple(map(int, k.strip('()').split(','))): float(v) 
                    for k,v in data.items()}
            # Get dimensions from the loaded tensor
            keys = list(st.session_state.w_tensor.keys())
            st.session_state.n_variables = max(max(k[0], k[1]) for k in keys) + 1
            st.session_state.dimension = max(max(k[2], k[3]) for k in keys) + 1
            st.session_state.dimensions = np.full(st.session_state.n_variables, st.session_state.dimension, dtype=int)
        else:  # .npy file
            tensor = np.load(uploaded_file)
            st.session_state.w_tensor = {}
            st.session_state.n_variables = tensor.shape[0]
            st.session_state.dimension = tensor.shape[2]
            st.session_state.dimensions = np.full(st.session_state.n_variables, st.session_state.dimension, dtype=int)
            for i in range(st.session_state.n_variables):
                for j in range(i, min(i+2, st.session_state.n_variables)):
                    for xi in range(st.session_state.dimension):
                        for xj in range(st.session_state.dimension):
                            if tensor[i,j,xi,xj] != 0:
                                st.session_state.w_tensor[(i,j,xi,xj)] = float(tensor[i,j,xi,xj])
        st.success(f"✅ File loaded successfully! Number of variables: {st.session_state.n_variables}, Dimension: {st.session_state.dimension}")
        
    if st.session_state.w_tensor != {}:
        if st.button("🚀 Solve Tensor QUDO", use_container_width=True):
            try:
                start_time = time.perf_counter()
                with st.spinner("🔄 Optimizing tensor network..."):
                    solution = tn_qudo_solver_tensor(st.session_state.w_tensor, st.session_state.n_variables, st.session_state.dimension, st.session_state.tau)
                    cost = sum(st.session_state.w_tensor[(i,j,solution[i],solution[j])] 
                            for i,j,_,_ in st.session_state.w_tensor.keys())
                    computation_time = time.perf_counter() - start_time

                # Display results in an elegant container
                with st.container():
                    st.markdown("### 🎯 Optimization Results")
                    
                    # Create metrics in columns
                    metric_col1, metric_col2 = st.columns(2)
                    with metric_col1:
                        st.metric(label="Objective Value 💎", value=f"{cost:.6e}")
                    with metric_col2:
                        st.metric(label="Computation Time ⏱️", value=f"{computation_time:.2e}s")
                    
                    if st.session_state.n_variables < 1001:
                        st.markdown("#### 📊 Solution Vector")
                        ROW_AMOUNT = 20
                        num_rows = int(np.ceil(st.session_state.n_variables/ROW_AMOUNT))
                        var_cols = []
                        for row in range(num_rows):
                            var_cols.append(st.columns(len(solution[ROW_AMOUNT*row:ROW_AMOUNT*(row+1)])))

                            # Create boxes for each variable
                            for i, val in enumerate(solution[ROW_AMOUNT*row:ROW_AMOUNT*(row+1)]):
                                with var_cols[row][i]:
                                    # Start variable container
                                    st.markdown(f"""
                                        <div style='
                                            width: 100%;
                                            text-align: center;
                                            display: flex;
                                            flex-direction: row;
                                            align-items: center;
                                            justify-content: center;
                                        '>
                                            <div style='font-weight: bold; margin-right: 5px;'>x<sub>{ROW_AMOUNT*row+i}</sub></div>
                                            <div style='
                                                display: flex;
                                                flex-direction: row;
                                                gap: 4px;
                                                flex: 1;
                                                justify-content: center;
                                            '>
                                    """, unsafe_allow_html=True)
                                    
                                    # Create boxes for each possible value
                                    for d in range(st.session_state.dimension):
                                        background = "#ff0000" if d == val else "#0000ff"
                                        text_color = "white"
                                        st.markdown(f"""
                                            <div style='
                                                background-color: {background};
                                                color: {text_color};
                                                border: 1px solid #ccc;
                                                padding: 8px;
                                                margin: 0;
                                                border-radius: 4px;
                                                font-size: 1.1em;
                                                flex: 1;
                                                text-align: center;
                                            '>{d}</div>
                                        """, unsafe_allow_html=True)
                            
                    #     # Close variable container
                    #     st.markdown("</div></div>", unsafe_allow_html=True)
                    # # Close horizontal container
                    # st.markdown("</div>", unsafe_allow_html=True)
                    # Show detailed dataframe
                    st.markdown("#### 📋 Detailed Solution")
                    solution_df = pd.DataFrame({
                        'Variable': [f'x{i}' for i in range(len(solution))],
                        'Value': solution
                    }).transpose()
                    st.dataframe(solution_df, use_container_width=True)
                    
                    # Prepare download data with additional metadata
                    download_data = {
                        "metadata": {
                            "n_variables": st.session_state.n_variables,
                            "dimension": st.session_state.dimension,
                            "computation_time": computation_time
                        },
                        "w_tensor": {str(k): float(v) for k,v in st.session_state.w_tensor.items()},
                        "solution": solution,
                        "cost": float(cost)
                    }

                    # Create download button with improved styling
                    st.download_button(
                        "⬇️ Download Complete Results",
                        data=json.dumps(download_data, indent=2),
                        file_name=f"qudo_results.json",
                        mime="application/json",
                        help="Download the complete solution including problem instance and metadata"
                    )

            except Exception as e:
                st.error(f"❌ Error during optimization: {str(e)}")
                st.info("ℹ️ Please check your input parameters and try again.")