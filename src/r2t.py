import sqlite3
import os
import pandas as pd
import numpy as np
from datetime import datetime
from ortools.linear_solver import pywraplp
import matplotlib.pyplot as plt



query = """
    SELECT Supplier.S_SUPPKEY, Customer.C_CUSTKEY, 
           L_EXTENDEDPRICE * (1 - L_DISCOUNT) as calculated_price
    FROM Supplier, Lineitem, Orders, Customer
    WHERE Supplier.S_SUPPKEY = Lineitem.L_SUPPKEY 
          AND Lineitem.L_ORDERKEY = Orders.O_ORDERKEY
          AND Orders.O_CUSTKEY = Customer.C_CUSTKEY
          AND Orders.O_ORDERDATE <= '1997-03-01'
    """

def create_database_and_tables():
    """Create SQLite database and import table data"""
    # Connect to SQLite database (create if not exists)
    conn = sqlite3.connect(':memory:')
    
    # Get project root directory path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    tbl_dir = os.path.join(project_root, 'tbl')
    
    # Read table data
    print("Reading table data...")
    
    # Read supplier table (TPC-H: S_SUPPKEY, S_NAME, S_ADDRESS, S_NATIONKEY, S_PHONE, S_ACCTBAL, S_COMMENT)
    supplier_df = pd.read_csv(os.path.join(tbl_dir, 'supplier.tbl'), sep='|', header=None, 
                             names=['S_SUPPKEY', 'S_NAME', 'S_ADDRESS', 'S_NATIONKEY', 'S_PHONE', 'S_ACCTBAL', 'S_COMMENT'],
                             usecols=range(7))
    
    # Read customer table (TPC-H: C_CUSTKEY, C_NAME, C_ADDRESS, C_NATIONKEY, C_PHONE, C_ACCTBAL, C_MKTSEGMENT, C_COMMENT)
    customer_df = pd.read_csv(os.path.join(tbl_dir, 'customer.tbl'), sep='|', header=None,
                             names=['C_CUSTKEY', 'C_NAME', 'C_ADDRESS', 'C_NATIONKEY', 'C_PHONE', 'C_ACCTBAL', 'C_MKTSEGMENT', 'C_COMMENT'],
                             usecols=range(8))
    
    # Read orders table (TPC-H: O_ORDERKEY, O_CUSTKEY, O_ORDERSTATUS, O_TOTALPRICE, O_ORDERDATE, O_ORDERPRIORITY, O_CLERK, O_SHIPPRIORITY, O_COMMENT)
    orders_df = pd.read_csv(os.path.join(tbl_dir, 'orders.tbl'), sep='|', header=None,
                           names=['O_ORDERKEY', 'O_CUSTKEY', 'O_ORDERSTATUS', 'O_TOTALPRICE', 'O_ORDERDATE', 'O_ORDERPRIORITY', 'O_CLERK', 'O_SHIPPRIORITY', 'O_COMMENT'],
                           usecols=range(9))
    
    # Read lineitem table (TPC-H: L_ORDERKEY, L_PARTKEY, L_SUPPKEY, L_LINENUMBER, L_QUANTITY, L_EXTENDEDPRICE, L_DISCOUNT, L_TAX, L_RETURNFLAG, L_LINESTATUS, L_SHIPDATE, L_COMMITDATE, L_RECEIPTDATE, L_SHIPINSTRUCT, L_SHIPMODE, L_COMMENT)
    lineitem_df = pd.read_csv(os.path.join(tbl_dir, 'lineitem.tbl'), sep='|', header=None,
                             names=['L_ORDERKEY', 'L_PARTKEY', 'L_SUPPKEY', 'L_LINENUMBER', 'L_QUANTITY', 'L_EXTENDEDPRICE', 'L_DISCOUNT', 'L_TAX', 'L_RETURNFLAG', 'L_LINESTATUS', 'L_SHIPDATE', 'L_COMMITDATE', 'L_RECEIPTDATE', 'L_SHIPINSTRUCT', 'L_SHIPMODE', 'L_COMMENT'],
                             usecols=range(16))
    
    # Write data to SQLite database
    print("\nWriting data to database...")
    supplier_df.to_sql('Supplier', conn, if_exists='replace', index=False)
    customer_df.to_sql('Customer', conn, if_exists='replace', index=False)
    orders_df.to_sql('Orders', conn, if_exists='replace', index=False)
    lineitem_df.to_sql('Lineitem', conn, if_exists='replace', index=False)
    
    print("Database creation completed!")
    return conn


def solve_linear_programming(result_df, tau):
    """
    Use OR-Tools to solve linear programming optimal solution
    
    Parameters:
        result_df: DataFrame containing query results with S_SUPPKEY, C_CUSTKEY, calculated_price columns
        tau: constraint parameter
    
    Returns:
        optimal_u: optimal u value array
        optimal_value: optimal objective function value
    """
    n = len(result_df)  # Number of query results
    
    if n == 0:
        return [], 0
    
    # Create solver
    solver = pywraplp.Solver.CreateSolver('GLOP')
    if not solver:
        return np.zeros(n), 0
    
    # Create variables u_i, each query result corresponds to one variable
    u = []
    for i in range(n):
        # u_i >= 0, upper bound is the corresponding calculated_price
        u.append(solver.NumVar(0, result_df.iloc[i]['calculated_price'], f'u_{i}'))
    
    # Constraint 1: u_i <= calculated_price (already included in variable definition)
    
    # Constraint 2: For each customer key, sum of all corresponding supplier key u_i <= tau
    customer_keys = result_df['C_CUSTKEY'].unique()
    for customer in customer_keys:
        # Find all rows corresponding to this customer
        customer_indices = result_df[result_df['C_CUSTKEY'] == customer].index
        # Add constraint: sum(u_i for i in customer_indices) <= tau
        solver.Add(solver.Sum([u[i] for i in customer_indices]) <= tau)
    
    # Constraint 3: For each supplier key, sum of all corresponding customer key u_i <= tau
    supplier_keys = result_df['S_SUPPKEY'].unique()
    for supplier in supplier_keys:
        # Find all rows corresponding to this supplier
        supplier_indices = result_df[result_df['S_SUPPKEY'] == supplier].index
        # Add constraint: sum(u_i for i in supplier_indices) <= tau
        solver.Add(solver.Sum([u[i] for i in supplier_indices]) <= tau)
    
    # Objective function: maximize sum(u_i)
    # solver.Maximize(solver.Sum(u))
    objective = solver.Objective()
    for i in range(n):
        objective.SetCoefficient(u[i], 1)
    objective.SetMaximization()
    
    # Solve
    status = solver.Solve()
    
    if status == pywraplp.Solver.OPTIMAL:
        optimal_u = [u[i].solution_value() for i in range(n)]
        optimal_value = solver.Objective().Value()
        # optimal_value = sum(optimal_u)
        return optimal_value
    else:
        print(f"Solving failed, status: {status}")
        return np.zeros(n), 0

def differential_privacy(optimal_value, tau, gsq, epsilon, beta):
    """
    Calculate race to top value
    """
    logOfGsq = np.log2(gsq)
    lapTarget = logOfGsq * tau / epsilon
    noise = np.random.laplace(loc=0.0, scale=lapTarget)
    bias = logOfGsq * np.log(logOfGsq / beta) * (tau / epsilon)
    r2t_result = optimal_value + noise - bias
    return r2t_result

def draw_graph(taus, optimal_values, r2t_results, sql_price_sum):
    """
    Draw comparison graph of optimal_value and r2t_result under different tau values, and show SQL sum dashed line
    
    Parameters:
        taus: list of tau values
        optimal_values: corresponding optimal_value list
        r2t_results: corresponding r2t_result list
        sql_price_sum: SQL direct sum result
    """
    plt.figure(figsize=(12, 8))
    
    # Create evenly spaced x-axis positions
    x_positions = list(range(len(taus)))
    
    # Filter data points where r2t_result >= 0
    filtered_data = [(i, tau, opt_val, r2t_val) for i, (tau, opt_val, r2t_val) in enumerate(zip(taus, optimal_values, r2t_results)) if r2t_val >= 0]
    
    if filtered_data:
        filtered_x_positions, filtered_taus, filtered_optimal_values, filtered_r2t_results = zip(*filtered_data)
    else:
        filtered_x_positions, filtered_taus, filtered_optimal_values, filtered_r2t_results = [], [], [], []
    
    # Plot optimal_value (blue dots)
    plt.scatter(x_positions, optimal_values, color='blue', label=r"$Q(I, \tau)$", alpha=0.7, s=50)
    
    # Plot r2t_result (red dots, only plot values >= 0)
    if filtered_r2t_results:
        plt.scatter(filtered_x_positions, filtered_r2t_results, color='red', label=r"$\tilde{Q}(I, \tau)$", alpha=0.7, s=50)
    
    # Plot SQL direct sum dashed line
    plt.axhline(sql_price_sum, color='grey', linestyle='--', linewidth=2, label=r"$Q(I)$")
    
    # Set graph properties
    plt.xlabel(r"$\tau$", fontsize=12)
    plt.ylabel('Calculated Price', fontsize=12)
    plt.title('Optimal Value vs R2T Result under Different tau Values', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Set x-axis ticks, each tau value corresponds to one tick, same spacing
    plt.xticks(x_positions, [str(tau) for tau in taus], rotation=45)
    
    # Adjust layout
    plt.tight_layout()
    
    # Show graph
    plt.show()

def main():
    """Main function"""
    print("Starting to create database and tables...")

    # Parameters setting
    gsq = 10 ** 9
    epsilon = 0.8
    beta = 1.2

    # Create database and tables
    conn = create_database_and_tables()
    
    result_df = pd.read_sql_query(query, conn)

    # Execute query
    print(result_df)
    print(f"\nNumber of query results: {len(result_df)}")
    
    # Calculate SQL direct sum
    sql_price_sum = result_df['calculated_price'].sum()

    
    # Close connection
    conn.close()
    
    taus = [2**i for i in range(1, int(np.log2(gsq)))]
    
    # Solve linear programming for different tau values
    print(f"\nLinear programming solution results:")

    r2t_results_list = []
    error_list = []
    optimal_values = []
    opt_values_list = {}

    for tau in taus:
        optimal_value = solve_linear_programming(result_df, tau)
        opt_values_list[tau] = optimal_value
        optimal_values.append(optimal_value)

    for i in range(1, 101):
         # Store results for plotting
        r2t_results = []
        used_taus = []
        
        for tau in taus[:28]:  # Only test first 28 tau values
            # Get optimal_value from pre-computed dictionary
            optimal_value = opt_values_list[tau]
            print(f"tau = {tau:>8}: optimal_value = {optimal_value:>12.2f}")

            r2t_result = differential_privacy(optimal_value, tau, gsq, epsilon, beta)
            print(f"R2T tau = {tau:>8}: r2t_result = {r2t_result:>12.2f}")
            
            # Store results
            used_taus.append(tau)
            if r2t_result < 0:
                break
            r2t_results.append(r2t_result)

        r2t_results_list.append(max(r2t_results))
        error = sql_price_sum - max(r2t_results)
        error_list.append(error)
        
        print(f"\n--The {i} round--")
        print(f"sql_price_sum = {sql_price_sum}")
        print(f"optimal_ui_sum = {sum(optimal_values)}")
        print(f"max_r2t_result = {max(r2t_results)}")
        print(f"error = {error}")

    print(f"\n\n-------final result-------")
    print(f"sql_price_sum = {sql_price_sum}")
    # print(f"r2t_results_list = {r2t_results_list}")
    print(f"average_r2t_result = {sum(r2t_results_list) / len(r2t_results_list)}")
    # print(f"error_list = {error_list}")
    sorted_errors = sorted(error_list)
    # Remove the largest 20 values and smallest 20 values
    trimmed_errors = sorted_errors[20:-20]
    avg_error = sum(trimmed_errors) / len(trimmed_errors)
    print(f"Average error after removing largest 20 values and smallest 20 values: {avg_error:.2f}")
    


    # Draw graph
    print("\nDrawing graph...")
    # Extract tau values and optimal values from dictionary for plotting
    plot_taus = list(opt_values_list.keys())[:28]  # Use first 28 tau values
    plot_optimal_values = [opt_values_list[tau] for tau in plot_taus]
    # Calculate final r2t_results for plotting
    final_r2t_results = []
    for tau in plot_taus:
        r2t_result = differential_privacy(opt_values_list[tau], tau, gsq, epsilon, beta)
        final_r2t_results.append(r2t_result)
    
    draw_graph(plot_taus, plot_optimal_values, final_r2t_results, sql_price_sum)
  
    print("\nProgram execution completed!")

if __name__ == "__main__":
    main()
