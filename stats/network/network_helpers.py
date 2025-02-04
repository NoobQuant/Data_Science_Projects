import scipy as sc
import numpy as np
import pandas as pd
from copy import deepcopy

def run_dynamics(args):
    """Helper to run DebtRank implementation in parallel."""
    node, W, ids, nodes_dr = args
    ds = debtrank_dynamics(
        W_in=W,
        ids=ids,
        S_f=[node],
    )
    dr = debtrank_calculate(ds, nodes_dr)
    return {"ds": ds, "dr": dr, "S_f": node}

def prepare_data_for_debtrank(edges, nodes):
    """Prepare network data for DebtRank calculation."""
    df = edges.copy()

    # Get data
    df = (
        edges[["source", "target", "equity_investment"]]
        .copy()
        .rename(columns={
            "source": "j",
            "target": "i",
            "equity_investment": "A_ji",
        })
    )
    df = pd.merge(
        df,
        (
            nodes
            .copy()
            .reset_index()
            .loc[:, ["node_id", "equity"]]
            .rename(columns={
                "node_id": "j",
                "equity": "E_j",
            })
        ),
        how="left",
        left_on="j",
        right_on="j"
    )

    # The total value of the assets invested by j in funding activities.
    # Again, notice that in our case \sum_l A_jl = \sum_l Z_jl.
    df = pd.merge(
        df,
        df.groupby("j").agg(A_j=("A_ji", "sum")).reset_index(),
        how="left",
        left_on="j",
        right_on="j",
    )

    # Impact of i on j can be calculated directly
    df["W_ij"] = np.where(
        df["A_ji"] / df["E_j"] > 1,
        1,
        df["A_ji"] / df["E_j"]
    )

    # Relative economic value of j; notice denominator is equivalent to
    # \sum_j A_j when taken over edges frame
    df["v_j"] = df["A_j"] / df["A_ji"].sum()

    # Multiplication of W_ij and v_j
    df["W_ij_x_v_j"] = df["W_ij"] * df["v_j"]

    # Direct part of the impact of i on its neighbors
    df = pd.merge(
        df,
        df.groupby("i").agg(I_i_direct=("W_ij_x_v_j", "sum")).reset_index(),
        how="left",
        left_on="i",
        right_on="i",
    )

    # Append v_j to nodes frame
    nodes_out = pd.merge(
        nodes,
        (
            df
            .groupby("j")
            .agg({"v_j": "first"})
            .rename_axis("node_id")
            .reindex(nodes.index, fill_value=0)
        ),
        how="outer",
        left_index=True,
        right_index=True,
    )

    edges_out = df.copy()
    return edges_out, nodes_out

def get_adjacency_matrix(df):
    """Get adjacency matrix W from prepared edges frame.

    Parameters
    ----------
    df : pandas.DataFrame
        Edges frame with columns j, i, and W_ij.

    Returns
    -------
    W : scipy.sparse._csr.csr_matrix
        Adjacency matrix with values W_ij. In same sparse as
        nx.adjacency_matrix() would return.
    ids : list
        Node IDs in the order they appear in rows/columns of W.
    """
    # Sorted list of all IDs
    ids = (
        pd
        .concat([df["j"], df["i"]], ignore_index=True)
        .drop_duplicates().sort_values()
        .tolist()
    )
 
    # Index position for each entity in the list
    entity_index = {entity: idx for idx, entity in enumerate(ids)}

    # Initialize lists to store row indices, column indices, and data
    row_indices = []
    col_indices = []
    data = []

    # Iterate over edges
    for _, row in df.iterrows():
        # Get indices of j and i, will be non-zero elements in W
        row_indices.append(entity_index[row["i"]])
        col_indices.append(entity_index[row["j"]])
        data.append(row["W_ij"])

    # Create the CSR matrix
    W = sc.sparse.csr_matrix(
        (data, (row_indices, col_indices)),
        shape=(len(ids), len(ids))
    )

    return W, ids

def debtrank_dynamics(
    W_in,
    ids,
    S_f,
    **kwargs
):
    """Simulate DebtRank distress propagation.

    Parameters
    ----------
    W_in : scipy.sparse.csr_matrix
        Adjacency matrix of the impacts.
    ids : list
        Node IDS.
    S_f : list
        List of initially stressed node IDs.
    psi : float, optional
        Amount of stress. Applies uniformly to all stressed nodes. 
    iter_limit : in, optional
        Number of maximum iterations.

    Returns
    -------
    pd.DataFrame
        Frame with distress and state per node at the end of simulation.
    """
    assert isinstance(W_in, sc.sparse.csr_matrix)
    psi = kwargs.get("psi", 1.0) # Initial stress amount
    iter_limit = kwargs.get("iter_limit", 100) # Iteration limit

    ###################
    # Step 1
    ##################
    # Transpose W and make sure we operate with csr_matrix afterwards
    W = deepcopy(W_in).T
    W = W.tocsr()

    # Initial stress vector
    statevec_h = np.array([0]*W.shape[0])
    statevec_h[[ids.index(x) for x in S_f]] = psi
    statevec_h_init = deepcopy(statevec_h)

    # Initial state vector
    statevec_s = np.array(["U"]*W.shape[0])
    statevec_s[[ids.index(x) for x in S_f]] = "D"

    ###################
    # Helper function for W alterations
    ##################
    def copy_csr_data(X):
        """Copying array directly is slow, hence this helper function."""
        return X.data.copy(), X.indices.copy(), X.indptr.copy()

    def return_csr_data(data, indices, indptr, shape):
        """Converse of copy_csr_data."""
        return(
            sc.sparse.csr_matrix(
                (data.copy(), indices.copy(), indptr.copy()),
                shape=shape,
            )
        )
    def zero_constrained_w(X, statevec_s):
        """Create zero-constrained version of the input matrix."""
        # Copy data of the original matrix
        X2_data = X.data.copy()

        # Columns from state vector to be constrained to zero
        cols_to_zero = np.where(statevec_s!="D")[0]

        # Populate data for the constrained matrix 
        for i in range(X.shape[0]):
            start_idx = X.indptr[i]
            end_idx = X.indptr[i + 1]
            for j in range(start_idx, end_idx):
                if X.indices[j] in cols_to_zero:
                    X2_data[j] = 0

        # Return new sparse matrix with constrained data
        return sc.sparse.csr_matrix(
            (X2_data, X.indices.copy(), X.indptr.copy()),
            shape=X.shape
        )

    ###################
    # Step >=2
    ##################
    for counter in range(iter_limit):
        # This does not seem too slow
        statevec_h_prev = deepcopy(statevec_h)
        
        # Copy data of W
        original_data, original_indices, original_indptr = copy_csr_data(W)

        # Zero-constrained W
        W = zero_constrained_w(W, statevec_s)

        # Update stress vector
        statevec_h = statevec_h + W.dot(statevec_h)
        statevec_h = np.where(statevec_h > 1.0, 1.0, statevec_h)

        # Return data of W
        W = return_csr_data(
            original_data, original_indices, original_indptr, W.shape)

        # Update vector of visit state. Condition order matters: I needs to come
        # before D, as we want to designated cases h(t-1) > 0 and s(t-1)!=D as I
        # instead of D.
        statevec_s = np.where(
            (statevec_s=="D"), "I", np.where(
                (statevec_h > 0.0) & (statevec_s != "I"), "D", statevec_s
            )
        )

        # Check if stress vector did not change
        if np.all(np.isclose(statevec_h_prev, statevec_h)):
            break

        # Error if hit the iteration limit
        if counter==(iter_limit-1):
            print(ValueError("iter_limit encountered!"))
            break
    
    # Merge results and return
    return (
        pd.concat(
            [
                pd.DataFrame(statevec_h, index=ids, columns=["h"]),
                pd.DataFrame(statevec_s, index=ids, columns=["s"]),
                pd.DataFrame(statevec_h_init, index=ids, columns=["h_init"]),
            ],
            axis=1,
        )
        .rename_axis("node_id", axis=0)
    )

def debtrank_calculate(distress, nodes):
    """Calculate DebtRank from simulated distress.

    distress : pd.DataFrame
        Output from debtrank_dynamics.
    nodes : pd.DataFrame
        Nodes frame.
    """
    df = pd.merge(
        distress,
        nodes,
        how="outer",
        left_index=True,
        right_index=True,
    )
    df["dr_T"] = df["h"] * df["v_j"]
    df["dr_1"] = df["h_init"] * df["v_j"]
    return df["dr_T"].sum() - df["dr_1"].sum()
