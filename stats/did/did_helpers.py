import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

def simulate_did_data(
    param_datasettype,
    **kwargs,
):
    r"""Function to simulate data for difference-in-differences settings.

    Can create either panel or repeated cross-section samples of data. See
    notebook did_simulated_datasets.ipynb for description of the data-
    generating process.

    Parameters
    ----------
    param_datasettype : str
        Either 'panel' or 'repeated cross-section'.
    param_no_t : int, optional
        Number of time points.
    param_last_pre_timepoint : int, optional
        Which time point is the last in 'pre'.
    param_N : int, optional
        Total number of individuals.
    param_treatprob : float, optional
        Probability of being in group j=T.
    param_tau : float, optional
        Treatment effect.
    param_treat_eff_trend : boolean, optional
        Is treatment effect linearly growing? Defaults to False.
    param_gamma_c : float, optional
        Control group specific constant effect.
    param_gamma_t : float, optional
        Treatment group specific constant effect.
    param_xi : lambda, optional
        Time-specific effect.
    param_xtype : str, optional
        Type of covariateX form. Default to None, in which case X=0.
    param_x_kwargs : dict, optional
        Default kwargs for different param_xtype choices.
    param_sigma_e : float, optional
        Standard deviation in epsilon ~ N(0, param_sigma_e).

    Returns
    -------
    dict
        Frames with observed data for control and treatment groups,
        counterfactual treatment group data, as well as means calculated for
        treatment/control/counterfactual treatment.

    Notes
    -------
    Currently panel data does not allow for systematic individual effects. That
    is, panel data is basically repeated cross-sample with observations in
    time-domain lumped under unique individual IDs. Should be made more flexible
    in terms of panel data.

    Currently no staggered treatment is allowed.
    """

    # Optional parameters
    param_no_t = kwargs.get("param_no_t", 6)
    param_last_pre_timepoint = kwargs.get(
        "param_last_pre_timepoint", int(param_no_t/2) - 1)
    param_N = kwargs.get("param_N", 600)
    param_treatprob = kwargs.get("param_treatprob", 0.5)
    param_tau = kwargs.get("param_tau", -2)
    param_treat_eff_trend = kwargs.get("param_treat_eff_trend", False)
    param_gamma_c = kwargs.get("param_gamma_c", 0)
    param_gamma_t = kwargs.get("param_gamma_t", 0)
    param_xi = kwargs.get("param_xi", lambda t: t)
    param_xtype = kwargs.get("param_xtype", None)
    param_x_kwargs = kwargs.get("param_x_kwargs", {
        "param_mu_T": 2,
        "param_sigma_T": 1,
        "param_mu_C": 1,
        "param_sigma_C": 1,
        "param_trendcoef_T": 0.5,
        "param_trendcoef_C": -0.5
    })
    param_sigma_e = kwargs.get("param_sigma_e", 1)

    # Check input parameters
    assert param_datasettype in ["panel", "repeated cross-section"], \
        "Invalid parameter dataset_type!"
    assert param_xtype in [None, "X1", "X5a"], \
        "Invalid parameter param_xtype!"

    # Initiate data frame for individuals by allocating them to T and C groups
    df = pd.DataFrame(
        data={
            "t": np.repeat(range(param_no_t), param_N),
            "treatment_group": pd.Series(
                np.tile(
                    np.random.binomial(
                        n=1,
                        p=param_treatprob,
                        size=param_N
                    ),
                    reps=param_no_t
                )
            ).map({0: "control", 1: "treatment"}),
            "epsilon": np.random.normal(
                loc=0,
                scale=param_sigma_e,
                size=param_N*param_no_t
            )
        }
    )

    # Append pre/post time group
    df["time_group"] = np.where(
        df["t"]>param_last_pre_timepoint,
        "post",
        "before"
    )

    # Append individual IDs
    if param_datasettype == "panel":
        df["id"] = np.tile(range(param_N), param_no_t)
    elif param_datasettype == "repeated cross-section":
        df["id"] = [x for x in range(param_N*param_no_t)]
    df["id"] = df["id"].astype(str)

    # Append group-specific constant effect
    df["gamma"] = np.where(
        df["treatment_group"]=="control",
        param_gamma_c,
        param_gamma_t
    )

    # Append time-specific effect
    df["xi"] = df["t"].apply(param_xi)

    # Model individual-group-time specific effect
    # Note that this effect is called "X" in the frame, but actually it
    # encompasses the whole term lambda*X
    if param_xtype is None:
        df["X"] = 0
    elif param_xtype in ["X1", "X5a"]:
        to_be_asserted = (
            "param_mu_T", "param_sigma_T", "param_mu_C", "param_sigma_C")
        if param_xtype == "X5a":
            to_be_asserted = to_be_asserted + (
                "param_trendcoef_T", "param_trendcoef_C")

        assert all(k in param_x_kwargs for k in to_be_asserted), (
            "param_x_kwargs does not contain needed parameters for covariate "
            "type {}"
        ).format(param_xtype)

        df["X_T"] = np.tile(
            np.random.normal(
                loc=param_x_kwargs["param_mu_T"],
                scale=param_x_kwargs["param_sigma_T"],
                size=param_N
            ),
            reps=param_no_t
        )
        df["X_C"] = np.tile(
            np.random.normal(
                loc=param_x_kwargs["param_mu_C"],
                scale=param_x_kwargs["param_sigma_C"],
                size=param_N
            ),
            reps=param_no_t
        )
        if param_xtype == "X1":
            df["X"] = np.where(
                df["treatment_group"]=="control",
                df["X_C"],
                df["X_T"]
            )
        elif param_xtype == "X5a":
            df["X"] = np.where(
                df["treatment_group"]=="control",
                df["X_C"] + param_x_kwargs["param_trendcoef_C"]*df["t"],
                df["X_T"] + param_x_kwargs["param_trendcoef_T"]*df["t"],
            )
        del df["X_T"]
        del df["X_C"]

    # Model the outcome

    # Non-treated potential outcomes. For group C these also observed values,
    # but for group T these are couterfactual values for the post period
    df["Y"] = df["gamma"] + df["xi"] + df["X"] + df["epsilon"]

    # Save frame with counterfactual treated values
    df_cf = df[df["treatment_group"]=="treatment"].copy()

    # Treatment effect
    if param_treat_eff_trend:
        # Linear trend in treatment effect
        df["treatment_effect"] = np.where(
            (df["treatment_group"]=="treatment") & (df["time_group"]=="post"),
            (df["t"] - param_last_pre_timepoint) * param_tau,
            0,
        )
    else:
        # Constant treatment effect
        df["treatment_effect"] = np.where(
            (df["treatment_group"]=="treatment") & (df["time_group"]=="post"),
            param_tau,
            0,
        )

    # Now append interaction to non-treated potential outcomes to get observed
    # values frame
    df["Y"] = df["Y"].copy() + df["treatment_effect"].copy()

    # Calculate and save mean values of outcome from observed values
    means = {}
    for tg in ["control", "treatment"]:
        for pg in ["before", "post"]:
            means[tg + "_" + pg] = df.loc[
                (df["treatment_group"]==tg) &
                (df["time_group"]==pg),
                "Y"
            ].mean()
        for t in range(param_no_t):
            means[tg + "_t" + str(t)] = df.loc[
                (df["treatment_group"]==tg) &
                (df["t"]==t),
                "Y"
            ].mean()

    # Calculate and save mean values of outcome from counterfactual values
    for tg in ["treatment"]:
        for pg in ["post"]:
            means[tg + "_cf" + "_" + pg] = df_cf.loc[
                (df_cf["treatment_group"]==tg) &
                (df_cf["time_group"]==pg),
                "Y"
            ].mean()
        for t in range(param_last_pre_timepoint+1, param_no_t):
            means[tg + "_cf" + "_t" + str(t)] = df_cf.loc[
                (df_cf["treatment_group"]==tg) &
                (df_cf["t"]==t),
                "Y"
            ].mean()

    # Return frames and calculated means
    return {
        "observed": df,
        "treatment_cf": df_cf,
        "means": means
    }

def plot_panel_data(data, selected_individuals=None, **kwargs):
    r"""Plots simulated panel data.

    Parameters
    ----------
    data : dict
        Output from simulate_did_data() with dataset_type='panel'.
    selected_individuals : list/None, optional
        If list, contains IDs of individuals to plot.
    same_ylim : optional
        Use same ylim to control and treatment plots?.
    **kwargs : dict
    """

    figsize = kwargs.get("figsize", (12, 6))
    same_ylim = kwargs.get("same_ylim", False)

    df = data["observed"].copy()
    df_cf = data["treatment_cf"].copy()
    if selected_individuals is not None:
        df = df[df["id"].isin(selected_individuals)]
        df_cf = df_cf[df_cf["id"].isin(selected_individuals)]

    # Infer last pre time point from control group of observed data
    last_pre_timepoint = data["observed"].loc[
        data["observed"]["time_group"]=="before",
        "t"
    ].drop_duplicates().iloc[-1]

    # Figure skeleton
    fig = plt.figure(figsize=figsize)

    # Treatment group
    ax = fig.add_subplot(2, 1, 2)
    df_t = df[df["treatment_group"]=="treatment"].copy().pivot_table(
        values="Y",
        index="t",
        columns="id"
    )
    df_t.plot(
        ax=ax,
        linestyle="-",
        marker="o",
        linewidth=1.0,
        color=cm.rainbow(np.linspace(0, 1, len(df_t.columns))),
    )

    # Counterfactual treatment
    df_t = df_cf.copy().pivot_table(
        values="Y",
        index="t",
        columns="id"
    )
    df_t.plot(
        ax=ax,
        linestyle="--",
        marker="o",
        linewidth=1.0,
        alpha=0.5,
        color=cm.rainbow(np.linspace(0, 1, len(df_t.columns))),
        legend=False,
    )
    ax.axvline(x=last_pre_timepoint+0.5, color="black")
    ax.set_title("Treatment group (with observed and counterfactuals)")
    axylims_t = ax.get_ylim()

    # Control group
    ax = fig.add_subplot(2, 1, 1)
    df_t = df[df["treatment_group"]=="control"].copy().pivot_table(
        values="Y",
        index="t",
        columns="id"
    )
    df_t.plot(ax=ax, linestyle="-", marker="o", linewidth=0.5)
    ax.axvline(x=last_pre_timepoint+0.5, color="black")
    if same_ylim:
        ax.set_ylim(axylims_t)
    ax.set_title("Control group")

    fig.tight_layout()

def plot_repcrossec_data(data, prints="skip time points", **kwargs):
    r"""Plots simulated repeated cross-section data.

    Parameters
    ----------
    data : dict
        Output from simulate_did_data() with dataset_type='repeated
        cross-section'.
    prints : str, optional
        Which prints to display. Default to 'skip time points'.
    **kwargs : dict
    """

    figsize = kwargs.get("figsize", (14, 10))
    
    # Check inputs
    assert prints in ["all", "none", "skip time points"], \
        "Invalid parameter prints!"

    # Infer some parameters from the input data
    no_periods = len(data["observed"]["t"].unique())
    last_pre_timepoint = data["observed"].loc[
        data["observed"]["time_group"]=="before",
        "t"
    ].drop_duplicates().iloc[-1]

    # Get frames from input dict
    treatment = data["observed"].query("treatment_group=='treatment'").copy()
    control = data["observed"].query("treatment_group=='control'").copy()
    treatment_cf = data["treatment_cf"]
    means = data["means"]

    # Helpers for plots
    time_fe_mean_line_pos = [
        (0, 0.1),
        (0.175, 0.275),
        (0.375, 0.475),
        (0.55, 0.65),
        (0.75, 0.825),
        (0.90, 1)
    ]
    colors = return_colors()
    def format_ax(ax):
        ax.set_ylabel("Y")
        ax.axvline(last_pre_timepoint+0.5, linestyle="-", c=colors[0], linewidth=2)
        #ax.set_ylim(-4, 10)
        ax.legend()

    # Figure skeleton
    fig = plt.figure(figsize=figsize)

    # Control group
    ax = fig.add_subplot(2, 1, 1)
    control.plot.scatter(
        x="t", y="Y", ax=ax, s=1, c=colors[3], label="Realized observations")
    ax.axhline(y=means["control_before"], xmin=0, xmax=0.5, linestyle="--",
        c=colors[3], label="Realized pre/post-period means and individual time-point means")
    ax.axhline(y=means["control_post"], xmin=0.5, xmax=1.0, linestyle="--",
        c=colors[3])
    for t, xminmax in zip(range(no_periods), time_fe_mean_line_pos):
        ax.axhline(y=means["control_t" + str(t)], xmin=xminmax[0],
            xmax=xminmax[1], linestyle="--", c=colors[3])
    format_ax(ax)
    ax.set_title("Control group")
    ax.set_xlabel(None)

    # Treatment group
    ax = fig.add_subplot(2, 1, 2)
    # Realized
    treatment.plot.scatter(x="t", y="Y", ax=ax, s=1, c=colors[3],
        label="Realized observations")
    ax.axhline(
        y=means["treatment_before"],
        xmin=0,
        xmax=0.5,
        linestyle="--",
        c=colors[3],
        label="Realized pre/post-period means and individual time-point means"
    )
    ax.axhline(y=means["treatment_post"], xmin=0.5, xmax=1.0, linestyle="--",
        c=colors[3])
    for t, xminmax in zip(range(no_periods), time_fe_mean_line_pos):
        ax.axhline(y=means["treatment_t" + str(t)], xmin=xminmax[0],
            xmax=xminmax[1], linestyle="--", c=colors[3])
    # Counterfactual (unobserved)
    treatment_cf.query("time_group == 'after'").plot.scatter(
        x="t",
        y="Y",
        ax=ax,
        color=colors[6],
        s=1,
        label="Counterfactual observations (unobserved)"
    )
    ax.axhline(y=means["treatment_cf_post"], xmin=0.5, xmax=1.0, linestyle="--",
        c=colors[6], label="Counterfactual post-period mean (unobserved)")
    for t, xminmax in zip(
        range(last_pre_timepoint+1, no_periods),
        time_fe_mean_line_pos[last_pre_timepoint+1:]
    ):
        ax.axhline(y=means["treatment_cf_t" + str(t)], xmin=xminmax[0],
            xmax=xminmax[1], linestyle="--", color=colors[6])
    # Counterfactual (naively estimated)
    ax.axhline(
        y=means["treatment_before"] + (means["control_post"] - means["control_before"]),
        xmin=0.5,
        xmax=1.0,
        linestyle="--",
        color=colors[4],
        label="Counterfactual post-period mean (naively estimated)"
    )
    format_ax(ax)
    ax.set_title("Treatment group")
    fig.tight_layout()

    # Prints
    if prints in ["all", "skip time points"]:
        print("Realized control pre-period mean {:.3f}".format(
            means["control_before"]))
        print("Realized control post-period mean {:.3f}".format(
            means["control_post"]))
        print("Realized treated pre-period mean {:.3f}".format(
            means["treatment_before"]))
        print("Realized treated post-period mean {:.3f}".format(
            means["treatment_post"]))
        print(("Counterfactual (unobserved) treatment post-period "
            "mean {:.3f}").format(means["treatment_cf_post"]))

        # Per invidual time points
        if prints != "skip time points":
            for el1, el2, crt_periods in zip(
                ["control", "treatment", "treatment_cf"],
                ["Realized control", "Realized treated", "Counterfactual treated (unobserved)"],
                [range(no_periods), range(no_periods), range(last_pre_timepoint+1, no_periods)]
            ):
                for t in crt_periods:
                    print("{} t_{} mean {:.3f}".format(
                        el2,
                        t,
                        means[el1 + "_t" + str(t)],
                    ))

        # Naive DiD estimates
        print("Counterfactual (naively estimated) treated post-period mean {:.3f}".format(
            means["treatment_before"] + (means["control_post"] - means["control_before"])
        ))
        print("Naive DiD-estimate {:.3f}".format(
            (means["treatment_post"] - means["treatment_before"]) -
            (means["control_post"] - means["control_before"])
        ))

def parallel_trends_plot(data, **kwargs):
    r"""Plots group averages for parallel trends investigation.

    Parameters
    ----------
    data : dict
        Output from simulate_did_data().
    """

    figsize = kwargs.get("figsize", (12, 8))

    # Get frames separately for different groups
    treatment = data["observed"].query("treatment_group=='treatment'").copy()
    control = data["observed"].query("treatment_group=='control'").copy()

    # Infer some parameters from the input data
    last_pre_timepoint = data["observed"].loc[
        data["observed"]["time_group"]=="before",
        "t"
    ].drop_duplicates().iloc[-1]

    # Plot colors
    colors = return_colors()

    # Figure skeleton
    fig = plt.figure(figsize=figsize)

    # First axis: averages in absolute units
    ax = fig.add_subplot(2, 1, 1)
    treatment.groupby(["t"]).agg({"Y": "mean"}) \
        .rename(columns={"Y": "Treatment group"}) \
        .plot(ax=ax, linestyle="--", marker="o", c=colors[3])
    control.groupby(["t"]).agg({"Y": "mean"}) \
        .rename(columns={"Y": "Control group"}) \
        .plot(ax=ax, linestyle="--", marker="o", c=colors[5])
    ax.axvline(last_pre_timepoint+0.5, c=colors[0])
    ax.set_title("Average Y")

    # Second axis: averages in de-meaned units
    ax = fig.add_subplot(2, 1, 2)
    treatment.groupby(["t"]).agg({"Y": "mean"}) \
        .subtract(treatment.query("time_group == 'before'").loc[:, "Y"].mean()) \
        .rename(columns={"Y": "Treatment group"}) \
        .plot(ax=ax, linestyle="--", marker="o", c=colors[3])
    control.groupby(["t"]).agg({"Y": "mean"}) \
        .subtract(control.query("time_group == 'before'").loc[:, "Y"].mean()) \
        .rename(columns={"Y": "Control group"}) \
        .plot(ax=ax, linestyle="--", marker="o", c=colors[5])
    ax.axvline(last_pre_timepoint+0.5, c=colors[0])
    ax.set_title("De-meaned (using group's pre-period average) average Y")

    fig.tight_layout()

def return_colors():
    """# https://www.learnui.design/tools/data-color-picker.html"""
    return [
        "#003f5c",
        "#2f4b7c",
        "#665191",
        "#a05195",
        "#d45087",
        "#f95d6a",
        "#ff7c43",
        "#ffa600",
    ]
