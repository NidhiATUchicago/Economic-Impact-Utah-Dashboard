#Importing all the libraries
from shiny import App, ui, render, reactive
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Defining helper functions for all economic calculations

# Defining a function for money multiplier effect
def bank_multiplier(deposits, money_multiplier, reserve_ratio):
    available_loans = deposits * (1 - reserve_ratio)
    total_deposits = available_loans * money_multiplier
    new_loans = total_deposits * (1 - reserve_ratio)
    return available_loans, total_deposits, new_loans

#Defining function for income multiplier effect
def compute_income_multiplier(deposits, saving_rate, tax_rate, income_multiplier):
    """
    Returns: initial_spending, new_income, final_personal_income, tax_revenue
    """
    initial_spending = deposits * (1 - saving_rate)
    new_income = initial_spending * income_multiplier      # fixed spelling
    final_personal_income = new_income * (1 - saving_rate)
    tax_revenue = final_personal_income * tax_rate
    return initial_spending, new_income, final_personal_income, tax_revenue

# Defining the function for Utah Institution's Revenue as per Wong and Haslag (2006) and Kansas State Study
def marginal_revenue(deposits, tax_rate, mpk, m, m_star, i_in, i_out, mpk_mode="auto"):
    # Convert MPK if needed
    if mpk_mode == "gross" or (mpk_mode == "auto" and mpk > 1):
        mpk_net = mpk - 1
    else:
        mpk_net = mpk

    # Components of the model 
    tax_return = tax_rate * mpk_net * (m - m_star) * deposits
    interest_income = i_in * deposits
    interest_out = i_out * deposits
    marginal_revenue = tax_return + interest_income - interest_out

    return mpk_net, tax_return, interest_income, interest_out, marginal_revenue


# -----------------------------
# UI
# -----------------------------

app_ui = ui.page_navbar(
    ui.nav_panel(
        "Multiplier effect",
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_numeric("mm_deposits", "Base deposits", 14_000_000, min=0),
                ui.input_numeric("mm_ratio", "Reserve ratio (rr)", 0.20, min=0.0, max=0.99, step=0.005),
                ui.input_numeric("mm_mult", "Money multiplier (exogenous)", 2.0, min=0.1, step=0.1),
                ui.input_action_button("btn_mm", "Recalculate"),
                ui.hr(),
                ui.markdown(
                    """
                    **Model**: available loans = D·(1−rr); total deposits = available·MM; new loans = total·(1−rr).  
                    Use your empirically estimated **money multiplier** if you don't want to assume 1/rr.
                    """
                ),
                width=350,
            ),
            ui.card(
                ui.card_header("Summary"),
                ui.output_table("mm_table"),
            ),
            ui.card(
                ui.card_header("Flows from Base Deposits"),
                ui.output_plot("mm_plot", height="380px"),
            ),
        ),
    ),
    ui.nav_panel(
        "Income effect",
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_numeric("inc_deposits", "Deposits", 14_000_000, min=0),
                ui.input_numeric("inc_s", "Saving rate (s)", 0.20, min=0.0, max=1.0, step=0.01),
                ui.input_numeric("inc_t", "Tax rate (t)", 0.1009, min=0.0, max=1.0, step=0.0001),
                ui.input_numeric("inc_k", "Income multiplier (k)", 2.0, min=0.1, step=0.1),
                ui.input_action_button("btn_inc", "Recalculate"),
                ui.hr(),
                ui.markdown(
                    "**Kahn/Keynes multiplier framing**: deposits → spending → income; tax revenue = (1−s)·Y·t."
                ),
                width=350,
            ),
            ui.card(
                ui.card_header("Components"),
                ui.output_table("inc_table"),
            ),
            ui.card(
                ui.card_header("Income Cascade"),
                ui.output_plot("inc_plot", height="380px"),
            ),
        ),
    ),
    ui.nav_panel(
        "Sensitivity model",
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_numeric("s_dep", "Deposits", 14_000_000, min=0),
                ui.input_numeric("s_tax", "Tax rate", 0.1009, min=0.0, max=1.0, step=0.0001),
                ui.input_numeric("s_mpk", "Marginal Product of Capital", 1.5, step=0.001),
                ui.input_select("s_mpk_mode", "Marginal Product of Capital -  mode", {"auto":"auto","net":"net","gross":"gross"}, selected="auto"),
                ui.hr(),
                ui.h4("A) Relationship betweem Marginal Revenue of Utah Institutions and Utah Loan to Deposit Ratio (ceteris paribus)"),
                ui.input_slider("s_m_min", "Utah Loans to Deposit Ratio - min", 0.2, 1.0, 0.0, step=0.01),
                ui.input_slider("s_m_max", "Utah Loans to Deposit Ratio - max", 0.6, 1.0, 0.9, step=0.01),
                ui.input_numeric("s_m_steps", "Utah Loans to Deposit Ratio -  steps", 10, min=5, step=1),
                ui.input_numeric("s_i_out_fixed", "Interest Rate on Out of State Investment (fixed)", 0.05, step=0.0005),
                ui.input_numeric("s_mstar", "m* (benchmark)", 0.00, step=0.01),
                ui.input_numeric("s_i_in", "Interest Rate by Utah Banks (internal)", 0.04, step=0.0005),
                ui.input_action_button("btn_sens", "Build m-curve"),
                ui.hr(),
                ui.h4("B) Impact of Interest Rate Differential on Marginal Revenue of Utah Institutions (ceteris paribus"),
                ui.input_slider("s_m_fixed", "Utah Loan to Deposit (fixed)", 0.2, 1.0, 0.30, step=0.01),
                ui.input_numeric("s_iout_base", "Interest rate on Out of State Institutions base", 0.05, step=0.0005),
                ui.input_numeric("s_spread_min", "Δi min (i_in−i_out)", -0.02, step=0.0005),
                ui.input_numeric("s_spread_max", "Δi max (i_in−i_out)", 0.02, step=0.0005),
                ui.input_numeric("s_spread_steps", "Δi steps", 41, min=5, step=1),
                ui.input_action_button("btn_spread", "Build Δi-curve"),
                width=350,
            ),
            ui.card(
                ui.card_header("MR vs Utah Loan to Deposit"),
                ui.output_plot("sens_plot", height="380px"),
            ),
            ui.card(
                ui.card_header("Grid (first 30 rows) for m-curve"),
                ui.output_table("sens_table"),
            ),
            ui.card(
                ui.card_header("MR vs Interest Rate Differential)"),
                ui.output_plot("spread_plot", height="380px"),
            ),
            ui.card(
                ui.card_header("Grid (first 30 rows) for Δi-curve"),
                ui.output_table("spread_table"),
            ),
        ),
    ),
    title="Economic Impacts of PTIF Deposits in Utah Banks - Dashboard",
)

# -----------------------------
# Server logic
# -----------------------------

def server(input, output, session):

    # ---- Multiplier tab (bank_multiplier) ----
    @reactive.calc
    def mm_vals():
        _ = input.btn_mm()
        avail, tot_dep, new_loans = bank_multiplier(
            input.mm_deposits(), input.mm_mult(), input.mm_ratio()
        )
        return avail, tot_dep, new_loans

    @output
    @render.table
    def mm_table():
        avail, tot_dep, new_loans = mm_vals()
        df = pd.DataFrame({
            "Metric": ["Available loans", "Total deposits", "New loans"],
            "Value": [avail, tot_dep, new_loans],
        })
        for i in range(len(df)):
            df.loc[i, "Value"] = f"${df.loc[i,'Value']:,.0f}"
        return df

    @output
    @render.plot
    def mm_plot():
        avail, tot_dep, new_loans = mm_vals()
        fig, ax = plt.subplots(figsize=(7.5, 3.8))
        ax.bar(["Available loans", "Total deposits", "New loans"], [avail, tot_dep, new_loans])
        ax.set_ylabel("Dollars")
        ax.set_title("Money Multiplier Flows")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        return fig

    # ---- Income tab (compute_income_multiplier) ----
    @reactive.calc
    def inc_vals():
        _ = input.btn_inc()
        return compute_income_multiplier(
            input.inc_deposits(), input.inc_s(), input.inc_t(), input.inc_k()
        )

    @output
    @render.table
    def inc_table():
        init_spend, new_inc, fin_income, tax_rev = inc_vals()
        df = pd.DataFrame({
            "Component": [
                "Initial spending",
                "New income",
                "Final personal income",
                "Tax revenue",
            ],
            "Value": [init_spend, new_inc, fin_income, tax_rev],
        })
        for i in range(len(df)):
            df.loc[i, "Value"] = f"${df.loc[i,'Value']:,.0f}"
        return df

    @output
    @render.plot
    def inc_plot():
        init_spend, new_inc, fin_income, tax_rev = inc_vals()
        labels = ["Initial spending", "New income", "Final personal income", "Tax revenue"]
        vals = [init_spend, new_inc, fin_income, tax_rev]
        fig, ax = plt.subplots(figsize=(7.5, 3.8))
        ax.bar(labels, vals)
        ax.set_ylabel("Dollars")
        ax.set_title("Income Effect Cascade")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        ax.tick_params(axis="x", rotation=15)
        return fig

    # ---- Sensitivity tab (marginal_revenue): MR as function of m ----
    @reactive.calc
    def sens_df():
        _ = input.btn_sens()
        m_vals = np.linspace(input.s_m_min(), input.s_m_max(), int(input.s_m_steps()))
        rows = []
        for m in m_vals:
            mpk_net, tax_ret, inc_in, inc_out, mr = marginal_revenue(
                input.s_dep(), input.s_tax(), input.s_mpk(), m, input.s_mstar(), input.s_i_in(), input.s_i_out_fixed(), input.s_mpk_mode()
            )
            rows.append({
                "m": m,
                "mpk_net": mpk_net,
                "tax_return": tax_ret,
                "interest_income": inc_in,
                "interest_out": inc_out,
                "MR": mr,
            })
        return pd.DataFrame(rows)

    @output
    @render.plot
    def sens_plot():
        df = sens_df()
        fig, ax = plt.subplots(figsize=(8.0, 3.8))
        ax.plot(df["m"], df["MR"], marker="o")
        ax.axhline(0, linestyle="--", linewidth=1)
        ax.set_xlabel("m (share to productive lending)")
        ax.set_ylabel("MR")
        ax.set_title("MR Sensitivity to Utah Loan to Deposit Ratio")
        ax.grid(True, linestyle="--", alpha=0.3)
        return fig

    @output
    @render.table
    def sens_table():
        return sens_df().head(30)

    # ---- Sensitivity: MR vs interest-rate differential Δi = i_in − i_out (m fixed) ----
    @reactive.calc
    def spread_df():
        _ = input.btn_spread()
        spreads = np.linspace(float(input.s_spread_min()), float(input.s_spread_max()), int(input.s_spread_steps()))
        rows = []
        for s in spreads:
            i_in = float(input.s_iout_base()) + float(s)
            mpk_net, tax_ret, inc_in, inc_out, mr = marginal_revenue(
                float(input.s_dep()), float(input.s_tax()), float(input.s_mpk()), float(input.s_m_fixed()), float(input.s_mstar()), i_in, float(input.s_iout_base()), input.s_mpk_mode()
            )
            rows.append({
                "spread": s,
                "i_in": i_in,
                "i_out": float(input.s_iout_base()),
                "MR": mr,
            })
        return pd.DataFrame(rows)

    @output
    @render.plot
    def spread_plot():
        df = spread_df()
        fig, ax = plt.subplots(figsize=(8.0, 3.8))
        ax.plot(df["spread"], df["MR"], marker="o")
        ax.axhline(0, linestyle="--", linewidth=1)
        ax.axvline(0, linestyle=":", linewidth=1)
        ax.set_xlabel("Δi = i_in − i_out")
        ax.set_ylabel("MR")
        ax.set_title("MR Senstivity to Interest-Rate Differential")
        ax.grid(True, linestyle="--", alpha=0.3)
        return fig

    @output
    @render.table
    def spread_table():
        return spread_df().head(30)


app = App(app_ui, server)























