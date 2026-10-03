# Carbon portfolio optimization

A carbon-credit portfolio calculator for the Optiver AdaHack 2026 challenge. The model minimizes acquisition cost while meeting carbon, inventory, budget, risk-allocation and diversification constraints.

## Run the app

Requires Python 3.11 or later. From the project folder:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

On Windows, activate the environment using `.venv/Scripts/activate`. Open the localhost URL printed by Streamlit, normally http://localhost:8501.

Choose risk appetite and maximum budget, then select **Calculate portfolio**. Results include geographic and sector allocations, correlated failure scenarios, comparisons with simpler portfolios and CSV/JSON downloads. Infeasible inputs are reported without relaxing constraints.

## Model

- Carbon target: 100,000 tonnes CO₂e. Maximum budget: $1,000,000.
- Objective: minimize spending, allowing unused budget.
- Inventory: purchased tonnes cannot exceed the project's available credits.
- Risk appetite sets spending ranges across three rating bands.
- Concentration limits apply globally and within each risk band.
- A lower-tail CVaR constraint requires the mean delivery in the worst 5% of 1,024 training scenarios to exceed 112,000 tonnes.
- The 12% cushion is calibrated on separate validation scenarios. Reported performance uses 10,000 independent test scenarios.

Prices and ratings are synthetic challenge inputs. Failure correlations, concentration limits and risk-allocation policy are explicit assumptions. Simulations are not historical backtests or financial-return forecasts; a feasible training allocation does not guarantee future carbon delivery.

## Project files

- `app.py`: Streamlit interface.
- `portfolio_model.py`: optimization, simulation and constraint checks.
- `carbon_portfolio.ipynb`: exploratory analysis, methodology and interactive notebook.
- `data/CREDITS.csv`: 4,355 project records.
- `data/source_rules.txt`: organiser's rules.
- `.streamlit/config.toml`: interface theme and local server settings.

Run the notebook with `jupyter lab carbon_portfolio.ipynb` and execute the cells in order.

## Data attribution

[Organiser workbook](https://docs.google.com/spreadsheets/d/1d3YKqXgVyYUkYA9aYVlmE6ryc5MrrasY/edit), retrieved 3 October 2026. Underlying project data: Pamela Quartson, Barbara K Haya, Tyler Bernard, Aline Abayo, Xinyun Rong, Ivy S So, Micah Elias (2026), [Voluntary Registry Offsets Database v2026-06](https://gspp.berkeley.edu/berkeley-carbon-trading-project/offsets-database), Berkeley Carbon Trading Project, University of California, Berkeley, CC BY 4.0. Synthetic prices and ratings were added by the organiser.

The brief's 100,000 target is interpreted as tonnes CO₂e, consistent with the dataset's inventory units.
