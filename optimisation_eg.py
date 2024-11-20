import pyomo.environ as pyo # requires to have Pyomo installed
import pandas # requires to have pandas and openpyxl installed

l_t = list(range(24))
load = pandas.read_excel('dades_dem_cat.xlsx',sheet_name="Full2", header=None, index_col=0)

Pload = {t: float(load[1][t+1]) for t in l_t}

model = pyo.ConcreteModel()

model.t = pyo.Set(initialize=l_t)

model.Pnuc = pyo.Var(within=pyo.NonNegativeReals)
model.Ppgas = pyo.Var(within=pyo.NonNegativeReals)
model.Pgas_t = pyo.Var(model.t, within=pyo.NonNegativeReals)

def Constraint_balance(m, t):
    return m.Pnuc + m.Pgas_t[t] == Pload[t]

model.Constr_balance = pyo.Constraint(model.t, rule=Constraint_balance)

def Constraint_gas_size(m, t):
    return m.Pgas_t[t] <= m.Ppgas
model.Constr_gas_size = pyo.Constraint(model.t, rule=Constraint_gas_size)

def func_obj(m):
    CAPEX_nuc = m.Pnuc * 3600 * 10**3
    OPEX_nuc = m.Pnuc * 0.02 * 10**3 * 8760
    CAPEX_gas = m.Ppgas * 823 * 10**3
    OPEX_gas = sum(m.Pgas_t[t] * 1 for t in l_t) *365 * 0.15 * 10**3
    return CAPEX_nuc + OPEX_nuc * 50 + CAPEX_gas + OPEX_gas * 50

model.goal = pyo.Objective(rule=func_obj, sense=pyo.minimize)

opt = pyo.SolverFactory('highs')
results = opt.solve(model)

print(model.Pnuc.get_values())
model.Pgas_t.get_values()