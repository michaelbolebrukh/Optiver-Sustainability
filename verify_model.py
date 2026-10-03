"""Check allocation constraints, infeasibility and failure recovery."""
import ast
import nbformat
import numpy as np

nb = nbformat.read('carbon_portfolio.ipynb', as_version=4)
ns = {}
exec(nb.cells[2].source.split("print(f'{len(df)")[0], ns)
for i in [6, 8, 10]:
    tree = ast.parse(nb.cells[i].source)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if i == 8:
        nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.Assign))]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<model>', 'exec'), ns)

for appetite in [0, 100]:
    result = ns['optimize_portfolio'](appetite=appetite)
    assert result['success'], result
    assert ns['audit'](result).all()
    if appetite == 0:
        assert not result['portfolio'].risk_band.eq('High').any()
    print(f'Appetite {appetite}: all constraints verified; cost ${result["cost"]:,.2f}', flush=True)

assert not ns['optimize_portfolio'](budget=10_000)['success']
for kwargs in [{'appetite':-1},{'appetite':101},{'budget':0},{'budget':1_000_001},{'target':0}]:
    try: ns['optimize_portfolio'](**kwargs)
    except ValueError: pass
    else: raise AssertionError(f'Invalid input accepted: {kwargs}')

tiny = ns['df'].iloc[:3].copy()
tiny['p_fail'] = [0.,1.,1.]
tiny['recovery'] = [0.,0.,.5]
retention = ns['retention_scenarios'](tiny,1000,123)
assert np.all(retention == np.array([1.,0.,.5]))
assert np.isclose(ns['tail_mean']([0,10,20,30], .375), 10/3)
print('Infeasibility, invalid inputs, certain failure, buffer recovery and fractional-tail checks passed.')
