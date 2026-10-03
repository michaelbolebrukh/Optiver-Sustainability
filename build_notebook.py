from pathlib import Path
import textwrap
import nbformat as nbf

cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(s).strip()))
def code(s): cells.append(nbf.v4.new_code_cell(textwrap.dedent(s).strip()))

md(r'''
# Carbon portfolios that survive project failures
### Optiver · AdaHack 2026

**Decision:** buy the least expensive diversified portfolio that meets the carbon target, within the budget and chosen risk mix. Measure survival under correlated failures separately.

**Start here:** run all cells, then use the **risk appetite** and **budget** controls near the end. Default target: **100,000 tonnes CO₂e**; challenge spending ceiling: **$1,000,000**. The brief omits a unit after 100,000; tonnes CO₂e is our explicit interpretation, consistent with `available_tonnes`.

These are purchases of carbon credits, with no supplied coupons, redemption cash flows, resale prices or return history. We use credit-risk mathematics for *carbon delivery*, not bond valuation. Simulations are **model-based stress tests, not historical backtests**. A carbon-credit purchase also does not directly reduce the buyer's gross emissions.
''')
md('''
## 1 · Sources, rules and assumptions

- Local `data/CREDITS.csv`: all 4,355 rows exported from the [organiser's database](https://docs.google.com/spreadsheets/d/1d3YKqXgVyYUkYA9aYVlmE6ryc5MrrasY/edit), accessed 3 October 2026. Its README is preserved in `data/source_rules.txt`.
- Underlying projects: Quartson et al. (2026), [Berkeley Voluntary Registry Offsets Database v2026-06](https://gspp.berkeley.edu/berkeley-carbon-trading-project/offsets-database), CC BY 4.0. Prices and ratings are **synthetic organiser inputs**.
- Organiser failure probabilities: AAA 1%, AA 2%, A 4%, BBB 7%, BB 12%, B 20%, CCC 35%, unrated 15%. A recorded reversal multiplies probability by 1.5, capped at 100%. A buffer pool recovers 50% of losses upon failure.
- No time horizon or correlations are supplied. Probabilities apply to one unspecified **challenge horizon**. Correlation strengths and risk-allocation policy below are explicit, editable modelling assumptions.
- The supplied GHG conversion-factor workbook converts activity into emissions. Without activity quantities it cannot determine this portfolio's target; the challenge already supplies that target.

We retain every project status: the organiser describes these rows as having unretired credits. Registry-specific eligibility and additionality would require further due diligence beyond the supplied rules.
''')
code('''
from pathlib import Path
import json, hashlib, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
import pycountry
from scipy import sparse
from scipy.optimize import linprog
from scipy.special import ndtr, ndtri
from IPython.display import display, Markdown
import ipywidgets as widgets

ROOT = Path.cwd()
DATA = ROOT / 'data' / 'CREDITS.csv'
OUT = ROOT / 'outputs'
OUT.mkdir(exist_ok=True)
SEED = 20261003
TARGET = 100_000.0
CHALLENGE_BUDGET = 1_000_000.0
TRAIN_SCENARIOS = 1024
TEST_SCENARIOS = 10_000
TAIL_ALPHA = 0.05  # Mean of the worst 5% of modelled outcomes
SAFETY_MARGIN = 0.12  # Selected on validation seeds; final test uses a separate seed
COLORS = ['#087f8c', '#e3a23b', '#b5485d']
plt.rcParams.update({'figure.figsize': (11, 4.8), 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.titleweight': 'bold', 'font.size': 10,
    'axes.prop_cycle': plt.cycler(color=COLORS), 'figure.facecolor': '#ffffff'})
pio.templates.default = 'plotly_white'
pio.renderers.default = 'plotly_mimetype+notebook'

raw = pd.read_csv(DATA)
required = ['credit_id','project_name','registry','status','scope','project_type',
 'reduction_or_removal','region','country','developer','vintage_year',
 'available_tonnes','price_usd_per_t','risk_rating','has_buffer_pool','had_reversal']
assert set(required).issubset(raw.columns), 'Missing required columns'
assert raw.credit_id.is_unique, 'Duplicate project IDs need investigation'
df = raw.copy()
for col in ['price_usd_per_t','available_tonnes','vintage_year']:
    df[col] = pd.to_numeric(df[col], errors='raise')
    assert np.isfinite(df[col]).all(), f'Non-finite {col}'
assert (df.price_usd_per_t > 0).all() and (df.available_tonnes > 0).all()
for col in set(required) - {'risk_rating'}:
    assert df[col].notna().all(), f'Missing values in {col}'
for col in ['has_buffer_pool','had_reversal']:
    assert set(df[col]) <= {'Yes','No'}, f'Unknown boolean labels: {col}'
df['risk_rating'] = df.risk_rating.fillna('Unrated')
BASE_P = {'AAA':.01,'AA':.02,'A':.04,'BBB':.07,'BB':.12,'B':.20,'CCC':.35,'Unrated':.15}
assert set(df.risk_rating) <= BASE_P.keys(), 'Unknown ratings'
df['p_fail'] = (df.risk_rating.map(BASE_P) * np.where(df.had_reversal.eq('Yes'),1.5,1)).clip(upper=1)
df['recovery'] = np.where(df.has_buffer_pool.eq('Yes'),.5,0.)
df['expected_retention'] = 1 - df.p_fail * (1-df.recovery)
df['usd_per_expected_tonne'] = df.price_usd_per_t / df.expected_retention
RISK_BANDS = {'AAA':'Low','AA':'Low','A':'Low','BBB':'Medium','BB':'Medium',
              'Unrated':'Medium','B':'High','CCC':'High'}
df['risk_band'] = df.risk_rating.map(RISK_BANDS)
BANDS = ['Low','Medium','High']
print(f'{len(df):,} projects · {df.country.nunique()} countries · {df.scope.nunique()} scopes')
print(f"{raw.risk_rating.isna().sum():,} unrated projects retained with the organiser's 15% base failure probability.")
print('CSV SHA-256:', hashlib.sha256(DATA.read_bytes()).hexdigest())
display(raw.describe(include='all').T)
''')
md('''
## 2 · Explore geography and every source field

Country counts describe the opportunity set, not our portfolio. Credit availability is an inventory limit, not project quality. Prices and risk ratings reflect the organiser's simulation. Project IDs/names are identifiers; we audit uniqueness rather than fit them as features.
''')
code('''
country_aliases = {'DRC':'COD','Democratic Republic of the Congo':'COD','Republic of the Congo':'COG',
                  'South Korea':'KOR','North Korea':'PRK','Taiwan':'TWN','Tanzania':'TZA',
                  'Vietnam':'VNM','Laos':'LAO','Bolivia':'BOL','Venezuela':'VEN',
                  'Russia':'RUS','Kosovo':None}
def iso3(name):
    if name in country_aliases: return country_aliases[name]
    try: return pycountry.countries.lookup(name).alpha_3
    except LookupError: return None

def world_map(frame, value, title):
    g = frame.groupby('country',as_index=False)[value].sum()
    g['iso3'] = g.country.map(iso3)
    missing = g.loc[g.iso3.isna(),'country'].tolist()
    if missing: print('Unmapped countries (still included in totals):', ', '.join(missing))
    fig = px.choropleth(g.dropna(subset=['iso3']), locations='iso3', color=value,
        hover_name='country', color_continuous_scale='Teal', title=title,
        projection='natural earth')
    fig.update_layout(margin=dict(l=0,r=0,t=55,b=0),height=460)
    fig.show()
    return fig

universe_map = world_map(df.assign(projects=1),'projects','Project universe · number of projects by country')
fig, axes = plt.subplots(1,3,figsize=(15,4))
for ax,col,label,log in zip(axes,['price_usd_per_t','available_tonnes','vintage_year'],
                          ['Price (USD / tCO₂e)','Available credits (tCO₂e)','Vintage year'],[True,True,False]):
    vals = np.log10(df[col]) if log else df[col]
    ax.hist(vals,bins=35,color=COLORS[0],alpha=.85)
    ax.set(xlabel=('log₁₀ ' if log else '')+label,ylabel='Projects',title=label)
fig.tight_layout(); plt.show()

cats = ['country','region','registry','scope','project_type','reduction_or_removal',
        'developer','status','risk_rating','has_buffer_pool','had_reversal']
fig,axes = plt.subplots(4,3,figsize=(17,20))
for ax,col in zip(axes.flat,cats):
    counts = df[col].value_counts()
    top = counts.head(8)
    if len(counts)>8: top = pd.concat([top,pd.Series({'Other':counts.iloc[8:].sum()})])
    top.sort_values().plot.barh(ax=ax,color=COLORS[0])
    ax.set(title=col.replace('_',' ').title(),xlabel='Number of projects')
axes.flat[-1].axis('off')
fig.tight_layout(); plt.show()
display(raw.isna().sum().rename('Missing values').to_frame())
''')
md('''
## 3 · Group by scope and choose a risk policy

`scope` already supplies ten interpretable clusters of projects. We use this taxonomy directly: clustering categorical labels with arbitrary numeric encodings would add no scientific value. Here scope means **project sector**, not corporate GHG Scope 1/2/3.

The appetite slider interpolates spending targets: **0 → 80% Low / 20% Medium / 0% High**, **100 → 20% Low / 30% Medium / 50% High**. These are our policy choices, not statistically estimated optimal weights. Unrated projects remain visible within Medium with their own 15% probability. Recovery and reversals enter the delivery model separately.

Targets allow ±5 percentage points, except a zero High target has an exact zero upper bound. Percentages apply to **money actually spent**; unspent budget stays available. Appetite changes the mix; it does not promise a monotonic realised failure rate.
''')
code('''
scope_summary = df.groupby('scope').agg(projects=('credit_id','size'),
    available_tonnes=('available_tonnes','sum'),median_price=('price_usd_per_t','median'),
    mean_failure_probability=('p_fail','mean')).sort_values('projects',ascending=False)
display(scope_summary)
cross = pd.crosstab(df.scope,df.risk_band).reindex(columns=BANDS,fill_value=0)
cross.plot.barh(stacked=True,color=COLORS,figsize=(11,5))
plt.xlabel('Projects'); plt.ylabel('Project scope'); plt.title('Scope groups and rating-based risk bands')
plt.tight_layout(); plt.show()

def risk_policy(appetite):
    if not np.isfinite(appetite) or not 0 <= appetite <= 100: raise ValueError('Appetite must be 0–100')
    a=appetite/100
    target=np.array([.8-.6*a,.2+.1*a,.5*a])
    low=np.maximum(0,target-.05)
    high=np.minimum(1,target+.05)
    high[target==0]=0
    return pd.DataFrame({'target':target,'minimum':low,'maximum':high},index=BANDS)
display(risk_policy(50).style.format('{:.0%}'))
''')
md(r'''
## 4 · Correlated delivery and portfolio optimization

One latent Gaussian factor is shared by projects in each country, scope, project type, developer and registry, plus a market-wide factor and project-specific noise. A threshold turns this into a failure event with the supplied marginal probability. The coefficients below are **latent variance shares**, not observed default correlations. Recovery is immediate and follows the game rules.

For purchased tonnes $q_i$, price $c_i$, recovery $r_i$ and failure indicator $F_{si}$, scenario delivery is
$$D_s=\sum_i q_i[1-(1-r_i)F_{si}].$$
Minimize $\sum_i c_iq_i$, subject to:

- Budget, inventory and risk-band spending limits.
- Global spending caps: project 8%, country 35%, scope 60%, developer 15%, registry 70%.
- **Within every risk band:** project 20%, country 55%, scope 75%, developer 35% of that band's spending. This prevents a diversified total from hiding a concentrated sleeve.
- Expected delivery ≥ target; mean delivery in the **worst 5% of training scenarios** ≥ 112% of target by default.

The last condition is a lower-tail CVaR constraint, expressed linearly using $\eta-z_s\le D_s$, $z_s\ge0$ and $\eta-\frac{1}{\alpha S}\sum_s z_s\ge1.12T$. This is a conservative **sample constraint**, not a guaranteed real-world success probability. The 12% cushion is calibrated on validation scenarios using seeds 20261503 and 20261603. The final reported test uses a separate seed 20270003. [SciPy's HiGHS solver](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html) handles the linear program.
''')
code('''
FACTOR_SHARES = {'global':.05,'country':.15,'scope':.05,'project_type':.10,'developer':.15,'registry':.05}
GLOBAL_CAPS = {'credit_id':.08,'country':.35,'scope':.60,'developer':.15,'registry':.70}
SLEEVE_CAPS = {'credit_id':.20,'country':.55,'scope':.75,'developer':.35}

def correlated_uniforms(data, n, seed, factor_scale=1.):
    rng=np.random.default_rng(seed)
    shares={k:v*factor_scale for k,v in FACTOR_SHARES.items()}
    total=sum(shares.values())
    if any(v<0 for v in shares.values()) or total>=1: raise ValueError('Factor variance shares must sum to <1')
    latent=np.sqrt(1-total)*rng.standard_normal((n,len(data)))
    for key,share in shares.items():
        if key=='global': latent+=np.sqrt(share)*rng.standard_normal((n,1))
        else:
            labels,unique=pd.factorize(data[key],sort=True)
            latent+=np.sqrt(share)*rng.standard_normal((n,len(unique)))[:,labels]
    return ndtr(latent)

def retention_scenarios(data,n,seed,factor_scale=1.,probability_multiplier=1.):
    u=correlated_uniforms(data,n,seed,factor_scale)
    p=np.minimum(1,data.p_fail.to_numpy()*probability_multiplier)
    return 1-(u<p)*(1-data.recovery.to_numpy())

TRAIN_R = retention_scenarios(df,TRAIN_SCENARIOS,SEED)

def optimize_portfolio(appetite=50,budget=CHALLENGE_BUDGET,target=TARGET,
                       diversified=True,tail=True,margin=SAFETY_MARGIN):
    if not np.isfinite(budget) or not 0<budget<=CHALLENGE_BUDGET:
        raise ValueError('Budget must be positive and no more than $1,000,000')
    if not np.isfinite(target) or target<=0: raise ValueError('Target must be positive')
    if margin<0 or not np.isfinite(margin): raise ValueError('Margin must be nonnegative')
    policy=risk_policy(appetite)
    n=len(df); m=len(TRAIN_R) if tail else 0
    # q/target; total cost/budget; three sleeve costs/budget; eta/target; z/target.
    total=n; bandpos={b:n+1+j for j,b in enumerate(BANDS)}
    eta=n+4; z0=n+5; size=n+5+m
    c=df.price_usd_per_t.to_numpy()*target/budget
    rr=[]; cc=[]; vv=[]; rhs=[]
    def row(indices,values,b=0.):
        k=len(rhs); rr.extend([k]*len(indices)); cc.extend(indices); vv.extend(values); rhs.append(b)
    def group_cap(indices,cap,denominator):
        row(list(indices)+[denominator],list(c[indices])+[-cap])
    row(list(range(n)),list(-df.expected_retention),-1.)
    for band in BANDS:
        pos=bandpos[band]
        row([pos,total],[1,-policy.loc[band,'maximum']])
        row([pos,total],[-1,policy.loc[band,'minimum']])
    if diversified:
        for col,cap in GLOBAL_CAPS.items():
            for indices in df.groupby(col).indices.values(): group_cap(indices,cap,total)
        for band in BANDS:
            for col,cap in SLEEVE_CAPS.items():
                for indices in df[df.risk_band.eq(band)].groupby(col).groups.values():
                    group_cap(np.asarray(indices),cap,bandpos[band])
    A=sparse.coo_matrix((vv,(rr,cc)),shape=(len(rhs),size)).tocsr()
    if tail:
        scenario=sparse.hstack([-sparse.csr_matrix(TRAIN_R),sparse.csr_matrix((m,4)),
            sparse.csr_matrix(np.ones((m,1))),-sparse.eye(m)],format='csr')
        tailrow=sparse.csr_matrix(([ -1.]+[1/(TAIL_ALPHA*m)]*m,
            ([0]*(m+1),[eta]+list(range(z0,size)))),shape=(1,size))
        A=sparse.vstack([A,scenario,tailrow],format='csr')
        rhs=rhs+[0.]*m+[-(1+margin)]
    er=[]; ec=[]; ev=[]
    for k,band in enumerate([None]+BANDS):
        indices=np.arange(n) if band is None else np.where(df.risk_band.eq(band))[0]
        er.extend([k]*(len(indices)+1)); ec.extend(list(indices)+[total if band is None else bandpos[band]])
        ev.extend(list(c[indices])+[-1.])
    E=sparse.coo_matrix((ev,(er,ec)),shape=(4,size)).tocsr()
    bounds=list(zip(np.zeros(n),df.available_tonnes.to_numpy()/target))+[(0,1)]+[(0,None)]*3+[(None,None)]+[(0,None)]*m
    objective=np.zeros(size); objective[:n]=c
    result=linprog(objective,A_ub=A,b_ub=np.array(rhs),A_eq=E,b_eq=np.zeros(4),bounds=bounds,method='highs')
    if not result.success:
        return {'success':False,'message':result.message,'appetite':appetite,'budget':budget,'target':target}
    q=np.maximum(result.x[:n],0)*target
    p=df.copy(); p['tonnes']=q; p['spend_usd']=q*df.price_usd_per_t
    p['expected_tonnes']=q*df.expected_retention
    p=p[p.tonnes>1e-7].copy()
    p['weight']=p.spend_usd/p.spend_usd.sum()
    return dict(success=True,portfolio=p,quantities=q,cost=float(p.spend_usd.sum()),
        appetite=appetite,budget=budget,target=target,policy=policy,diversified=diversified,tail=tail,
        margin=margin,solver_max_violation=float(max(0,np.max(A@result.x-np.array(rhs)))))
''')
md(r'''
## 5 · Independent validation and portfolio results

We report the probability of retaining 100,000 tonnes, its 95% Wilson sampling interval, expected delivery, 5th percentile, mean of the worst 5%, carbon shortfall and acquisition cost. The interval measures Monte Carlo sampling uncertainty **conditional on the assumed model**; it does not cover uncertain ratings or correlations.

Concentration uses the Herfindahl index: $1/\sum w_i^2$ is an effective count, not proof of independent exposures. Dollar-weighted and tonne-weighted project counts can differ.
''')
code('''
def tail_mean(values,alpha=TAIL_ALPHA):
    x=np.sort(np.asarray(values)); mass=len(x)*alpha; k=int(np.floor(mass))
    return float((x[:k].sum()+(mass-k)*x[min(k,len(x)-1)])/mass)

def evaluate(result,n=TEST_SCENARIOS,seed=SEED+9000,factor_scale=1.,probability_multiplier=1.):
    p=result['portfolio']
    retained=retention_scenarios(p,n,seed,factor_scale,probability_multiplier)@p.tonnes.to_numpy()
    target=result['target']; successes=int(np.sum(retained>=target)); phat=successes/n; z=1.96
    center=(phat+z*z/(2*n))/(1+z*z/n)
    half=z*np.sqrt(phat*(1-phat)/n+z*z/(4*n*n))/(1+z*z/n)
    stats={'cost_usd':result['cost'],'nominal_tonnes':p.tonnes.sum(),
      'analytic_expected_tonnes':p.expected_tonnes.sum(),'simulated_mean_tonnes':retained.mean(),
      'p05_tonnes':np.quantile(retained,.05),'worst_5pct_mean_tonnes':tail_mean(retained),
      'target_success_probability':phat,'success_ci_low':center-half,'success_ci_high':center+half,
      'expected_shortfall_tonnes':np.maximum(target-retained,0).mean(),
      'projects':len(p),'countries':p.country.nunique(),'scopes':p.scope.nunique(),
      'effective_projects_spend':1/np.sum(p.weight**2),
      'effective_projects_tonnes':1/np.sum((p.tonnes/p.tonnes.sum())**2)}
    return stats,retained

def audit(result):
    p=result['portfolio']; spend=p.spend_usd.sum(); target=result['target']
    checks={'Budget':spend<=result['budget']+.01,
       'Inventory':bool((p.tonnes<=p.available_tonnes+1e-3).all()),
       'Nonnegative quantities':bool((p.tonnes>=0).all()),
       'Expected carbon target':p.expected_tonnes.sum()>=target-.01}
    for band in BANDS:
        share=p.loc[p.risk_band.eq(band),'spend_usd'].sum()/spend
        lo,hi=result['policy'].loc[band,['minimum','maximum']]
        checks[f'{band} spending band']=lo-1e-6<=share<=hi+1e-6
    if result['diversified']:
        for col,cap in GLOBAL_CAPS.items():
            checks[f'Global {col} cap']=p.groupby(col).spend_usd.sum().max()/spend<=cap+1e-6
        for band,g in p.groupby('risk_band'):
            for col,cap in SLEEVE_CAPS.items():
                checks[f'{band} {col} cap']=g.groupby(col).spend_usd.sum().max()/g.spend_usd.sum()<=cap+1e-6
    if result['tail']:
        checks['Training lower-tail carbon target']=tail_mean(TRAIN_R@result['quantities'])>=target*(1+result['margin'])-.1
    assert all(checks.values()), {k:v for k,v in checks.items() if not v}
    return pd.Series(checks,name='Satisfied')

def plot_portfolio(result,retained):
    p=result['portfolio']; target=result['target']
    fig,axes=plt.subplots(1,3,figsize=(16,4.5))
    for ax,col in zip(axes[:2],['risk_band','scope']):
        s=p.groupby(col).spend_usd.sum()/result['cost']*100
        s.sort_values().plot.barh(ax=ax,color=COLORS[0]); ax.set(xlabel='Share of spending (%)',ylabel='',title='Allocation by '+col.replace('_',' '))
    axes[2].hist(retained,bins=45,color=COLORS[0],alpha=.8)
    axes[2].axvline(target,color=COLORS[2],ls='--',label='Carbon target')
    axes[2].set(xlabel='Retained credits (tCO₂e)',ylabel='Simulated outcomes',title='Independent correlated stress test')
    axes[2].legend(); fig.tight_layout(); plt.show()
    return world_map(p,'spend_usd','Selected portfolio · spending by country (USD)')

def present(result,make_map=True):
    if not result['success']:
        display(Markdown('**No feasible allocation under these constraints.** '+result['message']+
          ' Increase the budget or explicitly revise the risk/diversification policy. No constraint was silently relaxed.'))
        return None
    checks=audit(result); stats,retained=evaluate(result)
    p=result['portfolio']
    display(Markdown(f"**${stats['cost_usd']:,.0f} spent** · {stats['nominal_tonnes']:,.0f} tonnes purchased · "
       f"{stats['projects']} projects · {stats['countries']} countries · {stats['scopes']} scopes"))
    display(Markdown(f"**Modelled target success: {stats['target_success_probability']:.1%}** "
       f"(95% sampling interval {stats['success_ci_low']:.1%}–{stats['success_ci_high']:.1%}). "
       f"Independent worst-5% mean: **{stats['worst_5pct_mean_tonnes']:,.0f} tCO₂e**."))
    if stats['worst_5pct_mean_tonnes']<result['target']:
        display(Markdown('**Independent tail test falls below target.** The training solution passed, but robustness did not generalise. Increase scenario count or the safety margin before relying on it.'))
    if make_map: plot_portfolio(result,retained)
    display(pd.DataFrame({'Target':result['policy'].target,
       'Actual':p.groupby('risk_band').spend_usd.sum().reindex(BANDS,fill_value=0)/result['cost']}).style.format('{:.1%}'))
    display(p[['credit_id','project_name','country','scope','risk_rating','tonnes','spend_usd','weight']]
            .sort_values('spend_usd',ascending=False).head(20).style.format({'tonnes':'{:,.1f}','spend_usd':'${:,.0f}','weight':'{:.1%}'}))
    print(f'All {len(checks)} deterministic / training checks satisfied. Full allocation exported below.')
    return stats,retained

selected=optimize_portfolio()
presentation=present(selected)
''')
md('''
## 6 · Benchmarks, correlation sensitivity and failure paths

Compare to a cost-only expected-delivery portfolio, then add tail protection, then diversification. Benchmarks keep the same risk-band policy. These are simulated alternatives, not historical returns; diversification's benefit must be measured. Higher acquisition cost can also purchase a larger carbon cushion, so the ablation is not a pure causal estimate.

The path chart tracks **retained carbon**, not a security price. Failure times use a constant-hazard interpolation over 12 equal fractions of the unspecified challenge horizon. The supplied probabilities are terminal probabilities, not monthly or annual rates. A failed project's recovered fraction remains available under the game's buffer rule.
''')
code('''
benchmarks={'Expected delivery only':optimize_portfolio(diversified=False,tail=False),
            'Tail protection only':optimize_portfolio(diversified=False,tail=True),
            'Tail + diversification':selected}
benchmark_rows=[]
for name,result in benchmarks.items():
    if result['success']:
        stats,_=evaluate(result,seed=SEED+200)
        benchmark_rows.append(dict(strategy=name,**stats))
comparison=pd.DataFrame(benchmark_rows).set_index('strategy')
display(comparison[['cost_usd','target_success_probability','p05_tonnes','worst_5pct_mean_tonnes','effective_projects_spend']])
fig,axes=plt.subplots(1,2,figsize=(13,4))
comparison.cost_usd.plot.barh(ax=axes[0],color=COLORS[0]); axes[0].set(xlabel='Acquisition cost (USD)',ylabel='')
(100*comparison.target_success_probability).plot.barh(ax=axes[1],color=COLORS[0]); axes[1].set(xlabel='Modelled target success (%)',ylabel='',xlim=(0,102))
fig.tight_layout(); plt.show()

if selected['success']:
    stress=[]
    for label,scale,mult in [('Independent failures',0,1),('Base correlation',1,1),
                             ('Stronger shared shocks',1.5,1),('Failure probabilities +50%',1,1.5)]:
        stats,_=evaluate(selected,seed=SEED+300,factor_scale=scale,probability_multiplier=mult)
        stress.append(dict(scenario=label,**stats))
    stress_table=pd.DataFrame(stress).set_index('scenario')
    display(stress_table[['target_success_probability','p05_tonnes','worst_5pct_mean_tonnes','expected_shortfall_tonnes']])
    p=selected['portfolio']; u=correlated_uniforms(p,2000,SEED+400)
    horizon=np.linspace(0,1,13)
    paths=np.column_stack([(1-(u<(1-(1-p.p_fail.to_numpy())**t))*(1-p.recovery.to_numpy()))@p.tonnes.to_numpy() for t in horizon])
    q05,q50,q95=np.quantile(paths,[.05,.5,.95],axis=0)
    fig,ax=plt.subplots(figsize=(12,4.5))
    ax.plot(horizon,paths[:30].T,color=COLORS[0],alpha=.10)
    ax.fill_between(horizon,q05,q95,color=COLORS[0],alpha=.18,label='5th–95th percentile')
    ax.plot(horizon,q50,color=COLORS[0],lw=2,label='Median')
    ax.axhline(TARGET,color=COLORS[2],ls='--',label='Target')
    ax.set(xlabel='Fraction of challenge horizon (time scale assumed)',ylabel='Retained carbon (tCO₂e)',title='Correlated failure paths · no historical price data')
    ax.legend(); fig.tight_layout(); fig.savefig(OUT/'carbon_failure_paths.png',dpi=160); plt.show()
''')
md('''
## 7 · Portfolio calculator

Choose appetite and maximum spending, then click **Build portfolio**. The optimizer may leave money unspent because the competition rewards minimum acquisition cost. A small budget can be infeasible; the interface reports this explicitly. The saved outputs above remain visible in a static viewer; controls require an active Jupyter kernel.
''')
code('''
risk_control=widgets.IntSlider(value=50,min=0,max=100,step=1,description='Risk appetite',continuous_update=False,style={'description_width':'110px'},layout=widgets.Layout(width='600px'))
budget_control=widgets.IntSlider(value=1_000_000,min=10_000,max=1_000_000,step=10_000,description='Budget (USD)',continuous_update=False,style={'description_width':'110px'},layout=widgets.Layout(width='600px'))
run_button=widgets.Button(description='Build portfolio',button_style='info')
live_output=widgets.Output()

def export_result(result,stats,prefix='portfolio'):
    result['portfolio'].sort_values('spend_usd',ascending=False).to_csv(OUT/f'{prefix}_allocation.csv',index=False)
    payload=dict(stats=stats,appetite=result['appetite'],budget=result['budget'],target=result['target'],
        assumptions=dict(factor_variance_shares=FACTOR_SHARES,global_spending_caps=GLOBAL_CAPS,
        within_band_spending_caps=SLEEVE_CAPS,tail_alpha=TAIL_ALPHA,training_cushion=result['margin'],
        training_scenarios=TRAIN_SCENARIOS,test_scenarios=TEST_SCENARIOS,seed=SEED),
        source_sha256=hashlib.sha256(DATA.read_bytes()).hexdigest())
    (OUT/f'{prefix}_summary.json').write_text(json.dumps(payload,indent=2,default=float))

def rebuild(_):
    run_button.disabled=True
    try:
        with live_output:
            live_output.clear_output(wait=True)
            result=optimize_portfolio(risk_control.value,budget_control.value)
            report=present(result)
            if report:
                export_result(result,report[0])
                print('Updated outputs/portfolio_allocation.csv and outputs/portfolio_summary.json')
    finally: run_button.disabled=False

run_button.on_click(rebuild)
display(widgets.VBox([risk_control,budget_control,run_button,live_output]))
if selected['success']: export_result(selected,presentation[0],prefix='default_portfolio')
''')
md('''
## 8 · Model interpretation and limitations

The allocation minimizes acquisition cost subject to carbon, inventory, budget and risk-allocation constraints. Diversification limits concentration within each risk band, while correlated simulations account for common developer, sector, country and registry exposures.

Simulation results are conditional on the supplied failure probabilities and assumed dependence structure. They do not establish real-world failure correlations, guarantee environmental outcomes or represent historical strategy performance. No financial return is estimated because the dataset contains no credit-price history or project cash flows.

Further calibration requires a defined scoring horizon, empirical failure dependence and project-level evidence on eligibility, additionality and permanence. Source data and model assumptions are documented above.
''')
nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}})
nbf.write(nb,Path('carbon_portfolio.ipynb'))
print('Created carbon_portfolio.ipynb')
