"""Carbon-credit cost optimization, correlated failure simulation and constraint validation."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog
from scipy.special import ndtr

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'CREDITS.csv'
SEED = 20261003
TARGET = 100_000.0
CHALLENGE_BUDGET = 1_000_000.0
TRAIN_SCENARIOS = 1024
TEST_SCENARIOS = 10_000
TAIL_ALPHA = 0.05  # Mean of the worst 5% of modelled outcomes
SAFETY_MARGIN = 0.12  # Selected on validation seeds; final test uses a separate seed

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

def risk_policy(appetite):
    if not np.isfinite(appetite) or not 0 <= appetite <= 100: raise ValueError('Appetite must be 0–100')
    a=appetite/100
    target=np.array([.8-.6*a,.2+.1*a,.5*a])
    low=np.maximum(0,target-.05)
    high=np.minimum(1,target+.05)
    high[target==0]=0
    return pd.DataFrame({'target':target,'minimum':low,'maximum':high},index=BANDS)

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
