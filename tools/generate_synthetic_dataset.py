"""Reproducible, wholly synthetic HS-GAW benchmark. No clinical calibration.
Run: python generate_synthetic_dataset.py --output Synthetic_Health_Dataset
Requires numpy and pandas. Default seed 20261104, 400 people, 241 daily rows/person.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

SCENARIOS = ('independent_auxiliaries', 'weight_related', 'both_related', 'mixed_timescales')

def ar(rng, n, rho):
    x = np.empty(n)
    x[0] = rng.normal()
    for t in range(1, n):
        x[t] = rho*x[t-1] + np.sqrt(1-rho*rho)*rng.normal()
    return x

def build(output, people=400, seed=20261104):
    output.mkdir(parents=True, exist_ok=True)
    n = 241
    dates = pd.date_range('2024-01-01', periods=n)
    split_rng = np.random.default_rng(seed+1)
    order = split_rng.permutation(people)
    labels = np.empty(people, dtype=object)
    edges = [0, int(people*.6), int(people*.75), int(people*.85), people]
    for name, lo, hi in zip(('train','validation','calibration','test'),edges[:-1],edges[1:]):
        labels[order[lo:hi]] = name
    participants = []
    summary = {}
    for scenario in SCENARIOS:
        rng = np.random.default_rng(seed)
        records, examples = [], []
        for p in range(people):
            pid = f'SYN_{p+1:04d}'
            age = int(rng.integers(18,76))
            sex = str(rng.choice(['female','male']))
            mass = ar(rng,n,.995)
            activity = ar(rng,n,.94)
            independent = ar(rng,n,.98)
            hb_base = rng.uniform(11.5,15.5)
            weight_base = rng.uniform(45,95)
            step_base = rng.uniform(3500,10000)
            hb_noise = rng.normal(0,.13,n)
            w_noise = rng.normal(0,.15,n)
            step_noise = rng.normal(0,650,n)
            if scenario == 'independent_auxiliaries':
                hb_signal = .55*independent
            elif scenario == 'weight_related':
                hb_signal = .5*mass + .3*independent
            elif scenario == 'both_related':
                hb_signal = .4*mass + .3*activity + .3*independent
            else:
                hb_signal = .3*mass + .5*activity + .3*independent
            hb_true = np.clip(hb_base+hb_signal+hb_noise,7.5,18.5)
            weight_true = np.clip(weight_base+2*mass+w_noise,35,125)
            steps_true = np.rint(np.clip(step_base+1400*activity+step_noise,0,22000))
            # Scheduled sparse observations, jittered dates, independent missingness.
            hb_days = np.unique(np.clip(np.arange(0,n,30)+rng.integers(-3,4,len(np.arange(0,n,30))),0,n-1))
            w_days = np.unique(np.clip(np.arange(0,n,14)+rng.integers(-2,3,len(np.arange(0,n,14))),0,n-1))
            hb_obs = np.full(n,np.nan); w_obs = np.full(n,np.nan)
            hb_keep = rng.random(len(hb_days))>.15
            w_keep = rng.random(len(w_days))>.10
            hb_obs[hb_days[hb_keep]] = hb_true[hb_days[hb_keep]]
            w_obs[w_days[w_keep]] = weight_true[w_days[w_keep]]
            hb_obs[0] = hb_true[0]; w_obs[0] = weight_true[0]
            steps_obs = steps_true.copy()
            steps_obs[rng.random(n)<.10] = np.nan
            raw = pd.DataFrame({'participant_id':pid,'date':dates.strftime('%Y-%m-%d'),
                'hb':np.round(hb_obs,3),'weight':np.round(w_obs,3),'steps':steps_obs,
                'age':age,'gender':sex,'split':labels[p],'is_synthetic':1})
            records.append(raw)
            if scenario == SCENARIOS[0]:
                participants.append({'participant_id':pid,'split':labels[p],'age':age,'gender':sex,'is_synthetic':1})
            for t in np.flatnonzero(np.isfinite(hb_obs)):
                if t<4:
                    continue
                wt = np.flatnonzero(np.isfinite(w_obs[:t+1]))[-1]
                hist = steps_obs[t-4:t+1]
                if np.isfinite(hist).sum()<3:
                    continue
                base = {'participant_id':pid,'anchor_date':str(dates[t].date()),
                    'hb0':round(hb_obs[t],3),'weight0':round(w_obs[wt],3),
                    'hb_age_days':0,'weight_age_days':int(t-wt),'age':age,'gender':sex,
                    'steps0':float(np.nanmean(hist)),'valid_step_days':int(np.isfinite(hist).sum()),
                    'split':labels[p],'is_synthetic':1}
                for j,v in enumerate(hist):
                    base[f'steps_day_{j-4}'] = v
                    base[f'steps_missing_day_{j-4}'] = int(not np.isfinite(v))
                for h in (30,90):
                    target = t+h
                    if target+7>=n: # complete target search window required
                        continue
                    row = dict(base, horizon_days=h, target_date=str(dates[target].date()))
                    for k,obs in [('hb',hb_obs),('weight',w_obs)]:
                        eligible = np.flatnonzero(np.isfinite(obs) & (np.abs(np.arange(n)-target)<=7) & (np.arange(n)>t))
                        if len(eligible):
                            chosen = int(eligible[np.argmin(np.abs(eligible-target))])
                            row[f'y_{k}'] = round(obs[chosen],3)
                            row[f'{k}_target_observed_date'] = str(dates[chosen].date())
                            row[f'{k}_target_offset_days'] = chosen-target
                        else:
                            row[f'y_{k}'] = np.nan
                            row[f'{k}_target_observed_date'] = ''
                            row[f'{k}_target_offset_days'] = np.nan
                    future_steps = steps_obs[target-6:target+1]
                    row['target_step_valid_days'] = int(np.isfinite(future_steps).sum())
                    row['y_steps'] = float(np.nanmean(future_steps)) if row['target_step_valid_days']>=5 else np.nan
                    for k in ('hb','weight','steps'):
                        row[f'mask_{k}'] = int(np.isfinite(row[f'y_{k}']))
                    examples.append(row)
        folder = output/scenario
        folder.mkdir(exist_ok=True)
        daily = pd.concat(records,ignore_index=True)
        anchors = pd.DataFrame(examples)
        daily.to_csv(folder/'daily_records.csv',index=False,float_format='%.3f')
        anchors.to_csv(folder/'forecast_examples.csv',index=False,float_format='%.3f')
        # Meaningful structural checks: split integrity, temporal provenance, masks.
        assert len(daily)==people*n
        assert not daily.duplicated(['participant_id','date']).any()
        assert daily.groupby('participant_id')['split'].nunique().max()==1
        assert not anchors.duplicated(['participant_id','anchor_date','horizon_days']).any()
        assert (pd.to_datetime(anchors.target_date)-pd.to_datetime(anchors.anchor_date)).dt.days.eq(anchors.horizon_days).all()
        for k in ('hb','weight','steps'):
            assert anchors[f'mask_{k}'].eq(anchors[f'y_{k}'].notna().astype(int)).all()
        for k in ('hb','weight'):
            obs_dates = pd.to_datetime(anchors[f'{k}_target_observed_date'],errors='coerce')
            observed = obs_dates.notna()
            assert (obs_dates[observed]>pd.to_datetime(anchors.loc[observed,'anchor_date'])).all()
            assert anchors.loc[observed,f'{k}_target_offset_days'].abs().le(7).all()
        # Reconstruct representative examples directly from raw observations.
        indexed = daily.set_index(['participant_id','date'])
        for _,e in anchors.iloc[::max(1,len(anchors)//25)].iterrows():
            assert abs(indexed.loc[(e.participant_id,e.anchor_date),'hb']-e.hb0)<.001
            for k in ('hb','weight'):
                if e[f'mask_{k}']:
                    actual=indexed.loc[(e.participant_id,e[f'{k}_target_observed_date']),k]
                    assert abs(actual-e[f'y_{k}'])<.001
        summary[scenario]={'daily_rows':len(daily),'forecast_rows':len(anchors),
            'observed_hb_targets_by_horizon':{str(h):int(anchors.loc[anchors.horizon_days==h,'mask_hb'].sum()) for h in (30,90)},
            'missing_daily_fraction':{k:round(float(daily[k].isna().mean()),4) for k in ('hb','weight','steps')},
            'participants_by_split':pd.DataFrame(participants)['split'].value_counts().to_dict()}
    pd.DataFrame(participants).to_csv(output/'participants_and_splits.csv',index=False)
    (output/'validation_summary.json').write_text(json.dumps({'seed':seed,'people_per_scenario':people,
        'days_per_person':n,'checks_passed':True,'scenarios':summary},indent=2))
    checksums={str(f.relative_to(output)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(output.rglob('*.csv'))}
    (output/'checksums.json').write_text(json.dumps(checksums,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('Synthetic_Health_Dataset'))
    parser.add_argument('--people',type=int,default=400)
    parser.add_argument('--seed',type=int,default=20261104)
    args=parser.parse_args()
    if args.people<20:
        parser.error('--people must be at least 20')
    build(args.output,args.people,args.seed)
