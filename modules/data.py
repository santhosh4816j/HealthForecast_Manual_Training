"""Module 1: CSV validation, dated examples, and participant-disjoint splits."""
import io
import numpy as np
import pandas as pd

TASKS = ['hb', 'weight', 'steps']
STEP_COLS = [f'steps_day_{i}' for i in range(-4, 1)]
NUMERIC = ['hb0', 'weight0', 'hb_age_days', 'weight_age_days', 'age', *STEP_COLS]
SPLITS = ['train', 'validation', 'calibration', 'test']

def split_people(df, seed=42):
    if 'split' in df:
        if df['split'].isna().any() or not set(df['split']).issubset(SPLITS):
            raise ValueError('split must contain train, validation, calibration or test for every row.')
        if df.groupby('participant_id')['split'].nunique().max() != 1:
            raise ValueError('Participant leakage: one participant occurs in multiple splits.')
    else:
        people = np.array(sorted(df.participant_id.unique()))
        np.random.default_rng(seed).shuffle(people)
        edges = [0, int(len(people)*.6), int(len(people)*.75), int(len(people)*.85), len(people)]
        mapping = {p:s for s,a,b in zip(SPLITS,edges[:-1],edges[1:]) for p in people[a:b]}
        df['split'] = df.participant_id.map(mapping)
    return df

def raw_to_examples(raw):
    required = ['participant_id','date','hb','weight','steps','age','gender']
    if not set(required).issubset(raw):
        raise ValueError('Daily CSV requires: '+', '.join(required))
    raw = raw.copy()
    raw['date'] = pd.to_datetime(raw.date, errors='raise').dt.normalize()
    if raw.duplicated(['participant_id','date']).any():
        raise ValueError('Duplicate participant/date rows. Upload ONE scenario only.')
    for c in ['hb','weight','steps','age']:
        raw[c] = pd.to_numeric(raw[c], errors='raise')
    rows = []
    for pid, group in raw.groupby('participant_id', sort=False):
        group = group.sort_values('date').set_index('date')
        for t, reading in group.loc[group.hb.notna()].iterrows():
            if t-pd.Timedelta(days=4)<group.index.min():
                continue
            prior = group.loc[:t]
            weights = prior.weight.dropna()
            hist = prior.steps.reindex(pd.date_range(t-pd.Timedelta(days=4),t))
            if hist.notna().sum()<3:
                continue
            r = dict(participant_id=pid, anchor_date=t, hb0=reading.hb,
                     weight0=weights.iloc[-1] if len(weights) else np.nan,
                     hb_age_days=0, weight_age_days=(t-weights.index[-1]).days if len(weights) else np.nan,
                     age=reading.age, gender=reading.gender)
            for c,v in zip(STEP_COLS,hist): r[c]=v
            if 'split' in group: r['split']=reading['split']
            if 'is_synthetic' in group: r['is_synthetic']=reading['is_synthetic']
            for h in [30,90]:
                target=t+pd.Timedelta(days=h)
                if target+pd.Timedelta(days=7)>group.index.max(): continue
                row=dict(r,horizon_days=h,target_date=target)
                for k in ['hb','weight']:
                    candidates=group.loc[(group.index>t)&(abs((group.index-target).days)<=7),k].dropna()
                    if len(candidates):
                        ix=int(np.argmin(abs((candidates.index-target).days)))
                        row['y_'+k]=candidates.iloc[ix]
                        row[k+'_target_observed_date']=candidates.index[ix]
                    else: row['y_'+k]=np.nan
                future=group.steps.reindex(pd.date_range(target-pd.Timedelta(days=6),target))
                row['y_steps']=future.mean() if future.notna().sum()>=5 else np.nan
                rows.append(row)
    if not rows: raise ValueError('No eligible examples. Need step history and observed future targets near 30/90 days.')
    return pd.DataFrame(rows)

def prepare_csv(blob, seed=42):
    df=pd.read_csv(io.BytesIO(blob),dtype={'participant_id':str})
    if len(df)>1000000: raise ValueError('Use a CSV of at most 1,000,000 rows.')
    if 'participant_id' not in df or df.participant_id.isna().any():
        raise ValueError('Every row needs participant_id.')
    kind='forecast examples' if 'horizon_days' in df else 'daily records'
    # Validate raw split consistency before constructing anchors.
    if kind=='daily records':
        df=raw_to_examples(split_people(df,seed))
    required=['participant_id','anchor_date','horizon_days','gender',*NUMERIC,*['y_'+k for k in TASKS]]
    missing=sorted(set(required)-set(df))
    if missing: raise ValueError('Missing columns: '+', '.join(missing))
    df=df.copy()
    df['anchor_date']=pd.to_datetime(df.anchor_date,errors='raise')
    for c in [*NUMERIC,'horizon_days',*['y_'+k for k in TASKS]]:
        df[c]=pd.to_numeric(df[c],errors='raise')
    if not set(df.horizon_days).issubset({30,90}): raise ValueError('Only 30- and 90-day horizons are supported.')
    if df.duplicated(['participant_id','anchor_date','horizon_days']).any():
        raise ValueError('Duplicate participant/anchor/horizon. Do not combine simulation scenarios.')
    if df.hb0.isna().any() or df.age.isna().any() or df.gender.isna().any():
        raise ValueError('Baseline Hb, age and gender must be supplied.')
    for c in [*NUMERIC,*['y_'+k for k in TASKS]]:
        if np.isinf(df[c].to_numpy(float)).any(): raise ValueError(c+' contains infinity.')
    for c in ['hb0','weight0','age','hb_age_days','weight_age_days',*STEP_COLS,'y_hb','y_weight','y_steps']:
        if (df[c].dropna()<0).any(): raise ValueError(c+' cannot be negative.')
    if (df[STEP_COLS].notna().sum(axis=1)<3).any():
        raise ValueError('Each example needs at least three observed days in its five-day step window.')
    if df['y_hb'].notna().sum()==0: raise ValueError('No observed future Hb targets.')
    if 'target_date' in df:
        delta=(pd.to_datetime(df.target_date,errors='raise')-df.anchor_date).dt.days
        if not delta.eq(df.horizon_days).all(): raise ValueError('target_date must equal anchor_date + horizon_days.')
    for k in TASKS:
        flag='mask_'+k
        if flag in df and not pd.to_numeric(df[flag],errors='raise').eq(df['y_'+k].notna().astype(int)).all():
            raise ValueError(flag+' disagrees with target missingness.')
        df[flag]=df['y_'+k].notna().astype(int)
    for k in ['hb','weight']:
        col=k+'_target_observed_date'
        if col in df:
            dates=pd.to_datetime(df[col],errors='coerce')
            present=df['y_'+k].notna()
            if dates[present].isna().any() or not (dates[present]>df.loc[present,'anchor_date']).all():
                raise ValueError(col+' must be a valid future measurement date.')
            nominal=df.anchor_date+pd.to_timedelta(df.horizon_days,unit='D')
            if ((dates[present]-nominal[present]).dt.days.abs()>7).any():
                raise ValueError(col+' must lie within +/-7 days of the horizon.')
    df=split_people(df,seed)
    if df.participant_id.nunique()<20: raise ValueError('At least 20 participants are needed for this workflow.')
    for h in sorted(df.horizon_days.unique()):
        part=df[df.horizon_days==h]
        for s in SPLITS:
            if part.loc[(part.split==s)&part.y_hb.notna(),'participant_id'].nunique()<2:
                raise ValueError(f'{h}-day {s} split needs at least two participants with observed future Hb.')
    summary={'input_type':kind,'rows':len(df),'participants':df.participant_id.nunique(),
             'synthetic':bool('is_synthetic' in df and df.is_synthetic.eq(1).all()),
             'splits':df.groupby('split').participant_id.nunique().to_dict()}
    return df.reset_index(drop=True),summary

def features(df):
    numeric=df[NUMERIC].to_numpy(float)
    gender=df.gender.astype(str).str.lower()
    onehot=np.column_stack([gender.eq(g) for g in ['female','male','other']]+[~gender.isin(['female','male','other'])]).astype(float)
    return np.column_stack([numeric,np.isnan(numeric).astype(float),onehot])

class Preprocessor:
    def fit(self,df):
        x=features(df)
        self.median=np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0 for c in x.T])
        x=np.where(np.isfinite(x),x,self.median)
        self.mean=x.mean(0); self.std=x.std(0)
        self.std[self.std<1e-8]=1
        self.weight_reference=float(df.weight0.median()) if df.weight0.notna().any() else 0.
        b=self.references(df)
        y=df[['y_'+k for k in TASKS]].to_numpy(float)
        self.scale=np.array([max(float(np.std(c[np.isfinite(c)])),floor) if np.isfinite(c).any() else floor
                             for c,floor in zip((y-b).T,[.1,.1,100])])
        return self
    def transform(self,df):
        x=features(df)
        return (np.where(np.isfinite(x),x,self.median)-self.mean)/self.std
    def references(self,df):
        return np.column_stack([df.hb0.to_numpy(float),df.weight0.fillna(self.weight_reference).to_numpy(float),
                                df[STEP_COLS].mean(axis=1).to_numpy(float)])
    def residuals(self,df): return (df[['y_'+k for k in TASKS]].to_numpy(float)-self.references(df))/self.scale

