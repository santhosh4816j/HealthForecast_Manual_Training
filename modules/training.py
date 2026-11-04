"""Module 3: held-out metrics, cluster bootstrap, conformal intervals and artifacts."""
import json
import platform
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from .data import TASKS,Preprocessor
from .model import GRID,JointMLP,fit_network,predict_network,gate

def participant_errors(df,truth,pred):
    valid=np.isfinite(truth)&np.isfinite(pred)
    return pd.DataFrame({'p':df.participant_id.to_numpy()[valid],
                         'ae':np.abs(truth[valid]-pred[valid])}).groupby('p').ae.mean()

def metrics(df,y,pred,base):
    valid=np.isfinite(y)&np.isfinite(pred)
    if not valid.any(): return {'mae':None,'rmse':None,'skill':None,'participants':0,'observations':0}
    mae=float(participant_errors(df,y,pred).mean())
    bmae=float(participant_errors(df,y,base).mean())
    return {'mae':mae,'rmse':float(np.sqrt(np.mean((y[valid]-pred[valid])**2))),
            'skill':1-mae/bmae if bmae>0 else None,'participants':int(df.loc[valid,'participant_id'].nunique()),
            'observations':int(valid.sum())}

def paired(df,y,baseline,selected,seed):
    # Positive means selected model has lower Hb error (one sign throughout).
    d=participant_errors(df,y,baseline)-participant_errors(df,y,selected)
    rng=np.random.default_rng(seed)
    means=rng.choice(d.to_numpy(),size=(2000,len(d)),replace=True).mean(1)
    lo,hi=np.quantile(means,[.025,.975])
    se=float(d.std(ddof=1)/np.sqrt(len(d)))
    # Approximate two-sided paired normal test; bootstrap interval is primary.
    p=float(2*norm.sf(abs(d.mean()/se))) if se>0 else (1. if d.mean()==0 else 0.)
    return {'improvement':float(d.mean()),'ci_low':float(lo),'ci_high':float(hi),'p_approx':p,
            'people':len(d),'bootstrap_resamples':2000}

def calibrate(df,truth,pred):
    scores=[]
    for k in range(3):
        tmp=pd.DataFrame({'p':df.participant_id.to_numpy(),'e':np.abs(truth[:,k]-pred[:,k])}).dropna()
        maxima=tmp.groupby('p').e.max().sort_values().to_numpy()
        rank=int(np.ceil((len(maxima)+1)*.9))
        scores.append(float(maxima[rank-1]) if 0<rank<=len(maxima) else None)
    return scores

def save_bundle(folder,prep,net,metadata):
    folder.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(folder/'model.npz',**net.p,median=prep.median,mean=prep.mean,std=prep.std,
                        scale=prep.scale,weight_reference=np.array(prep.weight_reference))
    (folder/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')

def load_bundle(folder):
    with np.load(Path(folder)/'model.npz',allow_pickle=False) as z:
        prep=Preprocessor()
        for k in ['median','mean','std','scale']: setattr(prep,k,z[k].copy())
        prep.weight_reference=float(z['weight_reference'])
        net=JointMLP(len(prep.mean))
        net.p={k:z[k].copy() for k in ['w1','b1','w2','b2']}
    meta=json.loads((Path(folder)/'metadata.json').read_text())
    return prep,net,meta

@threadpool_limits.wrap(limits=1)
def train_all(df,output,epochs=120,folds=3,seed=42,notify=lambda fraction,message:None,provenance=None):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    reports={}; all_metrics=[]; cv_rows=[]; pred_rows=[]
    horizons=sorted(int(h) for h in df.horizon_days.unique())
    total=len(horizons)*len(GRID)*folds; done=0
    for h in horizons:
        sub=df[df.horizon_days==h].reset_index(drop=True)
        dev=sub[sub.split.isin(['train','validation'])].reset_index(drop=True)
        calibration=sub[sub.split=='calibration'].reset_index(drop=True)
        test=sub[sub.split=='test'].reset_index(drop=True)
        splits=list(GroupKFold(folds).split(dev,groups=dev.participant_id))
        scores=[]
        for index,alpha in enumerate(GRID):
            candidate=[]
            for f,(tr,va) in enumerate(splits):
                train=dev.iloc[tr]; val=dev.iloc[va]
                if train.y_hb.notna().sum()==0 or val.y_hb.notna().sum()==0:
                    raise ValueError('A grouped fold has no Hb targets. Reduce folds or add data.')
                prep,net=fit_network(train,alpha,epochs,seed+f)
                pred=predict_network(prep,net,val)[:,0]
                score=float(participant_errors(val,val.y_hb.to_numpy(float),pred).mean())
                candidate.append(score)
                cv_rows.append({'horizon':h,'weight_alpha':alpha[0],'steps_alpha':alpha[1],
                                'fold':f+1,'hb_participant_mae':score})
                done+=1; notify(done/(total+len(horizons)),f'{h} days: weights {alpha}, fold {f+1}/{folds}')
            scores.append(candidate)
        chosen,decision=gate(scores); alpha=GRID[chosen]
        # Refit with the exact same seed and inputs for matched final comparisons.
        baseline_p,baseline_n=fit_network(dev,GRID[0],epochs,seed)
        if chosen==0: prep,net=baseline_p,baseline_n
        else: prep,net=fit_network(dev,alpha,epochs,seed)
        fixed_p,fixed_n=fit_network(dev,(1.,1.),epochs,seed)
        selected=predict_network(prep,net,test)
        base=prep.references(test)
        hb_only=predict_network(baseline_p,baseline_n,test)
        fixed=predict_network(fixed_p,fixed_n,test)
        x_train=prep.transform(dev); x_test=prep.transform(test)
        observed=dev.y_hb.notna().to_numpy()
        residual=(dev.y_hb.to_numpy(float)-prep.references(dev)[:,0])[observed]
        weights=1/dev.loc[observed,'participant_id'].map(dev.loc[observed,'participant_id'].value_counts()).to_numpy(float)
        ridge=Ridge(alpha=1.).fit(x_train[observed],residual,sample_weight=weights/weights.mean())
        boosting=HistGradientBoostingRegressor(max_iter=80,max_leaf_nodes=7,random_state=seed).fit(
            x_train[observed],residual,sample_weight=weights/weights.mean())
        candidates={'Persistence':base,'Hb-only MLP':hb_only,'Fixed multitask (1,1)':fixed,'HS-GAW':selected}
        for name,model in [('Ridge',ridge),('Gradient boosting',boosting)]:
            a=np.full_like(base,np.nan);a[:,0]=base[:,0]+model.predict(x_test);candidates[name]=a
        truth=test[['y_'+k for k in TASKS]].to_numpy(float)
        for name,pred in candidates.items():
            for k,task in enumerate(TASKS):
                if name in ['Ridge','Gradient boosting','Hb-only MLP'] and k>0: continue
                if name=='HS-GAW' and k>0 and alpha[k-1]==0: continue
                all_metrics.append(dict(horizon=h,model=name,task=task,**metrics(test,truth[:,k],pred[:,k],base[:,k])))
        comparison=paired(test,truth[:,0],hb_only[:,0],selected[:,0],seed)
        q=calibrate(calibration,calibration[['y_'+k for k in TASKS]].to_numpy(float),predict_network(prep,net,calibration))
        for k in (1,2):
            if alpha[k-1]==0:q[k]=None
        intervals={}
        for k,task in enumerate(TASKS):
            valid=np.isfinite(truth[:,k])
            if q[k] is not None and valid.any():
                inside=np.abs(truth[valid,k]-selected[valid,k])<=q[k]
                interval_df=pd.DataFrame({'p':test.loc[valid,'participant_id'].to_numpy(),'covered':inside})
                intervals[task]={'half_width':q[k],'anchor_coverage':float(inside.mean()),
                    'participant_all_anchor_coverage':float(interval_df.groupby('p').covered.all().mean())}
        metadata={'horizon':h,'alpha':alpha,'gate':decision,'comparison':comparison,'intervals':intervals,
                  'q':q,'epochs':epochs,'folds':folds,'seed':seed,'synthetic':bool((provenance or {}).get('synthetic',False)),
                  'auxiliary_outputs_active':[a>0 for a in alpha],
                  'loss_history':net.loss_history,'provenance':provenance or {},
                  'selection':'Grouped CV on train + validation participants; calibration and test excluded.'}
        save_bundle(output/f'horizon_{h}',prep,net,metadata)
        reports[str(h)]=metadata
        for i,row in test.iterrows():
            out={'participant_id':row.participant_id,'anchor_date':str(row.anchor_date.date()),'horizon':h}
            for k,task in enumerate(TASKS):
                out['actual_'+task]=None if not np.isfinite(truth[i,k]) else float(truth[i,k])
                out['predicted_'+task]=float(selected[i,k]) if k==0 or alpha[k-1]>0 else None
            pred_rows.append(out)
        notify(min(.99,done/(total+len(horizons))),f'{h}-day model fitted. Held-out evaluation complete.')
    # Holm correction over horizon-wise approximate p-values.
    ordered=sorted(reports,key=lambda h:reports[h]['comparison']['p_approx'])
    previous=0.
    for rank,h in enumerate(ordered):
        adjusted=min(1.,max(previous,(len(ordered)-rank)*reports[h]['comparison']['p_approx']))
        reports[h]['comparison']['p_holm_approx']=adjusted;previous=adjusted
        save_path=output/f'horizon_{h}'/'metadata.json'
        save_path.write_text(json.dumps(reports[h],indent=2))
    pd.DataFrame(all_metrics).to_csv(output/'metrics.csv',index=False)
    pd.DataFrame(cv_rows).to_csv(output/'cross_validation.csv',index=False)
    pd.DataFrame(pred_rows).to_csv(output/'test_predictions.csv',index=False)
    report={'horizons':reports,'versions':{'python':platform.python_version(),'numpy':np.__version__},
            'scope':'Synthetic simulation only' if (provenance or {}).get('synthetic') else 'User-provided data; provenance unverified',
            'limitations':['One generator seed and one training seed do not establish robust superiority.',
                          'Gate threshold is a heuristic and does not guarantee test improvement.',
                          'No GradNorm, uncertainty-weighting or shuffled-auxiliary comparison is implemented.',
                          'Paired p-values use a normal approximation; participant bootstrap intervals are primary.',
                          'Cluster conformal interpretation requires exchangeable participants and comparable anchor collection.']}
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    notify(1.,'Training finished. Models and measured results saved locally.')
    return report
