"""Module 2: report equations 3.3-3.11, analytic L1 gradients, Adam."""
import numpy as np
from .data import Preprocessor

GRID=[(0.,0.),(.3,0.),(0.,.3),(.3,.3),(1.,1.)]

class JointMLP:
    def __init__(self,features,hidden=24,seed=42):
        rng=np.random.default_rng(seed)
        self.p={'w1':rng.normal(0,np.sqrt(2/features),(features,hidden)), 'b1':np.zeros(hidden),
                'w2':rng.normal(0,np.sqrt(1/hidden),(hidden,3))*.02,'b2':np.zeros(3)}
    def predict(self,x): return np.tanh(x@self.p['w1']+self.p['b1'])@self.p['w2']+self.p['b2']
    def loss_grad(self,x,r,sample_weights,alpha):
        mask=np.isfinite(r); safe=np.where(mask,r,0.)
        weights=sample_weights[:,None]*mask
        weights=weights/np.maximum(weights.sum(0),1e-12)*np.array([1.,*alpha])[None,:]
        z=np.tanh(x@self.p['w1']+self.p['b1']); pred=z@self.p['w2']+self.p['b2']
        delta=np.sign(pred-safe)*weights
        dz=(delta@self.p['w2'].T)*(1-z*z)
        grad={'w2':z.T@delta,'b2':delta.sum(0),'w1':x.T@dz,'b1':dz.sum(0)}
        return float((np.abs(pred-safe)*weights).sum()),grad
    def fit(self,x,r,sample_weights,alpha,epochs=120,lr=.008):
        first={k:np.zeros_like(v) for k,v in self.p.items()}
        second={k:np.zeros_like(v) for k,v in self.p.items()}
        self.loss_history=[]
        for t in range(1,epochs+1):
            loss,grad=self.loss_grad(x,r,sample_weights,alpha)
            if not np.isfinite(loss): raise ValueError('Training diverged. Check input units and numeric values.')
            self.loss_history.append(loss)
            for k in self.p:
                first[k]=.9*first[k]+.1*grad[k]
                second[k]=.999*second[k]+.001*grad[k]**2
                self.p[k]-=lr*(first[k]/(1-.9**t))/(np.sqrt(second[k]/(1-.999**t))+1e-8)
        return self

def fit_network(df,alpha,epochs,seed):
    prep=Preprocessor().fit(df)
    x=prep.transform(df); r=prep.residuals(df)
    counts=df.participant_id.map(df.participant_id.value_counts()).to_numpy(float)
    net=JointMLP(x.shape[1],seed=seed).fit(x,r,1/counts,alpha,epochs)
    return prep,net

def predict_network(prep,net,df):
    y=prep.references(df)+prep.scale*net.predict(prep.transform(df))
    y[:,2]=np.maximum(y[:,2],0)
    return y

def gate(scores):
    """Return index and diagnostics. Eq 3.11 is a heuristic, not a test guarantee."""
    scores=np.asarray(scores,float)
    best=int(np.argmin(scores.mean(axis=1)))
    delta=scores[0]-scores[best]
    threshold=float(delta.std(ddof=1)/np.sqrt(len(delta)))
    accepted=bool(best!=0 and delta.mean()>threshold)
    return (best if accepted else 0),dict(best_candidate=best,accepted=accepted,
        mean_improvement=float(delta.mean()),standard_error=threshold)
