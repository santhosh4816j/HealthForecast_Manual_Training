import io
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from modules.data import prepare_csv
from modules.model import JointMLP,gate,predict_network
from modules.training import train_all,load_bundle

ROOT=Path(__file__).resolve().parents[1]

class ScientificChecks(unittest.TestCase):
    def test_masked_gradients_and_finite_difference(self):
        rng=np.random.default_rng(5);x=rng.normal(size=(5,4));r=rng.normal(size=(5,3))
        r[:,2]=np.nan
        net=JointMLP(4,hidden=3,seed=6)
        loss,grad=net.loss_grad(x,r,np.ones(5),(.3,1.))
        self.assertTrue(np.all(grad['w2'][:,2]==0));self.assertEqual(grad['b2'][2],0)
        original=net.p['w1'][0,0];eps=1e-6
        net.p['w1'][0,0]=original+eps;a=net.loss_grad(x,r,np.ones(5),(.3,1.))[0]
        net.p['w1'][0,0]=original-eps;b=net.loss_grad(x,r,np.ones(5),(.3,1.))[0]
        self.assertAlmostEqual(grad['w1'][0,0],(a-b)/(2*eps),places=6)

    def test_auxiliary_zero_equivalence(self):
        rng=np.random.default_rng(7);x=rng.normal(size=(9,4));r=rng.normal(size=(9,3))
        altered=r.copy();altered[:,1:]*=10000
        a=JointMLP(4,seed=2).fit(x,r,np.ones(9),(0,0),epochs=12)
        b=JointMLP(4,seed=2).fit(x,altered,np.ones(9),(0,0),epochs=12)
        np.testing.assert_array_equal(a.predict(x)[:,0],b.predict(x)[:,0])

    def test_gate(self):
        index,decision=gate([[1,1,1],[1.1,1.1,1.1]])
        self.assertEqual(index,0);self.assertFalse(decision['accepted'])
        index,decision=gate([[1,1,1],[.8,.8,.8]])
        self.assertEqual(index,1);self.assertTrue(decision['accepted'])
        index,decision=gate([[1,1,1],[.6,1.1,1.2]])
        self.assertEqual(index,0) # too little mean improvement relative to variability

    def test_end_to_end_and_serialization(self):
        blob=(ROOT/'data'/'sample_forecast_examples.csv').read_bytes()
        df,summary=prepare_csv(blob)
        self.assertEqual(summary['participants'],400)
        for s in ['train','validation','calibration','test']:
            self.assertGreater(df.loc[df.split==s,'participant_id'].nunique(),0)
        with tempfile.TemporaryDirectory() as temp:
            report=train_all(df,temp,epochs=20,folds=3,seed=42,provenance=summary)
            self.assertEqual(set(report['horizons']),{'30','90'})
            for h in [30,90]:
                prep,net,meta=load_bundle(Path(temp)/f'horizon_{h}')
                selected=df[(df.horizon_days==h)&(df.split=='test')].head(3)
                pred=predict_network(prep,net,selected)
                self.assertTrue(np.isfinite(pred).all())
                self.assertTrue((pred[:,2]>=0).all())
                self.assertEqual(meta['epochs'],20)
            results=pd.read_csv(Path(temp)/'metrics.csv')
            self.assertFalse(results.mae.isna().any())
            self.assertTrue((results.mae>=0).all())
            self.assertTrue((Path(temp)/'test_predictions.csv').exists())

    def test_reject_leakage_and_bad_masks(self):
        df=pd.read_csv(ROOT/'data'/'sample_forecast_examples.csv')
        pid=df.participant_id.iloc[0]
        ix=df.index[df.participant_id==pid][:2]
        df.loc[ix[0],'split']='train';df.loc[ix[1],'split']='test'
        with self.assertRaisesRegex(ValueError,'leakage'):
            prepare_csv(df.to_csv(index=False).encode())
        df=pd.read_csv(ROOT/'data'/'sample_forecast_examples.csv')
        df.loc[0,'mask_hb']=1-int(df.loc[0,'mask_hb'])
        with self.assertRaisesRegex(ValueError,'missingness'):
            prepare_csv(df.to_csv(index=False).encode())

if __name__=='__main__':unittest.main()
