"""Run with: python -m streamlit run app.py"""
import hashlib
import io
import json
from pathlib import Path
from datetime import datetime
import zipfile
import numpy as np
import pandas as pd
import streamlit as st
from modules.data import prepare_csv,STEP_COLS
from modules.model import predict_network
from modules.training import train_all,load_bundle

ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title='HealthForecast | HS-GAW',page_icon='🧬',layout='wide')
st.markdown('''<style>
.stApp {background:#f5f7fb;color:#182338} .block-container {padding-top:2rem;max-width:1250px}
h1,h2,h3 {letter-spacing:-.04em} [data-testid="stMetric"] {background:white;padding:18px;border:1px solid #e0e6f0;border-radius:14px}
[data-testid="stTabs"] button {font-weight:600} .hero {background:#12283f;color:white;border-radius:20px;padding:30px;margin-bottom:24px}
.hero p {color:#c3d4e5;margin-bottom:0}.eyebrow {font-size:12px;letter-spacing:.14em;color:#75dcc5}
</style>''',unsafe_allow_html=True)
st.markdown('<div class="hero"><div class="eyebrow">ADAPTIVE MULTIVARIATE HEALTH FORECASTING</div><h1>From your CSV to a trained forecast.</h1><p>Upload records. Train HS-GAW. Inspect the evidence.</p></div>',unsafe_allow_html=True)
st.caption('Research prototype • Runs on your computer • Synthetic sample data • No medical validation')

with st.sidebar:
    st.header('Your workspace')
    st.write('Training starts only when you click **Train Model**. No pretrained models are included.')
    runs=sorted((ROOT/'runs').glob('*/report.json'),reverse=True) if (ROOT/'runs').exists() else []
    restore=st.selectbox('Previously trained run',['Select a saved run']+[p.parent.name for p in runs])
    if st.button('Restore saved run',disabled=restore=='Select a saved run'):
        folder=ROOT/'runs'/restore
        st.session_state['run']=str(folder)
        st.session_state['report']=json.loads((folder/'report.json').read_text())
        st.session_state.pop('forecast_rows',None)
        st.success('Saved model restored. Open module 3.')
    st.divider()
    st.caption('NumPy MLP · analytic gradients · Adam optimiser · local model files')

one,two,three=st.tabs(['01  Data preparation','02  Train your algorithm','03  Evaluation & forecasts'])
with one:
    st.subheader('1. Upload and validate your dataset')
    st.write('Upload one **forecast_examples.csv** or **daily_records.csv**. Use one simulation scenario per run.')
    sample=ROOT/'data'/'sample_forecast_examples.csv'
    if sample.exists():
        st.download_button('Download sample CSV',sample.read_bytes(),'sample_forecast_examples.csv','text/csv')
    upload=st.file_uploader('Choose your CSV',type=['csv'],help='The CSV supplied with your synthetic dataset is supported directly.')
    if upload is not None:
        content=upload.getvalue(); digest=hashlib.sha256(content).hexdigest()
        if st.session_state.get('upload_hash')!=digest:
            for key in ['df','summary','run','report','forecast_rows']:
                st.session_state.pop(key,None)
            st.session_state['upload_hash']=digest
            try:
                with st.spinner('Checking columns, participants, dates and targets…'):
                    df,summary=prepare_csv(content)
                st.session_state['df']=df;st.session_state['summary']=summary
                st.session_state['source_name']=upload.name
            except Exception as exc:
                st.session_state['upload_error']=str(exc)
        if 'df' not in st.session_state:
            st.error(st.session_state.get('upload_error','CSV validation failed.'))
        else:
            df=st.session_state['df'];summary=st.session_state['summary']
            st.success('Dataset validated. Open module 2 to train.')
            c=st.columns(3)
            c[0].metric('Participants',summary['participants']);c[1].metric('Forecast examples',summary['rows'])
            c[2].metric('Horizons',', '.join(str(int(h))+' days' for h in sorted(df.horizon_days.unique())))
            st.dataframe(df.head(15),width='stretch',hide_index=True)
            st.write('**Participant allocation**')
            st.dataframe(pd.DataFrame([{'Split':s,'Participants':n} for s,n in summary['splits'].items()]),hide_index=True)
            coverage=df.groupby(['horizon_days','split'])[['mask_hb','mask_weight','mask_steps']].sum().reset_index()
            st.write('**Observed targets**');st.dataframe(coverage,width='stretch',hide_index=True)
            st.info('Train + validation people form the development pool for grouped cross-validation. Calibration and test people stay separate. Missing targets are never filled.')
    else: st.info('Start by uploading your CSV. The other modules will unlock after validation.')

with two:
    st.subheader('2. Train with your formulas and HS-GAW gate')
    st.latex(r'L=L_{Hb}+\alpha_W L_W+\alpha_S L_S')
    st.write('Tests **(0,0), (0.3,0), (0,0.3), (0.3,0.3), (1,1)** separately for each horizon. Hb-only and joint models receive identical inputs.')
    with st.expander('Show the implemented formulas'):
        st.latex(r'b_{Hb}=Hb_0,\quad b_W=W_0,\quad b_S=\mathrm{mean}(s_{-4:0})')
        st.latex(r'\sigma_k=\max(\mathrm{SD}_{train}(y_k-b_k),floor_k),\quad r_{ik}=(y_{ik}-b_{ik})/\sigma_k')
        st.latex(r'z=\tanh(W_1x^{\prime}+b_1),\quad \hat y_k=b_k+\sigma_k(W_2z+b_2)_k')
        st.latex(r'\omega_i=1/n_{p(i)},\quad L_k=\frac{\sum_i\omega_i m_{ik}|r_{ik}-\hat r_{ik}|}{\sum_i\omega_i m_{ik}}')
        st.latex(r'c^*=\arg\min_c\frac1K\sum_f e_f(c),\quad \Delta_f=e_f(0,0)-e_f(c^*)')
        st.latex(r'\mathrm{Accept\ if}\quad \overline\Delta>SD(\Delta)/\sqrt K')
        st.caption('An entirely missing target contributes zero loss. Scaling floors: 0.1 g/dL, 0.1 kg, 100 steps/day. Otherwise select (0,0). The gate is a heuristic, not a test-performance guarantee.')
    a,b,c=st.columns(3)
    epochs=a.number_input('Epochs per fit',min_value=20,max_value=1000,value=120,step=20)
    folds=b.selectbox('Participant-grouped folds',[3,5],index=0)
    seed=c.number_input('Training seed',min_value=0,max_value=1000000,value=42)
    label=st.checkbox('This upload contains synthetic data',value=st.session_state.get('summary',{}).get('synthetic',True))
    if st.button('Train Model',type='primary',disabled='df' not in st.session_state):
        progress=st.progress(0,text='Starting training…')
        folder=ROOT/'runs'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        def update(fraction,message):progress.progress(min(1.,float(fraction)),text=message)
        provenance={**st.session_state['summary'],'synthetic':bool(label or st.session_state['summary']['synthetic']),
                    'filename':st.session_state['source_name'],'csv_sha256':st.session_state['upload_hash']}
        try:
            with st.spinner('Training and evaluating. Keep this window open.'):
                report=train_all(st.session_state['df'],folder,int(epochs),int(folds),int(seed),update,provenance)
            st.session_state['run']=str(folder);st.session_state['report']=report
            st.session_state.pop('forecast_rows',None)
            st.success('Training complete. Open module 3 for measured results and forecasts.')
        except Exception as exc:
            st.error('Training did not complete: '+str(exc))
    st.caption('Training uses full-batch Adam, 24 tanh hidden units and a learning rate of 0.008. No external API key or GPU is needed.')

with three:
    st.subheader('3. Examine results and make a forecast')
    if 'report' not in st.session_state:
        st.info('Train your uploaded CSV in module 2, or restore a previously trained run.')
    else:
        report=st.session_state['report'];folder=Path(st.session_state['run'])
        st.warning(report['scope']+'. Forecasts are research outputs; they do not replace blood measurements.')
        results=pd.read_csv(folder/'metrics.csv')
        st.write('**Measured Hb performance on held-out participants**')
        st.dataframe(results[results.task=='hb'].drop(columns='task'),width='stretch',hide_index=True)
        for h,meta in report['horizons'].items():
            with st.expander(f'{h}-day selection and uncertainty',expanded=True):
                a,b=meta['alpha'];d=meta['comparison'];gate=meta['gate']
                st.write(f'Selected auxiliary weights: **weight = {a:g}, steps = {b:g}**. Gate accepted: **{gate["accepted"]}**.')
                st.write(f'Validation improvement: {gate["mean_improvement"]:.4f} g/dL. Gate threshold: {gate["standard_error"]:.4f} g/dL.')
                st.write(f'Test improvement over Hb-only: **{d["improvement"]:.4f} g/dL**; 95% participant-bootstrap CI **[{d["ci_low"]:.4f}, {d["ci_high"]:.4f}]**. Positive means HS-GAW is better.')
                st.caption(f'Approximate paired p-value after Holm correction: {d["p_holm_approx"]:.4g}. This is separate from the validation gate.')
                if meta['intervals']: st.json(meta['intervals'],expanded=False)
                st.line_chart(pd.DataFrame({'Training objective':meta['loss_history']}),height=160)
        st.caption('MAE averages within each participant before averaging across people. RMSE is observation-weighted. Compare both horizons; a zero gate or negative improvement is a valid finding.')
        with st.expander('Cross-validation details'):
            st.dataframe(pd.read_csv(folder/'cross_validation.csv'),width='stretch',hide_index=True)
        st.divider();st.write('**Enter measurements for a new forecast**')
        with st.form('forecast'):
            cols=st.columns(3)
            hb=cols[0].number_input('Latest Hb (g/dL)',min_value=1.,max_value=30.,value=12.5,step=.1)
            weight=cols[1].number_input('Latest weight (kg)',min_value=1.,max_value=400.,value=60.,step=.5)
            age=cols[2].number_input('Age (years)',min_value=18,max_value=110,value=22)
            cols=st.columns(3)
            gender=cols[0].selectbox('Sex / gender category',['male','female','other','unknown'])
            hb_age=cols[1].number_input('Days since Hb measurement',min_value=0,max_value=3650,value=0)
            weight_age=cols[2].number_input('Days since weight measurement',min_value=0,max_value=3650,value=0)
            steps=st.text_input('Five daily step counts, oldest to newest (comma separated; blank or NA for missing)','5000, 5500, 6000, 5200, 5800')
            submit=st.form_submit_button('Generate forecasts',type='primary')
        if submit:
            st.session_state.pop('forecast_rows',None)
            try:
                values=[np.nan if v.strip().lower() in ['','na','nan'] else float(v) for v in steps.split(',')]
                if len(values)!=5 or np.isfinite(values).sum()<3 or any(v<0 for v in values if np.isfinite(v)) or np.isinf(values).any():
                    raise ValueError('Enter five nonnegative daily step values, with at least three observed days.')
                row=dict(hb0=hb,weight0=weight,hb_age_days=hb_age,weight_age_days=weight_age,age=age,gender=gender)
                row.update(dict(zip(STEP_COLS,values)));input_df=pd.DataFrame([row])
                forecast=[]
                for h in report['horizons']:
                    prep,net,meta=load_bundle(folder/f'horizon_{h}');y=predict_network(prep,net,input_df)[0]
                    references=prep.references(input_df)[0]
                    for k,(name,unit) in enumerate([('Hb','g/dL'),('Weight','kg'),('Average daily steps','steps/day')]):
                        active=k==0 or meta['alpha'][k-1]>0
                        value=float(y[k]) if active else float(references[k])
                        q=meta['q'][k]
                        forecast.append({'Horizon':f'{h} days','Outcome':name,'Forecast':round(value,3),'Unit':unit,
                            'Method':'Trained HS-GAW' if active else 'Persistence fallback (auxiliary off)',
                            '90% interval':f'{value-q:.3f} to {value+q:.3f}' if active and q is not None else 'Unavailable'})
                st.session_state['forecast_rows']=forecast
                st.caption('Hb targets allow +/-7 days around the nominal horizon; steps forecast a seven-day average ending at the horizon. Inputs outside the training distribution can be unreliable. If an auxiliary task was switched off, its untrained output is hidden and a labelled persistence fallback is shown.')
                if hb_age>0:st.info('The supplied synthetic examples use same-day Hb anchors. Older Hb inputs are outside that training design.')
            except Exception as exc: st.error(str(exc))
        if st.session_state.get('forecast_rows'):
            forecast_df=pd.DataFrame(st.session_state['forecast_rows'])
            st.dataframe(forecast_df,width='stretch',hide_index=True)
            st.download_button('Download these forecasts',forecast_df.to_csv(index=False),'forecasts.csv','text/csv')
        st.divider()
        archive=io.BytesIO()
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(folder.rglob('*')):
                if p.is_file():z.write(p,p.relative_to(folder))
        st.download_button('Download trained models and reports',archive.getvalue(),'trained_models_and_reports.zip','application/zip')
        st.caption('Includes model.npz, preprocessing, chosen weights, CSV metrics, test predictions and report.json. Uploads are held in session memory; your CSV is not copied into the exported model archive. Test prediction files contain participant identifiers from your upload.')
        with st.expander('Study limitations'):
            for item in report['limitations']:st.write('• '+item)
