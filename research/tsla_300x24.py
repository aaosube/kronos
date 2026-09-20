"""300 calendar-verified 24-hour TSLA endpoint forecasts. Not a full 24-hour path test."""
from __future__ import annotations
import os,sys,json,time,hashlib,platform,math
from pathlib import Path
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import numpy as np
import pandas as pd
import torch
import yfinance as yf
import pandas_market_calendars as mcal
from huggingface_hub import snapshot_download
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model import Kronos,KronosTokenizer,KronosPredictor
from model.kronos import calc_time_stamps,sample_from_logits
MODEL=os.environ.get('MODEL_KIND','mini');assert MODEL in ('small','mini')
OUT=Path('results')/MODEL;OUT.mkdir(parents=True,exist_ok=True)
COLS=['open','high','low','close','volume','amount']
SPECS={'small':('NeoQuasar/Kronos-small','901c26c1332695a2a8f243eb2f37243a37bea320','NeoQuasar/Kronos-Tokenizer-base','0e0117387f39004a9016484a186a908917e22426'),'mini':('NeoQuasar/Kronos-mini','f4e68697d9d5aed55cef5c96aabc3376bcad9f81','NeoQuasar/Kronos-Tokenizer-2k','26966d0035065a0cae0ebad7af8ece35bc1fb51c')}
torch.set_num_threads(1);torch.set_num_interop_threads(1)
def stable_seed(date):
 return int.from_bytes(hashlib.sha256(('42|TSLA|'+str(date)).encode()).digest()[:4],'little')%(2**31-1)
def validate(d):
 assert d.index.is_unique and d.index.is_monotonic_increasing
 assert np.isfinite(d[['open','high','low','close','volume']].to_numpy(float)).all()
 assert (d[['open','high','low','close']]>0).all().all() and (d.volume>=0).all()
 assert (d.high+1e-4>=d[['open','low','close']].max(axis=1)).all()
 assert (d.low-1e-4<=d[['open','high','close']].min(axis=1)).all()
@torch.no_grad()
def draws_one(p,df,seed,n=5):
 v=df.copy();v['amount']=v.volume*v[['open','high','low','close']].mean(axis=1)
 x=v[COLS].to_numpy(np.float32);mean=x.mean(0);std=x.std(0)
 z=np.clip((x-mean)/(std+1e-5),-5,5);st=calc_time_stamps(pd.Series(df.index)).to_numpy(np.float32)
 tokens=p.tokenizer.encode(torch.from_numpy(z).unsqueeze(0),half=True)
 logits,context=p.model.decode_s1(tokens[0][:,-512:],tokens[1][:,-512:],torch.from_numpy(st).unsqueeze(0)[:,-512:])
 torch.manual_seed(seed)
 pre=sample_from_logits(logits[:,-1,:].expand(n,-1).clone(),temperature=.6,top_k=0,top_p=.9)
 fine_logits=p.model.decode_s2(context.expand(n,-1,-1),pre)[:,-1,:]
 post=sample_from_logits(fine_logits,temperature=.6,top_k=0,top_p=.9)
 pre_full=torch.cat([tokens[0].expand(n,-1),pre],1)[:,-512:];post_full=torch.cat([tokens[1].expand(n,-1),post],1)[:,-512:]
 zhat=p.tokenizer.decode([pre_full,post_full],half=True).cpu().numpy()
 samples=zhat[:,-1,:]*(std+1e-5)+mean;mu=zhat[:,-1,:].mean(0)*(std+1e-5)+mean
 reconstruction=zhat[0,-2,:]*(std+1e-5)+mean
 return samples,mu,float(df.iloc[-1].close-reconstruction[3])
raw=yf.download('TSLA',start='2022-01-01',end='2026-09-19',interval='1d',auto_adjust=False,actions=True,progress=False,threads=False)
if raw.empty:raise RuntimeError('No TSLA history returned')
if isinstance(raw.columns,pd.MultiIndex):raw.columns=raw.columns.get_level_values(0)
raw.index=pd.to_datetime(raw.index).tz_localize(None);raw.index.name='date';raw.to_csv(OUT/'source_daily.csv')
d=raw.rename(columns=str.lower)[['open','high','low','close','volume']].copy();validate(d)
cal=mcal.get_calendar('NYSE').schedule(start_date='2022-01-01',end_date='2026-09-18');cal.index=pd.to_datetime(cal.index).tz_localize(None);closes=pd.to_datetime(cal.market_close,utc=True)
origins=[]
for j in range(400,len(d)):
 a,b=d.index[j-1],d.index[j]
 if a not in closes.index or b not in closes.index:continue
 if closes[b]-closes[a]!=pd.Timedelta(hours=24):continue
 if closes[a].tz_convert('America/New_York').strftime('%H:%M')!='16:00':continue
 if closes[b].tz_convert('America/New_York').strftime('%H:%M')!='16:00':continue
 origins.append((j,a,b,closes[a],closes[b]))
if len(origins)<300:raise RuntimeError(f'Only {len(origins)} eligible pairs; refusing to relabel fewer as 300')
origins=origins[-300:]
protocol={'ticker':'TSLA','model':MODEL,'n_forecast_origins':300,'horizon_hours':24,'input':'daily RTH OHLCV','target':'next regular close exactly 24 elapsed hours after origin close','full_24h_path_evaluated':False,'overnight_intraday_bars_used':False,'context':400,'sample_count':5,'T':.6,'top_p':.9,'top_k':0,'selection':'Most recent 300 eligible pairs by calendar only, never by returns','source_code_commit':'67b630e67f6a18c9e9be918d9b4337c960db1e9a','source':'Yahoo via yfinance','vendor_vintage_PIT_verified':False,'weights_modified':False,'actual_execution_origin':'immediately after close is known; no executable trading PnL inferred','start_target':str(origins[0][2].date()),'end_target':str(origins[-1][2].date()),'all_300_temporal_OOS':False,'OOS_rule':'origins on or after 2025-07-02 reported separately','prior_exposure':'retrospective data, not a pristine prospective holdout','experimental_anchor':'separate history-only reconstruction shift; not used to select original model results'}
(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2));print('PROTOCOL',json.dumps(protocol),flush=True)
mid,mrev,tid,trev=SPECS[MODEL];weights={}
for rid,rev in [(mid,mrev),(tid,trev)]:
 path=Path(snapshot_download(rid,revision=rev,allow_patterns=['config.json','model.safetensors']))
 weights[rid]={'revision':rev,'path':str(path),'sha256':hashlib.sha256((path/'model.safetensors').read_bytes()).hexdigest()}
tok=KronosTokenizer.from_pretrained(weights[tid]['path']).eval();mdl=Kronos.from_pretrained(weights[mid]['path']).eval();p=KronosPredictor(mdl,tok,device='cpu',max_context=512)
qa=[]
for idx in [0,149,299]:
 j,a,b,ac,bc=origins[idx];hist=d.iloc[j-400:j].copy();seed=stable_seed(b.date());torch.manual_seed(seed)
 ref=p.predict(df=hist,x_timestamp=pd.Series(hist.index),y_timestamp=pd.Series([b]),pred_len=1,T=.6,top_p=.9,sample_count=5,verbose=False).to_numpy()[0]
 samples,mu,shift=draws_one(p,hist,seed);ok=bool(np.allclose(ref,mu,rtol=3e-6,atol=1e-4));qa.append({'case':idx,'passed':ok,'max_OHLC_abs_diff':float(abs(ref[:4]-mu[:4]).max())})
 if not ok:raise RuntimeError('Optimized inference differs from upstream')
j=origins[100][0];mutated=d.copy();mutated.iloc[j:,:4]=mutated.iloc[j:,:4]*10;assert np.array_equal(d.iloc[j-400:j].to_numpy(),mutated.iloc[j-400:j].to_numpy())
print('QA_PASSED',json.dumps(qa),flush=True)
rows=[];start=time.time()
for i,(j,a,b,ac,bc) in enumerate(origins):
 hist=d.iloc[j-400:j].copy();assert hist.index.max()<b
 prev=float(hist.iloc[-1].close);actual=float(d.iloc[j].close);r=hist.close.pct_change().dropna();vol=float(r.tail(20).std());trend=float(prev/hist.close.iloc[-21]-1)
 regime='uptrend' if trend>.5*vol*np.sqrt(20) else 'downtrend' if trend<-.5*vol*np.sqrt(20) else 'range'
 vol_reg='high' if vol>r.rolling(20).std().tail(252).median() else 'low'
 rr=r.tail(120).to_numpy();beta=float(np.cov(rr[:-1],rr[1:],ddof=0)[0,1]/np.var(rr[:-1])) if np.var(rr[:-1]) else 0.;ar=float(rr[1:].mean()-beta*rr[:-1].mean()+beta*rr[-1])
 samples,mu,shift=draws_one(p,hist,stable_seed(b.date()))
 if not np.isfinite(samples).all():raise RuntimeError('Nonfinite forecast')
 row={'date':str(b.date()),'origin_date':str(a.date()),'origin_utc':str(ac),'target_utc':str(bc),'horizon_hours':(bc-ac).total_seconds()/3600,'previous_close':prev,'actual_close':actual,'predicted_close':float(mu[3]),'anchor_close':float(mu[3]+shift),'actual_return':actual/prev-1,'predicted_return':float(mu[3])/prev-1,'anchor_return':float(mu[3]+shift)/prev-1,'predicted_open':float(mu[0]),'predicted_high':float(mu[1]),'predicted_low':float(mu[2]),'actual_open':float(d.iloc[j].open),'actual_high':float(d.iloc[j].high),'actual_low':float(d.iloc[j].low),'mom1_return':float(r.iloc[-1]),'drift20_return':float(r.tail(20).mean()),'ar1_return':ar,'trend_regime':regime,'volatility_regime':vol_reg,'post_release':a>=pd.Timestamp('2025-07-02'),'draws':samples[:,3].tolist()}
 rows.append(row)
 with open(OUT/'predictions.jsonl','a') as file:file.write(json.dumps(row,allow_nan=False)+'\n')
 if (i+1)%50==0:print('PROGRESS',MODEL,i+1,'elapsed_seconds',round(time.time()-start,1),flush=True)
f=pd.DataFrame(rows);f.drop(columns=['draws']).to_csv(OUT/'predictions.csv',index=False)
def metrics(g,col='predicted_close'):
 a=g.actual_close.to_numpy();pr=g[col].to_numpy();prev=g.previous_close.to_numpy();n=len(g);err=pr-a;loss=abs(err)/prev;ref=abs(a-prev)/prev;y=(a>prev).astype(int);s=(pr>prev).astype(int)
 accuracy=float((np.sign(pr-prev)==np.sign(a-prev)).mean())
 result={'n':n,'hits':int(round(accuracy*n)),'accuracy_pct':accuracy*100,'always_up_accuracy_pct':float(y.mean()*100),'always_down_accuracy_pct':float((1-y).mean()*100),'MAE_USD':float(abs(err).mean()),'RMSE_USD':float(np.sqrt((err**2).mean())),'MAPE_pct':float((abs(err)/a).mean()*100),'return_MAE_pp':float(loss.mean()*100),'persistence_MAE_USD':float(abs(a-prev).mean()),'persistence_return_MAE_pp':float(ref.mean()*100),'skill_vs_persistence_pct':float((1-loss.mean()/ref.mean())*100),'return_Pearson_IC':float(np.corrcoef(pr/prev-1,a/prev-1)[0,1]),'return_Spearman_IC':float(pd.Series(pr/prev-1).corr(pd.Series(a/prev-1),method='spearman')),'balanced_accuracy_pct':float(np.mean([(s[y==cls]==cls).mean() for cls in [0,1]])*100)}
 rng=np.random.default_rng(20260920);B=3000;length=5;starts=rng.integers(0,n,size=(B,math.ceil(n/length)));ix=(starts[:,:,None]+np.arange(length))%n;ix=ix.reshape(B,-1)[:,:n]
 skill=100*(1-loss[ix].mean(1)/ref[ix].mean(1));result['skill_block_bootstrap95_pct']=np.quantile(skill,[.025,.975]).tolist();acc=(np.sign(pr-prev)==np.sign(a-prev)).astype(float);result['accuracy_block_bootstrap95_pct']=(100*np.quantile(acc[ix].mean(1),[.025,.975])).tolist()
 return result
summary={'model':MODEL,'protocol':protocol,'all':metrics(f),'post_release':metrics(f[f.post_release]),'pre_release':metrics(f[~f.post_release]) if (~f.post_release).any() else None,'anchor_all':metrics(f,'anchor_close'),'regimes':{},'baselines':{}}
for c in ['trend_regime','volatility_regime']:
 for name,g in f.groupby(c):summary['regimes'][c+':'+name]=metrics(g)
for c in ['mom1_return','drift20_return','ar1_return']:
 g=f.copy();g['baseline_close']=g.previous_close*(1+g[c]);summary['baselines'][c]=metrics(g,'baseline_close')
close_samples=np.array(f.draws.tolist());lo,hi=np.quantile(close_samples,[.1,.9],axis=1);summary['interval80_empirical_coverage_pct']=float(((f.actual_close>=lo)&(f.actual_close<=hi)).mean()*100);summary['sample_direction_Brier']=float((((close_samples>f.previous_close.to_numpy()[:,None]).mean(1)-(f.actual_close>f.previous_close).astype(float))**2).mean())
summary['limits']=['Endpoint-only exactly 24-hour forecast; not full intraday or overnight path.','Five generated draws provide coarse interval/probability estimates.','Full 300 include pre-weight-release dates; only labeled post_release subset is temporal OOS.','One asset and retrospective data; no claim replication of the paper RankIC benchmark.','No trade execution costs or realized trading PnL.']
receipt={'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'pandas':pd.__version__,'yfinance':yf.__version__,'source_sha256':hashlib.sha256((OUT/'source_daily.csv').read_bytes()).hexdigest(),'weights':weights,'equivalence_tests':qa,'no_future_input_test_passed':True,'completed_forecasts':len(rows),'prediction_file_sha256':hashlib.sha256((OUT/'predictions.csv').read_bytes()).hexdigest()}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False));(OUT/'receipt.json').write_text(json.dumps(receipt,indent=2));print('FINAL_SUMMARY='+json.dumps(summary,separators=(',',':'),allow_nan=False),flush=True);print('COMPLETED_300_24H',MODEL,len(rows),flush=True)
