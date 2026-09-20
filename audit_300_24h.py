"""Frozen TSLA 300-origin, 24 elapsed hour Kronos audit. No trading or production writes."""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
EXPECTED_RAW_SHA='192ed91040f654900b8309d07992a9acd2d63a1fccfcccd85a23449f16d75c3b'
UPSTREAM='67b630e67f6a18c9e9be918d9b4337c960db1e9a'
CONFIG={
 'small':('Kronos-small','901c26c1332695a2a8f243eb2f37243a37bea320','b082dfcbd8e8c142a725c8bbb99781802f38fec81210e13479effb32b3c3e020','Kronos-Tokenizer-base','0e0117387f39004a9016484a186a908917e22426','59d85f6af76a2c3b8240ea06cb21db4213b4eeca053f246b23e29cf832fc6bee'),
 'mini':('Kronos-mini','f4e68697d9d5aed55cef5c96aabc3376bcad9f81','a7d5f37e2e9fbd9891f7d7d4f72574512dd1f704fee14223e0a8cd0fbf54197c','Kronos-Tokenizer-2k','26966d0035065a0cae0ebad7af8ece35bc1fb51c','b97ec46b3b72160509e289183eaf7bdf5f0dac5bb9b49522f6d46638a99a8717')}
COLS=['open','high','low','close','volume']
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def future_grid(start,end):
    first=pd.DatetimeIndex([start+pd.Timedelta(hours=i) for i in range(4)])
    ds=end.strftime('%Y-%m-%d')
    second=pd.DatetimeIndex([pd.Timestamp(ds+f' {h:02}:00',tz='America/New_York') for h in range(4,10)]+[pd.Timestamp(ds+f' {h:02}:30',tz='America/New_York') for h in range(9,16)])
    return first.append(second)
def prepare(rawpath):
    import pandas_market_calendars as mcal
    if sha(rawpath)!=EXPECTED_RAW_SHA:raise RuntimeError('Frozen data checksum mismatch')
    d=pd.read_csv(rawpath);d.columns=d.columns.str.lower();d['timestamp']=pd.to_datetime(d.timestamp,utc=True).dt.tz_convert('America/New_York');d=d.set_index('timestamp')[COLS]
    if d.index.duplicated().any() or not d.index.is_monotonic_increasing:raise RuntimeError('Duplicate or unsorted input')
    if not np.isfinite(d.to_numpy()).all() or (d[COLS[:4]]<=0).any().any() or (d.volume<0).any():raise RuntimeError('Invalid values')
    if ((d.high+1e-5<d[['open','close','low']].max(axis=1)) | (d.low-1e-5>d[['open','close','high']].min(axis=1))).any():raise RuntimeError('Invalid OHLC geometry')
    sched=mcal.get_calendar('NYSE').schedule(start_date='2024-09-23',end_date='2026-09-18')
    dates=set(sched.index[((sched.market_close-sched.market_open)==pd.Timedelta(hours=6.5))].strftime('%Y-%m-%d'))
    eligible=[];excluded=[]
    for ds in sorted(dates):
        end=pd.Timestamp(ds+' 16:00',tz='America/New_York');start=end-pd.Timedelta(hours=24)
        if str(start.date()) not in dates:continue
        hist=d.loc[d.index<start]
        last=start-pd.Timedelta(minutes=30);target=end-pd.Timedelta(minutes=30)
        if len(hist)<400:continue
        if last not in d.index or target not in d.index:
            excluded.append({'date':ds,'reason':'missing_origin_or_terminal_RTH_bar'});continue
        grid=future_grid(start,end)
        if len(grid)!=17 or (end-start)!=pd.Timedelta(hours=24):raise RuntimeError('Incorrect horizon')
        y=d.reindex(grid)
        eligible.append({'start':str(start),'end':str(end),'date':ds,'target_observed_bars':int(y.close.notna().sum()),'target_missing_times':[str(t) for t in y.index[y.close.isna()]],'post_release':start>=pd.Timestamp('2025-07-02',tz='America/New_York')})
    if len(eligible)<300:raise RuntimeError(f'Only {len(eligible)} eligible 24-hour windows, requires 300')
    chosen=eligible[-300:]
    meta={'ticker':'TSLA','n_origins':300,'first_origin':chosen[0]['start'],'last_target':chosen[-1]['end'],'forecast_horizon_elapsed_hours':24,'history_bars':400,'future_scheduled_bars':17,'models':['small','mini'],'samples_per_origin':5,'T':.6,'top_p':.9,'raw_sha256':EXPECTED_RAW_SHA,'upstream_commit':UPSTREAM,'n_post_release':sum(r['post_release'] for r in chosen),'n_before_release':sum(not r['post_release'] for r in chosen),'excluded_missing_endpoint':excluded,'n_total_eligible':len(eligible),'overnight_20_to_04_observed':False,'reported_extended_volume_zero_is_not_verified_true_zero':True,'vendor_bar_durations':'60 minutes except 09:00 and 15:30 bins are 30 minutes','actual_price_vintage_PIT_verified':False,'execution_is_forecast_benchmark_not_trading_simulation':True,'anchor_variant':'Separate unvalidated candidate: shift entire generated OHLC path by input close minus reconstruction of last known input close; no weights changed','selected_dates_before_inference':[r['date'] for r in chosen]}
    return d,chosen,meta
def features(hist,anchor):
    regular=hist.loc[hist.index.strftime('%H:%M')=='15:30','close'];ret=regular.pct_change().dropna()
    vol=float(ret.tail(20).std());vcut=float(ret.rolling(20).std().dropna().tail(252).median());past20=float(regular.iloc[-1]/regular.iloc[-21]-1)
    z=past20/(max(vol,1e-12)*np.sqrt(20));regime='uptrend' if z>.5 else 'downtrend' if z<-.5 else 'range'
    return {'trend':regime,'volatility':'high' if vol>vcut else 'low','vol20':vol,'past_return20':past20,'previous_day_return':float(ret.iloc[-1]),'drift20':float(ret.tail(20).mean())}
def run(args):
    import torch
    from huggingface_hub import hf_hub_download
    from model import Kronos,KronosTokenizer,KronosPredictor
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    d,origins,meta=prepare(args.raw);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    (out/f'protocol_{args.model}_{args.shard}.json').write_text(json.dumps(meta,indent=2))
    model_id,revision,mhash,tokid,trevision,thash=CONFIG[args.model];local=[]
    for rid,rev,hs in [(model_id,revision,mhash),(tokid,trevision,thash)]:
        dest=Path('audit_weights')/rid;dest.mkdir(parents=True,exist_ok=True)
        for filename in ['config.json','model.safetensors']:
            pp=hf_hub_download('NeoQuasar/'+rid,filename,revision=rev,local_dir=dest)
            if filename=='model.safetensors' and sha(pp)!=hs:raise RuntimeError('Weight checksum mismatch')
        local.append(dest)
    tok=KronosTokenizer.from_pretrained(str(local[1])).eval();model=Kronos.from_pretrained(str(local[0])).eval();p=KronosPredictor(model,tok,device='cpu',max_context=512)
    results=[];chosen=[(i,r) for i,r in enumerate(origins) if i%6==args.shard];t0=time.time();f=out/f'forecasts_{args.model}_{args.shard}.jsonl'
    with f.open('w') as handle:
        for j,(i,origin) in enumerate(chosen):
            start=pd.Timestamp(origin['start']);end=pd.Timestamp(origin['end']);grid=future_grid(start,end)
            history=d.loc[d.index<start].copy();hist=history.tail(400).copy()
            assert hist.index[-1]==start-pd.Timedelta(minutes=30)
            anchor=float(hist.close.iloc[-1]);regime=features(history,anchor)
            seed=int.from_bytes(hashlib.sha256(f'42|TSLA|{origin["date"]}|24h'.encode()).digest()[:4],'little')%(2**31-1)
            torch.manual_seed(seed)
            pred=p.predict(df=hist,x_timestamp=pd.Series(hist.index),y_timestamp=pd.Series(grid),pred_len=17,T=.6,top_p=.9,top_k=0,sample_count=5,verbose=False)
            if len(pred)!=17 or not np.isfinite(pred.to_numpy()).all():raise RuntimeError('Invalid forecast')
            # Reconstruct only known history with the same scaling, never target prices.
            h=hist.copy();h['amount']=h.volume*h[['open','high','low','close']].mean(axis=1)
            x=h[['open','high','low','close','volume','amount']].to_numpy(np.float32);mu=x.mean(0);sd=x.std(0);xx=np.clip((x-mu)/(sd+1e-5),-5,5)
            with torch.no_grad():
                ids=tok.encode(torch.from_numpy(xx).unsqueeze(0),half=True)
                recon=tok.decode(ids,half=True)[0,-1,:].cpu().numpy()*(sd+1e-5)+mu
            shift=anchor-float(recon[3]);actual=d.reindex(grid);a=actual[['open','high','low','close']].to_numpy();pv=pred[['open','high','low','close']].to_numpy()
            row={'origin_index':i,'model':args.model,'origin':origin,'anchor':anchor,'last_known_timestamp':str(hist.index[-1]),'future_times':[str(t) for t in grid],'actual_ohlc':[[float(v) if np.isfinite(v) else None for v in row] for row in a],'predicted_ohlc':pv.tolist(),'history_anchor_shift':shift,'seed':seed,'regime':regime,'input_high_abs_z':float(np.abs((x-mu)/(sd+1e-5)).max()),'original_pred_invalid_geometry_n':int(((pv[:,1]<np.max(pv[:,[0,2,3]],axis=1))|(pv[:,2]>np.min(pv[:,[0,1,3]],axis=1))).sum())}
            handle.write(json.dumps(row,allow_nan=False)+'\n');handle.flush();results.append(row)
            if (j+1)%10==0:print('COMPLETED',args.model,args.shard,j+1,'/',len(chosen),'seconds',round(time.time()-t0,1),flush=True)
    summary={'model':args.model,'shard':args.shard,'n_completed':len(results),'n_requested':len(chosen),'elapsed_seconds':time.time()-t0,'data_sha256':EXPECTED_RAW_SHA,'model_revision':revision,'model_weights_sha256':mhash,'tokenizer_revision':trevision,'tokenizer_sha256':thash,'torch':torch.__version__,'execution_commit':os.environ.get('GITHUB_SHA'),'status':'completed'}
    (out/f'run_{args.model}_{args.shard}.json').write_text(json.dumps(summary,indent=2));print('SHARD_COMPLETE='+json.dumps(summary),flush=True)
def bootstrap_metrics(g):
    rng=np.random.default_rng(77321);n=len(g);L=min(5,n);reps=4000
    starts=rng.integers(0,n,size=(reps,int(np.ceil(n/L))));idx=((starts[:,:,None]+np.arange(L))%n).reshape(reps,-1)[:,:n]
    hits=g.direction_correct.to_numpy(float);loss=g.endpoint_abs_error_pct.to_numpy(float);base=g.persistence_abs_error_pct.to_numpy(float)
    acc=hits[idx].mean(1)*100;sk=(1-loss[idx].mean(1)/base[idx].mean(1))*100
    return [float(x) for x in np.quantile(acc,[.025,.975])],[float(x) for x in np.quantile(sk,[.025,.975])]
def summarize(args):
    root=Path(args.out);rows=[];receipts=[]
    for f in sorted(root.rglob('forecasts_*.jsonl')):rows += [json.loads(x) for x in f.read_text().splitlines()]
    for f in sorted(root.rglob('run_*.json')):receipts.append(json.loads(f.read_text()))
    if len(receipts)!=12 or any(x['status']!='completed' or x['n_completed']!=50 for x in receipts):raise RuntimeError('Incomplete model execution')
    for model in ['small','mini']:
        rr=[r for r in rows if r['model']==model]
        if len(rr)!=300 or len({r['origin_index'] for r in rr})!=300:raise RuntimeError('Missing or duplicate origins')
    prot=json.loads(next(root.rglob('protocol_*.json')).read_text());flat=[]
    for r in rows:
        anchor=r['anchor'];a=np.asarray(r['actual_ohlc'],dtype=float);base=np.asarray(r['predicted_ohlc'],dtype=float)
        for variant,shift in [('original',0.),('history_anchor',r['history_anchor_shift'])]:
            p=base+shift;terminal_a=float(a[-1,3]);terminal_p=float(p[-1,3]);ar=terminal_a/anchor-1;pr=terminal_p/anchor-1
            ac=a[:,3];pc=p[:,3];ok=np.isfinite(ac)
            rec={'model':r['model'],'variant':variant,'date':r['origin']['date'],'origin_index':r['origin_index'],'post_release':r['origin']['post_release'],'anchor':anchor,'actual_close':terminal_a,'predicted_close':terminal_p,'actual_return_pct':ar*100,'predicted_return_pct':pr*100,'direction_correct':int(np.sign(ar)==np.sign(pr)),'endpoint_abs_error_usd':abs(terminal_p-terminal_a),'endpoint_abs_error_pct':abs(terminal_p-terminal_a)/anchor*100,'persistence_abs_error_pct':abs(ar)*100,'path_close_mae_pct':float(np.abs(pc[ok]-ac[ok]).mean()/anchor*100),'persistence_path_mae_pct':float(np.abs(ac[ok]-anchor).mean()/anchor*100),'path_observations':int(ok.sum()),'trend':r['regime']['trend'],'volatility':r['regime']['volatility'],'predicted_change_over_2pct':bool(abs(pr)>=.02),'actual_change_over_2pct':bool(abs(ar)>=.02),'correct_direction_actual_over_2pct':bool(np.sign(ar)==np.sign(pr)) if abs(ar)>=.02 else None,'high_abs_error_pct':float(abs(p[:,1].max()-np.nanmax(a[:,1]))/anchor*100),'low_abs_error_pct':float(abs(p[:,2].min()-np.nanmin(a[:,2]))/anchor*100),'rth_open_actual':float(a[10,0]),'rth_open_predicted':float(p[10,0]),'rth_open_abs_error_pct':float(abs(p[10,0]-a[10,0])/anchor*100),'invalid_geometry_bars':r['original_pred_invalid_geometry_n']}
            for name,k in [('after_hours',3),('premarket',9),('rth',16)]:
                rec[name+'_endpoint_abs_error_pct']=float(abs(p[k,3]-a[k,3])/anchor*100) if np.isfinite(a[k,3]) else None
            flat.append(rec)
    df=pd.DataFrame(flat).sort_values(['model','variant','origin_index']);df.to_csv(root/'session_results.csv',index=False);summaries=[]
    for (model,variant),g0 in df.groupby(['model','variant']):
        for split,g in [('all_300',g0),('after_release',g0.loc[g0.post_release])]:
            aci,sci=bootstrap_metrics(g);n=len(g);baseline=g.persistence_abs_error_pct.mean();ae=g.endpoint_abs_error_pct.mean()
            summaries.append({'model':model,'variant':variant,'split':split,'n':n,'correct':int(g.direction_correct.sum()),'direction_accuracy_pct':float(g.direction_correct.mean()*100),'direction_accuracy_block_bootstrap95':aci,'always_up_accuracy_pct':float((g.actual_return_pct>0).mean()*100),'always_down_accuracy_pct':float((g.actual_return_pct<0).mean()*100),'endpoint_MAE_usd':float(g.endpoint_abs_error_usd.mean()),'endpoint_MAE_pct':float(ae),'persistence_endpoint_MAE_pct':float(baseline),'endpoint_skill_vs_persistence_pct':float((1-ae/baseline)*100),'endpoint_skill_block_bootstrap95':sci,'path_MAE_pct':float(g.path_close_mae_pct.mean()),'persistence_path_MAE_pct':float(g.persistence_path_mae_pct.mean()),'return_pearson_corr':float(g.actual_return_pct.corr(g.predicted_return_pct)),'return_spearman_corr':float(g.actual_return_pct.corr(g.predicted_return_pct,method='spearman')),'actual_moves_over_2pct_n':int(g.actual_change_over_2pct.sum()),'direction_accuracy_on_actual_moves_over_2pct':float(g.loc[g.actual_change_over_2pct,'direction_correct'].mean()*100),'path_actual_coverage_pct':float(g.path_observations.sum()/(n*17)*100),'invalid_geometry_bars':int(g.invalid_geometry_bars.sum()),'rth_open_MAE_pct':float(g.rth_open_abs_error_pct.mean()),'AH_endpoint_MAE_pct':float(g.after_hours_endpoint_abs_error_pct.mean()),'PM_endpoint_MAE_pct':float(g.premarket_endpoint_abs_error_pct.mean())})
    regimes=[]
    for (model,var,trend,vol),g in df.groupby(['model','variant','trend','volatility']):
        regimes.append({'model':model,'variant':var,'trend':trend,'volatility':vol,'n':len(g),'hits':int(g.direction_correct.sum()),'accuracy_pct':float(g.direction_correct.mean()*100),'MAE_pct':float(g.endpoint_abs_error_pct.mean()),'naive_MAE_pct':float(g.persistence_abs_error_pct.mean())})
    pd.DataFrame(regimes).to_csv(root/'regimes.csv',index=False)
    result={'protocol':prot,'n_models_completed':2,'n_distinct_24h_sessions':300,'total_genuine_model_paths':600,'summary':summaries,'regimes':regimes,'receipts':receipts,'limitations':['No 20:00-04:00 historical trading observations; never fabricated or forward-filled.','Zero extended-session volumes reported by Yahoo are not confirmed true zero.','Not all 300 dates are later than pretrained weights. Evaluate after_release separately.','Bars are retrospective vendor history, not vintage point-in-time snapshots.','History-anchor candidate was fixed before this experiment, not fine-tuned or selected using these outcomes.','These are forecasting errors and hits, not executable option or equity trading returns.','Bootstrap confidence intervals use circular five-session blocks, 4000 replications.']}
    (root/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print('FINAL_SUMMARY='+json.dumps({'n_sessions':300,'n_paths':600,'protocol':prot,'summary':summaries},allow_nan=False),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--raw',default='evidence/TSLA_1h_extended_raw.csv');ap.add_argument('--model',choices=['small','mini']);ap.add_argument('--shard',type=int,default=0);ap.add_argument('--out',default='results');ap.add_argument('--summarize',action='store_true');args=ap.parse_args()
    if args.summarize:summarize(args)
    else:run(args)
