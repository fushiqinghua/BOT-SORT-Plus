"""Untrained short-time infrared evidence at fixed detector rows or projected boxes."""
from collections import defaultdict,deque
from pathlib import Path
import hashlib
import json
import cv2
import numpy as np

from .ir_pixels import Extractor,Stamp,INVALID


class ThermalObserver:
    def __init__(self,tiles_dir,seq,detector_fingerprint,policy,settings):
        self.policy=policy
        self.settings=settings
        self.extractor=Extractor(settings)
        self.shift=bool(policy.get('shift',False))
        path=Path(tiles_dir)/(seq+'.npz')
        self.meta=json.loads(path.with_suffix('.json').read_bytes())
        if (self.meta['detection_fingerprint']!=detector_fingerprint
                or hashlib.sha256(path.read_bytes()).hexdigest()!=self.meta['tiles_sha256']):
            raise ValueError('Raw IR tiles and frozen detections disagree')
        with np.load(path,allow_pickle=False) as z:
            self.tiles=z['tiles'][int(self.shift)].copy()
            self.flags=z['flags'][int(self.shift)].copy()
            self.offsets=z['offsets'];self.motion=z['motion']
        self.past={};self.frame=0;self.per_frame={};self.audit=[]
        self.current_image=None

    def begin(self,frame,image_path,num_detections):
        if frame!=self.frame+1 or self.offsets[frame]-self.offsets[frame-1]!=num_detections:
            raise ValueError('Misaligned IR frame/detection inputs')
        if self.meta['file_names'][frame-1]!=Path(image_path).name:
            raise ValueError('Infrared filename does not match current RGB frame')
        self.frame=frame;self.audit=[];self.current_image=None;self.pair_cache={}
        self.ir_path=Path(image_path).parent.parent/'01'/Path(image_path).name
        self.per_frame={key:v for key,v in self.per_frame.items() if key[0]>=frame-4}

    def detection(self,f,index):
        key=(int(f),int(index))
        if not self.frame-4<=f<=self.frame or f<1 or not 0<=index<self.offsets[f]-self.offsets[f-1]:
            raise ValueError('Invalid causal access to an IR detection tile')
        if key not in self.per_frame:
            row=self.offsets[f-1]+index;code=int(self.flags[row])
            self.per_frame[key]=(None,INVALID[code]) if code else (Stamp(self.tiles[row]),'ok')
        return self.per_frame[key]

    def prediction_stamp(self,box):
        if self.current_image is None:
            data=self.ir_path.read_bytes()
            if hashlib.sha256(data).hexdigest()!=self.meta['ir_frame_sha256'][self.frame-1]:
                raise ValueError('Changed IR image after registration/cache validation')
            self.current_image=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_GRAYSCALE)
        delta=self.motion[self.frame-1]
        patch,code=self.extractor.sample(self.current_image,box,delta if np.isfinite(delta).all() else None,self.shift)
        return (None,INVALID[code]) if code else (Stamp(patch),'ok')

    def compare(self,tid,current,last_seen=None):
        stamp,why=current
        if stamp is None:return None,why
        if not stamp.usable:return None,'low_local_contrast'
        bank=self.past.get(tid)
        if not bank or len(bank)<2:return None,'short_history'
        (t0,old),(t1,recent)=bank[-2:]
        if (self.frame-t1>8 or t1-t0!=1 or t1>=self.frame or
                (last_seen is not None and last_seen!=t1)):
            return None,'history_gap'
        if old is None or recent is None or not old.usable or not recent.usable:
            return None,'weak_history'
        stability=recent.similarity(old)
        if stability<.55 or old.contrast*recent.contrast<=0:return None,'unstable_history'
        # Both recent observations must see a similar pattern at one displacement.
        s0=cv2.matchTemplate(stamp.patch[3:25,3:25],old.template,cv2.TM_CCOEFF_NORMED)
        s1=cv2.matchTemplate(stamp.patch[3:25,3:25],recent.template,cv2.TM_CCOEFF_NORMED)
        sim=float(((s0+s1)*.5).max())
        flip=bool(stamp.contrast*recent.contrast<0 and min(stamp.salience,recent.salience)>=1.)
        return (sim,flip,stability), 'usable'

    @staticmethod
    def contradict(evidence,sensitive=False):
        sim,flip,_=evidence
        return sim<(.40 if sensitive else .25) or (flip and sim<(.60 if sensitive else .45))

    def check_output(self,measured_and_projected):
        mode=self.policy.get('presence')
        if not mode:return measured_and_projected
        keep=[]
        for row in measured_and_projected:
            if row[7]>=0:keep.append(True);continue
            tid=int(row[4])
            if mode=='position':
                from .spatial_presence import decide
                others=measured_and_projected[measured_and_projected[:,7]>=0,:4]
                accepted, reason, evidence=decide(self,tid,row[:4],others)
                self.audit.append(dict(event='ir_projection_check',frame=self.frame,track_id=tid,
                              status=reason,kept=accepted,similarity=None,spatial_evidence=evidence))
                keep.append(accepted)
                continue
            evidence,reason=self.compare(tid,self.prediction_stamp(row[:4]))
            rejected=False
            if evidence is not None:
                rejected=self.contradict(evidence,mode=='sensitive')
                reason='contradiction' if rejected else ('support' if evidence[0]>=.55 and not evidence[1] else 'inconclusive')
            accepted=(mode=='shadow' or not rejected)
            self.audit.append(dict(event='ir_projection_check',frame=self.frame,track_id=tid,
                          status=reason,kept=accepted,similarity=evidence[0] if evidence else None))
            keep.append(accepted)
        return measured_and_projected[np.asarray(keep,dtype=bool)]

    def finish(self,output):
        # Only current measured outputs enter history. No generated box feeds back.
        self.past={tid:hist for tid,hist in self.past.items() if self.frame-hist[-1][0]<=8}
        for row in output:
            index=int(row[7]);tid=int(row[4])
            if index<0:continue
            stamp,_=self.detection(self.frame,index)
            hist=self.past.get(tid,[])
            hist.append((self.frame,stamp));self.past[tid]=hist[-2:]

    def adjust_competition(self,costs,feasible,limit,score,stage,ids,indices,apply):
        # Only close competitors which the original matcher accepts spatially.
        if not min(costs.shape,default=0):return costs
        possible=np.where(feasible,costs,np.inf)
        groups=[]
        for axis in (0,1):
            counts=feasible.sum(axis=axis)
            for slot in np.flatnonzero(counts>=2):
                v=possible[:,slot] if axis==0 else possible[slot]
                near=np.flatnonzero(v<=v.min()+.15)
                if len(near)>1:groups.append([(int(p),int(slot)) if axis==0 else (int(slot),int(p)) for p in near])
        if not groups:return costs
        delta=defaultdict(set);scored={};comparable=0
        for group in groups:
            for pair in group:
                if pair not in scored:scored[pair]=score(*pair)
            if any(scored[p] is None for p in group):continue
            comparable+=1
            order=sorted(group,key=lambda p:scored[p],reverse=True)
            best,second=order[:2]
            if scored[best]<.50 or scored[best]-scored[second]<.18:continue
            delta[best].add(-1)
            for p in order[1:]:delta[p].add(1)
        result=costs.copy()
        for (i,j),directions in delta.items():
            if len(directions)!=1:continue
            result[i,j]=np.clip(costs[i,j]+.08*next(iter(directions)),0.,limit-1e-8)
        changed=np.argwhere(result!=costs)
        self.audit.append(dict(event='ir_competition',frame=self.frame,stage=stage,groups=len(groups),
                    comparable_groups=comparable,changed_pairs=len(changed),applied=bool(apply),
                    changes=[[int(ids[i]),int(indices[j]),float(costs[i,j]),float(result[i,j])] for i,j in changed]))
        return result if apply else costs

    def native(self,costs,geo,tracks,dets,threshold,prox):
        mode=self.policy.get('native')
        if not mode:return costs
        ids=[int(t.id) for t in tracks];inds=[int(d.det_ind) for d in dets]
        def score(i,j):
            cmp,_=self.compare(ids[i],self.detection(self.frame,inds[j]))
            return cmp[0] if cmp else None
        return self.adjust_competition(costs,(geo<=prox)&(costs<=threshold),threshold,score,'native_high',ids,inds,mode!='shadow')

    def recovery(self,costs,info,views,histories,cids,limit):
        mode=self.policy.get('recovery')
        if not mode:return costs
        table={}
        def value(i,j):
            if (i,j) not in table:
                view=views[i];hist=histories[j]
                if self.frame-view.last_seen>8 or len(hist)<2 or hist[-1].frame!=self.frame:
                    table[i,j]=None
                else:
                    dual=[self.compare(int(view.track_id),self.detection(o.frame,o.det_index),view.last_seen)[0] for o in hist[-2:]]
                    table[i,j]=dual if all(v is not None for v in dual) else None
            return table[i,j]
        if mode in ('delay','shadow'):
            for i,j in np.argwhere(np.isfinite(costs)):
                item=info[int(i),int(j)]
                if not item.get('two_hit_evidence'):continue
                dual=value(int(i),int(j));bad=dual is not None and all(self.contradict(v) for v in dual)
                self.audit.append(dict(event='ir_early_check',frame=self.frame,track_id=int(views[i].track_id),
                          candidate_id=int(cids[j]),comparable=dual is not None,would_defer=bool(bad),applied=mode=='delay'))
                if bad and mode=='delay':item['two_hit_evidence']=False
        if mode not in ('rerank','shadow'):return costs
        def score(i,j):
            dual=value(i,j)
            return float(np.mean([v[0] for v in dual])) if dual else None
        ans=self.adjust_competition(costs,np.isfinite(costs),limit,score,'B_recovery',[v.track_id for v in views],cids,mode=='rerank')
        if mode=='rerank':
            for i,j in np.argwhere(ans!=costs):
                info[int(i),int(j)]['ir_original_cost']=float(costs[i,j]);info[int(i),int(j)]['cost']=float(ans[i,j])
        return ans
