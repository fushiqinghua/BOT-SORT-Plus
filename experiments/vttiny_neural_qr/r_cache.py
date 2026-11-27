"""Extract current-frame visual features once, without loading test annotations."""
import hashlib
from pathlib import Path
import cv2
import numpy as np
from botsort_plus.io import detection_files, detection_fingerprint, read_detection_cache, resolve_image, sequence_name, forbid_ground_truth
from quality_features import frame_features
from qr_settings import ROOT, SRC, SETTINGS, read, write, sha
import sys

def main():
    sys.addaudithook(forbid_ground_truth)
    cv2.setNumThreads(1)
    settings=read(SETTINGS)
    root=Path(settings['images_root'])
    folder=ROOT/'quality_cache'
    folder.mkdir(exist_ok=True)
    files=detection_files(Path(settings['baseline_root'])/'detections')
    assert len(files)==30
    feature_sha=sha(SRC/'quality_features.py')
    for p in files:
        seq=sequence_name(p)
        output=folder/(seq+'.npz')
        manifest=folder/(seq+'.json')
        cache=read_detection_cache(p)
        det_hash=detection_fingerprint(cache)
        if manifest.exists() and output.exists():
            meta=read(manifest)
            if (meta['detection_fingerprint']==det_hash and meta['feature_sha256']==sha(output)
                    and meta['feature_source_sha256']==feature_sha):continue
        offsets=[0]
        rows=[]
        digest=hashlib.sha256()
        for t,frame in enumerate(cache['frames'],1):
            rgb=resolve_image(frame['image'],root)
            ir=resolve_image(frame['thermal'],root)
            if not (frame['frame']==t and rgb.name==ir.name and rgb.parent.name=='00' and ir.parent.name=='01'
                    and rgb.parent.parent==ir.parent.parent and rgb.parent.parent.name==seq):
                raise ValueError(f'Unpaired modalities: {seq}/{t}')
            images=[]
            for path,flags in ((rgb,cv2.IMREAD_COLOR),(ir,cv2.IMREAD_GRAYSCALE)):
                data=path.read_bytes()
                digest.update(hashlib.sha256(data).digest())
                image=cv2.imdecode(np.frombuffer(data,dtype=np.uint8),flags)
                if image is None or image.shape[:2]!=(512,640):raise ValueError(str(path))
                images.append(image)
            values=frame_features(*images,frame['detections'])
            rows.append(values)
            offsets.append(offsets[-1]+len(values))
        np.savez_compressed(output,features=np.concatenate(rows),offsets=np.asarray(offsets,dtype=np.int64))
        write(manifest,dict(sequence=seq,frames=len(rows),detections=offsets[-1],detection_fingerprint=det_hash,
                            feature_source_sha256=feature_sha,frame_pair_stream_sha256=digest.hexdigest(),
                            feature_sha256=sha(output)))
        print('Visual features',seq,len(rows),offsets[-1],flush=True)
if __name__=='__main__':main()
