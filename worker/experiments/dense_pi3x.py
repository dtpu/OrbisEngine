#!/usr/bin/env python3
"""At least 12 independently reconstructed person frames/second, shared static anchors.

This increases temporal density, not hidden-side reconstruction. The single-view
surface limitation remains explicit. All chunks share a measured similarity frame.
"""
from __future__ import annotations
import argparse, gc, hashlib, json, os, shutil, sys, time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from dense_environment import similarity
from dense_cameras import viewer_camera, fit_ray_intrinsics
from wander_worker.masks import people_masks
from wander_worker.ply import write_point_ply,home_distance


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--video',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--fps',type=float,default=12);ap.add_argument('--anchors',type=int,default=8);ap.add_argument('--batch',type=int,default=16)
    ap.add_argument('--repo',default=os.path.expanduser('~/pi3'));a=ap.parse_args()
    if a.fps<12:raise ValueError('Dense quality experiment requires >=12 fps')
    if a.anchors<2 or a.batch<1:raise ValueError('Need >=2 anchors and a positive batch size')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);framesdir=out/'input';framesdir.mkdir(exist_ok=True);temp=out/'batch-input';temp.mkdir(exist_ok=True)
    t0=time.time();cap=cv2.VideoCapture(a.video);srcfps=cap.get(cv2.CAP_PROP_FPS);count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));duration=count/srcfps
    if not np.isfinite(srcfps) or srcfps<=0 or count<2:raise ValueError('Invalid source video metadata')
    source_size=(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    requested=np.arange(int(np.ceil(duration*a.fps)))/a.fps
    indices=np.clip(np.round(requested*srcfps).astype(int),0,count-1)
    if len(np.unique(indices))!=len(indices):raise ValueError('Source frame rate below requested independent output density')
    times=indices/srcfps;N=len(indices)
    for i,idx in enumerate(indices):
        file=framesdir/f'f_{i:06d}.jpg'
        if file.exists():continue
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(idx));ok,bgr=cap.read()
        if not ok:raise RuntimeError(f'Decode failed: {idx}')
        if bgr.shape[1]>1024:bgr=cv2.resize(bgr,(1024,round(bgr.shape[0]*1024/bgr.shape[1])))
        cv2.imwrite(str(file),bgr,[cv2.IMWRITE_JPEG_QUALITY,96])
    cap.release();files=sorted(framesdir.glob('f_*.jpg'))
    import torch
    sys.path.insert(0,a.repo)
    from pi3.models.pi3x import Pi3X
    from pi3.utils.basic import load_images_as_tensor
    from pi3.utils.geometry import depth_normal_edge
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=True
    model=Pi3X.from_pretrained('yyfz233/Pi3X').eval().cuda();model.disable_multimodal()
    def predict(ids):
        for file in temp.glob('*.jpg'):file.unlink()
        for idx in ids:os.symlink(files[idx].resolve(),temp/files[idx].name)
        imgs=load_images_as_tensor(str(temp),verbose=False)[None].cuda()
        print('inference tensor',tuple(imgs.shape),flush=True)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):r=model(imgs=imgs)
        confidence=torch.sigmoid(r['conf'][0,...,0]);valid=confidence>.3
        edge=depth_normal_edge(r['local_points'],rtol=.03,mask=valid[None])[0]
        result=dict(points=r['points'][0].float().cpu().numpy(),poses=r['camera_poses'][0].float().cpu().numpy(),rays=r['rays'][0].float().cpu().numpy(),conf=confidence.float().cpu().numpy(),valid=(valid & ~edge).cpu().numpy(),rgb=(imgs[0].permute(0,2,3,1).float().cpu().numpy()*255).astype('uint8'))
        del imgs,r;torch.cuda.empty_cache();return result
    anchors=np.unique(np.linspace(0,N-1,min(a.anchors,N)).round().astype(int)).tolist()
    basefile=out/'anchors.npz'
    if basefile.exists():base=dict(np.load(basefile))
    else:
        base=predict(anchors);base['people']=people_masks(base['rgb'],dilate_px=2);np.savez_compressed(basefile,**base)
    inv=np.linalg.inv(base['poses'][0]);R0,t00=inv[:3,:3],inv[:3,3];flip=np.array([1.,-1.,-1.])
    zz=(base['points'][0]@R0.T+t00)[...,2];good=base['valid'][0]&(~base['people'][0])&(zz>0);scale=3/np.median(zz[good])
    rng=np.random.default_rng(13);matches=[]
    for j in range(len(anchors)):
        good=base['valid'][j] & (~base['people'][j]) & (base['conf'][j]>.5)
        idx=np.flatnonzero(good);matches.append(rng.choice(idx,min(1500,len(idx)),replace=False))
    cams=[None]*N;log=[];point_counts=[]
    camera_dir=out/'camera-batches';camera_dir.mkdir(exist_ok=True)
    for lo in range(0,N,a.batch):
        chunk=list(range(lo,min(N,lo+a.batch)));ids=sorted(set(anchors+chunk));print('Pi3X batch',lo,'/',N,'views',len(ids),flush=True)
        p=predict(ids)
        src=np.concatenate([p['points'][ids.index(ai)].reshape(-1,3)[matches[j]] for j,ai in enumerate(anchors)])
        dst=np.concatenate([base['points'][j].reshape(-1,3)[matches[j]] for j in range(len(anchors))])
        s,R,t,rms=similarity(src,dst)
        print('alignment',float(s),rms,flush=True)
        if not np.isfinite(rms) or not .1<s<10:raise RuntimeError('Invalid anchor alignment')
        np.savez_compressed(camera_dir/f'batch_{lo:03d}.npz',sample_ids=ids,source_indices=indices[ids],
            raw_opencv_camera_to_world=p['poses'],batch_rotation=R,batch_translation=t,batch_scale=s,
            reference_rotation=R0,reference_translation=t00,world_scale=scale)
        # Full-resolution masks preserve limbs; resize to the actual Pi3X pixel grid.
        full=np.stack([np.asarray(Image.open(files[i])) for i in ids]);pm=people_masks(full,dilate_px=1);H,W=p['rgb'].shape[1:3]
        pm=np.stack([cv2.resize(m.astype('uint8'),(W,H),interpolation=cv2.INTER_NEAREST)>0 for m in pm])
        for i in chunk:
            j=ids.index(i);valid=p['valid'][j]&pm[j]&np.isfinite(p['points'][j]).all(-1);xyz=p['points'][j][valid]@R.T*s+t;col=p['rgb'][j][valid]
            xyz=(xyz@R0.T+t00)*scale*flip
            if len(xyz)<100:raise RuntimeError(f'Person vanished at {times[i]}s ({len(xyz)} points)')
            write_point_ply(out/f'frame_{i:03d}.ply',xyz.astype('float32'),col)
            point_counts.append(len(xyz))
            camera=viewer_camera(p['poses'][j],s,R,t,R0,t00,scale)
            cams[i]=dict(position=camera[:3,3].tolist(),time=float(times[i]),sourceIndex=int(indices[i]),
                camera_to_world=camera.tolist(),world_to_camera=np.linalg.inv(camera).tolist(),
                **fit_ray_intrinsics(p['rays'][j],p['valid'][j],source_size))
            if i==0:np.savez_compressed(out/'source-camera-0.npz',rays=p['rays'][j],rgb=p['rgb'][j],valid=p['valid'][j],camera_to_world=camera)
            if i==0:home=home_distance(xyz)
        log.append(dict(first=lo,frames=len(chunk),scale=float(s),rotation=R.tolist(),translation=t.tolist(),rms=rms));(out/'alignment-batches.json').write_text(json.dumps(log,indent=2))
        del p,pm,full;gc.collect();torch.cuda.empty_cache()
    meta=dict(backend='pi3x-dense-anchors',frames=[f'frame_{i:03d}.ply' for i in range(N)],count=N,fps=a.fps,timestamps=times.tolist(),duration=duration,sourceFps=srcfps,sourceIndices=indices.tolist(),people_only=True,home_distance=home,sourceSha256=hashlib.sha256(Path(a.video).read_bytes()).hexdigest(),seconds=time.time()-t0,peakVRAMGB=torch.cuda.max_memory_allocated()/1e9,gpu=torch.cuda.get_device_name(),torch=torch.__version__,anchors=anchors,batch=a.batch,pointCounts=point_counts,note='Each source frame reconstructed with shared static anchors. >=12 independent frames/sec; single observed-facing surface, not complete human volume.')
    (out/'sequence.json').write_text(json.dumps(meta,indent=2));(out/'cameras.json').write_text(json.dumps(dict(
        coordinates='Both world and camera bases are OpenGL: x right, y up, -z forward. World matches exported PLY. Target camera zero is not assumed identity after batch registration.',
        reference=dict(rotation=R0.tolist(),translation=t00.tolist(),scale=float(scale),world_axis_flip=flip.tolist()),
        cameras=cams),indent=1));print(json.dumps({k:v for k,v in meta.items() if k not in ('frames','timestamps','sourceIndices')}),flush=True)

if __name__=='__main__':main()
