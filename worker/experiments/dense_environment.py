#!/usr/bin/env python3
"""All-frame DA3 in shared-anchor batches -> masked COLMAP + RGBD for mesh fusion.

One hypothesis: dense, registered observed pixels improve architectural continuity.
The masks exclude people; no view synthesis or image inpainting is used here.
Run with worker/.venv-da3/bin/python on CUDA. Outputs are resumable and kept isolated.
"""
from __future__ import annotations
import argparse, gc, hashlib, json, os, subprocess, sys, time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
from wander_worker.da3_backend import unproject_world
from wander_worker.masks import people_masks
from wander_worker.ply import write_point_ply


def similarity(src, dst):
    def fit(a, b):
        ma, mb = a.mean(0), b.mean(0); aa, bb = a-ma, b-mb
        U, S, V = np.linalg.svd(bb.T @ aa / len(a)); D = np.eye(3); D[2,2] = np.linalg.det(U @ V)
        R = U @ D @ V; s = np.sum(S*np.diag(D)) / np.mean(np.sum(aa*aa,1))
        return s, R, mb - s * R @ ma
    keep = np.ones(len(src), bool)
    for _ in range(4):
        s,R,t=fit(src[keep],dst[keep]);error=np.linalg.norm(src@R.T*s+t-dst,axis=1)
        keep=error <= np.percentile(error,75)
    return s,R,t,float(np.sqrt(np.mean(error[keep]**2)))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--video',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--anchors',type=int,default=12);ap.add_argument('--batch',type=int,default=16)
    ap.add_argument('--resolution',type=int,default=504);ap.add_argument('--width',type=int,default=1024)
    ap.add_argument('--model',default='depth-anything/DA3-LARGE-1.1')
    ap.add_argument('--max-frames',type=int,default=0,help='0 means every decoded frame')
    a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);start=time.time()
    for p in ('frames','rgbd','ds/images','ds/masks','ds/sparse/0'): (out/p).mkdir(parents=True,exist_ok=True)
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',a.video]))
    stream=next(s for s in probe['streams'] if s['codec_type']=='video');num,den=map(float,stream['avg_frame_rate'].split('/'));fps=num/den
    if not (out/'decoded.json').exists():
        cmd=['ffmpeg','-v','error','-i',a.video,'-vf',f'scale=min({a.width}\\,iw):-2','-vsync','0','-q:v','2']
        if a.max_frames:cmd+=['-frames:v',str(a.max_frames)]
        subprocess.run(cmd+[str(out/'frames/f_%06d.jpg')],check=True)
        (out/'decoded.json').write_text(json.dumps({'probe':probe,'sourceSha256':hashlib.sha256(Path(a.video).read_bytes()).hexdigest()}))
    files=sorted((out/'frames').glob('f_*.jpg'));N=len(files);print('decoded',N,'frames',fps,'fps',flush=True)
    anchors=np.unique(np.linspace(0,N-1,min(a.anchors,N)).round().astype(int)).tolist()
    import torch
    from depth_anything_3.api import DepthAnything3
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=True
    model=DepthAnything3.from_pretrained(a.model).to('cuda').eval()
    def predict(indices):
        return model.inference([str(files[i]) for i in indices],process_res=a.resolution,ref_view_strategy='first')
    anchor_file=out/'anchors.npz'
    if anchor_file.exists(): base=dict(np.load(anchor_file))
    else:
        p=predict(anchors)
        base=dict(depth=p.depth,conf=p.conf,extr=p.extrinsics,intr=p.intrinsics,rgb=p.processed_images)
        base['people']=people_masks(base['rgb'],dilate_px=8)
        np.savez_compressed(anchor_file,**base)
    base_world=unproject_world(base['depth'],base['extr'],base['intr'])
    R0,t0=base['extr'][0,:,:3],base['extr'][0,:,3]
    scale=3/np.median(base['depth'][0][~base['people'][0]])
    flip=np.diag([1.,-1.,-1.]);viewer_R=flip@R0
    rng=np.random.default_rng(7);matches=[]
    for j in range(len(anchors)):
        good=(~base['people'][j]) & (base['conf'][j]>np.percentile(base['conf'][j],45)) & (base['depth'][j]>0)
        idx=np.flatnonzero(good);matches.append(rng.choice(idx,min(1024,len(idx)),replace=False))
    log=[]
    for lo in range(0,N,a.batch):
        chunk=list(range(lo,min(N,lo+a.batch)))
        if all((out/f'rgbd/{i:06d}.npz').exists() for i in chunk):continue
        indices=sorted(set(anchors+chunk));print('batch',lo,'/',N,'views',len(indices),flush=True)
        p=predict(indices);wp=unproject_world(p.depth,p.extrinsics,p.intrinsics)
        src=np.concatenate([wp[indices.index(ai)].reshape(-1,3)[matches[j]] for j,ai in enumerate(anchors)])
        dst=np.concatenate([base_world[j].reshape(-1,3)[matches[j]] for j in range(len(anchors))])
        s,R,t,rms=similarity(src,dst);print('anchor alignment',s,'rms',rms,'relative',rms/np.median(base['depth']),flush=True)
        if not np.isfinite(rms) or not .1<s<10:raise RuntimeError('Invalid anchor alignment')
        pm=people_masks(p.processed_images,dilate_px=7)
        log.append(dict(first=lo,count=len(chunk),scale=float(s),rms=rms))
        for i in chunk:
            j=indices.index(i);E=p.extrinsics[j];C=-E[:,:3].T@E[:,3]
            c2w=np.eye(4);c2w[:3,:3]=R@E[:,:3].T;c2w[:3,3]=s*R@C+t
            # Express RGBD in the anchor frame, before the viewer-axis flip.
            depth=p.depth[j]*s;K=p.intrinsics[j];valid=(~pm[j]) & np.isfinite(depth) & (depth>0) & (p.conf[j]>np.percentile(p.conf[j],25))
            np.savez_compressed(out/f'rgbd/{i:06d}.npz',depth=depth,valid=valid,people=pm[j],conf=p.conf[j],c2w=c2w,intrinsics=K,rgb=p.processed_images[j],timestamp=i/fps)
            # DA3 upper_bound_resize uses anisotropic resizing, no crop. Scale K back to training resolution.
            im=Image.open(files[i]);W,H=im.size;h,w=depth.shape
            im.save(out/'ds/images'/files[i].name,quality=96)
            Image.fromarray((~pm[j]).astype('uint8')*255).resize((W,H),Image.Resampling.NEAREST).save(out/'ds/masks'/(files[i].stem+'.png'))
        del p,wp,pm;gc.collect();torch.cuda.empty_cache()
        (out/'batch-log.json').write_text(json.dumps(log,indent=2))
    del model;gc.collect();torch.cuda.empty_cache()
    cameras=[];images=[];points=[];colors=[];cam_views=[];c2ws=[];Ks=[]
    for i,file in enumerate(files):
        p=np.load(out/f'rgbd/{i:06d}.npz');K=p['intrinsics'].copy();c2w=p['c2w'];E=np.linalg.inv(c2w)[:3];W,H=Image.open(file).size;h,w=p['depth'].shape
        K[0]*=W/w;K[1]*=H/h
        cameras.append(f'{i+1} PINHOLE {W} {H} {K[0,0]} {K[1,1]} {K[0,2]} {K[1,2]}')
        q=Rotation.from_matrix(E[:,:3]).as_quat();q=q[[3,0,1,2]]
        images.extend([f'{i+1} '+ ' '.join(map(str,[*q,*E[:,3]]))+f' {i+1} {file.name}',''])
        world=unproject_world(p['depth'][None],E[None],p['intrinsics'][None])[0]
        ids=np.flatnonzero(p['valid']);ids=rng.choice(ids,min(2000,len(ids)),replace=False)
        points.append(world.reshape(-1,3)[ids]);colors.append(p['rgb'].reshape(-1,3)[ids])
        cam_views.append(dict(position=((R0@c2w[:3,3]+t0)*scale*np.array([1,-1,-1])).tolist(),name=file.name,time=i/fps))
        c2ws.append(c2w);Ks.append(p['intrinsics'])
    xyz=np.concatenate(points);rgb=np.concatenate(colors)
    # Voxel consolidate initialization to avoid repeated pixels dominating density.
    voxel=np.maximum(np.median(base['depth'])*.003,1e-5);_,indices=np.unique(np.floor(xyz/voxel).astype('int32'),axis=0,return_index=True)
    xyz,rgb=xyz[indices],rgb[indices]
    sparse=out/'ds/sparse/0';(sparse/'cameras.txt').write_text('\n'.join(cameras)+'\n');(sparse/'images.txt').write_text('\n'.join(images)+'\n')
    with (sparse/'points3D.txt').open('w') as f:
        for i,(point,col) in enumerate(zip(xyz,rgb)):f.write(f'{i+1} '+ ' '.join(map(str,point))+' '+ ' '.join(map(str,col))+' 1\n')
    viewer=(xyz@R0.T+t0)*scale*np.array([1,-1,-1]);write_point_ply(out/'points.ply',viewer.astype('float32'),rgb)
    np.savez(out/'poses.npz',c2w=c2ws,intrinsics=Ks,R0=R0,t0=t0,scale=scale,flip=np.array([1,-1,-1]),image_size=[w,h],timestamps=np.arange(N)/fps)
    (out/'cameras.json').write_text(json.dumps(dict(cameras=cam_views),indent=1))
    meta=dict(model=a.model,frames=N,fps=fps,seconds=time.time()-start,points=len(xyz),anchors=anchors,batch=a.batch,resolution=a.resolution,trainingWidth=a.width,viewerScale=float(scale),sourceSha256=hashlib.sha256(Path(a.video).read_bytes()).hexdigest(),hypothesis='Dense observed pixels, shared-anchor camera alignment; masks exclude people, no generated pixels.',torch=torch.__version__,gpu=torch.cuda.get_device_name(),peakVRAMGB=torch.cuda.max_memory_allocated()/1e9)
    (out/'result.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta),flush=True)

if __name__=='__main__':main()
