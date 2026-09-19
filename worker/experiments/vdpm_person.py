#!/usr/bin/env python3
"""V-DPM temporal observation fusion, initially one frozen human moment.

Uses the upstream encoder/decoder and pretrained checkpoint unchanged, but decodes
one target at a time to avoid retaining S squared point maps on the GPU.
Source RGB/masks move with each predicted observation. No synthetic front texture.
"""
from __future__ import annotations
import argparse, gc, hashlib, json, os, sys, time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from wander_worker.masks import people_masks
from wander_worker.ply import write_point_ply, write_gaussian_ply


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--video',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--views',type=int,default=16);ap.add_argument('--targets',default='0',help='comma-separated source-view indexes, or all')
    ap.add_argument('--start',type=float,default=0);ap.add_argument('--duration',type=float,default=0)
    ap.add_argument('--repo',default=os.path.expanduser('~/vdpm'))
    a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);t0=time.time()
    sys.path.insert(0,a.repo)
    import torch
    from omegaconf import OmegaConf
    from dpm.model import VDPM
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=True
    cap=cv2.VideoCapture(a.video);fps=cap.get(cv2.CAP_PROP_FPS);total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    first=round(a.start*fps);last=min(total-1,round((a.start+a.duration)*fps)-1) if a.duration else total-1
    indices=np.unique(np.linspace(first,last,a.views).round().astype(int));rgb=[]
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(idx));ok,bgr=cap.read()
        if not ok:raise RuntimeError(f'Cannot decode frame {idx}')
        im=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));w=518;h=round(im.height*w/im.width/14)*14
        im=im.resize((w,h),Image.Resampling.BICUBIC)
        if h>518:im=im.crop((0,(h-518)//2,w,(h-518)//2+518))
        rgb.append(np.asarray(im))
    cap.release();rgb=np.stack(rgb);N,H,W=rgb.shape[:3];print('decoded',N,'views',rgb.shape,flush=True)
    people=people_masks(rgb,dilate_px=0)
    # Save the segmentation used to define observation membership for review.
    for j in range(N):Image.fromarray((people[j]*255).astype('uint8')).save(out/f'mask_{j:03d}.png')
    model=VDPM(OmegaConf.create({'model':{'decoder_depth':4}})).eval()
    checkpoint=Path(torch.hub.get_dir())/'checkpoints/vdpm_model.pt'
    sd=torch.load(checkpoint,map_location='cpu',weights_only=True)
    print(model.load_state_dict(sd,strict=True),flush=True);del sd
    model=model.to('cuda');images=torch.from_numpy(rgb.copy()).permute(0,3,1,2)[None].float().cuda()/255
    targets=list(range(N)) if a.targets=='all' else [int(x) for x in a.targets.split(',')]
    frames=[];metrics=[]
    with torch.inference_mode():
        with torch.autocast('cuda',dtype=torch.bfloat16):tokens,patch=model.aggregator(images)
        # Matches upstream: DPT and camera heads run outside autocast.
        camera=model.camera_head(tokens)[-1]
        E,K=pose_encoding_to_extri_intri(camera,(H,W));E=E[0].float().cpu().numpy();K=K[0].float().cpu().numpy()
        R0,t00=E[0,:,:3],E[0,:,3]
        static,static_conf=model.point_head(tokens,images,patch)
        static=static[0].float().cpu().numpy();static_conf=static_conf[0].float().cpu().numpy()
        depths=(static[0]@R0.T+t00)[...,2];valid=(~people[0]) & (depths>0)
        scale=3/np.median(depths[valid]);flip=np.array([1,-1,-1]);cam=[]
        for i in range(N):
            C=-E[i,:,:3].T@E[i,:,3];cam.append(dict(position=((R0@C+t00)*scale*flip).tolist(),time=float(indices[i]/fps),sourceIndex=int(indices[i])))
        (out/'cameras.json').write_text(json.dumps(dict(cameras=cam),indent=1))
        np.savez_compressed(out/'static.npz',points=static,conf=static_conf,extrinsics=E,intrinsics=K,source_indices=indices,source_fps=fps,viewer_R0=R0,viewer_t0=t00,viewer_scale=scale)
        del static,static_conf
        for target in targets:
            print('decode target',target,'time',indices[target]/fps,flush=True)
            cond=torch.full((1,N),target,dtype=torch.long,device='cuda')
            with torch.autocast('cuda',dtype=torch.bfloat16):decoded=model.decoder(images,tokens,patch,cond)
            padded=[None]*len(tokens)
            for i,layer in enumerate(model.point_head.intermediate_layer_idx):padded[layer]=decoded[i]
            pts,conf=model.point_head(padded,images,patch)
            pts=pts[0].float().cpu().numpy();conf=conf[0].float().cpu().numpy()
            good=people & np.isfinite(pts).all(-1) & (conf>1.5)
            # Reject low-confidence limb streaks using each observation's own person confidence.
            for j in range(N):
                vals=conf[j][people[j]]
                if len(vals):good[j] &= conf[j]>=np.percentile(vals,25)
            raw=(pts@R0.T+t00)*scale*flip
            single=raw[target][good[target]];single_rgb=rgb[target][good[target]]
            xyz=raw[good];colors=rgb[good]
            if len(xyz)<100 or len(single)<100:raise RuntimeError('Too few valid person observations')
            # Statistical outliers only; never mirror or extrude a surface into a body.
            spacing=cKDTree(xyz).query(xyz,k=8)[0][:,-1];keep=spacing<=np.percentile(spacing,97)
            xyz,colors=xyz[keep],colors[keep]
            height=np.percentile(single[:,1],95)-np.percentile(single[:,1],5);voxel=max(height/200,1e-4)
            _,idx=np.unique(np.floor(xyz/voxel).astype('int32'),axis=0,return_index=True);xyz,colors=xyz[idx],colors[idx]
            name=f'frame_{len(frames):03d}.ply';write_point_ply(out/name,xyz.astype('float32'),colors);frames.append(name)
            write_point_ply(out/f'single_{target:03d}.ply',single.astype('float32'),single_rgb)
            write_gaussian_ply(out/f'fused_{target:03d}.ply',xyz.astype('float32'),colors,scale_mult=.6)
            metric=dict(target=target,sourceTime=float(indices[target]/fps),observations=N,points=len(xyz),singlePoints=len(single),fusedBounds=np.percentile(xyz,[2,98],axis=0).tolist(),singleBounds=np.percentile(single,[2,98],axis=0).tolist())
            metrics.append(metric);print(json.dumps(metric),flush=True)
            del decoded,padded,pts,conf,raw;gc.collect();torch.cuda.empty_cache()
    timestamps=[float(indices[i]/fps) for i in targets]
    meta=dict(backend='vdpm-temporal-fusion',count=len(frames),frames=frames,fps=(len(targets)-1)/(timestamps[-1]-timestamps[0]) if len(targets)>1 else 1,timestamps=timestamps,people_only=True,
              note='Multiple observed surfaces warped to each target by V-DPM. No unseen body completion. Sample-and-hold; not interpolated.',sourceSha256=hashlib.sha256(Path(a.video).read_bytes()).hexdigest(),sourceIndices=indices.tolist(),sourceFps=fps,metrics=metrics,seconds=time.time()-t0,peakVRAMGB=torch.cuda.max_memory_allocated()/1e9,gpu=torch.cuda.get_device_name())
    (out/'sequence.json').write_text(json.dumps(meta,indent=2));print(json.dumps({k:v for k,v in meta.items() if k not in ('metrics','frames')}),flush=True)

if __name__=='__main__':main()
