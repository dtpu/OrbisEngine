#!/usr/bin/env python3
"""Frozen original-LHM Gaussian human, posed from the supplied source image.

Uses public pretrained LHM-500M-HF and Multi-HMR. Canonical inference is followed
by native animation_infer_gs with measured source-pose parameters, never a stock
animation. Inferred hidden appearance is not recovered source identity.
"""
from __future__ import annotations
import argparse, gc, hashlib, importlib.util, json, os, sys, time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image


def pose_parameters(pose, betas=None):
    import torch
    rv=pose['rotvec']
    params=dict(betas=pose['shape'][None] if betas is None else betas,
        root_pose=rv[0][None,None],body_pose=rv[1:22][None,None],jaw_pose=rv[52][None,None],
        leye_pose=torch.zeros(1,1,3),reye_pose=torch.zeros(1,1,3),
        lhand_pose=rv[22:37][None,None],rhand_pose=rv[37:52][None,None],
        expr=torch.zeros(1,1,100),trans=pose['transl_pelvis'].reshape(1,1,3))
    return {k:v.float().cuda() for k,v in params.items()}


def preprocess(repo, source, mask):
    """The original LHM mask crop, white background, and 5:3 model framing."""
    import torch
    from engine.SegmentAPI.base import Bbox
    spec=importlib.util.spec_from_file_location('lhm_inference_helpers',repo/'LHM/runners/infer/utils.py')
    helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
    rgb=np.asarray(Image.open(source).convert('RGB'));h,w=mask.shape
    y,x=np.where(mask>127)
    if not len(x):raise ValueError('Empty input person mask')
    box=Bbox([int(x.min()),int(y.min()),int(x.max()),int(y.max())]).scale(1.1,width=w,height=h).get_box()
    rgb=rgb[box[1]:box[3],box[0]:box[2]];mask=mask[box[1]:box[3],box[0]:box[2]]
    h,w=rgb.shape[:2];aspect=5/3
    if w>=h:raise ValueError('Source mask is not a portrait person crop')
    target_w=int(min(w*(h/w/aspect),h))
    if target_w>w:
        pad=(target_w-w)//2;rgb=np.pad(rgb,((0,0),(pad,pad),(0,0)),constant_values=255);mask=np.pad(mask,((0,0),(pad,pad)))
    else:
        pad=int(w*aspect-h);rgb=np.pad(rgb,((pad,0),(0,0),(0,0)),constant_values=255);mask=np.pad(mask,((pad,0),(0,0)))
    mask=(mask/255>.5).astype(np.float32);rgb=rgb/255*mask[...,None]+(1-mask[...,None])
    rgb=helpers.resize_image_keepaspect_np(rgb,896);mask=helpers.resize_image_keepaspect_np(mask,896)
    rgb,mask,_,_=helpers.center_crop_according_to_mask(rgb,mask,aspect,[1.,1.])
    size,_,_=helpers.calc_new_tgt_size_by_aspect(rgb.shape[:2],aspect,1024,14)
    rgb=cv2.resize(rgb,(size[1],size[0]),interpolation=cv2.INTER_AREA)
    return torch.from_numpy(rgb).float().permute(2,0,1)[None]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prepared',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--repo',default='/opt/lhm');ap.add_argument('--model',required=True)
    a=ap.parse_args();repo=Path(a.repo);prepared=Path(a.prepared);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    os.chdir(repo);sys.path.insert(0,str(repo));t0=time.time()
    import torch
    from accelerate import Accelerator
    torch._dynamo.config.disable=True;torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    Accelerator()
    from engine.pose_estimation.pose_estimator import PoseEstimator
    source=prepared/'source.png';mask=np.asarray(Image.open(prepared/'mask.png').convert('L'))
    raw=np.asarray(Image.open(source).convert('RGB'))
    pose_rgb=raw.copy();pose_rgb[mask<128]=255
    Image.fromarray(pose_rgb).save(out/'pose-input.png')
    print('Estimating source pose with official Multi-HMR',flush=True)
    estimator=PoseEstimator('./pretrained_models/human_model_files',device='cuda')
    padded,offset_w,offset_h=estimator.img_center_padding(pose_rgb)
    tensor,annotation=estimator._preprocess(padded);K=estimator.get_camera_parameters()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
        people=estimator.mhmr_model(tensor,is_training=False,nms_kernel_size=3,det_thresh=.3,K=K,idx=None,max_dist=None)
    if not people:raise RuntimeError('No source person pose detected; refusing canonical-pose substitute')
    person=max(people,key=lambda p:float(p['scores']))
    pose={k:v.detach().float().cpu() for k,v in person.items()}
    pad_left,pad_top,factor,_,_=annotation
    source_K=K[0].float().cpu().numpy().copy();source_K[:2]/=factor
    source_K[0,2]-=pad_left/factor+offset_w;source_K[1,2]-=pad_top/factor+offset_h
    joints=(pose['j2d'].numpy()-[pad_left,pad_top])/factor-[offset_w,offset_h]
    overlay=raw.copy()
    for x,y in joints:cv2.circle(overlay,(round(float(x)),round(float(y))),5,(255,0,0),-1)
    Image.fromarray(overlay).save(out/'estimated-joints.png')
    torch.save(pose,out/'source-pose.pt')
    pose_record={k:v.numpy().tolist() for k,v in pose.items() if k not in ('v3d','j3d','j2d')}
    pose_record.update(detectedPeople=len(people),annotation=list(annotation),padOffsets=[offset_w,offset_h],K=K[0].cpu().tolist(),sourceIntrinsics=source_K.tolist(),method='Official Multi-HMR on masked source; highest detection score selected')
    (out/'source-pose.json').write_text(json.dumps(pose_record,indent=2))
    del estimator,tensor,K,people,person;gc.collect();torch.cuda.empty_cache()
    from LHM.utils.face_detector import FaceDetector
    detector=FaceDetector('./pretrained_models/gagatracker/vgghead/vgg_heads_l.trcd',device='cuda')
    try:
        box=detector(torch.from_numpy(raw.copy()).permute(2,0,1)).cpu().numpy()
        box[[0,2]]=box[[0,2]].clip(0,raw.shape[1]);box[[1,3]]=box[[1,3]].clip(0,raw.shape[0])
        head=raw[box[1]:box[3],box[0]:box[2]]
        head=cv2.resize(head,(112,112),interpolation=cv2.INTER_AREA);head_detected=True
    except Exception as error:
        print('No head input; upstream zero-head fallback:',str(error),flush=True)
        head=np.zeros((112,112,3),dtype=np.uint8);head_detected=False
    Image.fromarray(head).save(out/'head-input.png')
    del detector;gc.collect();torch.cuda.empty_cache()
    image=preprocess(repo,source,mask);Image.fromarray((image[0].permute(1,2,0).numpy()*255).astype('uint8')).save(out/'model-input.png')
    head_image=torch.from_numpy(head/255.).float().permute(2,0,1)[None,None]
    from LHM.models import model_dict
    from LHM.utils.hf_hub import wrap_model_hub
    print('Loading original LHM-500M-HF',flush=True)
    model=wrap_model_hub(model_dict['human_lrm_sapdino_bh_sd3_5']).from_pretrained(a.model)
    # Upstream overrides train() without returning self, so eval() is not chainable.
    model.eval();model.cuda()
    params=pose_parameters(pose)
    print('Reconstructing appearance and animating to source pose',flush=True)
    with torch.inference_mode():
        attrs,query,neutral=model.infer_single_view(image[None].cuda(),head_image.cuda(),None,None,None,None,None,smplx_params=params)
        params['transform_mat_neutral_pose']=neutral
        gaussians=model.animation_infer_gs(attrs,query,params)
        gaussians.save_ply(str(out/'person-posed.ply'))
        torch.save(dict(attrs=attrs,query=query,neutral=neutral,params=params),out/'canonical-state.pt')
    xyz=gaussians.xyz.detach().float().cpu().numpy()
    meta=dict(model='3DAIGC/LHM-500M-HF',modelRevision='dd6392905187a91fd67b3f6962aa74481e943764',codeRevision='4f88aaeb3629249fbbddb4d0784a06962d9e1338',
        seconds=time.time()-t0,peakVRAMGB=torch.cuda.max_memory_allocated()/1e9,gpu=torch.cuda.get_device_name(),torch=torch.__version__,
        points=len(xyz),bounds=np.percentile(xyz,[2,50,98],axis=0).tolist(),headDetected=head_detected,sourceImageSha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        prepared=json.loads((prepared/'prepared.json').read_text()),sourceIntrinsics=source_K.tolist(),nativeCoordinates='Multi-HMR source camera: x right, y down, z forward. Posed with estimated source joint rotations and translation.',
        coverage='All Gaussian geometry and appearance inferred by LHM from one observed image. Hidden face/front identity is not recorded evidence.',poseMethod=pose_record['method'])
    (out/'result.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta),flush=True)


if __name__=='__main__':main()
