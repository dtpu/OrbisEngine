#!/usr/bin/env python3
"""One canonical LHM appearance animated by independently estimated source poses."""
import argparse,gc,hashlib,json,os,sys,time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image


def main():
    ap=argparse.ArgumentParser()
    for key in ('video','canonical','reference','out','model'):ap.add_argument('--'+key,required=True)
    ap.add_argument('--repo',default='/opt/lhm');ap.add_argument('--fps',type=float,default=12)
    ap.add_argument('--cameras');ap.add_argument('--depth-reference')
    ap.add_argument('--seed-poses');ap.add_argument('--seed-motion')
    ap.add_argument('--track-only',action='store_true',help='Seed poses are one tracked person: animate exactly those samples, never re-detect (a re-detection would pick whichever person is nearest and swap identity)')
    ap.add_argument('--track-id',type=int,help='Track index recorded in sequence.json')
    ap.add_argument('--depth-roi',help='x0,y0,x1,y1 source-pixel box; the depth reference holds every masked person in the frame, so restrict the observed median to this track')
    ap.add_argument('--fixed-world-scale',type=float,help='Separate camera-model control: retain an explicitly supplied native/world scale without depth or floor refitting')
    ap.add_argument('--detection-threshold',type=float,default=.3)
    ap.add_argument('--overlay-all',action='store_true')
    ap.add_argument('--start',type=float,default=0,help='Bundled-video interval start; source bytes are not trimmed')
    ap.add_argument('--stop',type=float,help='Exclusive bundled-video interval end')
    ap.add_argument('--original-source-offset',type=float,default=0,help='Recorded original-recording time minus bundled-video time')
    a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    os.chdir(a.repo);sys.path.insert(0,a.repo);start=time.time()
    import torch
    from accelerate import Accelerator
    from pytorch3d.transforms import matrix_to_quaternion,quaternion_multiply
    from engine.pose_estimation.pose_estimator import PoseEstimator
    from lhm_person import pose_parameters
    torch._dynamo.config.disable=True;torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True;Accelerator()
    source_sha=hashlib.sha256(Path(a.video).read_bytes()).hexdigest()
    if not .1<=a.detection_threshold<=.3:raise ValueError('Detection threshold outside bounded experiment range')
    if a.fixed_world_scale is not None and (not a.cameras or not 0<a.fixed_world_scale<10):raise ValueError('Fixed scale requires supplied cameras and a finite positive scale')
    reference=json.loads(Path(a.reference).read_text())
    if source_sha!=reference['prepared']['sourceSha256']:raise ValueError('Canonical appearance and motion must come from the same source video')
    cap=cv2.VideoCapture(a.video);fps=cap.get(cv2.CAP_PROP_FPS);count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));duration=count/fps
    if a.fps<12 or fps<a.fps:raise ValueError('Require at least 12 distinct source frames per second')
    # Match the dense camera export's divide-then-multiply order at half-frame ties.
    stop=duration if a.stop is None else a.stop
    if not np.isfinite([a.start,stop,a.original_source_offset]).all() or not 0<=a.start<stop<=duration:raise ValueError('Invalid source interval')
    requested=a.start+np.arange(int(np.ceil((stop-a.start)*a.fps)))/a.fps
    requested=requested[requested<stop]
    indices=np.rint(requested*fps).astype(int)
    indices=np.clip(indices,0,count-1);times=indices/fps
    if np.any(times<a.start) or np.any(times>=stop):raise ValueError('Rounded samples leave the requested interval')
    if len(set(indices))!=len(indices):raise ValueError('Repeated source frames')
    camera_doc=json.loads(Path(a.cameras).read_text()) if a.cameras else None
    cameras=camera_doc['cameras'] if camera_doc else None
    if cameras and (len(cameras)!=len(indices) or any(c['sourceIndex']!=int(i) for c,i in zip(cameras,indices))):
        raise ValueError('Camera records must correspond exactly to sampled source indices')
    seed=torch.load(a.seed_poses,map_location='cpu',weights_only=False) if a.seed_poses else None
    seed_records={r['sample']:r for r in json.loads(Path(a.seed_motion).read_text())['frames']} if seed else {}
    if seed and not np.array_equal(seed['sourceIndices'],indices):raise ValueError('Seed pose source samples differ')
    if seed and not a.track_only and all(p is not None for p in seed['poses']):raise ValueError('Seed has no missing source poses to recover')
    if a.track_only and not seed:raise ValueError('Track mode requires the track\'s seed poses')
    estimator=None if a.track_only else PoseEstimator('./pretrained_models/human_model_files',device='cuda')
    poses=[];records=[];missing=[];last_center=None
    for sample,(index,timestamp) in enumerate(zip(indices,times)):
        if seed and seed['poses'][sample] is not None:
            pose=seed['poses'][sample];record=dict(seed_records[sample]);record['retainedOriginalPose']=True
            poses.append(pose);records.append(record)
            K0=np.array(record['source_intrinsics']);projected=pose['j3d'][:22].numpy()@K0.T
            # Source dimensions are fixed; project the retained person to track only missing samples.
            wh=np.array([reference['prepared']['sourceWidth'],reference['prepared']['sourceHeight']])
            last_center=np.median(projected[:,:2]/projected[:,2:3],axis=0)/wh
            continue
        if a.track_only:
            missing.append(dict(sample=sample,sourceIndex=int(index),time=float(timestamp),reason='Track absent from this sample (out of frame, undetected, or occluded)'))
            poses.append(None);records.append(None);continue
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(index));ok,bgr=cap.read()
        if not ok:raise RuntimeError(f'Unable to decode source frame {index}')
        raw=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);h,w=raw.shape[:2]
        padded,ow,oh=estimator.img_center_padding(raw);tensor,annotation=estimator._preprocess(padded)
        pl,pt,factor,_,_=annotation
        if cameras:
            source_K=np.array(cameras[sample]['source_intrinsics'])
            K=torch.tensor(np.array([[factor,0,factor*ow+pl],[0,factor,factor*oh+pt],[0,0,1]])@source_K,dtype=torch.float32,device='cuda')[None]
        else:
            K=estimator.get_camera_parameters();source_K=K[0].cpu().numpy().copy();source_K[:2]/=factor
            source_K[0,2]-=pl/factor+ow;source_K[1,2]-=pt/factor+oh
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
            detected=estimator.mhmr_model(tensor,is_training=False,nms_kernel_size=3,det_thresh=.1 if seed else a.detection_threshold,K=K,idx=None,max_dist=None)
        choices=[]
        for person in detected:
            joints=(person['j2d'].float().cpu().numpy()-[pl,pt])/factor-[ow,oh]
            center=np.median(joints[:22],axis=0)/[w,h]
            score=float(person['scores'])
            distance=float(np.linalg.norm(center-last_center)) if last_center is not None else 0
            choices.append((score-distance*2,person,joints,center,distance))
        choices.sort(key=lambda x:x[0],reverse=True)
        if not choices or (last_center is not None and choices[0][4]>.35):
            missing.append(dict(sample=sample,sourceIndex=int(index),time=float(timestamp),reason='No consistent source person detected'))
            print('missing pose',missing[-1],flush=True);poses.append(None);records.append(None);continue
        _,person,joints,last_center,_=choices[0]
        pose={k:v.detach().float().cpu() for k,v in person.items() if k in ('rotvec','shape','transl_pelvis','j3d','scores')}
        poses.append(pose)
        records.append(dict(sample=sample,sourceIndex=int(index),time=float(timestamp),detectedPeople=len(detected),score=float(pose['scores']),
            originalSourceTime=float(timestamp+a.original_source_offset),sourceFrameRGBSha256=hashlib.sha256(raw.tobytes()).hexdigest(),
            projectedBodyJoints=joints[:22].tolist(),jointProjectionInImage=((joints[:22,0]>=0)&(joints[:22,0]<w)&(joints[:22,1]>=0)&(joints[:22,1]<h)).tolist(),
            confidenceSemantics='Score is person detection confidence, not per-joint accuracy or visibility confidence.',
            source_intrinsics=source_K.tolist(),rootCamera=pose['j3d'][0].tolist(),smplTranslationCamera=pose['transl_pelvis'].reshape(3).tolist(),
            feetCamera=pose['j3d'][[7,8,10,11]].tolist(),footJointOrder=['leftAnkle','rightAnkle','leftFoot','rightFoot'],
            rootRotationVector=pose['rotvec'][0].tolist(),detectionThreshold=.1 if seed else a.detection_threshold,recoveredMissingPose=bool(seed)))
        if seed or a.overlay_all or sample in (0,len(indices)//2,len(indices)-1):
            overlay=raw.copy()
            for x,y in joints:cv2.circle(overlay,(round(float(x)),round(float(y))),4,(255,0,0),-1)
            Image.fromarray(overlay).save(out/f'pose-overlay-{sample:03d}.jpg')
        print('source pose',sample+1,'/',len(indices),'score',float(pose['scores']),flush=True)
    cap.release()
    if not a.track_only:del tensor,detected,K,choices
    del estimator;gc.collect();torch.cuda.empty_cache()
    torch.save(dict(poses=poses,sourceIndices=indices,timestamps=times),out/'source-poses.pt')
    (out/'missing-poses.json').write_text(json.dumps(missing,indent=2))
    first=next((i for i,p in enumerate(poses) if p is not None),None)
    if first is None:raise RuntimeError('Track has no usable source pose')
    if first!=0 and not a.track_only:raise RuntimeError('Source pose zero missing; cannot fit a fixed depth scale')
    from LHM.models import model_dict
    from LHM.utils.hf_hub import wrap_model_hub
    print('Loading canonical appearance and native LHM skinning',flush=True)
    model=wrap_model_hub(model_dict['human_lrm_sapdino_bh_sd3_5']).from_pretrained(a.model);model.eval();model.cuda()
    state=torch.load(a.canonical,map_location='cuda',weights_only=False)
    frames=[];hashes=[];export_times=[];used_indices=[];world_scale=a.fixed_world_scale;registration=None
    if world_scale is not None:
        registration=dict(uniformScale=world_scale,method='Explicit native/world scale retained from the control. No first-frame depth normalization, floor fitting or per-frame grounding adjustment.')
        (out/'registration.json').write_text(json.dumps(registration,indent=2))
    for sample,pose in enumerate(poses):
        if pose is None:continue
        params=pose_parameters(pose,betas=state['params']['betas']);params['transform_mat_neutral_pose']=state['neutral']
        with torch.inference_mode():gs=model.animation_infer_gs(state['attrs'],state['query'],params)
        if not all(torch.isfinite(x).all() for x in (gs.xyz,gs.opacity,gs.scaling,gs.rotation)):raise RuntimeError('Non-finite animated Gaussians')
        record=records[sample]
        if cameras:
            camera=np.array(cameras[sample]['camera_to_world']);rotation=camera[:3,:3]
            u,_,vt=np.linalg.svd(rotation);rotation=u@vt
            if np.linalg.det(rotation)<0:raise RuntimeError('Reflected source camera')
            cv_to_world=rotation@np.diag([1.,-1.,-1.])
            if world_scale is None:
                from plyfile import PlyData
                p=PlyData.read(a.depth_reference)['vertex'].data
                xyz=np.column_stack([p['x'],p['y'],p['z']]);local=(xyz-camera[:3,3])@np.linalg.inv(camera[:3,:3]).T
                roi_kept=None
                if a.depth_roi:
                    x0,y0,x1,y1=[float(v) for v in a.depth_roi.split(',')]
                    K0=np.array(record['source_intrinsics']);cam_cv=local*np.array([1.,-1.,-1.])
                    uvw=cam_cv@K0.T;front=uvw[:,2]>1e-6;uv=np.full((len(uvw),2),-1e9)
                    uv[front]=uvw[front,:2]/uvw[front,2:3]
                    inside=front&(uv[:,0]>=x0)&(uv[:,0]<=x1)&(uv[:,1]>=y0)&(uv[:,1]<=y1)
                    if inside.sum()<50:raise RuntimeError(f'Only {int(inside.sum())} depth-reference points inside the track ROI')
                    local=local[inside];roi_kept=int(inside.sum())
                observed_depth=-local[:,2];observed_depth=observed_depth[np.isfinite(observed_depth)&(observed_depth>0)]
                opaque=gs.opacity[:,0]>.5;native_depth=gs.xyz[opaque,2];native_depth=native_depth[torch.isfinite(native_depth)&(native_depth>0)]
                observed_median=float(np.median(observed_depth));native_median=float(native_depth.median());world_scale=observed_median/native_median
                if not np.isfinite(world_scale) or world_scale<=0:raise RuntimeError('Invalid first-frame depth registration')
                registration=dict(uniformScale=world_scale,observedPersonMedianDepth=observed_median,nativeGaussianMedianDepth=native_median,
                    registrationSample=sample,registrationSourceIndex=int(indices[sample]),depthRoi=a.depth_roi,depthRoiPoints=roi_kept,
                    method='One positive scale from first-frame person camera-space median depths, held fixed throughout. Pi3X observed person points versus opaque LHM body Gaussians; body-depth prior, not pixel-correspondence or floor fitting.')
                (out/'registration.json').write_text(json.dumps(registration,indent=2));print('registration',registration,flush=True)
            rotation_t=torch.tensor(cv_to_world,dtype=torch.float32,device='cuda');translation_t=torch.tensor(camera[:3,3],dtype=torch.float32,device='cuda')
            gs.xyz=(gs.xyz@rotation_t.T)*world_scale+translation_t
            gs.scaling=gs.scaling*world_scale
            gs.rotation=quaternion_multiply(matrix_to_quaternion(rotation_t)[None],gs.rotation)
            record['rootWorld']=(cv_to_world@np.array(record['rootCamera'])*world_scale+camera[:3,3]).tolist()
            record['feetWorld']=(np.array(record['feetCamera'])@cv_to_world.T*world_scale+camera[:3,3]).tolist()
            record['camera_to_world']=camera.tolist()
        name=f'frame_{sample:03d}.ply';gs.save_ply(str(out/name));frames.append(name)
        hashes.append(hashlib.sha256((out/name).read_bytes()).hexdigest());export_times.append(float(times[sample]));used_indices.append(int(indices[sample]))
        print('animated Gaussian frame',sample+1,'/',len(poses),flush=True)
    valid_records=[r for r in records if r is not None]
    for previous,current in zip(valid_records,valid_records[1:]):
        dt=current['time']-previous['time']
        for key in ('rootWorld','feetWorld'):
            if key in current:current[key+'Velocity']=((np.array(current[key])-previous[key])/dt).tolist()
    motion=dict(coordinates='Raw roots/feet: OpenCV source camera. World roots/feet: exported Gaussian OpenGL world when camera metadata supplied.',
        footContact=dict(status='unmeasured',reason='No measured floor plane or contact constraint. Feet are source-pose joint estimates; no snapping or synthetic stance.',
            smplxJointIndices=[7,8,10,11],semantics=['left ankle joint','right ankle joint','left foot joint','right foot joint'],soleOrHeelVertices=False),
        canonicalShapeFixed=True,frames=valid_records,missing=missing)
    (out/'motion.json').write_text(json.dumps(motion,indent=2))
    sequence=dict(frame_format='gaussian-ply',frames=frames,count=len(frames),fps=a.fps,timestamps=export_times,frame_sha256=hashes,
        sourceIndices=used_indices,sourceFps=fps,sourceSha256=source_sha,duration=duration,requestedSamples=len(indices),missingPoseSamples=missing,
        allRequestedSamplesReconstructed=not missing,canonicalStateSha256=hashlib.sha256(Path(a.canonical).read_bytes()).hexdigest(),
        poseRecovery=dict(originalPosesRetained=len(seed_records),lowerDetectionThreshold=.1,recoveredSamples=[r['sample'] for r in valid_records if r.get('recoveredMissingPose')]) if (seed and not a.track_only) else None,
        trackId=a.track_id,trackOnly=a.track_only,firstReconstructedSample=first,
        referenceTime=reference['prepared']['time'],canonicalShapeFixed=True,people_only=True,
        sourceInterval=dict(start=a.start,stopExclusive=stop,originalSourceOffset=a.original_source_offset,originalStart=a.start+a.original_source_offset,originalStopExclusive=stop+a.original_source_offset),
        coordinates=('OpenGL supplied fixed-SfM world; explicitly retained control scale' if a.fixed_world_scale is not None else 'OpenGL Pi3X world; one fixed first-frame body-depth scale') if cameras else 'OpenCV source camera for each timestamp; world motion unregistered',
        fixedWorldScaleControl=a.fixed_world_scale,sourcePosesReestimatedWithSuppliedIntrinsics=bool(a.fixed_world_scale is not None),
        cameraMetadataSha256=hashlib.sha256(Path(a.cameras).read_bytes()).hexdigest() if cameras else None,
        seconds=time.time()-start,peakVRAMGB=torch.cuda.max_memory_allocated()/1e9,gpu=torch.cuda.get_device_name(),torch=torch.__version__,
        note='One source-conditioned canonical appearance. Independent MultiHMR pose per decoded source sample; no stock motion, pose interpolation, or appearance regeneration. Hidden surfaces remain learned inference.')
    (out/'sequence.json').write_text(json.dumps(sequence,indent=2));print(json.dumps({k:v for k,v in sequence.items() if k not in ('frames','timestamps','frame_sha256','sourceIndices')}),flush=True)


if __name__=='__main__':main()
