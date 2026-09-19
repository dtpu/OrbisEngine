"""Original LHM frozen-human experiment on bounded Modal CUDA workers."""
from pathlib import Path
import modal

HERE = Path(__file__).resolve().parent
LHM_REV = "4f88aaeb3629249fbbddb4d0784a06962d9e1338"
MODEL_REV = "dd6392905187a91fd67b3f6962aa74481e943764"
LARGER_MODEL_REV = "92372582f660066b9f1b9513860744357265b3d5"
image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "build-essential", "ninja-build", "ffmpeg", "libgl1", "libglib2.0-0", "libegl1")
    .env({"TORCH_CUDA_ARCH_LIST":"8.9", "FORCE_CUDA":"1", "MAX_JOBS":"4",
          "PYTHONUNBUFFERED":"1", "HF_HOME":"/cache/huggingface", "TORCH_HOME":"/cache/torch",
          "PYOPENGL_PLATFORM":"egl"})
    .pip_install("torch==2.3.0", "torchvision==0.18.0", "xformers==0.0.26.post1", "numpy==1.23.5", "setuptools==74.0.0", "wheel", "ninja")
    .pip_install("numpy==1.23.5", "scipy==1.14.1", "pillow==10.4.0", "opencv-python-headless==4.11.0.86",
                 "einops", "roma", "accelerate", "smplx", "chumpy", "decord==0.6.0", "diffusers==0.32.0",
                 "gsplat==1.4.0", "huggingface_hub==0.28.1", "safetensors", "imageio==2.34.1", "imageio-ffmpeg",
                 "jaxtyping==0.2.38", "kornia==0.7.2", "loguru", "lpips", "matplotlib==3.8.4", "omegaconf",
                 "plyfile==1.0.3", "pygltflib", "pyrender", "pyyaml", "requests", "timm==1.0.15", "trimesh==4.4.9",
                 "transformers==4.41.2", "tqdm", "typeguard==2.13.3", "iopath", "fvcore", "rembg==2.0.63", "onnxruntime", "addict", "future", "lmdb", "scikit-image==0.22.0", "tb-nightly", "yapf")
    .run_commands(
        "git clone https://github.com/XPixelGroup/BasicSR.git /opt/basicsr && git -C /opt/basicsr checkout 8d56e3a045f9fb3e1d8872f92ee4a4f07f886b0a && pip install --no-deps /opt/basicsr",
        "pip install --no-deps gfpgan==1.3.8 facexlib==0.3.0 filterpy==1.4.5",
        "git clone https://github.com/facebookresearch/pytorch3d.git /opt/pytorch3d && git -C /opt/pytorch3d checkout 89653419d0973396f3eff1a381ba09a07fffc2ed && CC=gcc CXX=g++ CUB_HOME=/usr/local/cuda/include pip install --no-build-isolation --no-deps /opt/pytorch3d",
        "git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization.git /opt/diff-gaussian-rasterization && git -C /opt/diff-gaussian-rasterization checkout 8829d14f814fccdaf840b7b0f3021a616583c0a1 && CC=gcc CXX=g++ pip install --no-build-isolation --no-deps /opt/diff-gaussian-rasterization",
        "git clone https://github.com/camenduru/simple-knn.git /opt/simple-knn && git -C /opt/simple-knn checkout 60f461f4a56b7967e5d8045bf92f8c33f36976d0 && CC=gcc CXX=g++ pip install --no-build-isolation --no-deps /opt/simple-knn",
        f"git clone https://github.com/aigc3d/LHM.git /opt/lhm && git -C /opt/lhm checkout {LHM_REV}",
    )
    .add_local_dir(HERE / "experiments", "/root/experiments", ignore=["__pycache__"])
)
app = modal.App("wander-overnight-lhm")
cache = modal.Volume.from_name("wander-overnight-lhm-cache", create_if_missing=True)


@app.function(image=image, cpu=4, memory=32768, timeout=1800, retries=0,
              volumes={"/cache":cache}, scaledown_window=2)
def stage():
    import os, subprocess, time
    os.chdir('/opt/lhm')
    started=time.time()
    subprocess.run(['python','-c','import torch, pytorch3d, diff_gaussian_rasterization, gsplat; from LHM.models import model_dict; print(torch.__version__, list(model_dict))'],check=True)
    return dict(seconds=time.time()-started, codeRevision=LHM_REV, modelRevision=MODEL_REV)


@app.function(image=image,gpu='L4',cpu=4,memory=65536,timeout=1800,retries=0,
              max_containers=1,volumes={'/cache':cache},scaledown_window=2)
def frozen(inputs:dict,animate:bool=False,larger_model:bool=False):
    import hashlib,io,json,os,subprocess,tarfile,tempfile,time,traceback
    from huggingface_hub import snapshot_download
    start=time.time();root=Path(tempfile.mkdtemp(prefix='lhm-'));prepared=root/'prepared';prepared.mkdir();out=root/'output';out.mkdir()
    for name,data in inputs.items():
        if name not in ('source.png','mask.png','prepared.json','source.mp4','canonical-state.pt','reference.json','cameras.json','depth-reference.ply','seed-poses.pt','seed-motion.json','source-pose.pt','source-pose.json','head-input.png'):raise ValueError('Unexpected prepared input')
        (prepared/name).write_bytes(data)
    prior=Path('/cache/data/pretrained_models');link=Path('/opt/lhm/pretrained_models')
    if not prior.exists():raise RuntimeError('Stage public LHM prior assets before GPU inference')
    if not link.exists():link.symlink_to(prior,target_is_directory=True)
    os.chdir('/opt/lhm');error=None
    try:
        model_id='3DAIGC/LHM-1B-HF' if larger_model else '3DAIGC/LHM-500M-HF'
        model_revision=LARGER_MODEL_REV if larger_model else MODEL_REV
        model=snapshot_download(model_id,revision=model_revision,local_files_only=True)
        if (prepared/'source-pose.pt').exists():
            if animate:raise ValueError('Fixed input comparison is frozen')
            command=['python','/root/experiments/lhm_capacity.py','--prepared',str(prepared),'--out',str(out),'--model',model,'--model-id',model_id,'--model-revision',model_revision]
        elif animate:
            command=['python','/root/experiments/lhm_animate.py','--video',str(prepared/'source.mp4'),
                '--canonical',str(prepared/'canonical-state.pt'),'--reference',str(prepared/'reference.json'),'--out',str(out),'--model',model]
            if (prepared/'cameras.json').exists():command+=['--cameras',str(prepared/'cameras.json'),'--depth-reference',str(prepared/'depth-reference.ply')]
            if (prepared/'seed-poses.pt').exists():command+=['--seed-poses',str(prepared/'seed-poses.pt'),'--seed-motion',str(prepared/'seed-motion.json')]
        else:
            command=['python','/root/experiments/lhm_person.py','--prepared',str(prepared),'--out',str(out),'--model',model]
        with (out/'inference.log').open('w') as log:
            proc=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            for line in proc.stdout:print(line,end='',flush=True);log.write(line);log.flush()
            if proc.wait()!=0:raise RuntimeError(f'LHM inference exited {proc.returncode}')
    except Exception:
        error=traceback.format_exc();print(error,flush=True)
    auxiliary={
        '/cache/torch/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth':'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth',
        '/opt/lhm/gfpgan/weights/detection_Resnet50_Final.pth':'https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth',
        '/opt/lhm/gfpgan/weights/parsing_parsenet.pth':'https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth',
        '/usr/local/lib/python3.10/site-packages/gfpgan/weights/GFPGANv1.3.pth':'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
    }
    provenance=[]
    for path,url in auxiliary.items():
        file=Path(path)
        if file.exists():
            h=hashlib.sha256()
            with file.open('rb') as f:
                for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
            provenance.append(dict(url=url,bytes=file.stat().st_size,sha256=h.hexdigest()))
    (out/'auxiliary-weight-provenance.json').write_text(json.dumps(provenance,indent=2))
    cache.commit();buf=io.BytesIO()
    with tarfile.open(fileobj=buf,mode='w:gz') as archive:
        for file in sorted(out.iterdir()):
            if file.is_file():archive.add(file,arcname=file.name)
        elapsed=time.time()-start
        report=dict(seconds=elapsed,error=error,experiment='canonical-source-motion' if animate else 'frozen-source-pose',gpu='L4',cpu=4,memoryGiB=64,timeoutSeconds=1800,
                    estimatedComputeUSD=elapsed*(.000222+4*.0000131+64*.00000222),codeRevision=LHM_REV,model=model_id,modelRevision=model_revision)
        (out/'modal-run.json').write_text(json.dumps(report,indent=2));archive.add(out/'modal-run.json',arcname='modal-run.json')
    return dict(report=report,archive=buf.getvalue())


@app.local_entrypoint()
def main(prepared:str='',out:str='',video:str='',canonical:str='',cameras:str='',seed:str='',fixed_inputs:str='',larger_model:bool=False):
    if not prepared and not canonical:
        print(stage.remote());return
    import io,json,tarfile
    dest=Path(out);dest.mkdir(parents=True,exist_ok=False)
    if canonical:
        if not video:raise ValueError('Canonical animation requires its source video')
        folder=Path(canonical)
        inputs={'source.mp4':Path(video).read_bytes(),'canonical-state.pt':(folder/'canonical-state.pt').read_bytes(),'reference.json':(folder/'result.json').read_bytes()}
        if cameras:
            inputs['cameras.json']=Path(cameras).read_bytes();inputs['depth-reference.ply']=(Path(cameras).parent/'frame_000.ply').read_bytes()
        if seed:
            inputs['seed-poses.pt']=(Path(seed)/'source-poses.pt').read_bytes();inputs['seed-motion.json']=(Path(seed)/'motion.json').read_bytes()
    else:
        inputs={name:(Path(prepared)/name).read_bytes() for name in ('source.png','mask.png','prepared.json')}
        if fixed_inputs:
            inputs.update({name:(Path(fixed_inputs)/name).read_bytes() for name in ('source-pose.pt','source-pose.json','head-input.png')})
    if larger_model and not fixed_inputs:raise ValueError('Larger checkpoint comparison requires saved fixed inputs')
    result=frozen.remote(inputs,animate=bool(canonical),larger_model=larger_model);(dest/'artifacts.tar.gz').write_bytes(result['archive'])
    with tarfile.open(fileobj=io.BytesIO(result['archive']),mode='r:gz') as archive:archive.extractall(dest,filter='data')
    print(json.dumps(result['report'],indent=2))
    if result['report']['error']:raise RuntimeError('LHM failed; see downloaded inference.log')
