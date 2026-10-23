"""Kommt beim Ruhewinkel ein Gradient an -- und zeigt er bergab?"""
import json,math,os,sys
import numpy as np
_ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,os.path.join(_ROOT,"engines"))
from dlolab_path import DLO_LAB
sys.path.insert(0,str(DLO_LAB)); sys.path.insert(0,_ROOT)
import multiprocessing as mp
CLIPS=os.environ.get("LS_CLIP","20260602_170108,20260602_114019").split(",")
BPTT=int(os.environ.get("LS_BPTT","120"))
N_ITERS=int(os.environ.get("LS_ITERS","35"))
LR=float(os.environ.get("LS_LR","0.04")); LR_MIN=float(os.environ.get("LS_LRMIN","0.005"))
OUT=os.environ.get("LS_OUT",os.path.join(_ROOT,"data/genesis_rod_kappa_fit.json"))
STEPS=[0.02,0.05,0.10,0.20]

def worker(job, conn):
    CLIP, cable = job
    os.environ["CUDA_VISIBLE_DEVICES"]=str(abs(hash(job))%2)
    import torch, genesis as gs
    import genesis_rod_scene as R
    from genesis_rod_autodiff import LOSS_DOWN, gt_dt, dt_loss
    from mujoco_scene import track_clip, DIR
    from wire_model.perception.rectify import find_board
    from fit_clip import read
    import cv2
    from clip_set import CLIP_TRIMS
    cap=cv2.VideoCapture(f"{DIR}/{CLIP}.mp4"); fp=(cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 0) or 4.0
    t1s=CLIP_TRIMS.get(CLIP)
    shp,t0,_,fps=track_clip(CLIP,t0=8,t1=None if t1s is None else int(round(t1s*fp)))
    tf=find_board(read(cap,0)); cap.release()
    nf=len(shp[0]); spf=max(1,round((1.0/fps)/1e-3))
    H,W=2128//LOSS_DOWN,1600//LOSS_DOWN
    kf=sorted(set(np.linspace(max(1,nf//8),nf-1,6).round().astype(int).tolist()))
    gts={i:gt_dt(CLIP,t0,i,tf,H,W) for i in kf}
    names=["hub"]+[f"plug{k}" for k in range(4)]
    d=json.load(open(os.path.join(_ROOT,"data/genesis_rod_bigfit_bptt120.json")))
    K,E=float(d["K"]),float(d["E"]); hs=np.asarray(d["hinge_scale"],float); ss=np.asarray(d["stretch_scale"],float)
    conn.send("ready")
    def once(kap, want_grad):
        scene,ents,rods,_=R.build_rod_scene(shp,0,K=K,E=E,requires_grad=want_grad,
            bptt_window=BPTT,grad_clip=1.0,use_inextensible=False,only_cable=cable)
        scene.build(n_envs=1); rod=rods[0]; n_v=rod.n_vertices
        rod.attach_to_rigid_link(R._mod_link(ents,"hub"),list(range(R.N_BIND)))
        rod.attach_to_rigid_link(R._mod_link(ents,f"plug{cable}"),list(range(n_v-R.N_BIND,n_v)))
        rod.set_stretching_scale(torch.as_tensor(ss.astype(np.float32)))
        rod.set_bending_scale(torch.as_tensor(hs.astype(np.float32)))
        if kap is not None: rod.set_kappa_rest_offset(torch.as_tensor(kap.astype(np.float32)))
        dofs,dt_={},{}
        for n in names:
            e=ents[n]; dofs[n]=e.get_joint(f"{n}_x").dofs_idx_local+e.get_joint(f"{n}_y").dofs_idx_local
            e.set_dofs_kp(np.full(2,4000.0),dofs[n]); e.set_dofs_kv(np.full(2,120.0),dofs[n])
            et=ents[f"{n}_top"]
            xy=et.get_joint(f"{n}_x").dofs_idx_local+et.get_joint(f"{n}_y").dofs_idx_local
            tu=et.get_joint(f"{n}_turn").dofs_idx_local; dt_[n]=(xy,tu)
            et.set_dofs_kp(np.full(2,4000.0),xy); et.set_dofs_kv(np.full(2,120.0),xy)
            et.set_dofs_kp(np.full(1,60.0),tu); et.set_dofs_kv(np.full(1,4.0),tu)
        p0=R.MJ_poses(shp,0); base={n:np.asarray(p0[n][0],float) for n in names}
        by={n:float(p0[n][1]) for n in names}; prev=dict(by)
        loss=torch.zeros((),dtype=torch.float32)
        for t in range(nf):
            p=R.MJ_poses(shp,t)
            for n in names:
                pos,yaw=p[n]; prev[n]+=np.arctan2(np.sin(yaw-prev[n]),np.cos(yaw-prev[n]))
                rel=np.asarray(pos,float)-base[n]
                ents[n].control_dofs_position(rel,dofs[n])
                ents[f"{n}_top"].control_dofs_position(rel,dt_[n][0])
                ents[f"{n}_top"].control_dofs_position(np.array([prev[n]-by[n]]),dt_[n][1])
            for _ in range(spf): scene.step()
            if t in gts: loss=loss+dt_loss(rod,tf,gts[t],H,W)
        loss=loss/len(gts); L=float(loss); g=np.zeros((14,2))
        if want_grad:
            loss.backward(); g=rod.get_kappa_rest_grad()[:,:,0].numpy()
        gs.destroy(); return L,g.tolist()
    while True:
        m=conn.recv()
        if m is None: break
        kap,wg=m
        conn.send(once(None if kap is None else np.asarray(kap,float),wg))
    conn.close()

if __name__=="__main__":
    import math, time
    ctx=mp.get_context("spawn"); procs,conns=[],[]
    jobs=[(c,k) for c in CLIPS for k in range(4)]
    for j in jobs:
        a,b=ctx.Pipe(); pr=ctx.Process(target=worker,args=(j,b),daemon=True); pr.start()
        procs.append(pr); conns.append(a)
    for c in conns: c.recv()
    print(f"{len(jobs)} prozesse bereit ({len(CLIPS)} clips)", flush=True)
    def ev(kap,wg):
        for c in conns: c.send((kap.tolist(),wg))
        L=0.0; g=np.zeros((14,2))
        for c in conns:
            l,gg=c.recv(); L+=l; g+=np.asarray(gg,float)
        return L/len(conns),g
    kap=np.zeros((14,2)); mom=np.zeros((14,2)); hist=[]; best=None
    for it in range(N_ITERS):
        t0=time.time(); L,g=ev(kap,True)
        nrm=float(np.linalg.norm(g)); gn=g/nrm if nrm>1e-14 else g
        mom=0.3*mom+0.7*gn
        lr=LR_MIN+0.5*(LR-LR_MIN)*(1+math.cos(math.pi*it/max(1,N_ITERS-1)))
        rec={"it":it,"loss":L,"gnorm":nrm,"kappa":kap.tolist()}
        hist.append(rec)
        if best is None or L<best["loss"]: best=dict(rec)
        print(f"[it {it:03d}] loss={L:.5f}  |grad|={nrm:.3e}  lr={lr:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        kap=kap-lr*mom
        json.dump({"hist":hist,"best":best,"clips":CLIPS,"kappa":best["kappa"]},
                  open(OUT,"w"), indent=1)
    for c in conns: c.send(None)
    print(f"\nbester stand: loss={best['loss']:.5f} it {best['it']}", flush=True)
    print("KAPPA_FIT_DONE", flush=True)
