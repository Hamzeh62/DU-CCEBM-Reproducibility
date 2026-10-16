"""Experimental DU-CCEBM prototype built from archived CCEBM-v3 equations.

Cites algorithmic ancestry in supplied files `ccebm_v3.py` and `pu_lattice.py`:
 - spatial complex whitening and (channel,lag) stacking;
 - closed-form augmented/widely-linear Eq.13 filtering (stop-grad alternating step);
 - entropy-bound surrogate and four measuring functions;
 - full-FIR Vaidyanathan paraunitary lattice;
 - learned positive unrolled step sizes replacing the ORIGINAL Armijo line search.

Important distinctions from the ORIGINAL classical CCEBM implementation:
 * learnable tau rather than backtracking;
 * PyTorch autograd gradients of alternating entropy cost, not original hand gradient;
 * supervised 2-source Doppler-coherence head, not original max-over-sources;
 * the original 3-restart PU selection is NOT implemented here;
 * THIS FILE HAS NOT BEEN TRAINED OR EVALUATED ON REAL IPIX.

Do not claim numerical superiority or mathematical line-by-line identity with the
classical algorithm; this is research code for training/verification only.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

# Monte-Carlo Gaussian expectations: numerically identical construction to
# the measuring-function helpers referenced by supplied CCEBM-v3 source.
def _gaussian_moments():
    r=np.random.default_rng(0).standard_normal(400_000)
    gs=(lambda u:u**4,
        lambda u:np.abs(u)/(1+np.abs(u)),
        lambda u:u*np.abs(u)/(10+np.abs(u)),
        lambda u:u/(1+u**2))
    return torch.tensor([g(r).mean() for g in gs],dtype=torch.float64)
_EG=_gaussian_moments()


def gaussian_functions(x):
    return (x**4,torch.abs(x)/(1+torch.abs(x)),
           x*torch.abs(x)/(10+torch.abs(x)), x/(1+x**2))


def complex_spatial_whiten(X):
    if X.ndim!=2 or X.shape[0]!=2 or not torch.is_complex(X):
        raise ValueError('X must have complex shape (2,T), HH,VV at same pulses')
    xc=X-X.mean(dim=1,keepdim=True)
    C=xc@xc.conj().T/X.shape[1]
    eig, E=torch.linalg.eigh(C)
    if torch.min(eig).item()<=1e-10*torch.max(eig).item():
        raise ValueError('Whitening rank deficiency; original solver changes dimension')
    return (E/torch.sqrt(eig)).conj().T@xc


def lag_stack(X,L):
    T=X.shape[1]
    return torch.cat([torch.stack([X[c,L-1-l:T-l] for l in range(L)])
                      for c in range(X.shape[0])],dim=0)


def lattice_bank(U,Vs):
    n=U.shape[1]
    I=torch.eye(n,dtype=U.dtype,device=U.device)
    coeff=[U]
    for v in Vs:
        v=v/(torch.linalg.vector_norm(v)+1e-30)
        P=v[:,None]@v.conj()[None,:]
        A=I-P
        coeff_new=[]
        for l in range(len(coeff)+1):
            acc=None
            if l<len(coeff):acc=coeff[l]@A
            if l>0:acc=(acc if acc is not None else 0)+coeff[l-1]@P
            coeff_new.append(acc)
        coeff=coeff_new
    return torch.stack(coeff,dim=-1)


def full_fir_residual(W):
    n,_,L=W.shape
    eye=torch.eye(n,dtype=W.dtype,device=W.device)
    errors=[]
    for lag in range(L):
        g=sum(W[:,:,l]@W[:,:,l+lag].conj().T for l in range(L-lag))
        errors.append(torch.sum(torch.abs(g-(eye if lag==0 else 0))**2))
    return torch.sqrt(sum(errors)).real


def lagged(y,K):
    T=y.shape[-1]
    return torch.stack([y[K-k:T-k] for k in range(K+1)])


def eq13_filter_stopgrad(y,K):
    # Exactly the alternating eigenvector *formula*; gradient through the
    # inner eigenproblem is excluded, matching legacy hand-gradient semantics.
    with torch.no_grad():
        Y=lagged(y.detach(),K)
        A=torch.cat([Y,Y.conj()],dim=0)
        R=A@A.conj().T/A.shape[1]
        R=R+(1e-8*torch.trace(R).real/R.shape[0])*torch.eye(R.shape[0],dtype=R.dtype,device=R.device)
        D=torch.zeros(R.shape[0],dtype=R.real.dtype,device=R.device)
        D[0]=1;D[K+1]=-1
        try:
            lam,V=torch.linalg.eig(torch.linalg.solve(R,torch.diag(D).to(R.dtype)))
            b=V[:,torch.argmax(lam.real)]
            s=torch.real(torch.vdot(b,D*b))
            if abs(s.item())<1e-12:raise ValueError('singular normalization')
            b=b/torch.sqrt(torch.abs(s))
            if not torch.isfinite(b).all():raise ValueError('nonfinite eigenvector')
        except (RuntimeError,ValueError):
            b=torch.zeros(R.shape[0],dtype=R.dtype,device=R.device)
            b[0]=1
        return b


def widely_linear_apply(Z,b,K):
    T=Z.shape[1];p=b[:K+1];q=b[K+1:]
    a=sum(p[k].conj()*Z[:,K-k:T-k] for k in range(K+1))
    c=sum(q[k].conj()*Z[:,K-k:T-k].conj() for k in range(K+1))
    return a,c


def entropy_cost(W,Z,K):
    wf=W.reshape(2,-1)
    y=wf.conj()@Z
    cost=torch.zeros((),dtype=Z.real.dtype,device=Z.device)
    for i in range(2):
        b=eq13_filter_stopgrad(y[i],K)
        a,c=widely_linear_apply(Z,b,K)
        z=wf[i].conj()@a+wf[i]@c
        parts=(z.real,z.imag)
        for x in parts:
            std=torch.sqrt(torch.mean((x-x.mean())**2)+1e-16)
            u=x/std
            fs=gaussian_functions(u)
            ent=torch.stack([(g.mean()-_EG[t].to(x.device))**2 for t,g in enumerate(fs)])
            best=int(torch.argmax(ent.detach()))
            cost=cost+torch.log(std)-ent[best]
    return cost+2*math.log(2*math.pi*math.e)/2


class DUCCEBM(nn.Module):
    """Research-only unrolled Eq13/PU prototype; one (HH,VV) sample per forward.

    Input X (2,T) complex; output scalar score for declared Doppler f_hz.
    `f_hz` is a KNOWN task steering parameter for injected-target experiments;
    using injected ground-truth target Doppler in an unknown-Doppler benchmark
    would be unfair. Never choose f_hz from held-out target labels.
    """
    def __init__(self, L=6,K=4,depth=2,seed=7,tau0=0.001):
        super().__init__()
        if L<2 or K<1 or depth<1:raise ValueError('invalid L/K/depth')
        from pathlib import Path
        import sys
        repo=str(Path(__file__).resolve().parent)
        if repo not in sys.path:sys.path.insert(0,repo)
        from pu_lattice_original_reference import init as lattice_init
        U,Vs=lattice_init(2,2,L,np.random.default_rng(seed))
        self.U0=nn.Parameter(torch.as_tensor(U,dtype=torch.complex128).clone())
        self.V0=nn.ParameterList([nn.Parameter(torch.as_tensor(v,dtype=torch.complex128).clone()) for v in Vs])
        self.L,self.K,self.depth=L,K,depth
        self.raw_steps=nn.Parameter(torch.full((depth,), math.log(math.expm1(tau0)),dtype=torch.float64))
        self.head=nn.Linear(2,1,dtype=torch.float64)
        nn.init.zeros_(self.head.bias)
        nn.init.constant_(self.head.weight,0.5)

    def forward(self,X,f_hz,M=128,prf_hz=1000.):
        if not torch.is_complex(X):raise ValueError('Complex slow-time signal required')
        X=X.to(dtype=torch.complex128)
        Zw=lag_stack(complex_spatial_whiten(X),self.L)
        U=torch.linalg.qr(self.U0.conj().T).Q.conj().T
        vs=[v/(torch.linalg.vector_norm(v)+1e-30) for v in self.V0]
        costs=[]
        for k in range(self.depth):
            W=lattice_bank(U,vs)
            with torch.enable_grad():
                c=entropy_cost(W,Zw,self.K)
                g=torch.autograd.grad(c,[U]+vs,create_graph=self.training)
            step=torch.clamp(F.softplus(self.raw_steps[k]),min=1e-7,max=.02)
            U=torch.linalg.qr((U-step*g[0]).conj().T).Q.conj().T
            vs=[(v-step*gv)/(torch.linalg.vector_norm(v-step*gv)+1e-30) for v,gv in zip(vs,g[1:])]
            costs.append(c.detach().item())
        W=lattice_bank(U,vs)
        y=W.reshape(2,-1).conj()@Zw
        if y.shape[-1]<M:raise ValueError('M exceeds output window length')
        yy=y[:,-M:]
        n=torch.arange(M,dtype=torch.float64,device=X.device)
        v=torch.exp((-2j*math.pi*f_hz/prf_hz)*n)
        cohs=torch.abs(torch.sum(yy*v,dim=-1))**2/(torch.sum(torch.abs(yy)**2,dim=-1)+1e-30)
        score=torch.max(cohs.real)  # fixed original-compatible readout, differentiable
        return score,{'pu_residual':full_fir_residual(W),'coherence':cohs.detach(),'cost_history':costs}


if __name__=='__main__':
    torch.manual_seed(1)
    t=torch.arange(110,dtype=torch.float64)
    x=torch.stack([torch.exp(1j*(.1*t+.001*t*t)),torch.exp(1j*(.3*t+.003*t*t))])
    model=DUCCEBM(depth=2)
    score,aux=model(x,15.0,M=32)
    print('DEMONSTRATION ONLY: score',score.item(),'PU residual',aux['pu_residual'].item())
    (score**2).backward()
    print('finite gradient', torch.isfinite(model.raw_steps.grad).all().item())
