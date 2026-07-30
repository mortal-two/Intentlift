"""Independent target-native baselines for the Fresh14 RVV experiment.

These sources are deliberately case-authored and do not consume IntentIR MLIR,
search candidates, schedules, or target objects.  They share only the public
contiguous memref ABI and workload tensors with the backend measurement.
"""

from __future__ import annotations


NATIVE_RVV_BASELINE_PROTOCOL = "intentir.backend.riscv.native-baseline.v1"
NATIVE_RVV_BASELINE_SIDECAR_PROTOCOL = (
    "intentir.backend.riscv.native-baseline-sidecar.v1"
)
NATIVE_RVV_SYMBOL = "intentir_native_baseline"


_PREAMBLE = r"""
#include <math.h>
#include <stdint.h>
#include <stddef.h>
#include <riscv_vector.h>

typedef struct { void *allocated, *aligned; int64_t offset; } M0;
typedef struct { void *allocated, *aligned; int64_t offset, sizes[1], strides[1]; } M1;
typedef struct { void *allocated, *aligned; int64_t offset, sizes[2], strides[2]; } M2;
typedef struct { void *allocated, *aligned; int64_t offset, sizes[3], strides[3]; } M3;
typedef struct { void *allocated, *aligned; int64_t offset, sizes[4], strides[4]; } M4;
typedef struct { void *allocated, *aligned; int64_t offset, sizes[5], strides[5]; } M5;

#define DATA(D, T) (((T *)((D)->aligned)) + (D)->offset)
"""


def _source(arguments: str, body: str) -> str:
    return (
        _PREAMBLE
        + "\nvoid "
        + NATIVE_RVV_SYMBOL
        + "("
        + arguments
        + ") {\n"
        + body
        + "\n}\n"
    )


def _dyt() -> str:
    return _source(
        "M2 *x_d, M1 *alpha_d, M1 *gamma_d, M1 *beta_d, M2 *y_d",
        r"""
  float *x=DATA(x_d,float), *alpha=DATA(alpha_d,float);
  float *gamma=DATA(gamma_d,float), *beta=DATA(beta_d,float), *y=DATA(y_d,float);
  int64_t m=x_d->sizes[0], n=x_d->sizes[1]; float a=alpha[0];
  #pragma omp parallel for schedule(static)
  for (int64_t i=0;i<m;i++) for (int64_t j=0;j<n;j++) {
    int64_t k=i*n+j; y[k]=tanhf(a*x[k])*gamma[j]+beta[j];
  }
""",
    )


def _swiglu() -> str:
    return _source(
        "M2 *a_d, M2 *b_d, M0 *mult_d, M2 *out_d",
        r"""
  float *a=DATA(a_d,float), *b=DATA(b_d,float), *mult=DATA(mult_d,float);
  float *out=DATA(out_d,float); int64_t total=a_d->sizes[0]*a_d->sizes[1];
  float g=mult[0];
  #pragma omp parallel for schedule(static)
  for (int64_t k=0;k<total;k++) { float z=a[k]*g; out[k]=(z/(1.0f+expf(-z)))*b[k]; }
""",
    )


def _geglu() -> str:
    return _source(
        "M2 *a_d, M2 *b_d, M2 *out_d",
        r"""
  float *a=DATA(a_d,float), *b=DATA(b_d,float), *out=DATA(out_d,float);
  int64_t total=a_d->sizes[0]*a_d->sizes[1];
  const float c=0.7978845608028654f;
  #pragma omp parallel for schedule(static)
  for (int64_t k=0;k<total;k++) { float z=a[k]; float z3=z*z*z;
    out[k]=(0.5f*z*(1.0f+tanhf(c*(z+0.044715f*z3))))*b[k]; }
""",
    )


def _softmax() -> str:
    return _source(
        "M2 *x_d, M0 *n_d, M2 *y_d",
        r"""
  float *x=DATA(x_d,float), *y=DATA(y_d,float); int32_t n0=DATA(n_d,int32_t)[0];
  int64_t m=x_d->sizes[0], n=x_d->sizes[1]; if (n0<n) n=n0;
  #pragma omp parallel for schedule(static)
  for (int64_t i=0;i<m;i++) { float mx=-INFINITY;
    for (int64_t j=0;j<n;j++) if (x[i*x_d->sizes[1]+j]>mx) mx=x[i*x_d->sizes[1]+j];
    float sum=0.0f; for (int64_t j=0;j<n;j++) { float e=expf(x[i*x_d->sizes[1]+j]-mx); y[i*y_d->sizes[1]+j]=e; sum+=e; }
    for (int64_t j=0;j<n;j++) y[i*y_d->sizes[1]+j]/=sum;
  }
""",
    )


def _sparsemax() -> str:
    return _source(
        "M2 *x_d, M2 *sorted_d, M0 *n_d, M2 *y_d",
        r"""
  float *x=DATA(x_d,float), *z=DATA(sorted_d,float), *y=DATA(y_d,float);
  int32_t n0=DATA(n_d,int32_t)[0]; int64_t m=x_d->sizes[0], width=x_d->sizes[1];
  int64_t n=n0<width?n0:width;
  #pragma omp parallel for schedule(static)
  for (int64_t i=0;i<m;i++) { float prefix=0.0f, support_sum=0.0f; int64_t k=1;
    for (int64_t j=0;j<n;j++) { prefix+=z[i*width+j]; float t=(prefix-1.0f)/(float)(j+1);
      if (z[i*width+j]>t) { k=j+1; support_sum=prefix; } }
    float tau=(support_sum-1.0f)/(float)k;
    for (int64_t j=0;j<n;j++) { float v=x[i*width+j]-tau; y[i*width+j]=v>0.0f?v:0.0f; }
  }
""",
    )


def _layer_norm() -> str:
    return _source(
        "M2 *x_d, M1 *w_d, M1 *b_d, M0 *n_d, M0 *eps_d, M2 *y_d, M1 *mean_d, M1 *rstd_d",
        r"""
  float *x=DATA(x_d,float), *w=DATA(w_d,float), *b=DATA(b_d,float), *y=DATA(y_d,float);
  float *mean=DATA(mean_d,float), *rstd=DATA(rstd_d,float); int32_t n0=DATA(n_d,int32_t)[0];
  float eps=DATA(eps_d,float)[0]; int64_t m=x_d->sizes[0], width=x_d->sizes[1], n=n0<width?n0:width;
  #pragma omp parallel for schedule(static)
  for (int64_t i=0;i<m;i++) { float s=0.0f; for(int64_t j=0;j<n;j++) s+=x[i*width+j];
    float mu=s/(float)n, v=0.0f; for(int64_t j=0;j<n;j++){float d=x[i*width+j]-mu;v+=d*d;}
    float r=1.0f/sqrtf(v/(float)n+eps); mean[i]=mu; rstd[i]=r;
    for(int64_t j=0;j<n;j++) y[i*width+j]=(x[i*width+j]-mu)*r*w[j]+b[j]; }
""",
    )


def _rms_norm() -> str:
    return _source(
        "M2 *x_d, M1 *w_d, M0 *n_d, M0 *eps_d, M0 *off_d, M2 *y_d, M1 *rstd_d",
        r"""
  float *x=DATA(x_d,float), *w=DATA(w_d,float), *y=DATA(y_d,float), *rstd=DATA(rstd_d,float);
  int32_t n0=DATA(n_d,int32_t)[0]; float eps=DATA(eps_d,float)[0], off=DATA(off_d,float)[0];
  int64_t m=x_d->sizes[0], width=x_d->sizes[1], n=n0<width?n0:width;
  #pragma omp parallel for schedule(static)
  for(int64_t i=0;i<m;i++){float s=0.0f;for(int64_t j=0;j<n;j++){float v=x[i*width+j];s+=v*v;}
    float r=1.0f/sqrtf(s/(float)n+eps);rstd[i]=r;
    for(int64_t j=0;j<n;j++)y[i*width+j]=x[i*width+j]*r*(w[j]+off);}
""",
    )


def _fused_add_rms_norm() -> str:
    return _source(
        "M2 *x_d, M2 *res_d, M1 *w_d, M0 *n_d, M0 *eps_d, M0 *off_d, M2 *y_d, M2 *sum_d, M1 *rstd_d",
        r"""
  float *x=DATA(x_d,float),*res=DATA(res_d,float),*w=DATA(w_d,float),*y=DATA(y_d,float);
  float *sum=DATA(sum_d,float),*rstd=DATA(rstd_d,float);int32_t n0=DATA(n_d,int32_t)[0];
  float eps=DATA(eps_d,float)[0],off=DATA(off_d,float)[0];int64_t m=x_d->sizes[0],width=x_d->sizes[1],n=n0<width?n0:width;
  #pragma omp parallel for schedule(static)
  for(int64_t i=0;i<m;i++){float q=0.0f;for(int64_t j=0;j<n;j++){float s=x[i*width+j]+res[i*width+j];sum[i*width+j]=s;q+=s*s;}
    float r=1.0f/sqrtf(q/(float)n+eps);rstd[i]=r;for(int64_t j=0;j<n;j++)y[i*width+j]=sum[i*width+j]*r*(w[j]+off);}
""",
    )


def _xformers_rms_norm_add() -> str:
    return _source(
        "M2 *x_d, M2 *res_d, M1 *w_d, M0 *eps_d, M2 *sum_d, M2 *y_d",
        r"""
  float *x=DATA(x_d,float),*res=DATA(res_d,float),*w=DATA(w_d,float),*sum=DATA(sum_d,float),*y=DATA(y_d,float);
  float eps=DATA(eps_d,float)[0];int64_t m=x_d->sizes[0],n=x_d->sizes[1];
  #pragma omp parallel for schedule(static)
  for(int64_t i=0;i<m;i++){float q=0.0f;for(int64_t j=0;j<n;j++){float s=x[i*n+j]+res[i*n+j];sum[i*n+j]=s;q+=s*s;}
    float r=1.0f/sqrtf(q/(float)n+eps);for(int64_t j=0;j<n;j++)y[i*n+j]=sum[i*n+j]*r*w[j];}
""",
    )


def _kl_div() -> str:
    return _source(
        "M2 *p_d, M2 *q_d, M1 *loss_d",
        r"""
  float *p=DATA(p_d,float),*q=DATA(q_d,float),*loss=DATA(loss_d,float);int64_t m=p_d->sizes[0],n=p_d->sizes[1];
  #pragma omp parallel for schedule(static)
  for(int64_t i=0;i<m;i++){float s=0.0f;for(int64_t j=0;j<n;j++){int64_t k=i*n+j;s+=expf(q[k])*(q[k]-p[k]);}loss[i]=s;}
""",
    )


def _poly_norm() -> str:
    return _source(
        "M2 *x_d, M1 *coef_d, M1 *bias_d, M0 *n_d, M0 *eps_d, M2 *y_d, M2 *rstd_d",
        r"""
  float *x=DATA(x_d,float),*c=DATA(coef_d,float),bias=DATA(bias_d,float)[0],eps=DATA(eps_d,float)[0];
  float *y=DATA(y_d,float),*rs=DATA(rstd_d,float);int32_t n0=DATA(n_d,int32_t)[0];int64_t m=x_d->sizes[0],w=x_d->sizes[1],n=n0<w?n0:w;
  #pragma omp parallel for schedule(static)
  for(int64_t i=0;i<m;i++){float s1=0.0f,s2=0.0f,s3=0.0f;for(int64_t j=0;j<n;j++){float v=x[i*w+j],v2=v*v;s1+=v2;s2+=v2*v2;s3+=v2*v2*v2;}
    float r1=1.0f/sqrtf(s1/(float)n+eps),r2=1.0f/sqrtf(s2/(float)n+eps),r3=1.0f/sqrtf(s3/(float)n+eps);
    rs[i*3]=r3;rs[i*3+1]=r2;rs[i*3+2]=r1;for(int64_t j=0;j<n;j++){float v=x[i*w+j],v2=v*v;y[i*w+j]=c[0]*v2*v*r3+c[1]*v2*r2+c[2]*v*r1+bias;}}
""",
    )


def _flash_bwd_preprocess() -> str:
    return _source(
        "M4 *a_d, M4 *b_d, M3 *out_d",
        r"""
  _Float16 *a=DATA(a_d,_Float16),*b=DATA(b_d,_Float16);float *out=DATA(out_d,float);
  int64_t rows=a_d->sizes[0]*a_d->sizes[1]*a_d->sizes[2],d=a_d->sizes[3];
  #pragma omp parallel for schedule(static)
  for(int64_t i=0;i<rows;i++){float s=0.0f;for(int64_t j=0;j<d;j++)s+=(float)a[i*d+j]*(float)b[i*d+j];out[i]=s;}
""",
    )


def _splitk() -> str:
    return _source(
        "M5 *o0_d, M5 *o1_d, M4 *l0_d, M4 *l1_d, M5 *out_d, M4 *lse_d",
        r"""
  float *o0=DATA(o0_d,float),*o1=DATA(o1_d,float),*l0=DATA(l0_d,float),*l1=DATA(l1_d,float);
  float *out=DATA(out_d,float),*lse=DATA(lse_d,float);int64_t rows=l0_d->sizes[0]*l0_d->sizes[1]*l0_d->sizes[2]*l0_d->sizes[3],d=o0_d->sizes[4];
  #pragma omp parallel for schedule(static)
  for(int64_t i=0;i<rows;i++){float m=fmaxf(l0[i],l1[i]);if(isinf(m)&&m<0){lse[i]=-INFINITY;for(int64_t j=0;j<d;j++)out[i*d+j]=0.0f;continue;}
    float p0=exp2f((l0[i]-m)*1.4426950408889634f),p1=exp2f((l1[i]-m)*1.4426950408889634f),den=p0+p1;
    lse[i]=m+log2f(den)/1.4426950408889634f;for(int64_t j=0;j<d;j++)out[i*d+j]=(o0[i*d+j]*p0+o1[i*d+j]*p1)/den;}
""",
    )


def _flash_layer_norm() -> str:
    arguments = ", ".join(
        [
            "M2 *x_d", "M1 *w0_d", "M1 *b0_d", "M2 *r0_d", "M2 *r1_d",
            "M1 *w1_d", "M1 *b1_d", "M1 *unused7_d", "M1 *unused8_d",
            "M2 *unused9_d", "M2 *unused10_d", "M0 *n_d", "M0 *eps_d",
            "M0 *zero_centered_d", "M2 *y0_d", "M2 *y1_d", "M2 *resout_d",
            "M1 *mean_d", "M1 *rstd_d",
        ]
    )
    return _source(
        arguments,
        r"""
  _Float16 *x=DATA(x_d,_Float16),*w0=DATA(w0_d,_Float16),*b0=DATA(b0_d,_Float16);
  _Float16 *r0=DATA(r0_d,_Float16),*r1=DATA(r1_d,_Float16),*w1=DATA(w1_d,_Float16),*b1=DATA(b1_d,_Float16);
  _Float16 *y0=DATA(y0_d,_Float16),*y1=DATA(y1_d,_Float16),*ro=DATA(resout_d,_Float16);
  float *mean=DATA(mean_d,float),*rstd=DATA(rstd_d,float);int32_t n0=DATA(n_d,int32_t)[0];float eps=DATA(eps_d,float)[0];
  int zc=DATA(zero_centered_d,uint8_t)[0]!=0;int64_t m=x_d->sizes[0],width=x_d->sizes[1],n=n0<width?n0:width;
  #pragma omp parallel for schedule(static)
  for(int64_t i=0;i<m;i++){float s=0.0f;for(int64_t j=0;j<n;j++){float v=(float)x[i*width+j]+(float)r0[i*width+j]+(float)r1[i*width+j];ro[i*width+j]=(_Float16)v;s+=v;}
    float mu=s/(float)n,var=0.0f;for(int64_t j=0;j<n;j++){float v=(float)x[i*width+j]+(float)r0[i*width+j]+(float)r1[i*width+j]-mu;var+=v*v;}
    float rr=1.0f/sqrtf(var/(float)n+eps);mean[i]=mu;rstd[i]=rr;
    for(int64_t j=0;j<n;j++){float v=((float)x[i*width+j]+(float)r0[i*width+j]+(float)r1[i*width+j]-mu)*rr;
      float q0=v*((float)w0[j]+(zc?1.0f:0.0f))+(float)b0[j];float q1=v*((float)w1[j]+(zc?1.0f:0.0f))+(float)b1[j];y0[i*width+j]=(_Float16)q0;y1[i*width+j]=(_Float16)q1;}}
""",
    )


_SOURCES = {
    "dyt": _dyt,
    "swiglu": _swiglu,
    "geglu": _geglu,
    "softmax": _softmax,
    "sparsemax": _sparsemax,
    "layer_norm": _layer_norm,
    "liger_rms_norm": _rms_norm,
    "fused_add_rms_norm": _fused_add_rms_norm,
    "xformers_rms_norm_add": _xformers_rms_norm_add,
    "kl_div": _kl_div,
    "poly_norm": _poly_norm,
    "flash_bwd_preprocess": _flash_bwd_preprocess,
    "splitk_reduce_varargs": _splitk,
    "flash_layer_norm_1pass": _flash_layer_norm,
}


def native_rvv_baseline_record(kernel: str) -> dict[str, str]:
    try:
        source = _SOURCES[kernel]()
    except KeyError as exc:
        raise ValueError(f"no independent Native RVV baseline for {kernel!r}") from exc
    return {
        "protocol": NATIVE_RVV_BASELINE_PROTOCOL,
        "symbol": NATIVE_RVV_SYMBOL,
        "source": source,
        "implementation": "independent-case-authored-c-rvv-openmp",
    }


__all__ = [
    "NATIVE_RVV_BASELINE_PROTOCOL",
    "NATIVE_RVV_BASELINE_SIDECAR_PROTOCOL",
    "NATIVE_RVV_SYMBOL",
    "native_rvv_baseline_record",
]
