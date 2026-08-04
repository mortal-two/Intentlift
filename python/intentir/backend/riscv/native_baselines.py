"""Independent LLVM-RVV C and explicit-intrinsic external baselines.

These sources are deliberately case-authored and do not consume IntentIR MLIR,
search candidates, schedules, or target objects.  They share only the public
contiguous memref ABI and workload tensors with the backend measurement.

The historical ``native`` baseline is scalar C compiled by Clang/LLVM with RVV
enabled.  It is therefore reported as ``LLVM-RVV C`` rather than being
misrepresented as a library or hand-vectorized implementation.  A smaller
strictly matching subset additionally exposes explicit, VLEN-agnostic RVV
intrinsics as an expert-target baseline.
"""

from __future__ import annotations


NATIVE_RVV_BASELINE_PROTOCOL = "intentir.backend.riscv.native-baseline.v1"
NATIVE_RVV_BASELINE_SIDECAR_PROTOCOL = (
    "intentir.backend.riscv.native-baseline-sidecar.v1"
)
NATIVE_RVV_SYMBOL = "intentir_native_baseline"
EXTERNAL_RVV_BASELINE_PROTOCOL = "intentir.backend.riscv.external-baseline.v1"
EXTERNAL_RVV_BASELINE_SIDECAR_PROTOCOL = (
    "intentir.backend.riscv.external-baseline-sidecar.v1"
)
LLVM_RVV_C_BASELINE_ID = "llvm_rvv_c"
EXPERT_RVV_BASELINE_ID = "expert_rvv"


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
  int64_t groups=a_d->sizes[0]*a_d->sizes[1],rows=a_d->sizes[2],d=a_d->sizes[3];
  int64_t padded=out_d->sizes[2];
  #pragma omp parallel for schedule(static)
  for(int64_t g=0;g<groups;g++){for(int64_t i=0;i<rows;i++){float s=0.0f;
      int64_t row=g*rows+i;for(int64_t j=0;j<d;j++)s+=(float)a[row*d+j]*(float)b[row*d+j];
      out[g*padded+i]=s;}
    for(int64_t i=rows;i<padded;i++)out[g*padded+i]=0.0f;}
""",
    )


def _flash_attention_forward() -> str:
    return _source(
        (
            "M4 *q_d, M4 *k_d, M4 *v_d, M1 *bias_d, M3 *tmp_in_d, "
            "M0 *scale_d, M4 *out_d, M3 *lse_d, M3 *tmp_out_d"
        ),
        r"""
  _Float16 *q=DATA(q_d,_Float16),*k=DATA(k_d,_Float16),*v=DATA(v_d,_Float16);
  float *tmp_in=DATA(tmp_in_d,float),*out_lse=DATA(lse_d,float);
  float *tmp_out=DATA(tmp_out_d,float);_Float16 *out=DATA(out_d,_Float16);
  float scale=DATA(scale_d,float)[0];(void)bias_d;
  int64_t batch=q_d->sizes[0],queries=q_d->sizes[1],heads=q_d->sizes[2],d=q_d->sizes[3];
  int64_t keys=k_d->sizes[1],padded=tmp_out_d->sizes[2];
  int64_t scratch=batch*heads*padded;
  for(int64_t i=0;i<scratch;i++){tmp_out[i]=tmp_in[i];out_lse[i]=0.0f;}
  #pragma omp parallel for collapse(3) schedule(static)
  for(int64_t b=0;b<batch;b++)for(int64_t qi=0;qi<queries;qi++)for(int64_t h=0;h<heads;h++){
    float weights[keys];float maximum=-INFINITY;
    for(int64_t ki=0;ki<keys;ki++){float score=0.0f;
      int64_t qbase=((b*queries+qi)*heads+h)*d;
      int64_t kbase=((b*keys+ki)*heads+h)*d;
      for(int64_t j=0;j<d;j++)score+=(float)q[qbase+j]*(float)k[kbase+j];
      score*=scale;weights[ki]=score;if(score>maximum)maximum=score;}
    float denominator=0.0f;
    for(int64_t ki=0;ki<keys;ki++){float weight=expf(weights[ki]-maximum);weights[ki]=weight;denominator+=weight;}
    float inverse=1.0f/denominator;
    int64_t state=(b*heads+h)*padded+qi;
    tmp_out[state]=inverse;out_lse[state]=maximum+logf(denominator);
    int64_t obase=((b*queries+qi)*heads+h)*d;
    for(int64_t j=0;j<d;j++){float accumulator=0.0f;
      for(int64_t ki=0;ki<keys;ki++){
        int64_t vbase=((b*keys+ki)*heads+h)*d;
        accumulator+=weights[ki]*(float)v[vbase+j];}
      out[obase+j]=(_Float16)(accumulator*inverse);}
  }
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


def _jsd() -> str:
    return _source(
        "M2 *logp_d, M2 *logq_d, M1 *labels_d, M0 *denom_d, M2 *loss_d, M2 *dx_d",
        r"""
  float *logp=DATA(logp_d,float),*logq=DATA(logq_d,float);
  float *loss=DATA(loss_d,float),*dx=DATA(dx_d,float);
  int32_t denom_i=DATA(denom_d,int32_t)[0];float denom=(float)denom_i;
  int64_t m=logp_d->sizes[0],n=logp_d->sizes[1];(void)labels_d;
  #pragma omp parallel for schedule(static)
  for(int64_t i=0;i<m;i++){float mx=-INFINITY;
    for(int64_t j=0;j<n;j++){int64_t k=i*n+j;mx=fmaxf(mx,fmaxf(logp[k],logq[k]));}
    for(int64_t j=0;j<n;j++){int64_t k=i*n+j;float p=expf(logp[k]-mx)*expf(mx);float q=expf(logq[k]-mx)*expf(mx);
      float wp=0.5f*p,wq=0.5f*q,mix=wp+wq,logmix=logf(mix);
      loss[k]=(wq*logq[k]+wp*logp[k]-mix*logmix)/denom;
      dx[k]=wp*(logp[k]-logmix)/denom;}}
""",
    )


def _tvd() -> str:
    return _source(
        "M2 *a_d, M2 *b_d, M1 *labels_d, M0 *scale_d, M2 *loss_d, M2 *grad_d",
        r"""
  float *a=DATA(a_d,float),*b=DATA(b_d,float),scale=DATA(scale_d,float)[0];
  float *loss=DATA(loss_d,float),*grad=DATA(grad_d,float);(void)labels_d;
  int64_t total=a_d->sizes[0]*a_d->sizes[1];
  #pragma omp parallel for schedule(static)
  for(int64_t k=0;k<total;k++){loss[k]=0.5f*fabsf(a[k]-b[k]);grad[k]=(a[k]>b[k]?0.5f:-0.5f)*scale;}
""",
    )


def _xformers_rms_norm() -> str:
    return _source(
        "M2 *x_d, M1 *w_d, M0 *eps_d, M2 *y_d",
        r"""
  float *x=DATA(x_d,float),*w=DATA(w_d,float),eps=DATA(eps_d,float)[0],*y=DATA(y_d,float);
  int64_t m=x_d->sizes[0],n=x_d->sizes[1];
  #pragma omp parallel for schedule(static)
  for(int64_t i=0;i<m;i++){float sumsq=0.0f;for(int64_t j=0;j<n;j++){float v=x[i*n+j];sumsq+=v*v;}
    float r=1.0f/sqrtf(sumsq/(float)n+eps);for(int64_t j=0;j<n;j++)y[i*n+j]=x[i*n+j]*r*w[j];}
""",
    )


def _cross_entropy() -> str:
    return _source(
        "M2 *logits_d, M1 *labels_d, M1 *weight_d, M1 *zloss_d, M1 *accuracy_d, M1 *predicted_d, M0 *denom_d, M0 *ignore_d, M2 *logits_out_d, M1 *loss_d",
        r"""
  float *x=DATA(logits_d,float),*out=DATA(logits_out_d,float),*loss=DATA(loss_d,float);
  int64_t *labels=DATA(labels_d,int64_t);float denom=DATA(denom_d,float)[0];int64_t ignore=DATA(ignore_d,int64_t)[0];
  int64_t m=logits_d->sizes[0],n=logits_d->sizes[1];
  (void)weight_d;(void)zloss_d;(void)accuracy_d;(void)predicted_d;
  #pragma omp parallel for schedule(static)
  for(int64_t i=0;i<m;i++){int64_t label=labels[i];
    if(label==ignore){for(int64_t j=0;j<n;j++)out[i*n+j]=0.0f;loss[i]=0.0f;continue;}
    float mx=-INFINITY;for(int64_t j=0;j<n;j++){float v=x[i*n+j];out[i*n+j]=v;if(v>mx)mx=v;}
    float sum=0.0f;for(int64_t j=0;j<n;j++)sum+=expf(x[i*n+j]-mx);
    loss[i]=(mx+logf(sum)-x[i*n+label])/denom;}
""",
    )


def _group_norm() -> str:
    return _source(
        "M3 *x_d, M1 *weight_d, M1 *bias_d, M0 *hidden_d, M0 *channels_d, M0 *eps_d, M3 *y_d, M2 *mean_d, M2 *rstd_d",
        r"""
  float *x=DATA(x_d,float),*w=DATA(weight_d,float),*bias=DATA(bias_d,float),*y=DATA(y_d,float);
  float *mean=DATA(mean_d,float),*rstd=DATA(rstd_d,float),eps=DATA(eps_d,float)[0];
  int32_t hidden_i=DATA(hidden_d,int32_t)[0],channels_i=DATA(channels_d,int32_t)[0];
  int64_t batch=x_d->sizes[0],groups=x_d->sizes[1],width=x_d->sizes[2];
  int64_t hidden=hidden_i<width?hidden_i:width,channels=channels_i,spatial=hidden/channels;
  #pragma omp parallel for schedule(static)
  for(int64_t row=0;row<batch*groups;row++){float sum=0.0f,sumsq=0.0f;
    for(int64_t j=0;j<hidden;j++){float v=x[row*width+j];sum+=v;sumsq+=v*v;}
    float mu=sum/(float)hidden,var=sumsq/(float)hidden-mu*mu,rr=1.0f/sqrtf(var+eps);mean[row]=mu;rstd[row]=rr;
    int64_t group=row%groups;for(int64_t j=0;j<hidden;j++){int64_t channel=group*channels+j/spatial;
      y[row*width+j]=(x[row*width+j]-mu)*rr*w[channel]+bias[channel];}
    for(int64_t j=hidden;j<width;j++)y[row*width+j]=0.0f;}
""",
    )


def _liger_rope() -> str:
    return _source(
        "M4 *q_d, M4 *k_d, M3 *cos_d, M3 *sin_d, M4 *qout_d, M4 *kout_d",
        r"""
  _Float16 *q=DATA(q_d,_Float16),*k=DATA(k_d,_Float16),*c=DATA(cos_d,_Float16),*s=DATA(sin_d,_Float16);
  _Float16 *qo=DATA(qout_d,_Float16),*ko=DATA(kout_d,_Float16);
  int64_t b=q_d->sizes[0],seq=q_d->sizes[1],hq=q_d->sizes[2],hk=k_d->sizes[2],d=q_d->sizes[3],half=d/2;
  #pragma omp parallel for schedule(static)
  for(int64_t row=0;row<b*seq;row++){int64_t token=row%seq;
    for(int64_t h=0;h<hq;h++)for(int64_t j=0;j<half;j++){int64_t base=(row*hq+h)*d;float cv=(float)c[token*d+j],sv=(float)s[token*d+j];
      float a=(float)q[base+j],z=(float)q[base+half+j];qo[base+j]=(_Float16)(a*cv-z*sv);qo[base+half+j]=(_Float16)(z*cv+a*sv);}
    for(int64_t h=0;h<hk;h++)for(int64_t j=0;j<half;j++){int64_t base=(row*hk+h)*d;float cv=(float)c[token*d+j],sv=(float)s[token*d+j];
      float a=(float)k[base+j],z=(float)k[base+half+j];ko[base+j]=(_Float16)(a*cv-z*sv);ko[base+half+j]=(_Float16)(z*cv+a*sv);}}
""",
    )


def _llama4_rope() -> str:
    return _source(
        "M4 *q_d, M4 *k_d, M3 *freq_d, M0 *sign_d, M4 *qout_d, M4 *kout_d",
        r"""
  _Float16 *q=DATA(q_d,_Float16),*k=DATA(k_d,_Float16);
  _Float16 *freq=DATA(freq_d,_Float16),*qo=DATA(qout_d,_Float16),*ko=DATA(kout_d,_Float16);
  float sign=DATA(sign_d,float)[0];
  int64_t b=q_d->sizes[0],seq=q_d->sizes[1],hq=q_d->sizes[2],hk=k_d->sizes[2];
  int64_t d=q_d->sizes[3],half=d/2;
  for(int64_t row=0;row<b*seq;row++){int64_t token=row%seq;
    for(int64_t h=0;h<hq;h++)for(int64_t j=0;j<half;j++){
      int64_t base=(row*hq+h)*d,angle=(token*half+j)*2;
      _Float16 ae=q[base+2*j],ao=q[base+2*j+1],ch=freq[angle];
      float sf=(float)freq[angle+1]*sign;
      float low=(float)ae*(float)ch-(float)ao*sf;
      _Float16 rounded=(_Float16)(ao*ch);
      float high=(float)ae*sf+(float)rounded;
      qo[base+2*j]=(_Float16)low;qo[base+2*j+1]=(_Float16)high;}
    for(int64_t h=0;h<hk;h++)for(int64_t j=0;j<half;j++){
      int64_t base=(row*hk+h)*d,angle=(token*half+j)*2;
      _Float16 ae=k[base+2*j],ao=k[base+2*j+1],ch=freq[angle];
      float sf=(float)freq[angle+1]*sign;
      float low=(float)ae*(float)ch-(float)ao*sf;
      _Float16 rounded=(_Float16)(ao*ch);
      float high=(float)ae*sf+(float)rounded;
      ko[base+2*j]=(_Float16)low;ko[base+2*j+1]=(_Float16)high;}}
""",
    )


def _qwen2vl_mrope() -> str:
    return _source(
        "M4 *q_d, M4 *k_d, M4 *cos_d, M4 *sin_d, M4 *qout_d, M4 *kout_d",
        r"""
  _Float16 *q=DATA(q_d,_Float16),*k=DATA(k_d,_Float16),*c=DATA(cos_d,_Float16),*s=DATA(sin_d,_Float16);
  _Float16 *qo=DATA(qout_d,_Float16),*ko=DATA(kout_d,_Float16);
  int64_t b=q_d->sizes[0],seq=q_d->sizes[1],hq=q_d->sizes[2],hk=k_d->sizes[2],d=q_d->sizes[3],half=d/2;
  #pragma omp parallel for schedule(static)
  for(int64_t row=0;row<b*seq;row++){int64_t batch=row/seq,token=row%seq;
    for(int64_t j=0;j<half;j++){int64_t plane=j<16?0:(j<40?1:2);int64_t ci=((plane*b+batch)*seq+token)*d+j;
      float cv=(float)c[ci],sv=(float)s[ci];
      for(int64_t h=0;h<hq;h++){int64_t base=(row*hq+h)*d;float a=(float)q[base+j],z=(float)q[base+half+j];
        qo[base+j]=(_Float16)(a*cv-z*sv);qo[base+half+j]=(_Float16)(z*cv+a*sv);}
      for(int64_t h=0;h<hk;h++){int64_t base=(row*hk+h)*d;float a=(float)k[base+j],z=(float)k[base+half+j];
        ko[base+j]=(_Float16)(a*cv-z*sv);ko[base+half+j]=(_Float16)(z*cv+a*sv);}}}
""",
    )


def _flash_linear() -> str:
    return _source(
        "M2 *a_d, M2 *b_d, M1 *bias_d, M2 *out_d, M2 *act_d",
        r"""
  _Float16 *a=DATA(a_d,_Float16),*b=DATA(b_d,_Float16),*bias=DATA(bias_d,_Float16);
  _Float16 *out=DATA(out_d,_Float16),*act=DATA(act_d,_Float16);
  int64_t m=a_d->sizes[0],kdim=a_d->sizes[1],n=b_d->sizes[0];
  #pragma omp parallel for schedule(static)
  for(int64_t i=0;i<m;i++)for(int64_t j=0;j<n;j++){float acc=(float)bias[j];
    for(int64_t k=0;k<kdim;k++)acc+=(float)a[i*kdim+k]*(float)b[j*kdim+k];
    _Float16 value=(_Float16)acc;out[i*n+j]=value;act[i*n+j]=value;}
""",
    )


def _expert_xformers_rms_norm() -> str:
    return _source(
        "M2 *x_d, M1 *w_d, M0 *eps_d, M2 *y_d",
        r"""
  float *x=DATA(x_d,float),*w=DATA(w_d,float),eps=DATA(eps_d,float)[0],*y=DATA(y_d,float);
  int64_t m=x_d->sizes[0],n=x_d->sizes[1];
  for(int64_t i=0;i<m;i++){float sumsq=0.0f;int64_t j=0;
    while(j<n){size_t vl=__riscv_vsetvl_e32m1((size_t)(n-j));
      vfloat32m1_t value=__riscv_vle32_v_f32m1(x+i*n+j,vl);
      vfloat32m1_t square=__riscv_vfmul_vv_f32m1(value,value,vl);
      vfloat32m1_t seed=__riscv_vfmv_v_f_f32m1(0.0f,vl);
      vfloat32m1_t reduced=__riscv_vfredosum_vs_f32m1_f32m1(square,seed,vl);
      sumsq+=__riscv_vfmv_f_s_f32m1_f32(reduced);j+=(int64_t)vl;}
    float scale=1.0f/sqrtf(sumsq/(float)n+eps);j=0;
    while(j<n){size_t vl=__riscv_vsetvl_e32m1((size_t)(n-j));
      vfloat32m1_t value=__riscv_vle32_v_f32m1(x+i*n+j,vl);
      vfloat32m1_t weight=__riscv_vle32_v_f32m1(w+j,vl);
      value=__riscv_vfmul_vv_f32m1(value,weight,vl);
      value=__riscv_vfmul_vf_f32m1(value,scale,vl);
      __riscv_vse32_v_f32m1(y+i*n+j,value,vl);j+=(int64_t)vl;}
  }
""",
    )


def _expert_layer_norm() -> str:
    return _source(
        "M2 *x_d, M1 *w_d, M1 *b_d, M0 *n_d, M0 *eps_d, M2 *y_d, M1 *mean_d, M1 *rstd_d",
        r"""
  float *x=DATA(x_d,float),*w=DATA(w_d,float),*b=DATA(b_d,float),*y=DATA(y_d,float);
  float *mean=DATA(mean_d,float),*rstd=DATA(rstd_d,float);
  int32_t requested=DATA(n_d,int32_t)[0];float eps=DATA(eps_d,float)[0];
  int64_t rows=x_d->sizes[0],width=x_d->sizes[1],n=requested<width?requested:width;
  for(int64_t i=0;i<rows;i++){float sum=0.0f;int64_t j=0;
    while(j<n){size_t vl=__riscv_vsetvl_e32m1((size_t)(n-j));
      vfloat32m1_t value=__riscv_vle32_v_f32m1(x+i*width+j,vl);
      vfloat32m1_t seed=__riscv_vfmv_v_f_f32m1(0.0f,vl);
      vfloat32m1_t reduced=__riscv_vfredosum_vs_f32m1_f32m1(value,seed,vl);
      sum+=__riscv_vfmv_f_s_f32m1_f32(reduced);j+=(int64_t)vl;}
    float mu=sum/(float)n,var=0.0f;j=0;
    while(j<n){size_t vl=__riscv_vsetvl_e32m1((size_t)(n-j));
      vfloat32m1_t value=__riscv_vle32_v_f32m1(x+i*width+j,vl);
      value=__riscv_vfsub_vf_f32m1(value,mu,vl);
      vfloat32m1_t square=__riscv_vfmul_vv_f32m1(value,value,vl);
      vfloat32m1_t seed=__riscv_vfmv_v_f_f32m1(0.0f,vl);
      vfloat32m1_t reduced=__riscv_vfredosum_vs_f32m1_f32m1(square,seed,vl);
      var+=__riscv_vfmv_f_s_f32m1_f32(reduced);j+=(int64_t)vl;}
    float scale=1.0f/sqrtf(var/(float)n+eps);mean[i]=mu;rstd[i]=scale;j=0;
    while(j<n){size_t vl=__riscv_vsetvl_e32m1((size_t)(n-j));
      vfloat32m1_t value=__riscv_vle32_v_f32m1(x+i*width+j,vl);
      vfloat32m1_t weight=__riscv_vle32_v_f32m1(w+j,vl);
      vfloat32m1_t bias=__riscv_vle32_v_f32m1(b+j,vl);
      value=__riscv_vfsub_vf_f32m1(value,mu,vl);
      value=__riscv_vfmul_vf_f32m1(value,scale,vl);
      value=__riscv_vfmul_vv_f32m1(value,weight,vl);
      value=__riscv_vfadd_vv_f32m1(value,bias,vl);
      __riscv_vse32_v_f32m1(y+i*width+j,value,vl);j+=(int64_t)vl;}
  }
""",
    )


def _expert_flash_bwd_preprocess() -> str:
    return _source(
        "M4 *a_d, M4 *b_d, M3 *out_d",
        r"""
  _Float16 *a=DATA(a_d,_Float16),*b=DATA(b_d,_Float16);float *out=DATA(out_d,float);
  int64_t groups=a_d->sizes[0]*a_d->sizes[1],rows=a_d->sizes[2],width=a_d->sizes[3];
  int64_t padded=out_d->sizes[2];
  for(int64_t g=0;g<groups;g++){for(int64_t i=0;i<rows;i++){float sum=0.0f;int64_t j=0;
    int64_t row=g*rows+i;
    while(j<width){size_t vl=__riscv_vsetvl_e16m1((size_t)(width-j));
      vfloat16m1_t half_a=__riscv_vle16_v_f16m1(a+row*width+j,vl);
      vfloat16m1_t half_b=__riscv_vle16_v_f16m1(b+row*width+j,vl);
      vfloat32m2_t value_a=__riscv_vfwcvt_f_f_v_f32m2(half_a,vl);
      vfloat32m2_t value_b=__riscv_vfwcvt_f_f_v_f32m2(half_b,vl);
      vfloat32m2_t product=__riscv_vfmul_vv_f32m2(value_a,value_b,vl);
      vfloat32m1_t seed=__riscv_vfmv_v_f_f32m1(0.0f,vl);
      vfloat32m1_t reduced=__riscv_vfredosum_vs_f32m2_f32m1(product,seed,vl);
      sum+=__riscv_vfmv_f_s_f32m1_f32(reduced);j+=(int64_t)vl;}
    out[g*padded+i]=sum;}
    for(int64_t i=rows;i<padded;i++)out[g*padded+i]=0.0f;}
""",
    )


def _expert_tvd() -> str:
    return _source(
        "M2 *a_d, M2 *b_d, M1 *labels_d, M0 *scale_d, M2 *loss_d, M2 *grad_d",
        r"""
  float *a=DATA(a_d,float),*b=DATA(b_d,float),scale=DATA(scale_d,float)[0];
  float *loss=DATA(loss_d,float),*grad=DATA(grad_d,float);(void)labels_d;
  int64_t total=a_d->sizes[0]*a_d->sizes[1],k=0;
  while(k<total){size_t vl=__riscv_vsetvl_e32m1((size_t)(total-k));
    vfloat32m1_t va=__riscv_vle32_v_f32m1(a+k,vl);
    vfloat32m1_t vb=__riscv_vle32_v_f32m1(b+k,vl);
    vfloat32m1_t diff=__riscv_vfsub_vv_f32m1(va,vb,vl);
    vfloat32m1_t magnitude=__riscv_vfabs_v_f32m1(diff,vl);
    magnitude=__riscv_vfmul_vf_f32m1(magnitude,0.5f,vl);
    vbool32_t positive=__riscv_vmflt_vv_f32m1_b32(vb,va,vl);
    vfloat32m1_t negative_value=__riscv_vfmv_v_f_f32m1(-0.5f*scale,vl);
    vfloat32m1_t positive_value=__riscv_vfmv_v_f_f32m1(0.5f*scale,vl);
    vfloat32m1_t gradient=__riscv_vmerge_vvm_f32m1(
        negative_value,positive_value,positive,vl);
    __riscv_vse32_v_f32m1(loss+k,magnitude,vl);
    __riscv_vse32_v_f32m1(grad+k,gradient,vl);k+=(int64_t)vl;
  }
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
    "flash_attention_forward": _flash_attention_forward,
    "flash_bwd_preprocess": _flash_bwd_preprocess,
    "splitk_reduce_varargs": _splitk,
    "flash_layer_norm_1pass": _flash_layer_norm,
    "jsd": _jsd,
    "tvd": _tvd,
    "xformers_rms_norm": _xformers_rms_norm,
    "cross_entropy": _cross_entropy,
    "group_norm": _group_norm,
    "liger_rope": _liger_rope,
    "llama4_rope": _llama4_rope,
    "qwen2vl_mrope": _qwen2vl_mrope,
    "flash_linear": _flash_linear,
}

_EXPERT_SOURCES = {
    "flash_bwd_preprocess": _expert_flash_bwd_preprocess,
    "layer_norm": _expert_layer_norm,
    "tvd": _expert_tvd,
    "xformers_rms_norm": _expert_xformers_rms_norm,
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
        "implementation": "independent-case-authored-c-llvm-rvv-autovectorized",
    }


def _external_record(
    *,
    baseline_id: str,
    display_name: str,
    kind: str,
    source: str,
    implementation: str,
    build_profile: str,
    language: str = "c",
) -> dict[str, object]:
    return {
        "protocol": EXTERNAL_RVV_BASELINE_PROTOCOL,
        "baseline_id": baseline_id,
        "display_name": display_name,
        "kind": kind,
        "symbol": NATIVE_RVV_SYMBOL,
        "source": source,
        "language": language,
        "build_profile": build_profile,
        "implementation": implementation,
        "strict_match": True,
    }


def llvm_rvv_c_baseline_record(kernel: str) -> dict[str, object]:
    native = native_rvv_baseline_record(kernel)
    return _external_record(
        baseline_id=LLVM_RVV_C_BASELINE_ID,
        display_name="LLVM-RVV C",
        kind="portable_compiler",
        source=str(native["source"]),
        implementation=str(native["implementation"]),
        build_profile="clang-rvv-c-strict",
    )


def expert_rvv_baseline_record(kernel: str) -> dict[str, object]:
    try:
        source = _EXPERT_SOURCES[kernel]()
    except KeyError as exc:
        raise ValueError(f"no explicit-intrinsic RVV baseline for {kernel!r}") from exc
    return _external_record(
        baseline_id=EXPERT_RVV_BASELINE_ID,
        display_name="Expert RVV",
        kind="expert_intrinsic",
        source=source,
        implementation="independent-vlen-agnostic-rvv-intrinsics",
        build_profile="clang-rvv-intrinsic-strict",
    )


def external_rvv_baseline_records(kernel: str) -> tuple[dict[str, object], ...]:
    records = [llvm_rvv_c_baseline_record(kernel)]
    if kernel in _EXPERT_SOURCES:
        records.append(expert_rvv_baseline_record(kernel))
    return tuple(records)


__all__ = [
    "NATIVE_RVV_BASELINE_PROTOCOL",
    "NATIVE_RVV_BASELINE_SIDECAR_PROTOCOL",
    "NATIVE_RVV_SYMBOL",
    "EXTERNAL_RVV_BASELINE_PROTOCOL",
    "EXTERNAL_RVV_BASELINE_SIDECAR_PROTOCOL",
    "LLVM_RVV_C_BASELINE_ID",
    "EXPERT_RVV_BASELINE_ID",
    "expert_rvv_baseline_record",
    "external_rvv_baseline_records",
    "llvm_rvv_c_baseline_record",
    "native_rvv_baseline_record",
]
