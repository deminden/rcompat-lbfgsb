/* Original black-box probe of public BLAS DTRSV, upper N/T nonunit forms.
 * Compare dimensions 1..31 against mathematical unblocked fused solves.
 * No BLAS or optimizer implementation source is read or included.
 * Build with the reference compiler, -O3 -mcpu=native, and its BLAS dylib;
 * run with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1. */
#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
extern void dtrsv_(const char*,const char*,const char*,const int*,const double*,const int*,double*,const int*);
static uint64_t state;
static double sample(void){state=state*UINT64_C(6364136223846793005)+UINT64_C(1442695040888963407);return ((double)(state>>32)/2147483648.0-1.0);}
int main(void){int one=1;char u='U',diag='N';int failed=0,total=0;
for(int n=1;n<=31;n++)for(int seed=1;seed<=256;seed++)for(int form=0;form<2;form++){
 state=seed;double a[1024]={0},x[32],expected[32];
 for(int j=0;j<n;j++)for(int i=0;i<=j;i++)a[i+j*n]=i==j?2.0+sample():sample()*0.125;
 for(int i=0;i<n;i++)x[i]=expected[i]=sample()*10;
 char trans=form?'T':'N';dtrsv_(&u,&trans,&diag,&n,a,&n,x,&one);
 if(form){for(int i=0;i<n;i++){double sum=0;for(int j=0;j<i;j++)sum=fma(a[j+i*n],expected[j],sum);expected[i]=(expected[i]-sum)/a[i+i*n];}}
 else{for(int i=n-1;i>=0;i--){expected[i]/=a[i+i*n];for(int j=0;j<i;j++)expected[j]=fma(-expected[i],a[j+i*n],expected[j]);}}
 int mismatch=0;for(int i=0;i<n;i++)if(memcmp(&x[i],&expected[i],sizeof(double)))mismatch++;
 if(mismatch){if(failed<20)printf("mismatch n=%d seed=%d trans=%c coordinates=%d\n",n,seed,trans,mismatch);failed++;}total++;
}
printf("total=%d failed=%d\n",total,failed);return failed?1:0;}
