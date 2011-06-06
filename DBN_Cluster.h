/*
 *  DBN_Cluster.h
 *  
 *
 *  Created by Karen Hovsepian on 9/28/09.
 *  Copyright 2009 __MyCompanyName__. All rights reserved.
 *
 */

#undef _GLIBCXX_DEBUG

#define DO_OPENMP 0

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include <Accelerate/Accelerate.h>

#include <ctype.h>
#include <unistd.h>
#include <omp.h>

#define NUM_THREADS 2
#define CHUNK 1

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define Calloc(type,n) (type *)calloc(n,sizeof(type))
#define Realloc(ptr,type,n) (type *) realloc(ptr, sizeof(type)*(n));

#define MemCopy(dest,source,type,n)	memcpy(dest,source,sizeof(type)*(n))
#define swap(a,b,c) c t; t=a; a=b; b=t;
#define max(a,b) (a>b ? a : b)

#define weights(I,J,N) weights[(I)+(J)*(N)]

#define weights_decode(I,J,N) weights_decode[(I)+(J)*(N)]
#define weights_encode(I,J,N) weights_encode[(I)+(J)*(N)]
#define vishidinc(I,J,N) vishidinc[(I)+(J)*(N)]

#define CDiv_weight_diff(I,J,K,N,M) CDiv_weight_diff[(K)*(M)*(N)+(J)+(I)*(M)]

#define CDiv_hidbias_diff(I,J,N) CDiv_hidbias_diff[(J)+(I)*(N)]
#define CDiv_visbias_diff(I,J,N) CDiv_visbias_diff[(J)+(I)*(N)]

#define negdata(I,J,N) negdata[(J)+(I)*(N)]
#define poshidprobs(I,J,N) poshidprobs[(J)+(I)*(N)]
#define poshidstates(I,J,N) poshidstates[(J)+(I)*(N)]

#define neghidprobs(I,J,N) neghidprobs[(J)+(I)*(N)]

#define EPSILON_LG 0.01   // Learning rate for weights 
#define EPSILON_GA 0.000005   // Learning rate for weights 
#define INITIAL_MOMENT 0.5
#define FINAL_MOMENT 0.9
#define WEIGHTCOST 0.00002
#define T 1
#define RBMERROR 1e-6

#define SOFTMAX 0
#define LOGISTIC 1
#define GAUSSIAN 2

#define NILPOT 10e-300
#define LARGEVAL 100

#define LOSS_CROSSENTR 0
#define LOSS_SQUAREDERR 1
#define LOSS_MULTICROSSENTR 2

#define real_add_ones(i,start,end,mat)   	for(i=(start);i<(end);i++){mat[i] = 1;}
#define add_ones(i,start,end,complexmat)   	for(i=(start);i<(end);i+=2){complexmat[i] = 1;}//	complexmat[i+1] = 0;}
#define zadd_ones(i,start,end,complexmat)   	for(i=(start);i<(end);i++){complexmat[i].real = 1;}//	complexmat[i].imag = 0;}

typedef struct batchdatastruct
{
	double **batchdata;
	int numcases;
	int numcases_last;
	int numbatches;
	int numdims;
	int numrows;
} batchdata_str;

typedef struct batchdatastruct_flat
{
	double *batchdata;
	int numcases;
	int numcases_last;
	int numbatches;
	int numdims;
	int numrows;

} batchdata_str_flat;


typedef struct zomplexstruct
{
	double real;
	double imag;
} zomplex;   //the use of prefix z is borrowed from BLAS to denote double precision


typedef struct params_SMD
{
	int maxepoch;
	double errorthresh;
	double lambda;
	double mu;
	double nu0;
	int batchsizebp;
} SMDparams;


typedef struct datastruct
{
	double **data;
	double *targets;
	int numdims;
	int numrows;
} data_str;


typedef struct weights_data
{
	double *weights;
	int numweights;
} weights_str;

extern weights_str * DBN_deepauto(double **datatrain, int numrows, int maxepoch, int numlayers, int *numunits, int batchsizepre, char * typesoflayers, SMDparams *backpropparams);
extern weights_str * DBN_classify(double **datatrain, double **datatest, int *targetstrain, int *targetstest, int numrows, int maxepoch, int numlayers, int *numunits, int batchsizepre, char * typesoflayers, SMDparams *backpropparams);
extern weights_str * DBN_regress(double **datatrain, double **datatest, double *targetstrain, double *targetstest, int numrows, int maxepoch, int numlayers, int *numunits, int batchsizepre, char * typesoflayers, SMDparams *backpropparams);

extern double SMD(double **data, int numrows, double *weights, int *offset_weights, int *Dim, int numlayers, char typemiddle, char typetop, SMDparams * params);

int * randperm(int size);

batchdata_str * createbatch(double **data, int numrows, int batchsize,int numdims);
void randomize_batch(batchdata_str * batchdata);

extern batchdata_str_flat * createbatchflat(int numrows, int batchsize, int numdim);
extern void randomize_batchflat(double **data,  batchdata_str_flat * batchdata,int *randperms);

void deletebatchdata(batchdata_str * batchdata);
extern void deletebatchdata_flat(batchdata_str_flat * batchdata);

void rbm_autoencoder(batchdata_str * batchdata_rbm, int bottomlayer, int toplayer, double *weights_encode, double *weights_decode,int numvis, int numhid,char typevis,char typehid,int maxepoch);
void rbm_class_regress(batchdata_str * batchdata_rbm, int bottomlayer, int toplayer, double *weights,int numvis, int numhid,char typevis,char typehid,int maxepoch);

double *randn(int size, double weight);
void RBM_CORECODE_SM(double **data, double *poshidprobs, double *poshidstates, double *negdata, double *neghidprobs, double *temparrayforthreads_vis, double *temparrayforthreads_hid,  double *weights, double *hidbiases, double *visbiases, double *CDiv_weight_diff, double *CDiv_hidbias_diff, double *CDiv_visbias_diff, int numhid, int numvis, int typevis, int numcases);
void RBM_CORECODE_LG(double **data, double *poshidprobs, double *poshidstates, double *negdata, double *neghidprobs, double *temparrayforthreads_vis, double *temparrayforthreads_hid,  double *weights, double *hidbiases, double *visbiases, double *CDiv_weight_diff, double *CDiv_hidbias_diff, double *CDiv_visbias_diff, int numhid, int numvis, int typevis, int numcases);
void RBM_CORECODE_GA(double **data, double *poshidprobs, double *poshidstates, double *negdata, double *neghidprobs, double *temparrayforthreads_vis, double *temparrayforthreads_hid,  double *weights, double *hidbiases, double *visbiases, double *CDiv_weight_diff, double *CDiv_hidbias_diff, double *CDiv_visbias_diff, int numhid, int numvis, int typevis, int numcases);

extern double GRADIENT_HESSIAN_VEC_PROD_LOG(zomplex *weights, double *realweights, zomplex**all_activations, double **real_all_activations, int *Dim, double *data, int numlayers, double *dw, double *Hv, double *v, int N, int* offsets_weights, double **activationfuncs, char typetop);
extern double GRADIENT_HESSIAN_VEC_PROD_LIN(zomplex *weights, double *realweights, zomplex**all_activations, double **real_all_activations, int *Dim, double *data, int numlayers, double *dw, double *Hv, double *v, int N, int* offsets_weights, double **activationfuncs, char typetop);
extern double GRADIENT_HESSIAN_VEC_PROD_SOFTMAX(zomplex *weights, double *realweights, zomplex**all_activations, double **real_all_activations, int *Dim, double *data, int numlayers, double *dw, double *Hv, double *v, int N, int* offsets_weights, double **activationfuncs, char typetop);

