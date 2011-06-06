/*
 *  SMD.c
 *  
 *
 *  Created by Karen Hovsepian on 10/15/09.
 *  Copyright 2009 __MyCompanyName__. All rights reserved.
 *
 */
#include "DBN_Cluster.h"

double (*SMD_RBMCODEFUNCS[3])(zomplex *, double *, zomplex**, double **, int *, double *, int, double *, double *, double *, int, int* , double **, char) = {GRADIENT_HESSIAN_VEC_PROD_SOFTMAX, GRADIENT_HESSIAN_VEC_PROD_LOG, GRADIENT_HESSIAN_VEC_PROD_LIN};

double SMD(double **data, int numrows, double *weights, int *offset_weights, int *Dim, int numlayers, char typemiddle, char typetop, SMDparams * params)
{
	
	
	int numweights = offset_weights[numlayers-1];
	
	double *v = Malloc(double,numweights);
	double *nu = Malloc(double,numweights);
		
	int converged = 0;
	
	int i,j,epoch, batch;
	
//	for(i=0;i<numlayers;i++)
//		printf("%d\n",offset_weights[i]);
		
	int numdims = Dim[0];
	
	batchdata_str_flat * batchdata = createbatchflat(numrows,params->batchsizebp,numdims);
	double f;

	zomplex * compweights = Malloc(zomplex,numweights);
	
	int *randperms = Malloc(int,numrows);
	for(i=0;i<numrows;i++)
		randperms[i] = i;
	
	zomplex **all_activations = Malloc(zomplex *,numlayers);
	double  **real_all_activations = Malloc(double *,numlayers);
	
//	double **activationfuncs = Malloc(double *,numlayers-1);
	double **activationfuncs = Malloc(double *,numlayers-2);
	
	int N = batchdata->numcases;
	
	for(i=0;i<numlayers;i++)
	{
		int N1 = Dim[i]*N;
		all_activations[i] = Calloc(zomplex, (Dim[i]+1)*batchdata->numcases_last);
		zadd_ones(j,N1,N1+N,all_activations[i])

		if(i>0)
		{
			real_all_activations[i] = Malloc(double, (Dim[i]+1)*batchdata->numcases_last);
			real_add_ones(j,N1,N1+N,real_all_activations[i])
		}
	}
	
	for(i=0;i<numlayers-1;i++)
		activationfuncs[i] = Malloc(double, (Dim[i+1]+1)*batchdata->numcases_last);
//	activationfuncs[i] = Malloc(double, (Dim[i]+1)*batchdata->numcases_last);
	
	#pragma omp parallel for shared(numweights,v,nu,compweights,weights,params) private(i)  if(DO_OPENMP)
	for (i=0;i<numweights;i++)
	{
		compweights[i].real = weights[i];
		nu[i] = params->nu0;
		v[i] = 0;
	}
	
	double *dw = Malloc(double, offset_weights[numlayers-1]);
	double *Hv = Malloc(double, offset_weights[numlayers-1]);
	
	for(epoch=0;epoch<params->maxepoch;epoch++)
	{
		randomize_batchflat(data, batchdata, randperms);
		
		f = SMD_RBMCODEFUNCS[typemiddle](compweights, weights, all_activations, real_all_activations, Dim, batchdata->batchdata, numlayers, dw, Hv, v, N, offset_weights,activationfuncs, typetop);

		#pragma omp parallel for shared(compweights,weights,dw,nu,numweights) private(i)  if(DO_OPENMP)
		for (i=0;i<numweights;i++)
		{
			weights[i] -= dw[i]*nu[i];
			compweights[i].real = weights[i];
		}
		
		for (batch = 1;batchdata->numbatches-1;batch++)
		{
			converged = 1;
			#pragma omp parallel for shared(v,params,nu,dw, Hv,numweights) private(i)  if(DO_OPENMP)
			for (i=0;i<numweights;i++)
				v[i] = params->lambda*v[i] - nu[i]*(dw[i] + params->lambda*Hv[i]);
			
			f = SMD_RBMCODEFUNCS[typemiddle](compweights, weights, all_activations, real_all_activations, Dim, batchdata->batchdata+batch*N*(numdims+1),numlayers, dw, Hv, v, N, offset_weights,activationfuncs, typetop);
			
			double temp;
			#pragma omp parallel for shared(converged,v,dw, params, nu, compweights, numweights) private(i,temp)  if(DO_OPENMP)
			for (i=0;i<numweights;i++)
			{
				nu[i] *= max(0.5,1-params->mu*dw[i]*v[i]);
				temp = dw[i]*nu[i];
				if(fabs(temp) > params->errorthresh)
					converged = 0;

				weights[i] -= temp;
				compweights[i].real = weights[i];
			}

			if(converged) goto done;
		}

		for(i=0;i<numlayers;i++)
		{
			int N1 = Dim[i]*batchdata->numcases_last;
			zadd_ones(j,N1,N1+batchdata->numcases_last,all_activations[i])
			
			if(i>0)
				real_add_ones(j,N1,N1+batchdata->numcases_last,real_all_activations[i])
		}

		converged = 1;
		#pragma omp parallel for shared(v,params,nu,dw, numweights,Hv) private(i)  if(DO_OPENMP)
		for (i=0;i<numweights;i++)
			v[i] = params->lambda*v[i] - nu[i]*(dw[i] + params->lambda*Hv[i]);	

		f = SMD_RBMCODEFUNCS[typemiddle](compweights, weights, all_activations, real_all_activations, Dim, batchdata->batchdata+batch*N*(numdims+1),numlayers, dw, Hv, v,  batchdata->numcases_last, offset_weights,activationfuncs, typetop);

		double temp;
		#pragma omp parallel for shared(converged,v,dw, params, nu, compweights, numweights) private(i,temp)  if(DO_OPENMP)
		for (i=0;i<numweights;i++)
		{
			nu[i] *= max(0.5,1-params->mu*dw[i]*v[i]);
			temp = dw[i]*nu[i];
			if(fabs(temp) > params->errorthresh)
				converged = 0;
			
			compweights[i].real -= temp;
		}
		
		if(converged) goto done;
	}
	
done:
	
	deletebatchdata_flat(batchdata);
	printf("Optimization finished\n");
	return f;
}


