 /*
 *  CG_ENCODE_CLUST.c
 *  
 *
 *  Created by Karen Hovsepian on 10/10/09.
 *  Copyright 2009 __MyCompanyName__. All rights reserved.
 *
 */
#include "DBN_Cluster.h"



double GRADIENT_HESSIAN_VEC_PROD_LOG(zomplex *weights, double *realweights, zomplex**all_activations, double **real_all_activations, int *Dim, double *data, int numlayers, double *dw, double *Hv, double *v, int N, int* offset_weights, double **activationfuncs, char typetop)
{
	int i,j,k,l;
	zomplex alpha = {1.0,0.0};
	zomplex beta = {0.0,0.0};
	
	double *temparr;
	double *temparr2;
	double *temparr3;
	double temp;
	
	l = N*Dim[0];
	
	
//	cblas_dcopy(l,data,1,real_all_activations[0],1);

	real_all_activations[0] = data;
	real_add_ones(j,l,l+N,real_all_activations[0])

	cblas_dcopy(l,data,1,(double *) all_activations[0],2);

/*	
	#pragma omp parallel for shared(all_activations,data,l) private(i) num_threads(NUM_THREADS) schedule(dynamic,CHUNK) if(DO_OPENMP)
	for(i=0;i<l;i++)
	{
		all_activations[0][i].real = data[i];
		real_all_activations[0][i] = data[i];
		//don't need to run the next line because when we used Calloc, the space was allocated with zeros
		//		all_activations[0][i].imag = 0;
	}
*/
//	zadd_ones(j,l,l+N,all_activations[0])
//	real_add_ones(j,l,l+N,real_all_activations[0])
				  
	#pragma omp parallel for shared(weights,v,offset_weights,numlayers) private(i) num_threads(NUM_THREADS) schedule(dynamic,CHUNK) if(DO_OPENMP)
	for (i=0;i<offset_weights[numlayers-1];i++)
		weights[i].imag = v[i]*NILPOT;
	
	//%ENCODING PART
	
	for (i=1;i<(numlayers-1)/2;i++)
	{
//		cblas_zgemm(CblasColMajor,CblasNoTrans, CblasNoTrans, N, Dim[i-1]+1, Dim[i], &alpha, all_activations[i-1], N, weights + offset_weights[i-1], Dim[i-1]+1, &beta,all_activations[i], N);
		cblas_zgemm(CblasColMajor,CblasNoTrans, CblasNoTrans, N, Dim[i],Dim[i-1]+1, &alpha, all_activations[i-1], N, weights + offset_weights[i-1], Dim[i-1]+1, &beta,all_activations[i], N);
		
		temparr = (double *)all_activations[i];
		temparr2 = activationfuncs[i-1];
		temparr3 = real_all_activations[i];
		
		l = Dim[i]*N*2;
		#pragma omp parallel for shared(temparr,temparr2,temparr3,l) private(k) num_threads(NUM_THREADS) schedule(dynamic,CHUNK) if(DO_OPENMP)
		for(k=0;k<l;k+=2)
		{
			temparr[k] = 1.0/(1+exp(-temparr[k]));
			temparr3[k >> 1] = temparr[k];
			temparr2[k >> 1] = temparr[k]*(1-temparr[k]);
			temparr[k+1] *= temparr2[k>>1];
		}
//		add_ones(j,l,l+N*2,temparr)
//		l = l >> 1;
//		real_add_ones(j,l,l+N,temparr3)
	}
	
	
	//MIDDLE BOTTLENECK LAYER (LOGISTIC)
	i = (numlayers-1)/2;
//	cblas_zgemm(CblasColMajor,CblasNoTrans, CblasNoTrans, N, Dim[i-1]+1, Dim[i], &alpha, all_activations[i-1], N, weights + offset_weights[i-1], Dim[i-1]+1, &beta,all_activations[i], N);
	cblas_zgemm(CblasColMajor,CblasNoTrans, CblasNoTrans, N, Dim[i],Dim[i-1]+1,&alpha, all_activations[i-1], N, weights + offset_weights[i-1], Dim[i-1]+1, &beta,all_activations[i], N);
	
	temparr = (double *)all_activations[i];
	temparr2 = activationfuncs[i-1];
	temparr3 = real_all_activations[i];
	
	l = Dim[i]*N*2;
	#pragma omp parallel for shared(temparr,temparr2,l) private(k) num_threads(NUM_THREADS) schedule(dynamic,CHUNK) if(DO_OPENMP)
	for(k=0;k<l;k+=2)
	{
		temparr[k] = 1.0/(1+exp(-temparr[k]));
		temparr2[k >> 1] = temparr[k]*(1-temparr[k]);
		temparr[k+1] *= temparr2[k>>1];
		temparr3[k >> 1] = temparr[k];
	}
//	add_ones(j,l,l+N*2,temparr)
//	l = l >> 1;
//	real_add_ones(j,l,l+N,temparr3)	

	
	//%DECODING PART
	for (i=(numlayers-1)/2+1;i<numlayers-1;i++)	
	{
//		cblas_zgemm(CblasColMajor,CblasNoTrans, CblasNoTrans, N, Dim[i-1]+1, Dim[i], &alpha, all_activations[i-1], N, weights + offset_weights[i-1], Dim[i-1]+1, &beta,all_activations[i], N);
		cblas_zgemm(CblasColMajor,CblasNoTrans, CblasNoTrans, N, Dim[i],Dim[i-1]+1, &alpha, all_activations[i-1], N, weights + offset_weights[i-1], Dim[i-1]+1, &beta,all_activations[i], N);

		temparr = (double *)all_activations[i];
		temparr2 = activationfuncs[i-1];
		temparr3 = real_all_activations[i];
		
		l = Dim[i]*N*2;
		#pragma omp parallel for shared(temparr,temparr2,temparr3,l) private(k) num_threads(NUM_THREADS) schedule(dynamic,CHUNK) if(DO_OPENMP)
		for(k=0;k<l;k+=2)
		{
			temparr[k] = 1.0/(1+exp(-temparr[k]));
			temparr2[k >> 1] = temparr[k]*(1-temparr[k]);
			temparr[k+1] *= temparr2[k>>1];
			temparr3[k >> 1] = temparr[k];
		}
//		add_ones(j,l,l+N*2,temparr)
//		l = l >> 1;
//		real_add_ones(j,l,l+N,temparr3)	
	}
	
	double f = 0;
	printf("%d\n",N*Dim[numlayers-1]);
	double *Ixk = Malloc(double,N*Dim[numlayers-1]);
	double *Ixk_Hessian = Malloc(double,N*Dim[numlayers-1]);
	
	i = numlayers-1;

	
	//TOP LAYER COMPUTATION AND BACKPROP (NOTE THAT TOP LAYER ACTIVATION FUNCTION AND LOSS FUNCTION ARE MATCHING)
	
//	cblas_zgemm(CblasColMajor,CblasNoTrans, CblasNoTrans, N, Dim[i-1]+1, Dim[i], &alpha, all_activations[i-1], N, weights + offset_weights[i-1], Dim[i-1]+1, &beta,all_activations[i], N);
	cblas_zgemm(CblasColMajor,CblasNoTrans, CblasNoTrans, N, Dim[i],Dim[i-1]+1, &alpha, all_activations[i-1], N, weights + offset_weights[i-1], Dim[i-1]+1, &beta,all_activations[i], N);
	switch (typetop) {
		case LOGISTIC:	
			temparr = (double *)all_activations[i];
			
			l = Dim[i]*N*2;
			#pragma omp parallel for shared(temparr,l) private(k) num_threads(NUM_THREADS) schedule(dynamic,CHUNK) if(DO_OPENMP)
			for(k=0;k<l;k+=2)
			{
				temparr[k] = 1.0/(1+exp(-temparr[k]));
				temparr[k+1] *= temparr2[k>>1];
			}
			
			#pragma omp parallel for shared(all_activations,N,Dim,i,Ixk,Ixk_Hessian) private(k,temp) num_threads(NUM_THREADS) reduction(+:f) schedule(dynamic,CHUNK)	 if(DO_OPENMP)
			for(k=0;k<Dim[0]*N;k++)
			{
				temp = all_activations[i][k].real;
				
				f += data[k]*log(temp) + (1-data[k])*log(1-temp);
				
				Ixk[k] = (temp - data[k]);
				Ixk_Hessian[k] = all_activations[i][k].imag/NILPOT;
			}
			f /= -N;
			printf("%e\n",f);

			break;
		case GAUSSIAN:
			#pragma omp parallel for shared(all_activations,N,Dim,i,Ixk,Ixk_Hessian) private(k,temp) num_threads(NUM_THREADS) reduction(+:f) schedule(dynamic,CHUNK)	 if(DO_OPENMP)
			for(k=0;k<Dim[0]*N;k++)
			{
				temp = all_activations[i][k].real - data[k];
				f += temp*temp;
				Ixk[k] = temp;
				Ixk_Hessian[k] = all_activations[i][k].imag/NILPOT;
			}
			f /= 2*N;
			printf("%e\n",f);
			break;
		default:
			temparr = (double *)all_activations[i];
			
			l = Dim[i]*N*2;
			#pragma omp parallel for shared(temparr,l) private(k) num_threads(NUM_THREADS) schedule(dynamic,CHUNK) if(DO_OPENMP)
			for(k=0;k<l;k+=2)
				temparr[k] = exp(LARGEVAL*temparr[k]);
			
			double sumnums = 0;
			double sumnums1 = 0;
			
			#pragma omp parallel for firstprivate(sumnums,sumnums1) shared(N,Dim,i,all_activations) private(j,k) if(DO_OPENMP)
			for (j=0;j<N;j++)
			{
				for(k=0;k<Dim[i];k++)
					sumnums += all_activations[i][j+k*N].real;
				
				for(k=0;k<Dim[i];k++)
				{
					all_activations[i][j+k*N].real /= sumnums;
					sumnums1 += all_activations[i][j+k*N].real*all_activations[i][j+k*N].imag;
				}
				for(k=0;k<Dim[i];k++)
					all_activations[i][j+k*N].imag = LARGEVAL*all_activations[i][j+k*N].real*(all_activations[i][j+k*N].imag - sumnums1);
			}
			
			#pragma omp parallel for shared(all_activations,N,Dim,i,Ixk,Ixk_Hessian) private(k,temp) num_threads(NUM_THREADS) reduction(+:f) schedule(dynamic,CHUNK)	 if(DO_OPENMP)
			for(k=0;k<Dim[0]*N;k++)
			{
				temp = all_activations[i][k].real;
				
				f += data[k]*log(temp);
				Ixk[k] = LARGEVAL*(temp - data[k]);
				Ixk_Hessian[k] = LARGEVAL*all_activations[i][k].imag/NILPOT;
			}
			f /= -N;
			printf("%e\n",f);
			break;
	} 

	i = numlayers-2;
	
	cblas_dgemm(CblasColMajor,CblasTrans, CblasNoTrans, Dim[numlayers-2]+1, Dim[numlayers-1], N, 1.0/N, real_all_activations[numlayers-2], N, Ixk,N,0,dw + offset_weights[numlayers-2], Dim[numlayers-2]+1);
	cblas_dgemm(CblasColMajor,CblasTrans, CblasNoTrans, Dim[numlayers-2]+1, Dim[numlayers-1], N, 1.0/N, real_all_activations[numlayers-2], N, Ixk_Hessian,N,0,Hv + offset_weights[numlayers-2], Dim[numlayers-2]+1);
	
	//DECODING PART BACKPROP
	for (i=numlayers-3;i>=(numlayers-1)/2;i--)
	{
		cblas_dgemm(CblasColMajor,CblasNoTrans, CblasTrans, N, Dim[i+1], Dim[i+2], 1, Ixk, N, realweights+offset_weights[i+1],Dim[i+1],0,temparr, N);
		cblas_dgemm(CblasColMajor,CblasNoTrans, CblasTrans, N, Dim[i+1], Dim[i+2], 1, Ixk_Hessian, N, realweights+offset_weights[i+1],Dim[i+1],0,temparr2, N);
		
		#pragma omp parallel for shared(temparr,all_activations,N,Dim,i) private(j,k) num_threads(NUM_THREADS) schedule(dynamic,CHUNK) if(DO_OPENMP)
		for (j=0;j<Dim[i+1]*N;j++)
		{
			temparr[j] *= activationfuncs[i][j];
			temparr2[j] *= activationfuncs[i][j];
		}
				
		free(Ixk);
		Ixk = temparr;
		free(Ixk_Hessian);
		Ixk_Hessian = temparr2;
		
		cblas_dgemm(CblasColMajor,CblasTrans, CblasNoTrans, Dim[i]+1, Dim[i+1], N, 1.0/N, real_all_activations[i], N, Ixk,N,0,dw + offset_weights[i], Dim[i]+1);
		cblas_dgemm(CblasColMajor,CblasTrans, CblasNoTrans, Dim[i]+1, Dim[i+1], N, 1.0/N, real_all_activations[i], N, Ixk_Hessian,N,0,Hv + offset_weights[i], Dim[i]+1);
	}

	//MIDDLE BOTTLENECK LAYER BACKPROP
	cblas_dgemm(CblasColMajor,CblasNoTrans, CblasTrans, N, Dim[i+1], Dim[i+2], 1, Ixk, N, realweights+offset_weights[i+1],Dim[i+1],0,temparr, N);
	cblas_dgemm(CblasColMajor,CblasNoTrans, CblasTrans, N, Dim[i+1], Dim[i+2], 1, Ixk_Hessian, N, realweights+offset_weights[i+1],Dim[i+1],0,temparr2, N);
		
	#pragma omp parallel for shared(temparr,all_activations,N,Dim,i) private(j,k) num_threads(NUM_THREADS) schedule(dynamic,CHUNK) if(DO_OPENMP)
	for (j=0;j<Dim[i+1]*N;j++)
	{
		temparr[j] *= activationfuncs[i][j];
		temparr2[j] *= activationfuncs[i][j];
	}
		
	free(Ixk);
	Ixk = temparr;
	free(Ixk_Hessian);
	Ixk_Hessian = temparr2;
		
	cblas_dgemm(CblasColMajor,CblasTrans, CblasNoTrans, Dim[i]+1, Dim[i+1], N, 1.0/N, real_all_activations[i], N, Ixk,N,0,dw + offset_weights[i], Dim[i]+1);
	cblas_dgemm(CblasColMajor,CblasTrans, CblasNoTrans, Dim[i]+1, Dim[i+1], N, 1.0/N, real_all_activations[i], N, Ixk_Hessian,N,0,Hv + offset_weights[i], Dim[i]+1);

	//BOTTOM ENCODING PART BACKPROP
	for (i=(numlayers-5)/2;i>=0;i--)
	{
		cblas_dgemm(CblasColMajor,CblasNoTrans, CblasTrans, N, Dim[i+1], Dim[i+2], 1, Ixk, N, realweights+offset_weights[i+1],Dim[i+1],0,temparr, N);
		cblas_dgemm(CblasColMajor,CblasNoTrans, CblasTrans, N, Dim[i+1], Dim[i+2], 1, Ixk_Hessian, N, realweights+offset_weights[i+1],Dim[i+1],0,temparr2, N);
		
		#pragma omp parallel for shared(temparr,all_activations,N,Dim,i) private(j,k) num_threads(NUM_THREADS) schedule(dynamic,CHUNK) if(DO_OPENMP)
		for (j=0;j<Dim[i+1]*N;j++)
		{
			temparr[j] *= activationfuncs[i][j];
			temparr2[j] *= activationfuncs[i][j];
		}
		
		free(Ixk);
		Ixk = temparr;
		free(Ixk_Hessian);
		Ixk_Hessian = temparr2;
		
		cblas_dgemm(CblasColMajor,CblasTrans, CblasNoTrans, Dim[i]+1, Dim[i+1], N, 1.0/N, real_all_activations[i], N, Ixk,N,0,dw + offset_weights[i], Dim[i]+1);
		cblas_dgemm(CblasColMajor,CblasTrans, CblasNoTrans, Dim[i]+1, Dim[i+1], N, 1.0/N, real_all_activations[i], N, Ixk_Hessian,N,0,Hv + offset_weights[i], Dim[i]+1);
	}
	
	
	return f;
}


double GRADIENT_HESSIAN_VEC_PROD_LIN(zomplex *weights, double *realweights, zomplex**all_activations, double **real_all_activations, int *Dim, double *data, int numlayers, double *dw, double *Hv, double *v, int N, int* offset_weights, double **activationfuncs, char typetop)
{
	int i,j,k,l;
	zomplex alpha = {1.0,0.0};
	zomplex beta = {0.0,0.0};
	
	double *temparr;
	double *temparr2;
	double *temparr3;
	double temp;
	
	l = N*Dim[0];

	real_all_activations[0] = data;
	real_add_ones(j,l,l+N,real_all_activations[0])
	
	cblas_dcopy(l,data,1,(double *) all_activations[0],2);
/*	
	
#pragma omp parallel for shared(all_activations,data,l) private(i) num_threads(NUM_THREADS) schedule(dynamic,CHUNK) if(DO_OPENMP)
	for(i=0;i<l;i++)
	{
		all_activations[0][i].real = data[i];
		real_all_activations[0][i] = data[i];
		//don't need to run the next line because when we used Calloc, the space was allocated with zeros
		//		all_activations[0][i].imag = 0;
	}
//	zadd_ones(j,l,l+N,all_activations[0])
//	real_add_ones(j,l,l+N,real_all_activations[0])
*/
	
#pragma omp parallel for shared(weights,v,offset_weights,numlayers) private(i) num_threads(NUM_THREADS) schedule(dynamic,CHUNK) if(DO_OPENMP)
	for (i=0;i<offset_weights[numlayers-1];i++)
		weights[i].imag = v[i]*NILPOT;
	
	//%ENCODING PART
	
	for (i=1;i<(numlayers-1)/2;i++)
	{
//		cblas_zgemm(CblasColMajor,CblasNoTrans, CblasNoTrans, N, Dim[i-1]+1, Dim[i], &alpha, all_activations[i-1], N, weights + offset_weights[i-1], Dim[i-1]+1, &beta,all_activations[i], N);
		cblas_zgemm(CblasColMajor,CblasNoTrans, CblasNoTrans, N, Dim[i],Dim[i-1]+1, &alpha, all_activations[i-1], N, weights + offset_weights[i-1], Dim[i-1]+1, &beta,all_activations[i], N);
		
		temparr = (double *)all_activations[i];
		temparr2 = activationfuncs[i-1];
		temparr3 = real_all_activations[i];
		
		l = Dim[i]*N*2;
#pragma omp parallel for shared(temparr,temparr2,temparr3,l) private(k) num_threads(NUM_THREADS) schedule(dynamic,CHUNK) if(DO_OPENMP)
		for(k=0;k<l;k+=2)
		{
			temparr[k] = 1.0/(1+exp(-temparr[k]));
			temparr3[k >> 1] = temparr[k];
			temparr2[k >> 1] = temparr[k]*(1-temparr[k]);
			temparr[k+1] *= temparr2[k>>1];
		}
//		add_ones(j,l,l+N*2,temparr)
//		l = l >> 1;
//		real_add_ones(j,l,l+N,temparr3)	
	}
	
	
	//MIDDLE BOTTLENECK LAYER (LINEAR)
	i = (numlayers-1)/2;
//	cblas_zgemm(CblasColMajor,CblasNoTrans, CblasNoTrans, N, Dim[i-1]+1, Dim[i], &alpha, all_activations[i-1], N, weights + offset_weights[i-1], Dim[i-1]+1, &beta,all_activations[i], N);
	cblas_zgemm(CblasColMajor,CblasNoTrans, CblasNoTrans, N, Dim[i],Dim[i-1]+1, &alpha, all_activations[i-1], N, weights + offset_weights[i-1], Dim[i-1]+1, &beta,all_activations[i], N);
	
	l = Dim[i]*N;
	#pragma omp parallel for shared(temparr,temparr2,l) private(k) num_threads(NUM_THREADS) schedule(dynamic,CHUNK) if(DO_OPENMP)
	for(k=0;k<l;k++)
		real_all_activations[i][k] = all_activations[i][k].real;

//	zadd_ones(j,l,l+N,all_activations[i])
//	real_add_ones(j,l,l+N,real_all_activations[i])	
	
	
	//%DECODING PART
	for (i=(numlayers-1)/2+1;i<numlayers-1;i++)
	{
//		cblas_zgemm(CblasColMajor,CblasNoTrans, CblasNoTrans, N, Dim[i-1]+1, Dim[i], &alpha, all_activations[i-1], N, weights + offset_weights[i-1], Dim[i-1]+1, &beta,all_activations[i], N);
		cblas_zgemm(CblasColMajor,CblasNoTrans, CblasNoTrans, N, Dim[i],Dim[i-1]+1, &alpha, all_activations[i-1], N, weights + offset_weights[i-1], Dim[i-1]+1, &beta,all_activations[i], N);
		
		temparr = (double *)all_activations[i];
		temparr2 = activationfuncs[i-1];
		temparr3 = real_all_activations[i];
		
		l = Dim[i]*N*2;
#pragma omp parallel for shared(temparr,temparr2,temparr3,l) private(k) num_threads(NUM_THREADS) schedule(dynamic,CHUNK) if(DO_OPENMP)
		for(k=0;k<l;k+=2)
		{
			temparr[k] = 1.0/(1+exp(-temparr[k]));
			temparr2[k >> 1] = temparr[k]*(1-temparr[k]);
			temparr[k+1] *= temparr2[k>>1];
			temparr3[k >> 1] = temparr[k];
		}
//		add_ones(j,l,l+N*2,temparr)
//		l = l >> 1;
//		real_add_ones(j,l,l+N,temparr3)	
	}
	
	double f = 0;
	double *Ixk = Malloc(double,N*Dim[numlayers-1]);
	double *Ixk_Hessian = Malloc(double,N*Dim[numlayers-1]);
	
	i = numlayers-1;
	
	
	//TOP LAYER COMPUTATION AND BACKPROP (NOTE THAT TOP LAYER ACTIVATION FUNCTION AND LOSS FUNCTION ARE MATCHING)
	
//	cblas_zgemm(CblasColMajor,CblasNoTrans, CblasNoTrans, N, Dim[i-1]+1, Dim[i], &alpha, all_activations[i-1], N, weights + offset_weights[i-1], Dim[i-1]+1, &beta,all_activations[i], N);
	cblas_zgemm(CblasColMajor,CblasNoTrans, CblasNoTrans, N, Dim[i],Dim[i-1]+1, &alpha, all_activations[i-1], N, weights + offset_weights[i-1], Dim[i-1]+1, &beta,all_activations[i], N);
	switch (typetop) {
		case LOGISTIC:	
			temparr = (double *)all_activations[i];
			
			l = Dim[i]*N*2;
#pragma omp parallel for shared(temparr,l) private(k) num_threads(NUM_THREADS) schedule(dynamic,CHUNK) if(DO_OPENMP)
			for(k=0;k<l;k+=2)
			{
				temparr[k] = 1.0/(1+exp(-temparr[k]));
				temparr[k+1] *= temparr2[k>>1];
			}
			
#pragma omp parallel for shared(all_activations,N,Dim,i,Ixk,Ixk_Hessian) private(k,temp) num_threads(NUM_THREADS) reduction(+:f) schedule(dynamic,CHUNK)	 if(DO_OPENMP)
			for(k=0;k<Dim[0]*N;k++)
			{
				temp = all_activations[i][k].real;
				
				f += data[k]*log(temp) + (1-data[k])*log(1-temp);
				
				Ixk[k] = (temp - data[k]);
				Ixk_Hessian[k] = all_activations[i][k].imag/NILPOT;
			}
			f /= -N;
			printf("%e\n",f);
			
			break;
		case GAUSSIAN:
#pragma omp parallel for shared(all_activations,N,Dim,i,Ixk,Ixk_Hessian) private(k,temp) num_threads(NUM_THREADS) reduction(+:f) schedule(dynamic,CHUNK)	 if(DO_OPENMP)
			for(k=0;k<Dim[0]*N;k++)
			{
				temp = all_activations[i][k].real - data[k];
				f += temp*temp;
				Ixk[k] = temp;
				Ixk_Hessian[k] = all_activations[i][k].imag/NILPOT;
			}
			f /= 2*N;
			printf("%e\n",f);
			break;
		default:
			temparr = (double *)all_activations[i];
			
			l = Dim[i]*N*2;
#pragma omp parallel for shared(temparr,l) private(k) num_threads(NUM_THREADS) schedule(dynamic,CHUNK) if(DO_OPENMP)
			for(k=0;k<l;k+=2)
				temparr[k] = exp(LARGEVAL*temparr[k]);
			
			double sumnums = 0;
			double sumnums1 = 0;
			
#pragma omp parallel for firstprivate(sumnums,sumnums1) shared(N,Dim,i,all_activations) private(j,k) if(DO_OPENMP)
			for (j=0;j<N;j++)
			{
				for(k=0;k<Dim[i];k++)
					sumnums += all_activations[i][j+k*N].real;
				
				for(k=0;k<Dim[i];k++)
				{
					all_activations[i][j+k*N].real /= sumnums;
					sumnums1 += all_activations[i][j+k*N].real*all_activations[i][j+k*N].imag;
				}
				for(k=0;k<Dim[i];k++)
					all_activations[i][j+k*N].imag = LARGEVAL*all_activations[i][j+k*N].real*(all_activations[i][j+k*N].imag - sumnums1);
			}
			
#pragma omp parallel for shared(all_activations,N,Dim,i,Ixk,Ixk_Hessian) private(k,temp) num_threads(NUM_THREADS) reduction(+:f) schedule(dynamic,CHUNK)	 if(DO_OPENMP)
			for(k=0;k<Dim[0]*N;k++)
			{
				temp = all_activations[i][k].real;
				
				f += data[k]*log(temp);
				Ixk[k] = LARGEVAL*(temp - data[k]);
				Ixk_Hessian[k] = LARGEVAL*all_activations[i][k].imag/NILPOT;
			}
			f /= -N;
			printf("%e\n",f);
			break;
	} 
	
	i = numlayers-2;
	
	cblas_dgemm(CblasColMajor,CblasTrans, CblasNoTrans, Dim[numlayers-2]+1, Dim[numlayers-1], N, 1.0/N, real_all_activations[numlayers-2], N, Ixk,N,0,dw + offset_weights[numlayers-2], Dim[numlayers-2]+1);
	cblas_dgemm(CblasColMajor,CblasTrans, CblasNoTrans, Dim[numlayers-2]+1, Dim[numlayers-1], N, 1.0/N, real_all_activations[numlayers-2], N, Ixk_Hessian,N,0,Hv + offset_weights[numlayers-2], Dim[numlayers-2]+1);
	
	//DECODING PART BACKPROP
	for (i=numlayers-3;i>=(numlayers-1)/2;i--)
	{
		cblas_dgemm(CblasColMajor,CblasNoTrans, CblasTrans, N, Dim[i+1], Dim[i+2], 1, Ixk, N, realweights+offset_weights[i+1],Dim[i+1],0,temparr, N);
		cblas_dgemm(CblasColMajor,CblasNoTrans, CblasTrans, N, Dim[i+1], Dim[i+2], 1, Ixk_Hessian, N, realweights+offset_weights[i+1],Dim[i+1],0,temparr2, N);
		
#pragma omp parallel for shared(temparr,all_activations,N,Dim,i) private(j,k) num_threads(NUM_THREADS) schedule(dynamic,CHUNK) if(DO_OPENMP)
		for (j=0;j<Dim[i+1]*N;j++)
		{
			temparr[j] *= activationfuncs[i][j];
			temparr2[j] *= activationfuncs[i][j];
		}
		
		free(Ixk);
		Ixk = temparr;
		free(Ixk_Hessian);
		Ixk_Hessian = temparr2;
		
		cblas_dgemm(CblasColMajor,CblasTrans, CblasNoTrans, Dim[i]+1, Dim[i+1], N, 1.0/N, real_all_activations[i], N, Ixk,N,0,dw + offset_weights[i], Dim[i]+1);
		cblas_dgemm(CblasColMajor,CblasTrans, CblasNoTrans, Dim[i]+1, Dim[i+1], N, 1.0/N, real_all_activations[i], N, Ixk_Hessian,N,0,Hv + offset_weights[i], Dim[i]+1);
	}
	
	//MIDDLE BOTTLENECK LAYER BACKPROP
	cblas_dgemm(CblasColMajor,CblasNoTrans, CblasTrans, N, Dim[i+1], Dim[i+2], 1, Ixk, N, realweights+offset_weights[i+1],Dim[i+1],0,temparr, N);
	cblas_dgemm(CblasColMajor,CblasNoTrans, CblasTrans, N, Dim[i+1], Dim[i+2], 1, Ixk_Hessian, N, realweights+offset_weights[i+1],Dim[i+1],0,temparr2, N);
		
	free(Ixk);
	Ixk = temparr;
	free(Ixk_Hessian);
	Ixk_Hessian = temparr2;
	
	cblas_dgemm(CblasColMajor,CblasTrans, CblasNoTrans, Dim[i]+1, Dim[i+1], N, 1.0/N, real_all_activations[i], N, Ixk,N,0,dw + offset_weights[i], Dim[i]+1);
	cblas_dgemm(CblasColMajor,CblasTrans, CblasNoTrans, Dim[i]+1, Dim[i+1], N, 1.0/N, real_all_activations[i], N, Ixk_Hessian,N,0,Hv + offset_weights[i], Dim[i]+1);
	
	//BOTTOM ENCODING PART BACKPROP
	for (i=(numlayers-5)/2;i>=0;i--)
	{
		cblas_dgemm(CblasColMajor,CblasNoTrans, CblasTrans, N, Dim[i+1], Dim[i+2], 1, Ixk, N, realweights+offset_weights[i+1],Dim[i+1],0,temparr, N);
		cblas_dgemm(CblasColMajor,CblasNoTrans, CblasTrans, N, Dim[i+1], Dim[i+2], 1, Ixk_Hessian, N, realweights+offset_weights[i+1],Dim[i+1],0,temparr2, N);
		
#pragma omp parallel for shared(temparr,all_activations,N,Dim,i) private(j,k) num_threads(NUM_THREADS) schedule(dynamic,CHUNK) if(DO_OPENMP)
		for (j=0;j<Dim[i+1]*N;j++)
		{
			temparr[j] *= activationfuncs[i][j];
			temparr2[j] *= activationfuncs[i][j];
		}
		
		free(Ixk);
		Ixk = temparr;
		free(Ixk_Hessian);
		Ixk_Hessian = temparr2;
		
		cblas_dgemm(CblasColMajor,CblasTrans, CblasNoTrans, Dim[i]+1, Dim[i+1], N, 1.0/N, real_all_activations[i], N, Ixk,N,0,dw + offset_weights[i], Dim[i]+1);
		cblas_dgemm(CblasColMajor,CblasTrans, CblasNoTrans, Dim[i]+1, Dim[i+1], N, 1.0/N, real_all_activations[i], N, Ixk_Hessian,N,0,Hv + offset_weights[i], Dim[i]+1);
	}
	
	
	return f;
}

double GRADIENT_HESSIAN_VEC_PROD_SOFTMAX(zomplex *weights, double *realweights, zomplex**all_activations, double **real_all_activations, int *Dim, double *data, int numlayers, double *dw, double *Hv, double *v, int N, int* offset_weights, double **activationfuncs, char typetop)
{
	int i,j,k,l;
	zomplex alpha = {1.0,0.0};
	zomplex beta = {0.0,0.0};
	
	double *temparr;
	double *temparr2;
	double *temparr3;
	double temp;
	
	l = N*Dim[0];
	
	real_all_activations[0] = data;
	real_add_ones(j,l,l+N,real_all_activations[0])
	
	cblas_dcopy(l,data,1,(double *) all_activations[0],2);

	double sumnums = 0;
	double sumnums1 = 0;

	
	#pragma omp parallel for shared(weights,v,offset_weights,numlayers) private(i) num_threads(NUM_THREADS) schedule(dynamic,CHUNK) if(DO_OPENMP)
	for (i=0;i<offset_weights[numlayers-1];i++)
		weights[i].imag = v[i]*NILPOT;
	
	//%ENCODING PART
	
	
//	for (i=0;i<numlayers;i++)
//		printf("%d %d %d\n",Dim[i],offset_weights[i],N);
	

/*	zomplex *onetimetemp = Calloc(zomplex,(Dim[i-1]+1)*N);
	zomplex *onetimetemp1 = Calloc(zomplex,(Dim[i-1]+1)*Dim[i]);
	zomplex *onetimetemp2 = Calloc(zomplex,N*Dim[i]);
	
	cblas_zgemm(CblasColMajor,CblasNoTrans, CblasNoTrans, N, Dim[i],Dim[i-1]+1,&alpha, onetimetemp, N, onetimetemp1, Dim[i-1]+1, &alpha,onetimetemp2, N);
	
	free(onetimetemp);
	free(onetimetemp1);
	free(onetimetemp2);
*/	
	
	
	for (i=1;i<(numlayers-1)/2;i++)
	{
/*		zomplex *onetimetemp = Calloc(zomplex,(Dim[i-1]+1)*N);
		zomplex *onetimetemp1 = Calloc(zomplex,(Dim[i-1]+1)*Dim[i]);
		zomplex *onetimetemp2 = Calloc(zomplex,N*Dim[i]);
		
		cblas_zgemm(CblasColMajor,CblasNoTrans, CblasNoTrans, N, Dim[i],Dim[i-1]+1,&alpha, onetimetemp, N, onetimetemp1, Dim[i-1]+1, &alpha,onetimetemp2, N);
		
		free(onetimetemp);
		free(onetimetemp1);
		free(onetimetemp2);
*/
		
		cblas_zgemm(CblasColMajor,CblasNoTrans, CblasNoTrans, N, Dim[i],Dim[i-1]+1,&alpha, all_activations[i-1], N, weights + offset_weights[i-1], Dim[i-1]+1, &beta,all_activations[i], N);
 		
		temparr = (double *)all_activations[i];
		temparr2 = activationfuncs[i-1];
		temparr3 = real_all_activations[i];
		
		l = Dim[i]*N*2;
//		#pragma omp parallel for shared(temparr,temparr2,temparr3,l) private(k) num_threads(NUM_THREADS) schedule(dynamic,CHUNK) if(DO_OPENMP)
		for(k=0;k<l;k+=2)
		{
//			printf("%d %d->\n",k,k>>1);
			temparr[k] = 1.0/(1+exp(-temparr[k]));
//			printf("%lf ",temparr[k]);
//			printf("%lf ",temparr3[k>>1]);
			temparr3[k >> 1] = temparr[k];
//			printf("%lf\n",temparr2[k>>1]);
			temparr2[k >> 1] = temparr[k]*(1-temparr[k]);
			temparr[k+1] *= temparr2[k>>1];	
		}

	}
	
	
	//MIDDLE BOTTLENECK LAYER (SOFTMAX)
	i = (numlayers-1)/2;
	cblas_zgemm(CblasColMajor,CblasNoTrans, CblasNoTrans, N, Dim[i],Dim[i-1]+1,&alpha, all_activations[i-1], N, weights + offset_weights[i-1], Dim[i-1]+1, &beta,all_activations[i], N);
	
	temparr = (double *)all_activations[i];
	temparr3 = real_all_activations[i];
	
	l = Dim[i]*N*2;
//	#pragma omp parallel for shared(temparr,l) private(k) num_threads(NUM_THREADS) schedule(dynamic,CHUNK) if(DO_OPENMP)
	for(k=0;k<l;k+=2)
		temparr[k] = exp(LARGEVAL*temparr[k]);
	
	
//	#pragma omp parallel for firstprivate(sumnums,sumnums1) shared(N,Dim,i,all_activations) private(j,k) if(DO_OPENMP)
	for (j=0;j<N;j++)
	{
		for(k=0;k<Dim[i];k++)
			sumnums += all_activations[i][j+k*N].real;
		
		for(k=0;k<Dim[i];k++)
		{
			all_activations[i][j+k*N].real /= sumnums;
			temparr3[j+k*N] = all_activations[i][j+k*N].real;

			sumnums1 += all_activations[i][j+k*N].real*all_activations[i][j+k*N].imag;
		}
		for(k=0;k<Dim[i];k++)
			all_activations[i][j+k*N].imag = LARGEVAL*all_activations[i][j+k*N].real*(all_activations[i][j+k*N].imag - sumnums1);
	}


	//%DECODING PART
	for (i=(numlayers-1)/2+1;i<numlayers-1;i++)
	{
		cblas_zgemm(CblasColMajor,CblasNoTrans, CblasNoTrans, N, Dim[i],Dim[i-1]+1, &alpha, all_activations[i-1], N, weights + offset_weights[i-1], Dim[i-1]+1, &beta,all_activations[i], N);
		
		temparr = (double *)all_activations[i];
		temparr2 = activationfuncs[i-1];
		temparr3 = real_all_activations[i];
		
		l = Dim[i]*N*2;
//#pragma omp parallel for shared(temparr,temparr2,temparr3,l) private(k) num_threads(NUM_THREADS) schedule(dynamic,CHUNK) if(DO_OPENMP)
		for(k=0;k<l;k+=2)
		{
			temparr[k] = 1.0/(1+exp(-temparr[k]));
			temparr2[k >> 1] = temparr[k]*(1-temparr[k]);
			temparr[k+1] *= temparr2[k>>1];
			temparr3[k >> 1] = temparr[k];
		}

	}
	

	i = numlayers-1;
	double f = 0;
	printf("%d\n",N*Dim[numlayers-1]);
	double *Ixk = Malloc(double,N*Dim[numlayers-1]);
	double *Ixk_Hessian = Malloc(double,N*Dim[numlayers-1]);

	exit(0);

	
	//TOP LAYER COMPUTATION AND BACKPROP (NOTE THAT TOP LAYER ACTIVATION FUNCTION AND LOSS FUNCTION ARE MATCHING)
	
//	cblas_zgemm(CblasColMajor,CblasNoTrans, CblasNoTrans, N, Dim[i-1]+1, Dim[i], &alpha, all_activations[i-1], N, weights + offset_weights[i-1], Dim[i-1]+1, &beta,all_activations[i], N);
	cblas_zgemm(CblasColMajor,CblasNoTrans, CblasNoTrans, N, Dim[i],Dim[i-1]+1,&alpha, all_activations[i-1], N, weights + offset_weights[i-1], Dim[i-1]+1, &beta,all_activations[i], N);
	switch (typetop) {
		case LOGISTIC:	
			temparr = (double *)all_activations[i];
			
			l = Dim[i]*N*2;
#pragma omp parallel for shared(temparr,l) private(k) num_threads(NUM_THREADS) schedule(dynamic,CHUNK) if(DO_OPENMP)
			for(k=0;k<l;k+=2)
			{
				temparr[k] = 1.0/(1+exp(-temparr[k]));
				temparr[k+1] *= temparr2[k>>1];
			}
			
#pragma omp parallel for shared(all_activations,N,Dim,i,Ixk,Ixk_Hessian) private(k,temp) num_threads(NUM_THREADS) reduction(+:f) schedule(dynamic,CHUNK)	 if(DO_OPENMP)
			for(k=0;k<Dim[0]*N;k++)
			{
				temp = all_activations[i][k].real;
				
				f += data[k]*log(temp) + (1-data[k])*log(1-temp);
				
				Ixk[k] = (temp - data[k]);
				Ixk_Hessian[k] = all_activations[i][k].imag/NILPOT;
			}
			f /= -N;
			printf("%e\n",f);
			
			break;
		case GAUSSIAN:
#pragma omp parallel for shared(all_activations,N,Dim,i,Ixk,Ixk_Hessian) private(k,temp) num_threads(NUM_THREADS) reduction(+:f) schedule(dynamic,CHUNK)	 if(DO_OPENMP)
			for(k=0;k<Dim[0]*N;k++)
			{
				temp = all_activations[i][k].real - data[k];
				f += temp*temp;
				Ixk[k] = temp;
				Ixk_Hessian[k] = all_activations[i][k].imag/NILPOT;
			}
			f /= 2*N;
			printf("%e\n",f);
			break;
		default:
			temparr = (double *)all_activations[i];
			
			l = Dim[i]*N*2;
			#pragma omp parallel for shared(temparr,l) private(k) num_threads(NUM_THREADS) schedule(dynamic,CHUNK) if(DO_OPENMP)
			for(k=0;k<l;k+=2)
				temparr[k] = exp(LARGEVAL*temparr[k]);
			
			sumnums = 0;
			sumnums1 = 0;
			
			#pragma omp parallel for firstprivate(sumnums,sumnums1) shared(N,Dim,i,all_activations) private(j,k) if(DO_OPENMP)
			for (j=0;j<N;j++)
			{
				for(k=0;k<Dim[i];k++)
					sumnums += all_activations[i][j+k*N].real;
				
				for(k=0;k<Dim[i];k++)
				{
					all_activations[i][j+k*N].real /= sumnums;
					sumnums1 += all_activations[i][j+k*N].real*all_activations[i][j+k*N].imag;
				}
				for(k=0;k<Dim[i];k++)
					all_activations[i][j+k*N].imag = LARGEVAL*all_activations[i][j+k*N].real*(all_activations[i][j+k*N].imag - sumnums1);
			}
			
			#pragma omp parallel for shared(all_activations,N,Dim,i,Ixk,Ixk_Hessian) private(k,temp) num_threads(NUM_THREADS) reduction(+:f) schedule(dynamic,CHUNK)	 if(DO_OPENMP)
			for(k=0;k<Dim[0]*N;k++)
			{
				temp = all_activations[i][k].real;
				
				f += data[k]*log(temp);
				Ixk[k] = LARGEVAL*(temp - data[k]);
				Ixk_Hessian[k] = LARGEVAL*all_activations[i][k].imag/NILPOT;
			}
			f /= -N;
			printf("%e\n",f);
			break;
	} 
	
	i = numlayers-2;
	
	cblas_dgemm(CblasColMajor,CblasTrans, CblasNoTrans, Dim[numlayers-2]+1, Dim[numlayers-1], N, 1.0/N, real_all_activations[numlayers-2], N, Ixk,N,0,dw + offset_weights[numlayers-2], Dim[numlayers-2]+1);
	cblas_dgemm(CblasColMajor,CblasTrans, CblasNoTrans, Dim[numlayers-2]+1, Dim[numlayers-1], N, 1.0/N, real_all_activations[numlayers-2], N, Ixk_Hessian,N,0,Hv + offset_weights[numlayers-2], Dim[numlayers-2]+1);
	
	//DECODING PART BACKPROP
	for (i=numlayers-3;i>=(numlayers-1)/2;i--)
	{
		cblas_dgemm(CblasColMajor,CblasNoTrans, CblasTrans, N, Dim[i+1], Dim[i+2], 1, Ixk, N, realweights+offset_weights[i+1],Dim[i+1],0,temparr, N);
		cblas_dgemm(CblasColMajor,CblasNoTrans, CblasTrans, N, Dim[i+1], Dim[i+2], 1, Ixk_Hessian, N, realweights+offset_weights[i+1],Dim[i+1],0,temparr2, N);
		
		#pragma omp parallel for shared(temparr,all_activations,N,Dim,i) private(j,k) num_threads(NUM_THREADS) schedule(dynamic,CHUNK) if(DO_OPENMP)
		for (j=0;j<Dim[i+1]*N;j++)
		{
			temparr[j] *= activationfuncs[i][j];
			temparr2[j] *= activationfuncs[i][j];
		}
		
		free(Ixk);
		Ixk = temparr;
		free(Ixk_Hessian);
		Ixk_Hessian = temparr2;
		
		cblas_dgemm(CblasColMajor,CblasTrans, CblasNoTrans, Dim[i]+1, Dim[i+1], N, 1.0/N, real_all_activations[i], N, Ixk,N,0,dw + offset_weights[i], Dim[i]+1);
		cblas_dgemm(CblasColMajor,CblasTrans, CblasNoTrans, Dim[i]+1, Dim[i+1], N, 1.0/N, real_all_activations[i], N, Ixk_Hessian,N,0,Hv + offset_weights[i], Dim[i]+1);
	}
	
	//MIDDLE BOTTLENECK LAYER BACKPROP
	cblas_dgemm(CblasColMajor,CblasNoTrans, CblasTrans, N, Dim[i+1], Dim[i+2], 1, Ixk, N, realweights+offset_weights[i+1],Dim[i+1],0,temparr, N);
	cblas_dgemm(CblasColMajor,CblasNoTrans, CblasTrans, N, Dim[i+1], Dim[i+2], 1, Ixk_Hessian, N, realweights+offset_weights[i+1],Dim[i+1],0,temparr2, N);
	
	free(Ixk);
	Ixk = temparr;
	free(Ixk_Hessian);
	Ixk_Hessian = temparr2;

	temparr = (double *)all_activations[i];
	sumnums = 0;
	sumnums1 =0;
	#pragma omp parallel for firstprivate(sumnums,sumnums1) shared(N,Dim,i,all_activations) private(j,k) if(DO_OPENMP)
	for (j=0;j<N;j++)
	{
		for(k=0;k<Dim[i+1];k++)
		{
			sumnums += real_all_activations[i+1][j+k*N]*Ixk[j+k*N];
			sumnums1 += real_all_activations[i+1][j+k*N]*Ixk_Hessian[j+k*N];
		}

		for(k=0;k<Dim[i+1];k++)
		{
			Ixk[k] = LARGEVAL*real_all_activations[i+1][j+k*N]*(Ixk[k] - sumnums);
			Ixk_Hessian[k] = LARGEVAL*real_all_activations[i+1][j+k*N]*(Ixk_Hessian[k] - sumnums1);
		}
	}
	
	cblas_dgemm(CblasColMajor,CblasTrans, CblasNoTrans, Dim[i]+1, Dim[i+1], N, 1.0/N, real_all_activations[i], N, Ixk,N,0,dw + offset_weights[i], Dim[i]+1);
	cblas_dgemm(CblasColMajor,CblasTrans, CblasNoTrans, Dim[i]+1, Dim[i+1], N, 1.0/N, real_all_activations[i], N, Ixk_Hessian,N,0,Hv + offset_weights[i], Dim[i]+1);
	
	//BOTTOM ENCODING PART BACKPROP
	for (i=(numlayers-5)/2;i>=0;i--)
	{
		cblas_dgemm(CblasColMajor,CblasNoTrans, CblasTrans, N, Dim[i+1], Dim[i+2], 1, Ixk, N, realweights+offset_weights[i+1],Dim[i+1],0,temparr, N);
		cblas_dgemm(CblasColMajor,CblasNoTrans, CblasTrans, N, Dim[i+1], Dim[i+2], 1, Ixk_Hessian, N, realweights+offset_weights[i+1],Dim[i+1],0,temparr2, N);
		
		#pragma omp parallel for shared(temparr,all_activations,N,Dim,i) private(j,k) num_threads(NUM_THREADS) schedule(dynamic,CHUNK) if(DO_OPENMP)
		for (j=0;j<Dim[i+1]*N;j++)
		{
			temparr[j] *= activationfuncs[i][j];
			temparr2[j] *= activationfuncs[i][j];
		}
		
		free(Ixk);
		Ixk = temparr;
		free(Ixk_Hessian);
		Ixk_Hessian = temparr2;
		
		cblas_dgemm(CblasColMajor,CblasTrans, CblasNoTrans, Dim[i]+1, Dim[i+1], N, 1.0/N, real_all_activations[i], N, Ixk,N,0,dw + offset_weights[i], Dim[i]+1);
		cblas_dgemm(CblasColMajor,CblasTrans, CblasNoTrans, Dim[i]+1, Dim[i+1], N, 1.0/N, real_all_activations[i], N, Ixk_Hessian,N,0,Hv + offset_weights[i], Dim[i]+1);
	}
	
	
	return f;
}

