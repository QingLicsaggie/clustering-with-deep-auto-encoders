#include "SimpleRNG.h"
#include "DBN_Cluster.h"

SimpleRNG *rng;

void (*RBMCODEFUNCS[3])(double **, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, int, int , int , int ) = {RBM_CORECODE_SM, RBM_CORECODE_LG, RBM_CORECODE_GA};
	
weights_str * DBN_deepauto(double **datatrain, double **datatest, int numrowstrain, int numrowstest, int maxepoch, int numlayers, int *numunits, int batchsizepre, char * typesoflayers, SMDparams *backpropparams)
{
	int i,j;

	rng = new SimpleRNG();
	rng->SetSeedFromSystemTime();
	
	batchdata_str * batchdata = createbatch(datatrain, numrowstrain, batchsizepre,numunits[0]);	

	int *offsetstoweightvector_botthalf = Malloc(int,2*numlayers-1);    //array of offsets to the start of the weight matrix that corresponds to a given level in the encoding bottom part 
	int *offsetstoweightvector_tophalf = Malloc(int,numlayers);		//array of offsets to the start of the weight matrix that corresponds to a given level in the Decoding top part (counting from the start of the Decoding part)
						  
	offsetstoweightvector_botthalf[0] = 0; 
	offsetstoweightvector_tophalf[0] = 0;
	
	for(j=1;j<numlayers;j++)
	{
		offsetstoweightvector_botthalf[j] = offsetstoweightvector_botthalf[j-1] + (numunits[j-1]+1)*numunits[j];
		offsetstoweightvector_tophalf[j] = offsetstoweightvector_tophalf[j-1] + numunits[numlayers-j-1]*(numunits[numlayers-j]+1);
	}

	int weightsvectorsize = offsetstoweightvector_botthalf[numlayers-1] + offsetstoweightvector_tophalf[numlayers-1];

	double *weights = Malloc(double, weightsvectorsize);
	
	for (i=0;i<numlayers-1;i++)
	{
		printf("Pretraining Layer %d with RBM: %d-%d \n",i+1,numunits[i],numunits[i+1]);

		if(i == 0)
			rbm_autoencoder(batchdata,1,1,weights + offsetstoweightvector_botthalf[i],weights+ offsetstoweightvector_botthalf[numlayers-1]+offsetstoweightvector_tophalf[numlayers-2-i],numunits[i],numunits[i+1],typesoflayers[0],typesoflayers[1],maxepoch);
		else 
		{
			if(i < numlayers-2)
				rbm_autoencoder(batchdata,0,1,weights + offsetstoweightvector_botthalf[i],weights + offsetstoweightvector_botthalf[numlayers-1]+offsetstoweightvector_tophalf[numlayers-2-i],numunits[i],numunits[i+1],typesoflayers[1],typesoflayers[1],maxepoch);
			else
				rbm_autoencoder(batchdata,0,0,weights + offsetstoweightvector_botthalf[i],weights + offsetstoweightvector_botthalf[numlayers-1]+offsetstoweightvector_tophalf[numlayers-2-i],numunits[i],numunits[i+1],typesoflayers[1],typesoflayers[2],maxepoch);
		}
	}
	

	deletebatchdata(batchdata);
	
	//%%%%%%%%%% END OF PREINITIALIZATIO OF WEIGHTS  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	
	int *l = Malloc(int,2*numlayers-1);
	
	for (i=0;i<numlayers-1;i++)
	{	
		l[i]=numunits[i];
		l[2*numlayers-2-i]=numunits[i];
		
		offsetstoweightvector_botthalf[i+numlayers-1] = offsetstoweightvector_botthalf[numlayers-1] + offsetstoweightvector_tophalf[i];
	}

	offsetstoweightvector_botthalf[2*numlayers-2] = weightsvectorsize;
	l[numlayers-1]=numunits[numlayers-1];
		
	numlayers = 2*numlayers-1;

	double f = SMD(datatrain, numrowstrain, weights, offsetstoweightvector_botthalf, l,numlayers, typesoflayers[2],typesoflayers[0],backpropparams);
	printf("Reconstruction error is: %f\n",f);

	weights_str * weightsdata = Malloc(weights_str,1);
	weightsdata->weights = weights;
	weightsdata->numweights = weightsvectorsize;

	
	return weightsdata;
}


weights_str * DBN_regress(double **datatrain, double **datatest, double *targetstrain, double *targetstest, int numrowstrain, int numrowstest, int maxepoch, int numlayers, int *numunits, int batchsizepre, char * typesoflayers, SMDparams *backpropparams)
{
	int i,j;
	
	rng = new SimpleRNG();
	rng->SetSeedFromSystemTime();
	int numrows = numrowstrain+numrowstest;

	double **wholeinputdata = Malloc(double *,numrows);
	
	for (i=0;i<numrowstrain;i++)
		wholeinputdata[i] = datatrain[i];

	for (i=0;i<numrowstest;i++)
		wholeinputdata[i+numrowstrain] = datatest[i];
	
	
	batchdata_str * batchdata = createbatch(wholeinputdata, numrows, batchsizepre, numunits[0]);		
	
	int *offsetstoweightvector = Malloc(int,numlayers+1);    //array of offsets to the start of the weight matrix that corresponds to a given level PLUS top ouput weights  
	
	offsetstoweightvector[0] = 0; 
	
	for(j=1;j<numlayers;j++)
		offsetstoweightvector[j] = offsetstoweightvector[j-1] + (numunits[j-1]+1)*numunits[j];
	

	offsetstoweightvector[numlayers] = offsetstoweightvector[numlayers-1] + (numunits[numlayers-1]+1);

	int weightsvectorsize = offsetstoweightvector[numlayers];
	
	double *weights = Malloc(double, weightsvectorsize);
	
	for (i=0;i<numlayers-1;i++)
	{
		printf("Pretraining Layer %d with RBM: %d-%d \n",i+1,numunits[i],numunits[i+1]);
		
		if(i == 0)
			rbm_class_regress(batchdata,1,1,weights + offsetstoweightvector[i], numunits[i],numunits[i+1],typesoflayers[0],typesoflayers[1],maxepoch);
		else 
		{
			if(i < numlayers-2)
				rbm_class_regress(batchdata,0,1,weights + offsetstoweightvector[i],numunits[i],numunits[i+1],typesoflayers[1],typesoflayers[1],maxepoch);
			else
				rbm_class_regress(batchdata,0,0,weights + offsetstoweightvector[i],numunits[i],numunits[i+1],typesoflayers[1],typesoflayers[2],maxepoch);
		}
	}
	
	
	deletebatchdata(batchdata);
	
	//%%%%%%%%%% END OF PREINITIALIZATIO OF WEIGHTS  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	
	int *l = Malloc(int,numlayers);
	
	for (i=0;i<numlayers;i++)
		l[i]=numunits[i];
		
//	double f = SMD_REGRESS(datatrain, targetstrain, datatest, targetstest, numrows, weights, offsetstoweightvector, l,numlayers, typesoflayers[2],typesoflayers[0],backpropparams);
//	printf("Reconstruction error is: %f\n",f);
	
	weights_str * weightsdata = Malloc(weights_str,1);
	weightsdata->weights = weights;
	weightsdata->numweights = weightsvectorsize;
	
	
	return weightsdata;
}


extern int * randperm(int size) //random permutation code taken from http://www.algoblog.com/2007/06/04/permutation/
{
	int * tempdata = Malloc(int,size);
	int i;
	for(i=0;i<size;i++)
		tempdata[i] = i;
	
	for(i=1;i<size;i++)
	{
		int j = (rng->GetUint() % (i+1));
		swap(tempdata[i],tempdata[j],int)
	}
	return tempdata;
}


extern batchdata_str * createbatch(double **data, int numrows, int batchsize,int numdims)
{
	int i;
	batchdata_str * batchdata = Malloc(batchdata_str,1);
	
	batchdata->numcases = batchsize;
	batchdata->numcases_last = batchsize+numrows % batchsize;
	
	batchdata->numbatches = numrows/batchsize;
	batchdata->numdims = numdims;
	batchdata->batchdata = Malloc(double *,numrows);
	for (i=0;i<numrows;i++)
	{
		batchdata->batchdata[i] = Malloc(double,numdims);
		cblas_dcopy(numdims,data[i],1,batchdata->batchdata[i],1);
	}
	batchdata->numrows = numrows;
	return batchdata;
}

void randomize_batch(batchdata_str * batchdata)
{
	int i;
	
	for (i=1;i<batchdata->numrows;i++)
	{
		int j = (rng->GetUint() % (i+1));
		swap(batchdata->batchdata[i],batchdata->batchdata[j],double *)
	}
}


extern batchdata_str_flat * createbatchflat(int numrows, int batchsize, int numdims)
{
	batchdata_str_flat * batchdata = Malloc(batchdata_str_flat,1);
	batchdata->batchdata = Malloc(double,(numdims+1)*numrows);
	batchdata->numcases = batchsize;
	batchdata->numcases_last = batchsize+numrows % batchsize;
	batchdata->numbatches = numrows/batchsize;
	batchdata->numdims = numdims;
	batchdata->numrows = numrows;
	return batchdata;
}
	


extern void randomize_batchflat(double **data,  batchdata_str_flat * batchdata,int *randperms)
{
	int i,j,k;
	
	for (i=1;i<batchdata->numrows;i++)
	{
		j = (rng->GetUint() % (i+1));
		swap(randperms[i],randperms[j],int)
	}
	
	k=0;
	for (i=0;i<batchdata->numbatches-1;i++)
	{
		double * batchdatatemp = batchdata->batchdata+i*(batchdata->numdims+1)*batchdata->numcases;
		for (j=0;j<batchdata->numcases;j++)
			cblas_dcopy(batchdata->numdims,data[randperms[k++]],1,batchdatatemp+j,batchdata->numcases);			
	}
	double * batchdatatemp = batchdata->batchdata+i*(batchdata->numdims+1)*batchdata->numcases;
	for (j=0;j<batchdata->numcases_last;j++)
		cblas_dcopy(batchdata->numdims,data[randperms[k++]],1,batchdatatemp+j,batchdata->numcases_last);
}


extern void deletebatchdata(batchdata_str * batchdata)
{
	free(batchdata->batchdata);
	free(batchdata);
}

extern void deletebatchdata_flat(batchdata_str_flat * batchdata)
{
	free(batchdata->batchdata);
	free(batchdata);
}


extern void rbm_autoencoder(batchdata_str * batchdata_rbm, int bottomlayer, int nottoplayer, double *weights_encode, double *weights_decode,int numvis, int numhid,char typevis,char typehid,int maxepoch)
{
	int epoch;
	
	// Initializing symmetric weights and biases. 
	double *weights = randn(numvis*numhid, 0.1);
	double *hidbiases  = Calloc(double,numhid);
	double *visbiases  = Calloc(double,numvis);

	double *CDiv_weight_diff = Malloc(double,numvis*numhid*NUM_THREADS);
	double *CDiv_hidbias_diff = Malloc(double,numhid*NUM_THREADS);
	double *CDiv_visbias_diff = Malloc(double,numvis*NUM_THREADS);
	
	double *vishidinc = Calloc(double,numvis*numhid);
	double *hidbiasinc = Calloc(double,numhid);
	double *visbiasinc = Calloc(double,numvis);

	double learning_rate = (typehid == GAUSSIAN || typevis == GAUSSIAN ? EPSILON_GA : EPSILON_LG);
	
	double *poshidprobs = Malloc(double,numhid*batchdata_rbm->numcases_last);
	double *poshidstates = Malloc(double,numhid*batchdata_rbm->numcases_last);
	double *negdata = Malloc(double,numvis*batchdata_rbm->numcases_last);
	double *neghidprobs = Malloc(double,numhid*batchdata_rbm->numcases_last);

	double *temparrayforthreads_hid = Malloc(double,numhid*NUM_THREADS);
	double *temparrayforthreads_vis = Malloc(double,numvis*NUM_THREADS);
		
	int batch;

	int i,j,k;
	double momentum;

	
	for(epoch=0;epoch<maxepoch;epoch++)
	{
		double average_visbias = 0;
		double average_hidbias = 0;
		double average_weights = 0;
//		printf("epoch %d\r",epoch);
		randomize_batch(batchdata_rbm);		
		
		for(batch = 0;batch < batchdata_rbm->numbatches;batch++)
		{
//			printf("epoch %d batch %d\r",epoch,batch); 
			
			//%%%%%%%%% START POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

			int numcasesperbatch = (batch < batchdata_rbm->numbatches-1 ? batchdata_rbm->numcases : batchdata_rbm->numcases_last);
			
			double **data = batchdata_rbm->batchdata+batch*batchdata_rbm->numcases;

			RBMCODEFUNCS[typehid](data, poshidprobs, poshidstates, negdata, neghidprobs, temparrayforthreads_vis, temparrayforthreads_hid, weights, hidbiases, visbiases, CDiv_weight_diff, CDiv_hidbias_diff, CDiv_visbias_diff, numhid, numvis, typevis, numcasesperbatch);			
								
			if (epoch>maxepoch*0.75)
				momentum=FINAL_MOMENT;
			else
				momentum=INITIAL_MOMENT;
				
/*			for(k=0;k<numvis;k++)
				printf("%lf ",CDiv_visbias_diff[k]);
			printf("\n-------------\n");
			for(k=0;k<numhid;k++)
				printf("%lf ",CDiv_hidbias_diff[k]);
			printf("\n-------------\n");
			for(j=0;j<numhid;j++)
				for(k=0;k<numvis;k++)
					printf("%lf ",CDiv_weight_diff[k+j*numvis]);
			printf("\n-------------\n");
*/
			#pragma omp parallel shared(CDiv_visbias_diff,CDiv_hidbias_diff,CDiv_weight_diff,vishidinc,weights,visbiasinc,hidbiasinc,visbiases,hidbiases,momentum,numcasesperbatch,numvis,numhid) private(k,i,j) num_threads(NUM_THREADS) if(DO_OPENMP)
			{
				#pragma omp for schedule(dynamic,CHUNK) nowait
				for(k=0;k<numvis;k++)
				{
					for(i=1;i<NUM_THREADS;i++)
						CDiv_visbias_diff[k] += CDiv_visbias_diff(i,k,numvis);
					
					visbiasinc[k] = visbiasinc[k]*momentum + CDiv_visbias_diff[k]*(learning_rate/numcasesperbatch);
					visbiases[k] += visbiasinc[k];
				}
				
				#pragma omp for schedule(dynamic,CHUNK) nowait
				for(k=0;k<numhid;k++)
				{
					for(i=1;i<NUM_THREADS;i++)
						CDiv_hidbias_diff[k] += CDiv_hidbias_diff(i,k,numhid);
					
					hidbiasinc[k] = hidbiasinc[k]*momentum + CDiv_hidbias_diff[k]*(learning_rate/numcasesperbatch); 
					hidbiases[k] += hidbiasinc[k];
				}
				
				#pragma omp for schedule(dynamic,CHUNK)
				for(k=0;k<numvis;k++)
				{
					for(j=0;j<numhid;j++)
					{
						for(i=1;i<NUM_THREADS;i++)
							CDiv_weight_diff[j+k*numhid] += CDiv_weight_diff(k,j,i,numvis,numhid);
						
						vishidinc(k,j,numvis) = momentum*vishidinc(k,j,numvis) + learning_rate*(CDiv_weight_diff[j+k*numhid]/numcasesperbatch - WEIGHTCOST* weights(k,j,numvis));
						weights(k,j,numvis) += vishidinc(k,j,numvis);
					}
				}
			}
		
			if(epoch == maxepoch-1)
			{
				if(bottomlayer)
				{
					#pragma omp parallel for shared(data,numcasesperbatch) num_threads(NUM_THREADS) schedule(dynamic,CHUNK) if(DO_OPENMP)
					for(i=0;i<numcasesperbatch;i++)
						free(data[i]);
				}
				
				if(nottoplayer)
				{
					#pragma omp parallel for shared(data, poshidprobs,numcasesperbatch,numhid) num_threads(NUM_THREADS) schedule(dynamic,CHUNK) if(DO_OPENMP)
					for(i=0;i<numcasesperbatch;i++)
						data[i] = poshidprobs+i*numhid;
				}
			}
			//%%%%%%%%%%%%%%%% END OF UPDATES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
		}
/*		printf("\n---visbiasinc----------\n");
		for(k=0;k<numvis;k++)
		{
			printf("%lf ",visbiasinc[k]);
			average_visbias += visbiasinc[k];
		}
		printf("\n-----hidbiasinc--------\n");
		for(k=0;k<numhid;k++)
		{
			printf("%lf ",hidbiasinc[k]);
			average_hidbias += hidbiasinc[k];
		}
		printf("\n-----weightsinc--------\n");
		for(k=0;k<numvis;k++)
		{
			for(j=0;j<numhid;j++)
			{
				printf("%lf ",vishidinc(k,j,numvis));
				average_weights += vishidinc(k,j,numvis);
			}
		}
*/		
		printf("\n%lf %lf %lf\n",average_hidbias/numhid,average_visbias/numvis,average_weights/(numvis*numhid));
	}
	

#pragma omp parallel shared(weights, weights_encode, weights_decode, hidbiases, numhid, numvis) private(i,j) num_threads(NUM_THREADS)  if(DO_OPENMP)
	{
		#pragma omp for schedule(dynamic,CHUNK) nowait
		for(j=0;j<numhid;j++)
		{
			cblas_dcopy(numvis,weights+j*numvis,1,weights_encode+j*(numvis+1),1);
		}
		
		
		#pragma omp for schedule(dynamic,CHUNK) nowait
		for(i=0;i<numvis;i++)
		{
			for(j=0;j<numhid;j++)
				weights_decode(j,i,numhid+1) = weights(i,j,numvis);
		}
	}
	cblas_dcopy(numhid,hidbiases,1,weights_encode+numvis, numvis+1);
	cblas_dcopy(numvis,visbiases,1,weights_decode+numhid, numhid+1);
/*
	FILE *tempfile = fopen("preweightsoutput.dat","a");
	fprintf(tempfile,"\n---visbiases----------\n");
	for(k=0;k<numvis;k++)
	{
		fprintf(tempfile,"%g ",visbiases[k]);
	}
	fprintf(tempfile,"\n-----hidbiases--------\n");
	for(k=0;k<numhid;k++)
	{
		fprintf(tempfile,"%g ",hidbiases[k]);
	}
	fprintf(tempfile,"\n-----weights--------\n");
	for(j=0;j<numhid;j++)
	{
		for(k=0;k<numvis;k++)
		{
			fprintf(tempfile,"%g ",weights(k,j,numvis));
		}
	}
	fclose(tempfile);

	tempfile = fopen("preweightsoutput1.dat","a");
	fprintf(tempfile,"\n---weights_encode----------\n");
	for(j=0;j<numhid;j++)
	{
		for(k=0;k<numvis+1;k++)
		{
			fprintf(tempfile,"%g ",weights_encode(k,j,numvis+1));
		}
	}
	fprintf(tempfile,"\n---weights_decode----------\n");
	for(j=0;j<numvis;j++)
	{
		for(k=0;k<numhid+1;k++)
		{
			fprintf(tempfile,"%g ",weights_decode(k,j,numhid+1));
		}
	}

	fclose(tempfile);
*/
	
	if(!nottoplayer)
		free(poshidprobs);
	
	free(weights);
	free(hidbiases);
	free(visbiases);
	
	free(CDiv_weight_diff);
	
	free(vishidinc);
	free(hidbiasinc);
	free(visbiasinc);
	
	free(CDiv_hidbias_diff);
	free(CDiv_visbias_diff);
	
	free(poshidstates);
	free(negdata);
	free(neghidprobs);
	
	free(temparrayforthreads_hid);
	free(temparrayforthreads_vis);
}


extern void rbm_class_regress(batchdata_str * batchdata_rbm, int bottomlayer, int nottoplayer, double *weights_inp,int numvis, int numhid,char typevis,char typehid,int maxepoch)
{
	int epoch;
	
	// Initializing symmetric weights and biases. 
	double *weights = randn(numvis*numhid, 0.1);
	double *hidbiases  = Calloc(double,numhid);
	double *visbiases  = Calloc(double,numvis);
	
	double *CDiv_weight_diff = Malloc(double,numvis*numhid*NUM_THREADS);
	double *CDiv_hidbias_diff = Malloc(double,numhid*NUM_THREADS);
	double *CDiv_visbias_diff = Malloc(double,numvis*NUM_THREADS);
	
	double *vishidinc = Calloc(double,numvis*numhid);
	double *hidbiasinc = Calloc(double,numhid);
	double *visbiasinc = Calloc(double,numvis);
	
	double learning_rate = (typehid == GAUSSIAN || typevis == GAUSSIAN ? EPSILON_GA : EPSILON_LG);
	
	double *poshidprobs = Malloc(double,numhid*batchdata_rbm->numcases_last);
	double *poshidstates = Malloc(double,numhid*batchdata_rbm->numcases_last);
	double *negdata = Malloc(double,numvis*batchdata_rbm->numcases_last);
	double *neghidprobs = Malloc(double,numhid*batchdata_rbm->numcases_last);
	
	double *temparrayforthreads_hid = Malloc(double,numhid*NUM_THREADS);
	double *temparrayforthreads_vis = Malloc(double,numvis*NUM_THREADS);
	
	int batch;
	
	int i,j,k;
	double momentum;
	
	
	for(epoch=0;epoch<maxepoch;epoch++)
	{
		double average_visbias = 0;
		double average_hidbias = 0;
		double average_weights = 0;
		//		printf("epoch %d\r",epoch);
		randomize_batch(batchdata_rbm);		
		
		for(batch = 0;batch < batchdata_rbm->numbatches;batch++)
		{
			//			printf("epoch %d batch %d\r",epoch,batch); 
			
			//%%%%%%%%% START POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
			
			int numcasesperbatch = (batch < batchdata_rbm->numbatches-1 ? batchdata_rbm->numcases : batchdata_rbm->numcases_last);
			
			double **data = batchdata_rbm->batchdata+batch*batchdata_rbm->numcases;
			
			RBMCODEFUNCS[typehid](data, poshidprobs, poshidstates, negdata, neghidprobs, temparrayforthreads_vis, temparrayforthreads_hid, weights, hidbiases, visbiases, CDiv_weight_diff, CDiv_hidbias_diff, CDiv_visbias_diff, numhid, numvis, typevis, numcasesperbatch);			
			
			if (epoch>maxepoch*0.75)
				momentum=FINAL_MOMENT;
			else
				momentum=INITIAL_MOMENT;
			
			/*			for(k=0;k<numvis;k++)
			 printf("%lf ",CDiv_visbias_diff[k]);
			 printf("\n-------------\n");
			 for(k=0;k<numhid;k++)
			 printf("%lf ",CDiv_hidbias_diff[k]);
			 printf("\n-------------\n");
			 for(j=0;j<numhid;j++)
			 for(k=0;k<numvis;k++)
			 printf("%lf ",CDiv_weight_diff[k+j*numvis]);
			 printf("\n-------------\n");
			 */
#pragma omp parallel shared(CDiv_visbias_diff,CDiv_hidbias_diff,CDiv_weight_diff,vishidinc,weights,visbiasinc,hidbiasinc,visbiases,hidbiases,momentum,numcasesperbatch,numvis,numhid) private(k,i,j) num_threads(NUM_THREADS) if(DO_OPENMP)
			{
#pragma omp for schedule(dynamic,CHUNK) nowait
				for(k=0;k<numvis;k++)
				{
					for(i=1;i<NUM_THREADS;i++)
						CDiv_visbias_diff[k] += CDiv_visbias_diff(i,k,numvis);
					
					visbiasinc[k] = visbiasinc[k]*momentum + CDiv_visbias_diff[k]*(learning_rate/numcasesperbatch);
					visbiases[k] += visbiasinc[k];
				}
				
#pragma omp for schedule(dynamic,CHUNK) nowait
				for(k=0;k<numhid;k++)
				{
					for(i=1;i<NUM_THREADS;i++)
						CDiv_hidbias_diff[k] += CDiv_hidbias_diff(i,k,numhid);
					
					hidbiasinc[k] = hidbiasinc[k]*momentum + CDiv_hidbias_diff[k]*(learning_rate/numcasesperbatch); 
					hidbiases[k] += hidbiasinc[k];
				}
				
#pragma omp for schedule(dynamic,CHUNK)
				for(k=0;k<numvis;k++)
				{
					for(j=0;j<numhid;j++)
					{
						for(i=1;i<NUM_THREADS;i++)
							CDiv_weight_diff[j+k*numhid] += CDiv_weight_diff(k,j,i,numvis,numhid);
						
						vishidinc(k,j,numvis) = momentum*vishidinc(k,j,numvis) + learning_rate*(CDiv_weight_diff[j+k*numhid]/numcasesperbatch - WEIGHTCOST* weights(k,j,numvis));
						weights(k,j,numvis) += vishidinc(k,j,numvis);
					}
				}
			}
			
			if(epoch == maxepoch-1)
			{
				if(bottomlayer)
				{
#pragma omp parallel for shared(data,numcasesperbatch) num_threads(NUM_THREADS) schedule(dynamic,CHUNK) if(DO_OPENMP)
					for(i=0;i<numcasesperbatch;i++)
						free(data[i]);
				}
				
				if(nottoplayer)
				{
#pragma omp parallel for shared(data, poshidprobs,numcasesperbatch,numhid) num_threads(NUM_THREADS) schedule(dynamic,CHUNK) if(DO_OPENMP)
					for(i=0;i<numcasesperbatch;i++)
						data[i] = poshidprobs+i*numhid;
				}
			}
			//%%%%%%%%%%%%%%%% END OF UPDATES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
		}
		/*		printf("\n---visbiasinc----------\n");
		 for(k=0;k<numvis;k++)
		 {
		 printf("%lf ",visbiasinc[k]);
		 average_visbias += visbiasinc[k];
		 }
		 printf("\n-----hidbiasinc--------\n");
		 for(k=0;k<numhid;k++)
		 {
		 printf("%lf ",hidbiasinc[k]);
		 average_hidbias += hidbiasinc[k];
		 }
		 printf("\n-----weightsinc--------\n");
		 for(k=0;k<numvis;k++)
		 {
		 for(j=0;j<numhid;j++)
		 {
		 printf("%lf ",vishidinc(k,j,numvis));
		 average_weights += vishidinc(k,j,numvis);
		 }
		 }
		 */		
		printf("\n%lf %lf %lf\n",average_hidbias/numhid,average_visbias/numvis,average_weights/(numvis*numhid));
	}
	
	
#pragma omp parallel shared(weights, weights_inp, hidbiases, numhid, numvis) private(i,j) num_threads(NUM_THREADS)  if(DO_OPENMP)
	{
#pragma omp for schedule(dynamic,CHUNK) nowait
		for(j=0;j<numhid;j++)
		{
			cblas_dcopy(numvis,weights+j*numvis,1,weights_inp+j*(numvis+1),1);
		}
		
	}
	
/*#pragma omp for schedule(dynamic,CHUNK) nowait
		for(i=0;i<numvis;i++)
		{
			for(j=0;j<numhid;j++)
				weights_decode(j,i,numhid+1) = weights(i,j,numvis);
		}
	}
*/
	cblas_dcopy(numhid,hidbiases,1,weights_inp+numvis, numvis+1);
//	cblas_dcopy(numvis,visbiases,1,weights_decode+numhid, numhid+1);
	/*
	 FILE *tempfile = fopen("preweightsoutput.dat","a");
	 fprintf(tempfile,"\n---visbiases----------\n");
	 for(k=0;k<numvis;k++)
	 {
	 fprintf(tempfile,"%g ",visbiases[k]);
	 }
	 fprintf(tempfile,"\n-----hidbiases--------\n");
	 for(k=0;k<numhid;k++)
	 {
	 fprintf(tempfile,"%g ",hidbiases[k]);
	 }
	 fprintf(tempfile,"\n-----weights--------\n");
	 for(j=0;j<numhid;j++)
	 {
	 for(k=0;k<numvis;k++)
	 {
	 fprintf(tempfile,"%g ",weights(k,j,numvis));
	 }
	 }
	 fclose(tempfile);
	 
	 tempfile = fopen("preweightsoutput1.dat","a");
	 fprintf(tempfile,"\n---weights_encode----------\n");
	 for(j=0;j<numhid;j++)
	 {
	 for(k=0;k<numvis+1;k++)
	 {
	 fprintf(tempfile,"%g ",weights_encode(k,j,numvis+1));
	 }
	 }
	 fprintf(tempfile,"\n---weights_decode----------\n");
	 for(j=0;j<numvis;j++)
	 {
	 for(k=0;k<numhid+1;k++)
	 {
	 fprintf(tempfile,"%g ",weights_decode(k,j,numhid+1));
	 }
	 }
	 
	 fclose(tempfile);
	 */
	
	if(!nottoplayer)
		free(poshidprobs);
	
	free(weights);
	free(hidbiases);
	free(visbiases);
	
	free(CDiv_weight_diff);
	
	free(vishidinc);
	free(hidbiasinc);
	free(visbiasinc);
	
	free(CDiv_hidbias_diff);
	free(CDiv_visbias_diff);
	
	free(poshidstates);
	free(negdata);
	free(neghidprobs);
	
	free(temparrayforthreads_hid);
	free(temparrayforthreads_vis);
}


extern double *randn(int size, double weight)
{
	int i;
	double * temp = Malloc(double,size);
	
	for(i=0;i<size;i++)
		temp[i] = rng->GetNormal(0.0, weight);

	return temp;
}


extern void RBM_CORECODE_SM(double **data, double *poshidprobs, double *poshidstates, double *negdata, double *neghidprobs, double *temparrayforthreads_vis, double *temparrayforthreads_hid,  double *weights, double *hidbiases, double *visbiases, double *CDiv_weight_diff, double *CDiv_hidbias_diff, double *CDiv_visbias_diff, int numhid, int numvis, int typevis, int numcases)
{
	int i,j,k;
	
	int tid=0;
	double sum_prods;
	
	#pragma omp parallel firstprivate(temparrayforthreads_vis,temparrayforthreads_hid,CDiv_weight_diff, CDiv_hidbias_diff, CDiv_visbias_diff) shared(data,weights,hidbiases,visbiases,poshidprobs,poshidstates,negdata,neghidprobs,numhid,numvis,typevis,numcases) private(i,j,k,tid,sum_prods) num_threads(NUM_THREADS) if(DO_OPENMP)
	{
		tid = omp_get_thread_num();
		
		CDiv_weight_diff += tid*numvis*numhid;
		CDiv_hidbias_diff += tid*numhid;
		CDiv_visbias_diff += tid*numvis;
		
		for(j=0;j<numhid;j++)
		{
			CDiv_hidbias_diff[j] = 0;
			
			for(k=0;k<numvis;k++)
				CDiv_weight_diff[j+k*numhid] = 0;
		}
		
		for(k=0;k<numvis;k++)
			CDiv_visbias_diff[k] = 0;

		temparrayforthreads_hid += tid*numhid;
		temparrayforthreads_vis += tid*numvis;
		
		#pragma omp for schedule(dynamic,CHUNK)
		for(i=0;i<numcases;i++)
		{
			cblas_dcopy(numhid,hidbiases,1,temparrayforthreads_hid,1);
			cblas_dgemv(CblasColMajor,CblasTrans, numvis,numhid,1.0,weights,numvis,data[i], 1, 1.0,temparrayforthreads_hid,1);
			
			sum_prods = 0;
			vvexp(poshidprobs+i*numhid, temparrayforthreads_hid, &numhid);
			
			for(j=0;j<numhid;j++)
				sum_prods += poshidprobs(i,j,numhid);
			
			for(j=0;j<numhid;j++)
			{
				poshidprobs(i,j,numhid) /= sum_prods;
				poshidstates(i,j,numhid) = poshidprobs(i,j,numhid) > rng->GetUniform();
			}
			
			cblas_dcopy(numvis,visbiases,1,temparrayforthreads_vis,1);
			
			
			switch (typevis) {
				case SOFTMAX:
					cblas_dgemv(CblasColMajor,CblasNoTrans, numvis,numhid,1.0,weights,numvis,poshidstates+i*numhid, 1, 1.0,temparrayforthreads_vis,1);
					sum_prods = 0;
					vvexp(negdata+i*numvis, temparrayforthreads_vis, &numvis);
					
					for(j=0;j<numvis;j++)
						sum_prods += negdata(i,j,numvis);
					
					for(j=0;j<numvis;j++)
						negdata(i,j,numvis) /= sum_prods;
					break;
				case GAUSSIAN:
					cblas_dgemv(CblasColMajor,CblasNoTrans, numvis,numhid,1.0,weights,numvis,poshidstates+i*numhid, 1, 1.0,temparrayforthreads_vis,1);
					for(j=0;j<numvis;j++)
						negdata(i,j,numvis) = rng->GetNormal(temparrayforthreads_vis[j], 1.0);
					break;
				case LOGISTIC:
					cblas_dgemv(CblasColMajor,CblasNoTrans, numvis,numhid,-T,weights,numvis,poshidstates+i*numhid, 1, -T,temparrayforthreads_vis,1);
					vvexp(negdata+i*numvis, temparrayforthreads_vis, &numvis);
					
					for(j=0;j<numvis;j++)
						negdata(i,j,numvis) = 1.0/(1 + negdata(i,j,numvis));
					break;
				default:
					break;
			}
			
			cblas_dcopy(numhid,hidbiases,1,temparrayforthreads_hid,1);
			cblas_dgemv(CblasColMajor,CblasTrans, numvis,numhid,1.0,weights,numvis,negdata+i*numvis, 1, 1.0,temparrayforthreads_hid,1);
			
			sum_prods = 0;
			vvexp(neghidprobs+i*numhid, temparrayforthreads_hid, &numhid);
			for(j=0;j<numhid;j++)
				sum_prods += neghidprobs(i,j,numhid);
			
			for(j=0;j<numhid;j++)
			{
				neghidprobs(i,j,numhid) /= sum_prods;				
				CDiv_hidbias_diff[j] += poshidprobs(i,j,numhid) - neghidprobs(i,j,numhid);
				
				for(k=0;k<numvis;k++)
					CDiv_weight_diff[j+k*numhid] += poshidprobs(i,j,numhid) * data[i][k] - neghidprobs(i,j,numhid) * negdata(i,k,numvis);
			}
			
			for(k=0;k<numvis;k++)
				CDiv_visbias_diff[k] += data[i][k] - negdata(i,k,numvis);
		}
	}
}


extern void RBM_CORECODE_LG(double **data, double *poshidprobs, double *poshidstates, double *negdata, double *neghidprobs, double *temparrayforthreads_vis, double *temparrayforthreads_hid,  double *weights, double *hidbiases, double *visbiases, double *CDiv_weight_diff, double *CDiv_hidbias_diff, double *CDiv_visbias_diff, int numhid, int numvis, int typevis, int numcases)
{
	int i,j,k;
	
	int tid=0;
	double sum_prods;

#pragma omp parallel firstprivate(temparrayforthreads_vis,temparrayforthreads_hid,CDiv_weight_diff, CDiv_hidbias_diff, CDiv_visbias_diff) shared(data,weights,hidbiases,visbiases,poshidprobs,poshidstates,negdata,neghidprobs,numhid,numvis,typevis,numcases) private(i,j,k,tid) num_threads(NUM_THREADS) if(DO_OPENMP)
	{
		tid = omp_get_thread_num();
		CDiv_weight_diff += tid*numvis*numhid;
		CDiv_hidbias_diff += tid*numhid;
		CDiv_visbias_diff += tid*numvis;
		
		temparrayforthreads_hid += tid*numhid;
		temparrayforthreads_vis += tid*numvis;
		
		for(j=0;j<numhid;j++)
		{
			CDiv_hidbias_diff[j] = 0;
			
			for(k=0;k<numvis;k++)
				CDiv_weight_diff[j+k*numhid] = 0;
		}
		
		for(k=0;k<numvis;k++)
			CDiv_visbias_diff[k] = 0;

//		printf("numcases:%d\n",numcases);
#pragma omp for schedule(dynamic,CHUNK)
		for(i=0;i<numcases;i++)
		{
			cblas_dcopy(numhid,hidbiases,1,temparrayforthreads_hid,1);
/*			printf("\n----hidbiases[j]---------\n");
			
			for(j=0;j<numhid;j++)
				printf("%lf ",temparrayforthreads_hid[j]);
			
			printf("\n---weights(j,k,numvis)----------\n");
			for(j=0;j<numvis;j++)
			{
				for(k=0;k<numhid;k++)
					printf("%lf ",weights(j,k,numvis));
				printf(" | ");
			}
*/
			
			cblas_dgemv(CblasColMajor,CblasTrans, numvis,numhid,-T,weights,numvis,data[i], 1, -T,temparrayforthreads_hid,1);
			
//			printf("\n-- -(weights*data+hidbiases)----------\n");
				
//			for(j=0;j<numhid;j++)
//				printf("%lf ",temparrayforthreads_hid[j]);

			
			vvexp(poshidprobs+i*numhid, temparrayforthreads_hid, &numhid);
						
			for(j=0;j<numhid;j++)
			{
				poshidprobs(i,j,numhid) = 1.0/(1+poshidprobs(i,j,numhid));
				poshidstates(i,j,numhid) = poshidprobs(i,j,numhid) > rng->GetUniform();
			}

//			printf("\n----poshidprobs-----poshidstates[j]---------\n");
			
//			for(j=0;j<numhid;j++)
//				printf("%lf %lf | ",poshidprobs(i,j,numhid),poshidstates(i,j,numhid));
			
			
			cblas_dcopy(numvis,visbiases,1,temparrayforthreads_vis,1);
			
//			printf("\n----visbiases[j]---------\n");
			
//			for(j=0;j<numvis;j++)
//				printf("%lf ",temparrayforthreads_vis[j]);

			switch (typevis) {
				case SOFTMAX:
					cblas_dgemv(CblasColMajor,CblasNoTrans, numvis,numhid,1.0,weights,numvis,poshidstates+i*numhid, 1, 1.0,temparrayforthreads_vis,1);
					sum_prods = 0;
					vvexp(negdata+i*numvis, temparrayforthreads_vis, &numvis);
					
					for(j=0;j<numvis;j++)
						sum_prods += negdata(i,j,numvis);
					
					for(j=0;j<numvis;j++)
						negdata(i,j,numvis) /= sum_prods;
					break;
				case GAUSSIAN:
					cblas_dgemv(CblasColMajor,CblasNoTrans, numvis,numhid,1.0,weights,numvis,poshidstates+i*numhid, 1, 1.0,temparrayforthreads_vis,1);
//					printf("\n-- (weights'*poshidstates+visbiases)----------\n");
					
//					for(j=0;j<numvis;j++)
//						printf("%lf ",temparrayforthreads_vis[j]);
					
					for(j=0;j<numvis;j++)
						negdata(i,j,numvis) = rng->GetNormal(temparrayforthreads_vis[j], 1.0);
					break;
				case LOGISTIC:
					cblas_dgemv(CblasColMajor,CblasNoTrans, numvis,numhid,-T,weights,numvis,poshidstates+i*numhid, 1, -T,temparrayforthreads_vis,1);
					vvexp(negdata+i*numvis, temparrayforthreads_vis, &numvis);
					
					for(j=0;j<numvis;j++)
						negdata(i,j,numvis) = 1.0/(1 + negdata(i,j,numvis));					
					break;
				default:
					break;
			}
			
//			printf("\n--data negdata(i,k,numvis)----------\n");
//			for(k=0;k<numvis;k++)
//				printf("%lf %lf | ", data[i][k],negdata(i,k,numvis));			

			
			
			cblas_dcopy(numhid,hidbiases,1,temparrayforthreads_hid,1);


//			printf("\n-- -(weights*negdata+hidbiases)----------\n");

			cblas_dgemv(CblasColMajor,CblasTrans, numvis,numhid,-T,weights,numvis,negdata+i*numvis, 1, -T,temparrayforthreads_hid,1);

//			for(j=0;j<numhid;j++)
//				printf("%lf ",temparrayforthreads_hid[j]);

//			printf("\n----poshidprobs ---neghidprobs(i,j,numhid)---------\n");
				
			
			vvexp(neghidprobs+i*numhid, temparrayforthreads_hid, &numhid);

			
			for(j=0;j<numhid;j++)
			{
				neghidprobs(i,j,numhid) = 1.0/(1+neghidprobs(i,j,numhid));
//				printf("%lf %lf | ",poshidprobs(i,j,numhid),neghidprobs(i,j,numhid));
				
				CDiv_hidbias_diff[j] += poshidprobs(i,j,numhid) - neghidprobs(i,j,numhid);
				
				for(k=0;k<numvis;k++)
					CDiv_weight_diff[j+k*numhid] += poshidprobs(i,j,numhid) * data[i][k] - neghidprobs(i,j,numhid) * negdata(i,k,numvis);
			}


			for(k=0;k<numvis;k++)
				CDiv_visbias_diff[k] += data[i][k] - negdata(i,k,numvis);	
		}
	}
}


extern void RBM_CORECODE_GA(double **data, double *poshidprobs, double *poshidstates, double *negdata, double *neghidprobs, double *temparrayforthreads_vis, double *temparrayforthreads_hid,  double *weights, double *hidbiases, double *visbiases, double *CDiv_weight_diff, double *CDiv_hidbias_diff, double *CDiv_visbias_diff, int numhid, int numvis, int typevis, int numcases)
{
	int i,j,k;
	
	int tid=0;
	double sum_prods;
	
	#pragma omp parallel firstprivate(temparrayforthreads_vis,temparrayforthreads_hid,CDiv_weight_diff, CDiv_hidbias_diff, CDiv_visbias_diff) shared(data,weights,hidbiases,visbiases,poshidprobs,poshidstates,negdata,neghidprobs,numhid,numvis,typevis,numcases) private(i,j,k,tid) num_threads(NUM_THREADS) if(DO_OPENMP)
	{
		tid = omp_get_thread_num();
		
		
		CDiv_weight_diff += tid*numvis*numhid;
		CDiv_hidbias_diff += tid*numhid;
		CDiv_visbias_diff += tid*numvis;
		
		for(j=0;j<numhid;j++)
		{
			CDiv_hidbias_diff[j] = 0;
			
			for(k=0;k<numvis;k++)
				CDiv_weight_diff[j+k*numhid] = 0;
		}
		
		for(k=0;k<numvis;k++)
			CDiv_visbias_diff[k] = 0;

		
		temparrayforthreads_hid += tid*numhid;
		temparrayforthreads_vis += tid*numvis;
		
		#pragma omp for schedule(dynamic,CHUNK)
		for(i=0;i<numcases;i++)
		{
			cblas_dcopy(numhid,hidbiases,1,temparrayforthreads_hid,1);
			cblas_dgemv(CblasColMajor,CblasTrans, numvis,numhid,1.0,weights,numvis,data[i], 1, 1.0,temparrayforthreads_hid,1);

			cblas_dcopy(numhid,poshidprobs+i*numhid,1,temparrayforthreads_hid,1);

			for(j=0;j<numhid;j++)
				poshidstates(i,j,numhid) = rng->GetNormal(temparrayforthreads_hid[j], 1.0);
			
			cblas_dcopy(numvis,visbiases,1,temparrayforthreads_vis,1);
			
			switch (typevis) {
				case SOFTMAX:
					cblas_dgemv(CblasColMajor,CblasNoTrans, numvis,numhid,1.0,weights,numvis,poshidstates+i*numhid, 1, 1.0,temparrayforthreads_vis,1);
					sum_prods = 0;
					vvexp(negdata+i*numvis, temparrayforthreads_vis, &numvis);
					
					for(j=0;j<numvis;j++)
						sum_prods += negdata(i,j,numvis);
					
					for(j=0;j<numvis;j++)
						negdata(i,j,numvis) /= sum_prods;
					break;
				case GAUSSIAN:
					cblas_dgemv(CblasColMajor,CblasNoTrans, numvis,numhid,1.0,weights,numvis,poshidstates+i*numhid, 1, 1.0,temparrayforthreads_vis,1);
					for(j=0;j<numvis;j++)
						negdata(i,j,numvis) = rng->GetNormal(negdata(i,j,numvis), 1.0);
					break;
				case LOGISTIC:
					cblas_dgemv(CblasColMajor,CblasNoTrans, numvis,numhid,-T,weights,numvis,poshidstates+i*numhid, 1, -T,temparrayforthreads_vis,1);
					vvexp(negdata+i*numvis, temparrayforthreads_vis, &numvis);
					
					for(j=0;j<numvis;j++)
						negdata(i,j,numvis) = 1.0/(1 + negdata(i,j,numvis));					
					break;
				default:
					break;
			}
			
			cblas_dcopy(numhid,hidbiases,1,temparrayforthreads_hid,1);
			cblas_dgemv(CblasColMajor,CblasTrans, numvis,numhid,-1.0,weights,numvis,negdata+i*numvis, 1, -1.0,temparrayforthreads_hid,1);

			cblas_dcopy(numhid,neghidprobs+i*numhid,1,temparrayforthreads_hid,1);
			
			for(j=0;j<numhid;j++)
			{
				CDiv_hidbias_diff[j] += poshidprobs(i,j,numhid) - neghidprobs(i,j,numhid);
				
				for(k=0;k<numvis;k++)
					CDiv_weight_diff[j+k*numhid] += poshidprobs(i,j,numhid) * data[i][k] - neghidprobs(i,j,numhid) * negdata(i,k,numvis);
			}
			
			for(k=0;k<numvis;k++)
				CDiv_visbias_diff[k] += data[i][k] - negdata(i,k,numvis);
		}
	}
}

