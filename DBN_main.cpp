//#include <iostream>
#include <stdio.h>
#include "DBN_Cluster.h"

#define MAXNUMLAYERS 10
#define MAX_NUM_EXAMPLES 2000000
#define DEFAULT_NUM_FEATURES 100000

#define DBN_REGRESSION	0
#define DBN_AUTO		1

//#define DEFAULT_DATA_FILE "K562_preprocessed.dat"//_smallerset.dat"
#define DEFAULT_DATA_FILE "K562_only_random_smallerset.dat"

data_str * read_problem(const char *filename);
data_str * read_problem_withtargets(const char *filename);

int main (int argc, char * const argv[]) 
{	
	data_str * datatrain,*datatest;
	
	SMDparams params = {5,1e-5,0.99,0.05,0.1,16};
	char typesoflayers [3] = {2,1,0};
	int maxepochpre = 5;
	int numlayers = 4;
	int numunits [MAXNUMLAYERS] = {0,300,150,10,0,10,0,0,0,0};
	int batchsizepre = 100;
	int type;
	
	int i,j;
	weights_str *weights;
		
	for(i=1;i< argc-1;i++)
	{
		if(argv[i][0] == '-')
		{
			switch (argv[i][1]) 
			{
				case 't':
					type = atoi(argv[++i]);
					break;
				case 'm':
					maxepochpre = atoi(argv[++i]);
					break;
				case 'l':
					typesoflayers[0] = argv[++i][0];
					typesoflayers[1] = argv[++i][0];
					typesoflayers[2] = argv[++i][0];
					break;
				case 'n':
					numlayers = atoi(argv[++i]);
					break;
				case 'u':
					for(j=1;j<numlayers;j++)
						numunits[j] = atoi(argv[++i]);
					break;
				case 'b':
					batchsizepre = atoi(argv[++i]);
					break;
				case 'e':
					params.errorthresh = atof(argv[++i]);
					break;
				case 'M':
					params.maxepoch = atoi(argv[++i]);
					break;
				case 'L':
					params.lambda = atof(argv[++i]);
					break;
				case 'N':
					params.nu0 = atof(argv[++i]);
					break;
				case 'U':
					params.mu = atof(argv[++i]);
					break;
				case 'B':
					params.batchsizebp = atoi(argv[++i]);
					break;
				default:
					printf("Wrong parameter to function\n");
					exit(0);
					break;
			}
		}
	}


	switch(type)
	{
		case DBN_REGRESSION:
			datatrain = read_problem_withtargets(argv[argc-2]);
			datatest = read_problem_withtargets(argv[argc-1]);
			
			numunits[0] = datatrain->numdims;
			weights = DBN_regress(datatrain->data,datatest->data, datatrain->targets,datatest->targets, datatrain->numrows, datatest->numrows, maxepochpre, numlayers, numunits, batchsizepre, typesoflayers, &params);
//                      DBN_regress(double **datatrain, double **datatest, double *targetstrain, double *targetstest, int numrowstrain, int numrowstest, int maxepoch, int numlayers, int *numunits, int batchsizepre, char * typesoflayers, SMDparams *backpropparams);
			break;
		case DBN_AUTO:
			datatrain = read_problem(argv[argc-2]);
			datatest = read_problem(argv[argc-1]);
			
			numunits[0] = datatrain->numdims;
			weights = DBN_deepauto(datatrain->data,datatest->data, datatrain->numrows, datatest->numrows, maxepochpre, numlayers, numunits, batchsizepre, typesoflayers, &params);
			break;
	}
	
	FILE *tempfile = fopen("postweightsoutput.dat","w");
	for(i=0;i<weights->numweights;i++)
	{
		printf("%.10e\n",weights->weights[i]);
	}
	fclose(tempfile);
    return 0;
}



// read in a problem (in dense format)
data_str * read_problem(const char *filename)
{
	int i,j;
	FILE *fp = fopen(filename,"r");
	
	if(fp == NULL)
	{
		fprintf(stderr,"can't open input file %s\n",filename);
		exit(1);
	}
	
	int l = 0;
	int numdims = 0;
	
	while(1)
	{
		int c;
		do {
			c = getc(fp);
			if(c=='\n') goto out;
			else if(c==EOF) goto out2;
		} while(isspace(c));
		
		ungetc(c,fp);
		fscanf(fp,"%*f");
		numdims++;
	}
out:
	l = 1;
	
	while(1)
	{
		while(1)
		{
			int c;
			do {
				c = getc(fp);
				if(c=='\n') goto out1;
				else if(c==EOF) goto out2;
			} while(isspace(c));
		
			ungetc(c,fp);
			fscanf(fp,"%*f");
		}
out1:
		l++;
	}
	
out2:
	rewind(fp);
	data_str *datastr = Malloc(data_str,1);
	
	datastr->data = Malloc(double *,l);
	double **data = datastr->data;
	
	i = 0;
	l = 0;
	while(1)
	{
		data[l] = Malloc(double,numdims);
		while(1)
		{
			int c;
			do {
				c = getc(fp);
				if(c=='\n') goto out3;
				else if(c==EOF) goto out4;
			} while(isspace(c));
			
			ungetc(c,fp);
			fscanf(fp,"%lf",&data[l][i++]);
		}
		
	out3:
		l++;
		i = 0;
	}

out4:
	fclose(fp);
	datastr->numdims = numdims;
	datastr->numrows = l;
	return datastr;
}


// read in a problem with targets (in dense format)
data_str * read_problem_withtargets(const char *filename)
{
	int i,j;
	FILE *fp = fopen(filename,"r");
	
	if(fp == NULL)
	{
		fprintf(stderr,"can't open input file %s\n",filename);
		exit(1);
	}
	
	int l = 0;
	int numdims = 0;
	
	while(1)
	{
		int c;
		do {
			c = getc(fp);
			if(c=='\n') goto out;
			else if(c==EOF) goto out2;
		} while(isspace(c));
		
		ungetc(c,fp);
		fscanf(fp,"%*f");
		numdims++;
	}
out:
	l = 1;
	
	numdims--;
	
	while(1)
	{
		while(1)
		{
			int c;
			do {
				c = getc(fp);
				if(c=='\n') goto out1;
				else if(c==EOF) goto out2;
			} while(isspace(c));
			
			ungetc(c,fp);
			fscanf(fp,"%*f");
		}
	out1:
		l++;
	}
	
out2:
	rewind(fp);
	data_str *datastr = Malloc(data_str,1);
	
	datastr->data = Malloc(double *,l);
	datastr->targets = Malloc(double,l);
	
	double **data = datastr->data;
	double *targets = datastr->targets;
	
	
	for(i=0;i<l;i++)
	{
		data[i] = Malloc(double,numdims);
		fscanf(fp,"%lf ",&targets[i]);		
		for(j=0;j<numdims;j++)
			fscanf(fp,"%lf",&data[i][j]);		
		fscanf(fp,"\n");
	}

	fclose(fp);
	datastr->numdims = numdims;
	datastr->numrows = l;
	return datastr;
}


