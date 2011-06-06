//#include <iostream>
#include <stdio.h>
#include "DBN_Cluster.h"

#define MAXNUMLAYERS 10
#define MAX_NUM_EXAMPLES 2000000
#define DEFAULT_NUM_FEATURES 100000

//#define DEFAULT_DATA_FILE "K562_preprocessed.dat"//_smallerset.dat"
//#define DEFAULT_DATA_FILE "K562_only_random_smallerset.dat"

data_str * read_problem(const char *filename);

int main (int argc, char * const argv[]) 
{	
	data_str * datatrain,*datatest;
	//DEFAULT VALUES
	SMDparams params = {5,1e-5,0.99,0.05,0.1,16};
	char typesoflayers [3] = {2,1,0};
	int maxepochpre = 5;
	int numlayers = 4;
	int numunits [MAXNUMLAYERS] = {0,300,150,10,0,10,0,0,0,0};
	int batchsizepre = 100;

	int i,j;
	weights_str *weights;
		
	for(i=1;i< argc-2;i++)
	{
		if(argv[i][0] == '-')
		{
			switch (argv[i][1]) 
			{
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

	datatrain = read_problem(argv[argc-2]);
	datatest = read_problem(argv[argc-1]);

	numunits[0] = datatrain->numdims;
	
	weights = DBN_regress(datatrain->data,datatrain->targets,datatest->data,datatest->targets,datatrain->numrows, datatest->numrows, maxepochpre, numlayers, numunits, batchsizepre, typesoflayers, &params);
	
	FILE *tempfile = fopen("postweightsoutput.dat","w");
	for(i=0;i<weights->numweights;i++)
	{
		printf("%.10e\n",weights->weights[i]);
	}
	fclose(tempfile);
    return 0;
}
/*
void putinvals(double *temp)
{
	for(int i=0;i<10;i++)
		temp[i] = i;
}
*/

/*// read in a problem (in dense format)
data_str * read_problem(const char *filename)
{
	int i;
	FILE *fp = fopen(filename,"r");
	
	if(fp == NULL)
	{
		fprintf(stderr,"can't open input file %s\n",filename);
		exit(1);
	}
	
	int l = 0;
	int numdims = 0;

	data_str *datastr = Malloc(data_str,1);
	
	datastr->data = Malloc(double *,MAX_NUM_EXAMPLES);
	double **data = datastr->data;

	data[0] = Malloc(double,DEAFULT_NUM_FEATURES);
	
	i = 0;
	while(1)
	{
		int c;
		do {
			c = getc(fp);
			if(c=='\n') goto out;
			else if(c==EOF) goto out2;
		} while(isspace(c));
		
		ungetc(c,fp);
		fscanf(fp,"%lf",&data[0][i]);
		numdims++;
	}

out:
	data[++l] = Malloc(double,numdims);
	
	Realloc(data[0],double,numdims);

	i=0;
	
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
			fscanf(fp,"%lf",&data[l][i++]);
		}
out1:
		data[++l] = Malloc(double,numdims);
		i = 0;
	}

	
out2:
	Realloc(datastr->data,double *,l);
	
	fclose(fp);
	datastr->numdims = numdims;
	datastr->numrows = l;
	return datastr;
}
*/

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
	
//	double * tempval = Malloc(double,MAX_NUM_EXAMPLES*DEFAULT_NUM_FEATURES);
//	int numvals = 0;
	
	while(1)
	{
		int c;
		do {
			c = getc(fp);
			if(c=='\n') goto out;
			else if(c==EOF) goto out2;
		} while(isspace(c));
		
		ungetc(c,fp);
		fscanf(fp,"%*f");//,&tempval[numvals++]);
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
			fscanf(fp,"%*f");//,&tempval[numvals++]);
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
/*	
	i = 0;
	l = 0;
	while(1)
	{
		data[l] = Malloc(double,numdims);
		while(1)
		{
			fscanf(fp,"%lf",&targets[l]);
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
*/
	/*	
	
	for(i=0;i<l;i++)
	{
		data[i] = Malloc(double,numdims);
		for(j=0;j<numdims;j++)
		{
			data[i][j] = *tempval;
			tempval++;
		}
	}
*/	
	fclose(fp);
//	free(tempval);
	datastr->numdims = numdims;
	datastr->numrows = l;
	return datastr;
}

