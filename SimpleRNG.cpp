/*
 *  SimpleRNG.cpp
 *  DBNClust
 *
 *  Created by Karen Hovsepian on 11/2/09.
 *  Copyright 2009 __MyCompanyName__. All rights reserved.
 *
 */

#include <math.h>
#include <time.h>
#include <stdio.h>


#include "SimpleRNG.h"
#define TIME_CONSTANT 116444736000000000LLU  //134774*24*3600*10000000

/// <summary>
/// SimpleRNG is a simple random number generator based on 
/// George Marsaglia's MWC (multiply with carry) generator.
/// Although it is very simple, it passes Marsaglia's DIEHARD
/// series of random number generator tests.
/// 
/// Written by John D. Cook 
/// http://www.johndcook.com
/// </summary>

SimpleRNG::SimpleRNG():m_w(521288629), m_z(362436069)
{
	// These values are not magical, just the default values Marsaglia used.
	// Any pair of unsigned integers should be fine.
//	m_w = 521288629;
//	m_z = 362436069;
}

// The random generator seed can be set three ways:
// 1) specifying two non-zero unsigned integers
// 2) specifying one non-zero unsigned integer and taking a default value for the second
// 3) setting the seed from the system time

void SimpleRNG::SetSeed(unsigned long u, unsigned long v)
{
	if (u != 0) m_w = u; 
	if (v != 0) m_z = v;
}


void SimpleRNG::SetSeed(unsigned long u)
{
	m_w = u;
}

void SimpleRNG::SetSeedFromSystemTime()
{
//	System.DateTime dt = System.DateTime.Now;
	time_t now = time(NULL);
	
//	long x1 = 4294967295UL;
	unsigned long long x = TIME_CONSTANT + now*10000000LLU; //making x be equivalent to Windows File time, which is the number of 100 nanosecond intervals since 0:00 Jan 1st, 1601.  134774 is the number of days between 1601 and 1970. 
//	printf("%llu\n",x);
	
//	dt.ToFileTime();									
	SetSeed((unsigned long)(x >> 16), (unsigned long)(x % 4294967296LLU));
}

// Produce a uniform random sample from the open interval (0, 1).
// The method will not return either end point.
double SimpleRNG::GetUniform()
{
	// 0 <= u <= 2^32
	unsigned long u = GetUint();
	// The magic number below is 1/(2^32 + 2).
	// The result is strictly between 0 and 1.
	return (u + 1) * 2.328306435454494e-10;
}

// Get normal (Gaussian) random sample with mean 0 and standard deviation 1
double SimpleRNG::GetNormal()
{ 
	// Use Box-Muller algorithm
	double u1 = GetUniform();
	double u2 = GetUniform();
	double r = sqrt( -2.0*log(u1) );
	return r*sin(2.0*M_PI*u2);
}

// Get normal (Gaussian) random sample with specified mean and standard deviation
double SimpleRNG::GetNormal(double mean, double standardDeviation)
{
	return mean + standardDeviation*GetNormal();
}

// Get exponential random sample with mean 1
double SimpleRNG::GetExponential()
{
	return -log( GetUniform() );
}

// Get exponential random sample with specified mean
double SimpleRNG::GetExponential(double mean)
{
	return mean*GetExponential();
}

// This is the heart of the generator.
// It uses George Marsaglia's MWC algorithm to produce an unsigned integer.
// See http://www.bobwheeler.com/statistics/Password/MarsagliaPost.txt
unsigned long SimpleRNG::GetUint()
{
	m_z = 36969 * (m_z & 65535) + (m_z >> 16);
	m_w = 18000 * (m_w & 65535) + (m_w >> 16);
	return (m_z << 16) + m_w;
}
