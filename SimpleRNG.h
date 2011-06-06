class SimpleRNG
{
private:
	unsigned long m_w;
	unsigned long m_z;

public:
	SimpleRNG();
	// The random generator seed can be set three ways:
	// 1) specifying two non-zero unsigned longegers
	// 2) specifying one non-zero unsigned longeger and taking a default value for the second
	// 3) setting the seed from the system time

	void SetSeed(unsigned long u, unsigned long v);
	void SetSeed(unsigned long u);
	void SetSeedFromSystemTime();
	double GetUniform();
	double GetNormal();
	double GetNormal(double mean, double standardDeviation);
	double GetExponential();
	double GetExponential(double mean);
	unsigned long GetUint();
};

