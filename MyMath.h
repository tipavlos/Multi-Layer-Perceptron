#pragma once
#include <cmath>
#include <utility>
#include <algorithm>

#define PI 3.14159265

double percentile_z(double p);

inline double z2(double x, double mx, double sx) {
	return (x - mx) * (x - mx) / (sx * sx);
}

double bivariate_proba(double x, double y, double mx, double sx, double my, double sy);
double density(double x, double mx, double sx);
std::pair<float,float> get_mean_stdev(float *x,int size);
float *z_normalize(float *x, int size);
float *max_min_normalize(float *x, int size);
float *softmax(float *x, int size);

double tanh_d(double x);
double sigmoid(double x);
double sigmoid_d(double x);
inline double relu(double x) {
	return x > 0 ? x : 0;
}
inline double relu_d(double x) {
	return x > 0;
}
inline double linear(double x) {
	return x;
}
inline double linear_d(double x) {
	return 1;
}

