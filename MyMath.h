#pragma once
#include <cmath>


#define PI 3.14159265


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
double tanh_d(double x) {
	double tanhx = tanh(x);
	return 1 - tanhx * tanhx;
}
double sigmoid(double x) {
	return 1 / (1 + exp(-x));
}
double sigmoid_d(double x) {
	double sigx = sigmoid(x);
	return sigx * (1 - sigx);
}
