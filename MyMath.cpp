#include "pch.h"
#include "MyMath.h"
#include <cstdlib>



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