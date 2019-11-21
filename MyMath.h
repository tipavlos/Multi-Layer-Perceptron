#pragma once
#include <cmath>
#include <utility>
#include <algorithm>

#define PI 3.14159265



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

