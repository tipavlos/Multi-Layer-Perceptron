#pragma once
#include<iostream>
#include<string>
#include <algorithm>
#include <cstring>
#include "RandomSelect.h"
#include "MyMath.h"
#include <vector>
#include <ctime>

class MultiLayerPerceptron {
public:
	MultiLayerPerceptron() {}
	MultiLayerPerceptron(int *hidden_layer_sizes, int hidden_layers, int output_size, int feature_size, double learning_rate, std::string activation_function, int epochs, int batchsize) {
		h_layer_size = hidden_layers + 1;
		step = learning_rate;
		n_features = feature_size;
		n_iter = epochs;
		bsize = batchsize;
		if (activation_function == "relu") {
			act_function = relu;
			act_function_d = relu_d;
		}
		else if (activation_function == "sigmoid") {
			act_function = sigmoid;
			act_function_d = sigmoid_d;
		}
		else if (activation_function == "tanh") {
			act_function = tanh;
			act_function_d = tanh_d;
		}
		else {
			act_function = linear;
			act_function_d = linear_d;
		}

		lv = new Layer[h_layer_size + 1];
		int weights_size = hidden_layer_sizes[0] * n_features;
		for (int i = 1; i < h_layer_size - 1; i++) {
			weights_size += hidden_layer_sizes[i - 1] * hidden_layer_sizes[i];
		}
		weights_size += output_size * hidden_layer_sizes[h_layer_size - 2];
		pool = (double *)malloc(sizeof(double)*weights_size * 2);
		next_pool_pos = pool;
		lv[0].init(n_features, 0, next_pool_pos);
		lv[1].init(hidden_layer_sizes[0], n_features, next_pool_pos);
		for (int i = 2; i < h_layer_size; i++) {
			lv[i].init(hidden_layer_sizes[i - 1], hidden_layer_sizes[i - 2], next_pool_pos);
		}
		lv[h_layer_size].init(output_size, hidden_layer_sizes[h_layer_size - 2], next_pool_pos);
		output = new double[output_size];
		loss = new double[output_size];
		param_size = lv[1].nrsize * (n_features + 1);
		for (int i = 2; i < h_layer_size + 1; i++) {
			param_size += lv[i].nrsize*(lv[i - 1].nrsize + 1);
		}
	}
	virtual ~MultiLayerPerceptron() {
		delete output;
		free(pool);
		delete[] lv;
	}
	struct Neuron {
	public:
		Neuron() {}
		void init(int s, double *&next_pool_pos) {
			wsize = s;
			w = next_pool_pos;
			next_pool_pos += s;
			dw = next_pool_pos;
			next_pool_pos += s;
			db = 0;
			b = fastrand() % 2 ? (fastrand() % 10000 + 1) / 10000.0 : -(fastrand() % 10000 + 1) / 10000.0;
			for (int i = 0; i < s; i++) {
				w[i] = fastrand() % 2 ? (fastrand() % 10000 + 1) / 10000.0 : -(fastrand() % 10000 + 1) / 10000.0;
			}
		}
		void init_d() {
			db = 0;
			for (int i = 0; i < wsize; i++) {
				dw[i] = 0;
			}
		}
		void update_d(double len_t) {
			b -= db * len_t;
			for (int i = 0; i < wsize; i++) {
				w[i] -= dw[i] * len_t;
			}
		}
		double *w, *dw, val, val_d, b, db, dummy;
		int wsize;
	};
	class Layer {
	public:
		Layer() {}
		virtual ~Layer() {
			delete nr;
		}
		void init(int s, int ws, double *&next_pool_pos) {
			nrsize = s;
			nr = new Neuron[s];
			if (ws == 0) {
				return;
			}
			for (int i = 0; i < s; i++) {
				nr[i].init(ws, next_pool_pos);
			}
		}
		void init_d() {
			for (int i = 0; i < nrsize; i++) {
				nr[i].init_d();
			}
		}
		void update_d(double len_t) {
			for (int i = 0; i < nrsize; i++) {
				nr[i].update_d(len_t);
			}
		}
		Neuron *nr;
		int nrsize;
	};
	void divide_parameters(int length) {
		double d_length = 1.0 / length;
		for (int z = 1; z < h_layer_size + 1; z++) {
			lv[z].update_d(d_length);
		}
	}
	void reward_multiply(double reward) {
		for (int z = 1; z < h_layer_size + 1; z++) {
			lv[z].update_d(reward);
		}
	}
	void init_sums() {
		for (int z = 1; z < h_layer_size + 1; z++) {
			lv[z].init_d();
		}
	}
	template<typename Xtype = double, typename Ytype = double>
	void fit(const std::vector<std::vector<Xtype>> &X, const std::vector<std::vector<Ytype>> &y) {
		int N = X.size();
		for (int i = 0; i < n_iter; i++) {
			clock_t start = clock();
			for (int j = 0; j < N; j += bsize) {
				init_sums();
				for (int k = j; k < N && k < j + bsize; k++) {
					sum_dw < Xtype, Ytype >(X[k], y[k]);
				}
				divide_parameters(std::min(bsize, N - j));
			}
			float elapsed_time = float(clock() - start) / CLOCKS_PER_SEC;
			std::cout << "epoch " << i << " |elapsed time: "<< elapsed_time <<" seconds." << std::endl;
		}
	}
	template<typename Xtype = double>
	void predict_hidden(const std::vector<Xtype> &x) {
		for (int j = 0; j < n_features; j++) {
			lv[0].nr[j].val = x[j];
		}
		for (int i = 1; i < h_layer_size; i++) {
			for (int j = 0; j < lv[i].nrsize; j++) {
				double sum = lv[i].nr[j].b;
				for (int k = 0; k < lv[i - 1].nrsize; k++) {
					sum += lv[i].nr[j].w[k] * lv[i - 1].nr[k].val;
				}
				lv[i].nr[j].val = act_function(sum);
				lv[i].nr[j].val_d = act_function_d(sum);
			}
		}
	}
	template<typename Xtype = double>
	double *predict_all(const std::vector<Xtype> &x) {
		predict_hidden(x);
		for (int j = 0; j < lv[h_layer_size].nrsize; j++) {
			output[j] = lv[h_layer_size].nr[j].b;
			for (int k = 0; k < lv[h_layer_size - 1].nrsize; k++) {
				output[j] += lv[h_layer_size].nr[j].w[k] * lv[h_layer_size - 1].nr[k].val;
			}
		}
		return output;
	}
	template<typename Xtype = double>
	double predict_max(const std::vector<Xtype> &x) {
		predict_hidden(x);
		double max_output = -1000000;
		for (int j = 0; j < lv[h_layer_size].nrsize; j++) {
			double sum = lv[h_layer_size].nr[j].b;
			for (int k = 0; k < lv[h_layer_size - 1].nrsize; k++) {
				sum += lv[h_layer_size].nr[j].w[k] * lv[h_layer_size - 1].nr[k].val;
			}
			if (sum > max_output) {
				max_output = sum;
			}
		}
		return max_output;
	}
	template<typename Xtype = double>
	int predict_best_class(const std::vector<Xtype> &x) {
		predict_hidden(x);
		int best_class_id = 0;
		double max_output = 0;
		for (int j = 0; j < lv[h_layer_size].nrsize; j++) {
			double sum = lv[h_layer_size].nr[j].b;
			for (int k = 0; k < lv[h_layer_size - 1].nrsize; k++) {
				sum += lv[h_layer_size].nr[j].w[k] * lv[h_layer_size - 1].nr[k].val;
			}
			if (sum > max_output) {
				max_output = sum;
				best_class_id = j;
			}
		}
		return best_class_id;
	}
	void print() {
		for (int i = 1; i < h_layer_size + 1; i++) {
			std::cout << "Layer " << i << std::endl;
			for (int j = 0; j < lv[i].nrsize; j++) {
				std::cout << "Neuron " << j << " : " << lv[i].nr[j].b << ",";
				for (int k = 0; k < lv[i].nr[j].wsize; k++) {
					std::cout << lv[i].nr[j].w[k] << ",";
				}
				std::cout << std::endl;
			}
		}
	}
	inline int get_feature_size() { return n_features; }
	inline int get_target_size() { return lv[h_layer_size].nrsize; }
	inline double get_param_size() { return param_size; }
private:
	template<typename Xtype = double, typename Ytype = double>
	void sum_dw(const std::vector<Xtype> &X, const std::vector<Ytype> &y) {
		double *y_pred = predict_all(X);
		typename std::vector<Ytype>::const_iterator beg = y.cbegin();
		for (typename std::vector<Ytype>::const_iterator it = beg; it != y.cend(); ++it) {
			loss[it - beg] = 2 * step * (y_pred[it - beg] - *it);
		}
		for (int i = 1; i < h_layer_size - 1; i++) {
			for (int j = 0; j < lv[i].nrsize; j++) {
				for (int jj = 0; jj < lv[i + 1].nrsize; jj++) {
					lv[i + 1].nr[jj].dummy = lv[i + 1].nr[jj].w[j];
				}
				for (int ii = i + 2; ii < h_layer_size + 1; ii++) {
					for (int jj = 0; jj < lv[ii].nrsize; jj++) {
						lv[ii].nr[jj].dummy = 0;
						for (int k = 0; k < lv[ii - 1].nrsize; k++) {
							lv[ii].nr[jj].dummy += lv[ii].nr[jj].w[k] * lv[ii - 1].nr[k].val_d*lv[ii - 1].nr[k].dummy;
						}
					}
				}
				double total_dummy = 0;
				for (int k = 0; k < lv[h_layer_size].nrsize; k++) {
					total_dummy += loss[k] * lv[h_layer_size].nr[k].dummy;
				}
				for (int k = 0; k < lv[i].nr[j].wsize; k++) {
					lv[i].nr[j].dw[k] += lv[i].nr[j].val_d*lv[i - 1].nr[k].val * total_dummy;
				}
				lv[i].nr[j].db += lv[i].nr[j].val_d*total_dummy;
			}
		}

		for (int j = 0; j < lv[h_layer_size - 1].nrsize; j++) {
			double total_dummy = 0;
			for (int k = 0; k < lv[h_layer_size].nrsize; k++) {
				total_dummy += loss[k] * lv[h_layer_size].nr[k].w[j];
			}
			for (int k = 0; k < lv[h_layer_size - 1].nr[j].wsize; k++) {
				lv[h_layer_size - 1].nr[j].dw[k] += total_dummy * lv[h_layer_size - 1].nr[j].val_d*lv[h_layer_size - 2].nr[k].val;
			}
			lv[h_layer_size - 1].nr[j].db += total_dummy * lv[h_layer_size - 1].nr[j].val_d;
		}
		for (int j = 0; j < lv[h_layer_size].nrsize; j++) {
			for (int k = 0; k < lv[h_layer_size].nr[j].wsize; k++) {
				lv[h_layer_size].nr[j].dw[k] += loss[j] * lv[h_layer_size - 1].nr[k].val;
			}
			lv[h_layer_size].nr[j].db += loss[j];
		}

	}
	int neuron_size, h_layer_size, n_features, n_iter, bsize, param_size;
	double step, *pool, *output, *next_pool_pos, *loss;
	long double e_loss;
	double(*act_function)(double), (*act_function_d)(double);
	Layer *lv;
};