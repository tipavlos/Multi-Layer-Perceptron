#include "pch.h"
#include <iostream>
#include "NeuralNetwork.h"
#include "mnist\mnist_reader_less.hpp"
#include "mnist\mnist_utils.hpp"

using namespace std;

int main()
{
	auto dataset = mnist::read_dataset<double, uint8_t>();
	normalize_dataset(dataset);
	vector<vector<bool>> y;
	vector<bool> zeros(10, false);
	for (auto v : dataset.training_labels) {
		y.push_back(zeros);
		y.back()[v] = 1;
	}
	int hd[]{ 128 };
	MultiLayerPerceptron *nn = new MultiLayerPerceptron(hd, 1, 10, dataset.training_images[0].size(), 0.01, "tanh", 100, 50);
	nn->fit<double, bool>(dataset.training_images, y);

	float acc = 0;
	for (int i = 0; i < dataset.test_images.size(); i++) {
		acc += (nn->predict_best_class(dataset.test_images[i]) == dataset.test_labels[i]);
	}
	cout << "Accuracy : " << acc / dataset.test_images.size() << endl;
}
