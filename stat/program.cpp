#include <iostream>
#include <stdio.h>
#include "statarray.cuh"


#define ERRIF(pred, str, errcode) if(pred) { fprintf(stderr, "Error in %s at %d, errcode %d: %s\n", __FILE__, __LINE__, errcode, str); return errcode;}


int main()
{
	cudaError_t cuerr;
	ERRIF((cuerr = cudaSetDevice(0)) != cudaSuccess, "Failed to set cuda device", cuerr);
//	cudaFree(0);
	std::cout << "cuda device initialized\n";

//	auto a = statarray::range(0, 20);
//	print(*a->histogram(5));


	thrust::counting_iterator<float> i1(10);
	auto a1 = statarray::range(0, 1e2);
	auto a2 = statarray::range(15, 1e2 + 15);

//	a1->resize(1e5);
//	a2->resize(1e5);
//	thrust::copy_n(i1, 1e5, a1->begin());
//	thrust::copy_n(i1, 1e5, a2->begin());

	print(*a1);
	print(*a2);

	print(*(*a1 / *a2));
	printf("%f\n", (*a1 + *a2)->sum());
	printf("%f\n", (*a1 + *a2)->mean());


	*a1 / *a2;

	auto a3 = std::make_shared<statarray>("a3");
	a3->push_back(1);
	a3->push_back(2);
	a3->push_back(3);
	a3->push_back(4);

	print(*a3);
	printf("LQ: %f,\nMQ: %f,\nUQ: %f,\nIQR: %f\n\n", a3->lquart(), a3->median(), a3->uquart(), a3->iqr());

	a3->push_back(5);

	print(*a3);
	printf("LQ: %f,\nMQ: %f,\nUQ: %f,\nIQR: %f\n\n", a3->lquart(), a3->median(), a3->uquart(), a3->iqr());

	print(*(*a1^2));

	print(*(statarray(1.f) / *a2));

	print(*(a2->sorted(DESC)));

	print(*a2);

	print(*a2->sort());

	print(*a2);

	a3->to_csv("a3.csv");

	std::shared_ptr<statarray> a4 = statarray::from_csv("a3.csv");

	print(*a4);

	print(*a4->winsorized(0.25f));

	print(*statarray::rnorm(10000));

	statarray::range(0, 1e10);

	return EXIT_SUCCESS;
}
