#include <iostream>
#include <stdio.h>
#include "statarray.cuh"
#include <thrust/iterator/counting_iterator.h>
#include <cuda_runtime_api.h>


int main()
{
	cudaSetDevice(0);

	statarray a1("a1");
	statarray a2("a2");
	thrust::counting_iterator<float> i1(10);
	for(int i = 0; i < 10; i++)
	{
		a1.push_back(i1[i-10]);
		a2.push_back(i1[i+5]);
	}

	print(a1);
	print(a2);

	print(*(a1 / a2));
	printf("%f\n", (a1 + a2)->sum());
	printf("%f\n", (a1 + a2)->mean());

	print(*(a1^2));

	print(*(statarray(1.f) / a2));

	return EXIT_SUCCESS;
}
