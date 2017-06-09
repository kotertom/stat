#include <iostream>
#include <stdio.h>
#include "statarray.cuh"




int main()
{
	polynomial v(4, std::vector<float>({1,2,3}));
	polynomial v2(4, std::vector<float>({ -1, -2,-4 }));

	std::shared_ptr<polynomial> v3 = v + v2;
	for (int i = 0; i <= v3->degree; i++)
	{
		std::cout << (*v3)[i] << std::endl;
	}

	return EXIT_SUCCESS;
}