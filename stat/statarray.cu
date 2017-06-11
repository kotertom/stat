#include "statarray.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <thrust/extrema.h>
#include <algorithm>

std::shared_ptr<polynomial> polynomial::operator+(const polynomial& other) const
{
	thrust::device_vector<float> d1 = this->coefficients;
	thrust::device_vector<float> d2 = other.coefficients;

	transform(d1.begin(), d1.end(), d2.begin(), d2.begin(), thrust::plus<float>());

	std::shared_ptr<polynomial> ret = std::make_shared<polynomial>(this->degree);

	thrust::copy(d2.begin(), d2.end(), ret->coefficients.begin());
	
	return ret;
}

std::shared_ptr<polynomial> polynomial::operator-(const polynomial& other) const
{
	thrust::device_vector<float> d1 = this->coefficients;
	thrust::device_vector<float> d2 = other.coefficients;

	transform(d1.begin(), d1.end(), d2.begin(), d2.begin(), thrust::minus<float>());

	std::shared_ptr<polynomial> ret = std::make_shared<polynomial>(this->degree);

	thrust::copy(d2.begin(), d2.end(), ret->coefficients.begin());

	return ret;
}
//
//std::shared_ptr<polynomial> polynomial::operator+=(const polynomial& other)
//{
//}
//
//std::shared_ptr<polynomial> polynomial::operator-=(const polynomial& other)
//{
//}
//
//std::shared_ptr<polynomial> polynomial::operator*(const polynomial& other)
//{
//}
//
//std::shared_ptr<polynomial> polynomial::operator/(const polynomial& other)
//{
//}
//
//std::shared_ptr<polynomial> polynomial::cut(int degree)
//{
//}


//std::shared_ptr<statarray> statarray::rnorm(int n)
//{
//}
//
//std::shared_ptr<statarray> statarray::randint(int n, int minvalue, int maxvalue)
//{
//}
//
//std::shared_ptr<statarray> statarray::randfloat(int n, float minvalue, float maxvalue)
//{
//}


std::shared_ptr<statarray> statarray::rep(int nelements) const
{
	if(this->size() > nelements)
	{
		return std::make_shared<statarray>(this->begin(),this->begin()+(nelements-1));
	}
	else if(this->size() < nelements)
	{
		std::shared_ptr<statarray> ret = std::make_shared<statarray>(this->begin(), this->end());
		ret->reserve(nelements);
		while(ret->size() < nelements)
		{
			ret->insert(ret->end(), ret->begin(), ret->end());
		}
		ret->resize(nelements);
		ret->shrink_to_fit();
		return ret;
	}
	return std::make_shared<statarray>(this->begin(), this->end());
}

std::shared_ptr<statarray> statarray::operator[](const vector<int>& ids) const
{
	std::shared_ptr<statarray> ret = std::make_shared<statarray>("");
	ret->reserve(ids.size());
	for (auto id : ids)
	{
		ret->push_back(this->at(id));
	}
	return ret;
}

std::shared_ptr<statarray> statarray::operator[](const statarray& predicate_vector) const
{
	std::shared_ptr<statarray> ret = std::make_shared<statarray>("");
	ret->reserve(predicate_vector.size());
	auto i = this->begin();
	for(auto belongs : predicate_vector)
	{
		if(belongs)
		{
			ret->push_back(*i);
		}
		++i;
	}
	std::vector<float>(ret->begin(), ret->end()).swap(*ret);
	return ret;
}

std::shared_ptr<statarray> statarray::operator*(const statarray& other) const
{
	int size = std::max(this->size(), other.size());
	thrust::device_vector<float> d1 = *this->rep(size);
	thrust::device_vector<float> d2 = *other.rep(size);

	thrust::transform(d1.begin(), d1.end(), d2.begin(), d1.begin(), thrust::multiplies<float>());

	std::shared_ptr<statarray> ret = std::make_shared<statarray>(this->get_name().append("*").append(other.get_name()));
	ret->resize(size);

	thrust::copy(d1.begin(), d1.end(), ret->begin());

	return ret;
}

std::shared_ptr<statarray> statarray::operator+(const statarray& other) const
{
	int size = std::max(this->size(), other.size());
	thrust::device_vector<float> d1 = *this->rep(size);
	thrust::device_vector<float> d2 = *other.rep(size);

	thrust::transform(d1.begin(), d1.end(), d2.begin(), d1.begin(), thrust::plus<float>());

	std::shared_ptr<statarray> ret = std::make_shared<statarray>(this->get_name().append("+").append(other.get_name()));
	ret->resize(size);

	thrust::copy(d1.begin(), d1.end(), ret->begin());

	return ret;
}

std::shared_ptr<statarray> statarray::operator-(const statarray& other) const
{
	int size = std::max(this->size(), other.size());
	thrust::device_vector<float> d1 = *this->rep(size);
	thrust::device_vector<float> d2 = *other.rep(size);

	thrust::transform(d1.begin(), d1.end(), d2.begin(), d1.begin(), thrust::minus<float>());

	std::shared_ptr<statarray> ret = std::make_shared<statarray>(this->get_name().append("-").append(other.get_name()));
	ret->resize(size);

	thrust::copy(d1.begin(), d1.end(), ret->begin());

	return ret;
}

std::shared_ptr<statarray> statarray::operator/(const statarray& other) const
{
	int size = std::max(this->size(), other.size());
	thrust::device_vector<float> d1 = *this->rep(size);
	thrust::device_vector<float> d2 = *other.rep(size);

	thrust::transform(d1.begin(), d1.end(), d2.begin(), d1.begin(), thrust::divides<float>());

	std::shared_ptr<statarray> ret = std::make_shared<statarray>(this->get_name().append("/").append(other.get_name()));
	ret->resize(size);

	thrust::copy(d1.begin(), d1.end(), ret->begin());

	return ret;
}

struct power_functor
{
	__host__ __device__
	float operator()(float a, float k) const
	{
		return powf(a, k);
	}
};


std::shared_ptr<statarray> statarray::operator^(const statarray& other) const
{
	int size = std::max(this->size(), other.size());
	thrust::device_vector<float> d1 = *this->rep(size);
	thrust::device_vector<float> d2 = *other.rep(size);

	thrust::transform(d1.begin(), d1.end(), d2.begin(), d1.begin(), power_functor());

	std::shared_ptr<statarray> ret = std::make_shared<statarray>(this->get_name().append("^").append(other.get_name()));
	ret->resize(size);

	thrust::copy(d1.begin(), d1.end(), ret->begin());

	return ret;
}



//std::shared_ptr<polynomial> polynomial::operator+=(const polynomial& other)
//{
//
//}


struct mirror_div_func
{
	__host__ __device__
	float operator()(float a, float b) const
	{
		return b / a;
	}
};

struct mirror_pow_func
{
	__host__ __device__
		float operator()(float a, float b) const
	{
		return powf(b, a);
	}
};

std::shared_ptr<statarray> statarray::operator-() const
{
	thrust::device_vector<float> d1 = *this;

	thrust::transform(d1.begin(), d1.end(), d1.begin(), thrust::negate<float>());

	std::shared_ptr<statarray> ret = std::make_shared<statarray>(std::string("-").append(this->get_name()));
	ret->resize(this->size());

	thrust::copy(d1.begin(), d1.end(), ret->begin());

	return ret;
}

float statarray::product() const
{
	thrust::device_vector<float> d = *this;

	return thrust::reduce(d.begin(), d.end(), 1, thrust::multiplies<float>());
}

//std::shared_ptr<polynomial> polynomial::operator-=(const polynomial& other)
//{
//}
//
//std::shared_ptr<polynomial> polynomial::operator*(const polynomial& other)
//{
//}
//
//std::shared_ptr<polynomial> polynomial::operator/(const polynomial& other)
//{
//}
//
//std::shared_ptr<polynomial> polynomial::cut(int degree)
//{
//}


std::shared_ptr<statarray> statarray::operator<(const statarray& other) const
{
	int size = std::max(this->size(), other.size());
	thrust::device_vector<float> d1 = *this->rep(size);
	thrust::device_vector<float> d2 = *other.rep(size);

	thrust::transform(d1.begin(), d1.end(), d2.begin(), d1.begin(), thrust::less<float>());

	std::shared_ptr<statarray> ret = std::make_shared<statarray>(this->get_name().append("<").append(other.get_name()));
	ret->resize(size);

	thrust::copy(d1.begin(), d1.end(), ret->begin());

	return ret;
}

std::shared_ptr<statarray> statarray::operator<=(const statarray& other) const
{
	int size = std::max(this->size(), other.size());
	thrust::device_vector<float> d1 = *this->rep(size);
	thrust::device_vector<float> d2 = *other.rep(size);

	thrust::transform(d1.begin(), d1.end(), d2.begin(), d1.begin(), thrust::less_equal<float>());

	std::shared_ptr<statarray> ret = std::make_shared<statarray>(this->get_name().append("<=").append(other.get_name()));
	ret->resize(size);

	thrust::copy(d1.begin(), d1.end(), ret->begin());

	return ret;
}

std::shared_ptr<statarray> statarray::operator>=(const statarray& other) const
{
	int size = std::max(this->size(), other.size());
	thrust::device_vector<float> d1 = *this->rep(size);
	thrust::device_vector<float> d2 = *other.rep(size);

	thrust::transform(d1.begin(), d1.end(), d2.begin(), d1.begin(), thrust::greater_equal<float>());

	std::shared_ptr<statarray> ret = std::make_shared<statarray>(this->get_name().append(">=").append(other.get_name()));
	ret->resize(size);

	thrust::copy(d1.begin(), d1.end(), ret->begin());

	return ret;
}

std::shared_ptr<statarray> statarray::operator>(const statarray& other) const
{
	int size = std::max(this->size(), other.size());
	thrust::device_vector<float> d1 = *this->rep(size);
	thrust::device_vector<float> d2 = *other.rep(size);

	thrust::transform(d1.begin(), d1.end(), d2.begin(), d1.begin(), thrust::greater<float>());

	std::shared_ptr<statarray> ret = std::make_shared<statarray>(this->get_name().append(">").append(other.get_name()));
	ret->resize(size);

	thrust::copy(d1.begin(), d1.end(), ret->begin());

	return ret;
}

std::shared_ptr<statarray> statarray::operator==(const statarray& other) const
{
	int size = std::max(this->size(), other.size());
	thrust::device_vector<float> d1 = *this->rep(size);
	thrust::device_vector<float> d2 = *other.rep(size);

	thrust::transform(d1.begin(), d1.end(), d2.begin(), d1.begin(), thrust::equal_to<float>());

	std::shared_ptr<statarray> ret = std::make_shared<statarray>(this->get_name().append("==").append(other.get_name()));
	ret->resize(size);

	thrust::copy(d1.begin(), d1.end(), ret->begin());

	return ret;
}

std::shared_ptr<statarray> statarray::operator*=(const statarray& other)
{
	int size = std::max(this->size(), other.size());
	thrust::device_vector<float> d1 = *this->rep(size);
	thrust::device_vector<float> d2 = *other.rep(size);

	thrust::transform(d1.begin(), d1.end(), d2.begin(), d1.begin(), thrust::multiplies<float>());

	this->resize(size);
	thrust::copy(d1.begin(), d1.end(), this->begin());

	return std::make_shared<statarray>(*this);
}

std::shared_ptr<statarray> statarray::operator+=(const statarray& other)
{
	int size = std::max(this->size(), other.size());
	thrust::device_vector<float> d1 = *this->rep(size);
	thrust::device_vector<float> d2 = *other.rep(size);

	thrust::transform(d1.begin(), d1.end(), d2.begin(), d1.begin(), thrust::plus<float>());

	this->resize(size);
	thrust::copy(d1.begin(), d1.end(), this->begin());

	return std::make_shared<statarray>(*this);
}

std::shared_ptr<statarray> statarray::operator-=(const statarray& other)
{
	int size = std::max(this->size(), other.size());
	thrust::device_vector<float> d1 = *this->rep(size);
	thrust::device_vector<float> d2 = *other.rep(size);

	thrust::transform(d1.begin(), d1.end(), d2.begin(), d1.begin(), thrust::minus<float>());

	this->resize(size);
	thrust::copy(d1.begin(), d1.end(), this->begin());

	return std::make_shared<statarray>(*this);
}

std::shared_ptr<statarray> statarray::operator/=(const statarray& other)
{
	int size = std::max(this->size(), other.size());
	thrust::device_vector<float> d1 = *this->rep(size);
	thrust::device_vector<float> d2 = *other.rep(size);

	thrust::transform(d1.begin(), d1.end(), d2.begin(), d1.begin(), thrust::divides<float>());

	this->resize(size);
	thrust::copy(d1.begin(), d1.end(), this->begin());

	return std::make_shared<statarray>(*this);
}

std::shared_ptr<statarray> statarray::operator^=(const statarray& other)
{
	int size = std::max(this->size(), other.size());
	thrust::device_vector<float> d1 = *this->rep(size);
	thrust::device_vector<float> d2 = *other.rep(size);

	thrust::transform(d1.begin(), d1.end(), d2.begin(), d1.begin(), power_functor());

	this->resize(size);
	thrust::copy(d1.begin(), d1.end(), this->begin());

	return std::make_shared<statarray>(*this);
}

std::shared_ptr<statarray> statarray::operator|(const statarray& other) const
{
	int size = std::max(this->size(), other.size());
	thrust::device_vector<float> d1 = *this->rep(size);
	thrust::device_vector<float> d2 = *other.rep(size);

	thrust::transform(d1.begin(), d1.end(), d2.begin(), d1.begin(), thrust::logical_or<float>());

	std::shared_ptr<statarray> ret = std::make_shared<statarray>(this->get_name().append("|").append(other.get_name()));
	ret->resize(size);

	thrust::copy(d1.begin(), d1.end(), ret->begin());

	return ret;
}

std::shared_ptr<statarray> statarray::operator&(const statarray& other) const
{
	int size = std::max(this->size(), other.size());
	thrust::device_vector<float> d1 = *this->rep(size);
	thrust::device_vector<float> d2 = *other.rep(size);

	thrust::transform(d1.begin(), d1.end(), d2.begin(), d1.begin(), thrust::logical_and<float>());

	std::shared_ptr<statarray> ret = std::make_shared<statarray>(this->get_name().append("&").append(other.get_name()));
	ret->resize(size);

	thrust::copy(d1.begin(), d1.end(), ret->begin());

	return ret;
}

std::shared_ptr<statarray> statarray::operator!() const
{
	thrust::device_vector<float> d1 = *this;

	thrust::transform(d1.begin(), d1.end(), d1.begin(), thrust::logical_not<float>());

	std::shared_ptr<statarray> ret = std::make_shared<statarray>(std::string("1").append(this->get_name()));
	ret->resize(this->size());

	thrust::copy(d1.begin(), d1.end(), ret->begin());

	return ret;
}

//std::shared_ptr<statarray> statarray::sample(int n) const
//{
//}

std::shared_ptr<statarray> statarray::sort(sortorder order)
{
	thrust::swap(*this, *this->sorted(order));
	return std::make_shared<statarray>(*this);
}

std::shared_ptr<statarray> statarray::sorted(sortorder order) const
{
	//TODO: implement
	return nullptr;
}

//std::shared_ptr<std::vector<int>> statarray::order(sortorder order) const
//{
//}
//
float statarray::sum() const
{
	thrust::device_vector<float> d(this->begin(), this->end());

	return thrust::reduce(d.begin(), d.end(), 0, thrust::plus<float>());
}

float statarray::mean() const
{
	return this->sum() / this->size();
}
//
//float statarray::mode() const
//{
//}
//
//float statarray::iqm() const
//{
//}

float statarray::min() const
{
	thrust::device_vector<float> d(this->begin(), this->end());

	return *thrust::min_element(d.begin(), d.end());
}

float statarray::max() const
{
	thrust::device_vector<float> d(this->begin(), this->end());

	return *thrust::max_element(d.begin(), d.end());
}

float statarray::median() const
{
	return this->quantile(0.5);
}

float statarray::lquart() const
{
	return this->quantile(0.25);
}

float statarray::uquart() const
{
	return this->quantile(0.75);
}

float statarray::quantile(float q) const
{
	q = std::min<float>(1, std::max<float>(0, q));
	std::shared_ptr<statarray> sorted = this->sorted();
	int before = floor(q * this->size());
	int after = floor((1 - q) * this->size());

	return ((*sorted)[before] + (*sorted)[this->size() - after - 1]) * 0.5;
}

float statarray::iqr() const
{
	return this->uquart() - this->lquart();
}

float statarray::expected_value() const
{
	return this->mean();
}

float statarray::stdev() const
{
	return sqrt(this->variance());
}

float statarray::variance() const
{
	return (*this ^ 2)->expected_value() - pow(this->expected_value(), 2);
}

float statarray::skewness() const
{
	return 3 * (this->mean() - this->median()) / this->stdev();
}

float statarray::kurtosis() const
{
	return (*(*this - this->expected_value()) ^ 4)->expected_value();
}

//float statarray::covariance(const statarray& other) const
//{
//}
//
//float statarray::correlation(const statarray& other) const
//{
//}

float statarray::harmonic_mean() const
{
	return float(this->size()) / ((*this) ^ (-1))->sum();
}

float statarray::geometric_mean() const
{
	return powf(this->product(), 1.0 / this->size());
}

//float statarray::generalized_mean(int k) const
//{
//}
//
//float statarray::winsorized_mean(float fraction) const
//{
//}
//
//float statarray::truncated_mean(float fraction) const
//{
//}
//
//float statarray::weighted_arithmetic_mean(const vector<float>& weights) const
//{
//}
//
//bool statarray::shapiro_wilk_test() const
//{
//}
//
//bool statarray::t_test() const
//{
//}
//
//std::shared_ptr<statarray> statarray::standardized() const
//{
//}
//
//std::shared_ptr<std::vector<int>> statarray::histogram(int nbins) const
//{
//}
//
//std::shared_ptr<polynomial> statarray::least_squares(const statarray& other) const
//{
//}
//
//void statarray::to_csv(std::string filename) const
//{
//}
//
//void statarray::from_csv(std::string filename) const
//{
//}
//
//float correlation(const statarray& v1, const statarray& v2)
//{
//}
//
//std::shared_ptr<polynomial> least_squares(const statarray& v1, const statarray& v2)
//{
//}

//std::shared_ptr<statarray> statarray::transform(const statarray& v, const thrust::binary_function<float, float, float>& binary_functor) const
//{
//	thrust::device_vector<float> d1(this->begin(), this->end());
//	thrust::device_vector<float> d2(v.begin(), v.end());
//
//	thrust::transform(d1.begin(), d1.end(), d2.begin(), d1.begin(), binary_functor);
//
//	std::shared_ptr<statarray> ret = std::make_shared<statarray>();
//	ret->resize(this->size());
//
//	thrust::copy(d1.begin(), d1.end(), ret->begin());
//
//	return ret;
//}
//
//std::shared_ptr<statarray> statarray::transform(const thrust::unary_function<float, float>& unary_functor) const
//{
//	thrust::device_vector<float> d(this->begin(), this->end());
//
//	thrust::transform(d.begin(), d.end(), d.begin(),unary_functor);
//
//	std::shared_ptr<statarray> ret = std::make_shared<statarray>();
//	ret->resize(this->size());
//
//	thrust::copy(d.begin(), d.end(), ret->begin());
//
//	return ret;
//}
//
//std::shared_ptr<statarray> statarray::transform_modify(const statarray& v, const thrust::binary_function<float, float, float>& binary_functor)
//{
//	thrust::device_vector<float> d1(this->begin(), this->end());
//	thrust::device_vector<float> d2(v.begin(), v.end());
//
//	thrust::transform(d1.begin(), d1.end(), d2.begin(), d1.begin(), binary_functor);
//
//	thrust::copy(d1.begin(), d1.end(), this->begin());
//
//	return std::make_shared<statarray>(*this);
//}
//
//std::shared_ptr<statarray> statarray::transform_modify(const thrust::unary_function<float,float>& unary_functor)
//{
//	thrust::device_vector<float> d(this->begin(), this->end());
//
//	thrust::transform(d.begin(), d.end(), d.begin(), unary_functor);
//
//	thrust::copy(d.begin(), d.end(), this->begin());
//
//	return std::make_shared<statarray>(*this);
//}

std::shared_ptr<std::vector<bool>> or(std::vector<bool>& v1, std::vector<bool>& v2)
{
	thrust::device_vector<bool> d1 = v1;
	thrust::device_vector<bool> d2 = v2;

	thrust::transform(d1.begin(), d1.end(), d2.begin(), d1.begin(), thrust::logical_or<bool>());

	std::shared_ptr<std::vector<bool>> ret;
	ret->resize(v1.size());

	thrust::copy(d1.begin(), d1.end(), ret->begin());

	return ret;
}

std::shared_ptr<std::vector<bool>> and(std::vector<bool>& v1, std::vector<bool>& v2)
{
	thrust::device_vector<bool> d1 = v1;
	thrust::device_vector<bool> d2 = v2;

	thrust::transform(d1.begin(), d1.end(), d2.begin(), d1.begin(), thrust::logical_and<bool>());

	std::shared_ptr<std::vector<bool>> ret;
	ret->resize(v1.size());

	thrust::copy(d1.begin(), d1.end(), ret->begin());

	return ret;
}

std::shared_ptr<std::vector<bool>> not(std::vector<bool>& v)
{
	thrust::device_vector<bool> d = v;

	thrust::transform(d.begin(), d.end(), d.begin(), thrust::logical_not<bool>());

	std::shared_ptr<std::vector<bool>> ret;
	ret->resize(v.size());

	thrust::copy(d.begin(), d.end(), ret->begin());

	return ret;
}

void print(const statarray& v)
{
	std::cout << "Statarray " << v.get_name() << "\n";
	std::cout << "Size: " << v.size() << "\n";
	for (auto value : v)
	{
		std::cout << value << std::endl;
	}
	printf("\n");
}
