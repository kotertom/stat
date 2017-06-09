#include "statarray.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>

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


std::shared_ptr<statarray> statarray::operator[](const vector<int>& ids) const
{
	std::shared_ptr<statarray> ret;
	ret->reserve(ids.size());
	for (auto id : ids)
	{
		ret->push_back(this->at(id));
	}
	return ret;
}

std::shared_ptr<statarray> statarray::operator[](const statarray& predicate_vector) const
{
	std::shared_ptr<statarray> ret;
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


std::shared_ptr<statarray> statarray::operator*(float f) const
{
	thrust::device_vector<float> d1 = *this;
	thrust::constant_iterator<float> fiterator(f);

	transform(d1.begin(), d1.end(), fiterator, d1.begin(), thrust::multiplies<float>());

	std::shared_ptr<statarray> ret;
	ret->reserve(this->size());

	thrust::copy(d1.begin(), d1.end(), ret->begin());

	return ret;
}

std::shared_ptr<statarray> statarray::operator*(const statarray& other) const
{
	thrust::device_vector<float> d1 = *this;
	thrust::device_vector<float> d2 = other;

	transform(d1.begin(), d1.end(), d2.begin(), d1.begin(), thrust::multiplies<float>());

	std::shared_ptr<statarray> ret;
	ret->reserve(this->size());

	thrust::copy(d1.begin(), d1.end(), ret->begin());

	return ret;
}

std::shared_ptr<statarray> statarray::operator+(float f) const
{
	thrust::device_vector<float> d1 = *this;
	thrust::constant_iterator<float> fiterator(f);

	transform(d1.begin(), d1.end(), fiterator, d1.begin(), thrust::multiplies<float>());

	std::shared_ptr<statarray> ret;
	ret->reserve(this->size());

	thrust::copy(d1.begin(), d1.end(), ret->begin());

	return ret;
}

std::shared_ptr<statarray> statarray::operator+(const statarray& other) const
{
	thrust::device_vector<float> d1 = *this;
	thrust::device_vector<float> d2 = other;

	transform(d1.begin(), d1.end(), d2.begin(), d1.begin(), thrust::plus<float>());

	std::shared_ptr<statarray> ret;
	ret->reserve(this->size());

	thrust::copy(d1.begin(), d1.end(), ret->begin());

	return ret;
}

std::shared_ptr<statarray> statarray::operator-(float f) const
{
	thrust::device_vector<float> d1 = *this;
	thrust::constant_iterator<float> fiterator(f);

	transform(d1.begin(), d1.end(), fiterator, d1.begin(), thrust::minus<float>());

	std::shared_ptr<statarray> ret;
	ret->reserve(this->size());

	thrust::copy(d1.begin(), d1.end(), ret->begin());

	return ret;
}

std::shared_ptr<statarray> statarray::operator-(const statarray& other) const
{
	thrust::device_vector<float> d1 = *this;
	thrust::device_vector<float> d2 = other;

	transform(d1.begin(), d1.end(), d2.begin(), d1.begin(), thrust::minus<float>());

	std::shared_ptr<statarray> ret;
	ret->reserve(this->size());

	thrust::copy(d1.begin(), d1.end(), ret->begin());

	return ret;
}

std::shared_ptr<statarray> statarray::operator/(float f) const
{
	thrust::device_vector<float> d1 = *this;
	thrust::constant_iterator<float> fiterator(f);

	transform(d1.begin(), d1.end(), fiterator, d1.begin(), thrust::divides<float>());

	std::shared_ptr<statarray> ret;
	ret->reserve(this->size());

	thrust::copy(d1.begin(), d1.end(), ret->begin());

	return ret;
}

std::shared_ptr<statarray> statarray::operator/(const statarray& other) const
{
	thrust::device_vector<float> d1 = *this;
	thrust::device_vector<float> d2 = other;

	transform(d1.begin(), d1.end(), d2.begin(), d1.begin(), thrust::divides<float>());

	std::shared_ptr<statarray> ret;
	ret->reserve(this->size());

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

std::shared_ptr<statarray> statarray::operator^(float f) const
{
	thrust::device_vector<float> d1 = *this;
	thrust::constant_iterator<float> fiterator(f);

	transform(d1.begin(), d1.end(), fiterator, d1.begin(), power_functor());

	std::shared_ptr<statarray> ret;
	ret->reserve(this->size());

	thrust::copy(d1.begin(), d1.end(), ret->begin());

	return ret;
}

std::shared_ptr<statarray> statarray::operator^(const statarray& other) const
{
	thrust::device_vector<float> d1 = *this;
	thrust::device_vector<float> d2 = other;

	transform(d1.begin(), d1.end(), d2.begin(), d1.begin(), power_functor());

	std::shared_ptr<statarray> ret;
	ret->reserve(this->size());

	thrust::copy(d1.begin(), d1.end(), ret->begin());

	return ret;
}

std::shared_ptr<statarray> statarray::operator<(float f) const
{
	thrust::device_vector<float> d1 = *this;
	thrust::constant_iterator<float> fiterator(f);

	transform(d1.begin(), d1.end(), fiterator, d1.begin(), thrust::less<float>());

	std::shared_ptr<statarray> ret;
	ret->reserve(this->size());

	thrust::copy(d1.begin(), d1.end(), ret->begin());

	return ret;
}

//std::shared_ptr<polynomial> polynomial::operator+=(const polynomial& other)
//{
//
//}

std::shared_ptr<statarray> operator*(const float& f, const statarray& right)
{
	thrust::device_vector<float> d1 = right;
	thrust::constant_iterator<float> fiterator(f);

	transform(d1.begin(), d1.end(), fiterator, d1.begin(), thrust::multiplies<float>());

	std::shared_ptr<statarray> ret;
	ret->reserve(right.size());

	thrust::copy(d1.begin(), d1.end(), ret->begin());

	return ret;
}

std::shared_ptr<statarray> operator+(const float& f, const statarray& right)
{
	thrust::device_vector<float> d1 = right;
	thrust::constant_iterator<float> fiterator(f);

	transform(d1.begin(), d1.end(), fiterator, d1.begin(), thrust::plus<float>());

	std::shared_ptr<statarray> ret;
	ret->reserve(right.size());

	thrust::copy(d1.begin(), d1.end(), ret->begin());

	return ret;
}



std::shared_ptr<statarray> operator-(const float& f, const statarray& right)
{
	thrust::device_vector<float> d1 = right;
	thrust::constant_iterator<float> fiterator(f);

	transform(thrust::make_transform_iterator(d1.begin(), thrust::negate<float>()), 
		thrust::make_transform_iterator(d1.end(), thrust::negate<float>()),
		fiterator, 
		d1.begin(), 
		thrust::plus<float>()
	);

	std::shared_ptr<statarray> ret;
	ret->reserve(right.size());

	thrust::copy(d1.begin(), d1.end(), ret->begin());

	return ret;
}

struct mirror_div_func
{
	__host__ __device__
	float operator()(float a, float b) const
	{
		return b / a;
	}
};

std::shared_ptr<statarray> operator/(const float& f, const statarray& right)
{
	thrust::device_vector<float> d1 = right;
	thrust::constant_iterator<float> fiterator(f);

	transform(d1.begin(), d1.end(), fiterator, d1.begin(), mirror_div_func());

	std::shared_ptr<statarray> ret;
	ret->reserve(right.size());

	thrust::copy(d1.begin(), d1.end(), ret->begin());

	return ret;
}

struct mirror_pow_func
{
	__host__ __device__
		float operator()(float a, float b) const
	{
		return powf(b, a);
	}
};

std::shared_ptr<statarray> operator^(const float& f, const statarray& right)
{
	thrust::device_vector<float> d1 = right;
	thrust::constant_iterator<float> fiterator(f);

	transform(d1.begin(), d1.end(), fiterator, d1.begin(), mirror_pow_func());

	std::shared_ptr<statarray> ret;
	ret->reserve(right.size());

	thrust::copy(d1.begin(), d1.end(), ret->begin());

	return ret;
}

std::shared_ptr<statarray> statarray::operator-() const
{
	thrust::device_vector<float> d1 = *this;

	transform(d1.begin(), d1.end(), d1.begin(), thrust::negate<float>());

	std::shared_ptr<statarray> ret;
	ret->reserve(this->size());

	thrust::copy(d1.begin(), d1.end(), ret->begin());

	return ret;
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

std::shared_ptr<statarray> operator<(const float& f, const statarray& right)
{
	return right > f;
}

std::shared_ptr<statarray> operator>(const float& f, const statarray& right)
{
	return right < f;
}

std::shared_ptr<statarray> operator<=(const float& f, const statarray& right)
{
	return right >= f;
}

std::shared_ptr<statarray> operator>=(const float& f, const statarray& right)
{
	return right <= f;
}

std::shared_ptr<statarray> operator==(const float& f, const statarray& right)
{
	return right == f;
}

std::shared_ptr<statarray> statarray::operator<(const statarray& other) const
{
	thrust::device_vector<float> d1 = *this;
	thrust::device_vector<float> d2 = other;

	transform(d1.begin(), d1.end(), d2.begin(), d1.begin(), thrust::less<float>());

	std::shared_ptr<statarray> ret;
	ret->reserve(this->size());

	thrust::copy(d1.begin(), d1.end(), ret->begin());

	return ret;
}

std::shared_ptr<statarray> statarray::operator<=(float f) const
{
	thrust::device_vector<float> d1 = *this;
	thrust::constant_iterator<float> fiterator(f);

	transform(d1.begin(), d1.end(), fiterator, d1.begin(), thrust::less_equal<float>());

	std::shared_ptr<statarray> ret;
	ret->reserve(this->size());

	thrust::copy(d1.begin(), d1.end(), ret->begin());

	return ret;
}

std::shared_ptr<statarray> statarray::operator<=(const statarray& other) const
{
	thrust::device_vector<float> d1 = *this;
	thrust::device_vector<float> d2 = other;

	transform(d1.begin(), d1.end(), d2.begin(), d1.begin(), thrust::less_equal<float>());

	std::shared_ptr<statarray> ret;
	ret->reserve(this->size());

	thrust::copy(d1.begin(), d1.end(), ret->begin());

	return ret;
}

std::shared_ptr<statarray> statarray::operator>=(float f) const
{
	thrust::device_vector<float> d1 = *this;
	thrust::constant_iterator<float> fiterator(f);

	transform(d1.begin(), d1.end(), fiterator, d1.begin(), thrust::greater_equal<float>());

	std::shared_ptr<statarray> ret;
	ret->reserve(this->size());

	thrust::copy(d1.begin(), d1.end(), ret->begin());

	return ret;
}

std::shared_ptr<statarray> statarray::operator>=(const statarray& other) const
{
	thrust::device_vector<float> d1 = *this;
	thrust::device_vector<float> d2 = other;

	transform(d1.begin(), d1.end(), d2.begin(), d1.begin(), thrust::greater_equal<float>());

	std::shared_ptr<statarray> ret;
	ret->reserve(this->size());

	thrust::copy(d1.begin(), d1.end(), ret->begin());

	return ret;
}

std::shared_ptr<statarray> statarray::operator>(float f) const
{
	thrust::device_vector<float> d1 = *this;
	thrust::constant_iterator<float> fiterator(f);

	transform(d1.begin(), d1.end(), fiterator, d1.begin(), thrust::greater<float>());

	std::shared_ptr<statarray> ret;
	ret->reserve(this->size());

	thrust::copy(d1.begin(), d1.end(), ret->begin());

	return ret;
}

std::shared_ptr<statarray> statarray::operator>(const statarray& other) const
{
	thrust::device_vector<float> d1 = *this;
	thrust::device_vector<float> d2 = other;

	transform(d1.begin(), d1.end(), d2.begin(), d1.begin(), thrust::greater<float>());

	std::shared_ptr<statarray> ret;
	ret->reserve(this->size());

	thrust::copy(d1.begin(), d1.end(), ret->begin());

	return ret;
}

std::shared_ptr<statarray> statarray::operator==(float f) const
{
	thrust::device_vector<float> d1 = *this;
	thrust::constant_iterator<float> fiterator(f);

	transform(d1.begin(), d1.end(), fiterator, d1.begin(), thrust::equal_to<float>());

	std::shared_ptr<statarray> ret;
	ret->reserve(this->size());

	thrust::copy(d1.begin(), d1.end(), ret->begin());

	return ret;
}

std::shared_ptr<statarray> statarray::operator==(const statarray& other) const
{
	thrust::device_vector<float> d1 = *this;
	thrust::device_vector<float> d2 = other;

	transform(d1.begin(), d1.end(), d2.begin(), d1.begin(), thrust::equal_to<float>());

	std::shared_ptr<statarray> ret;
	ret->reserve(this->size());

	thrust::copy(d1.begin(), d1.end(), ret->begin());

	return ret;
}

std::shared_ptr<statarray> statarray::operator*=(float f)
{
	thrust::device_vector<float> d1 = *this;
	thrust::constant_iterator<float> fiterator(f);

	transform(d1.begin(), d1.end(), fiterator, d1.begin(), thrust::multiplies<float>());

	thrust::copy(d1.begin(), d1.end(), this->begin());

	return std::make_shared<statarray>(*this);
}

std::shared_ptr<statarray> statarray::operator*=(const statarray& other)
{
	thrust::device_vector<float> d1 = *this;
	thrust::device_vector<float> d2 = other;

	transform(d1.begin(), d1.end(), d2.begin(), d1.begin(), thrust::multiplies<float>());

	thrust::copy(d1.begin(), d1.end(), this->begin());

	return std::make_shared<statarray>(*this);
}

std::shared_ptr<statarray> statarray::operator+=(float f)
{
	thrust::device_vector<float> d1 = *this;
	thrust::constant_iterator<float> fiterator(f);

	transform(d1.begin(), d1.end(), fiterator, d1.begin(), thrust::plus<float>());

	thrust::copy(d1.begin(), d1.end(), this->begin());

	return std::make_shared<statarray>(*this);
}

std::shared_ptr<statarray> statarray::operator+=(const statarray& other)
{
	thrust::device_vector<float> d1 = *this;
	thrust::device_vector<float> d2 = other;

	transform(d1.begin(), d1.end(), d2.begin(), d1.begin(), thrust::plus<float>());

	thrust::copy(d1.begin(), d1.end(), this->begin());

	return std::make_shared<statarray>(*this);
}

std::shared_ptr<statarray> statarray::operator-=(float f)
{
	thrust::device_vector<float> d1 = *this;
	thrust::constant_iterator<float> fiterator(f);

	transform(d1.begin(), d1.end(), fiterator, d1.begin(), thrust::minus<float>());

	thrust::copy(d1.begin(), d1.end(), this->begin());

	return std::make_shared<statarray>(*this);
}

std::shared_ptr<statarray> statarray::operator-=(const statarray& other)
{
	thrust::device_vector<float> d1 = *this;
	thrust::device_vector<float> d2 = other;

	transform(d1.begin(), d1.end(), d2.begin(), d1.begin(), thrust::minus<float>());

	thrust::copy(d1.begin(), d1.end(), this->begin());

	return std::make_shared<statarray>(*this);
}

std::shared_ptr<statarray> statarray::operator/=(float f)
{
	thrust::device_vector<float> d1 = *this;
	thrust::constant_iterator<float> fiterator(f);

	transform(d1.begin(), d1.end(), fiterator, d1.begin(), thrust::divides<float>());

	thrust::copy(d1.begin(), d1.end(), this->begin());

	return std::make_shared<statarray>(*this);
}

std::shared_ptr<statarray> statarray::operator/=(const statarray& other)
{
	thrust::device_vector<float> d1 = *this;
	thrust::device_vector<float> d2 = other;

	transform(d1.begin(), d1.end(), d2.begin(), d1.begin(), thrust::divides<float>());

	thrust::copy(d1.begin(), d1.end(), this->begin());

	return std::make_shared<statarray>(*this);
}

std::shared_ptr<statarray> statarray::operator^=(float f)
{
	thrust::device_vector<float> d1 = *this;
	thrust::constant_iterator<float> fiterator(f);

	transform(d1.begin(), d1.end(), fiterator, d1.begin(), power_functor());

	thrust::copy(d1.begin(), d1.end(), this->begin());

	return std::make_shared<statarray>(*this);
}

std::shared_ptr<statarray> statarray::operator^=(const statarray& other)
{
	thrust::device_vector<float> d1 = *this;
	thrust::device_vector<float> d2 = other;

	transform(d1.begin(), d1.end(), d2.begin(), d1.begin(), power_functor());

	thrust::copy(d1.begin(), d1.end(), this->begin());

	return std::make_shared<statarray>(*this);
}

std::shared_ptr<statarray> statarray::operator|(const statarray& other) const
{
	thrust::device_vector<float> d1 = *this;
	thrust::device_vector<float> d2 = other;

	transform(d1.begin(), d1.end(), d2.begin(), d1.begin(), thrust::logical_or<float>());

	std::shared_ptr<statarray> ret;
	ret->reserve(this->size());

	thrust::copy(d1.begin(), d1.end(), ret->begin());

	return ret;
}

std::shared_ptr<statarray> statarray::operator&(const statarray& other) const
{
	thrust::device_vector<float> d1 = *this;
	thrust::device_vector<float> d2 = other;

	transform(d1.begin(), d1.end(), d2.begin(), d1.begin(), thrust::logical_and<float>());

	std::shared_ptr<statarray> ret;
	ret->reserve(this->size());

	thrust::copy(d1.begin(), d1.end(), ret->begin());

	return ret;
}

std::shared_ptr<statarray> statarray::operator!() const
{
	thrust::device_vector<float> d1 = *this;

	transform(d1.begin(), d1.end(), d1.begin(), thrust::logical_not<float>());

	std::shared_ptr<statarray> ret;
	ret->reserve(this->size());

	thrust::copy(d1.begin(), d1.end(), ret->begin());

	return ret;
}

std::shared_ptr<statarray> statarray::sample(int n) const
{
}

std::shared_ptr<statarray> statarray::sorted(sortorder order) const
{
}

std::shared_ptr<std::vector<int>> statarray::order(sortorder order) const
{
}

float statarray::sum() const
{
}

float statarray::mean() const
{
}

float statarray::mode() const
{
}

float statarray::iqm() const
{
}

float statarray::min() const
{
}

float statarray::max() const
{
}

float statarray::median() const
{
}

float statarray::lquart() const
{
}

float statarray::uquart() const
{
}

float statarray::quantile(float q) const
{
}

float statarray::iqr() const
{
}

float statarray::stdev() const
{
}

float statarray::variance() const
{
}

float statarray::skewness() const
{
}

float statarray::kurtosis() const
{
}

float statarray::covariance(const statarray& other) const
{
}

float statarray::correlation(const statarray& other) const
{
}

float statarray::harmonic_mean() const
{
}

float statarray::geometric_mean(int k) const
{
}

float statarray::generalized_mean(int k) const
{
}

float statarray::winsorized_mean(float fraction) const
{
}

float statarray::truncated_mean(float fraction) const
{
}

float statarray::weighted_arithmetic_mean(const vector<float>& weights) const
{
}

bool statarray::shapiro_wilk_test() const
{
}

bool statarray::t_test() const
{
}

std::shared_ptr<statarray> statarray::standardized() const
{
}

std::shared_ptr<std::vector<int>> statarray::histogram(int nbins) const
{
}

std::shared_ptr<polynomial> statarray::least_squares(const statarray& other) const
{
}

void statarray::to_csv(std::string filename) const
{
}

void statarray::from_csv(std::string filename) const
{
}

float correlation(const statarray& v1, const statarray& v2)
{
}

std::shared_ptr<polynomial> least_squares(const statarray& v1, const statarray& v2)
{
}

std::shared_ptr<std::vector<bool>> or(std::vector<bool>& v1, std::vector<bool>& v2)
{
}

std::shared_ptr<std::vector<bool>> and(std::vector<bool>& v1, std::vector<bool>& v2)
{
}

std::shared_ptr<std::vector<bool>> not(std::vector<bool>& v1, std::vector<bool>& v2)
{
}
