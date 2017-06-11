#include <memory>
#include <thrust/functional.h>
#ifndef STATARRAY_H
#define STATARRAY_H

#include <vector>

enum sortorder
{
	ASC,
	DESC
};

class polynomial
{
private:
	std::vector<float> coefficients;
public:

	explicit polynomial(int degree, std::vector<float>& coefficients) 
	:polynomial(degree)
	{
		this->coefficients = coefficients;
		this->coefficients.resize(degree + 1);
	}

	polynomial(std::vector<float>& coefficients)
	:polynomial(coefficients.size()-1, coefficients)
	{
		
	}

	explicit polynomial(int degree)
	:degree(degree)
	{
		coefficients.resize(degree + 1);
	}

	polynomial(float f)
	:polynomial(0)
	{
		coefficients[0] = f;
	}

	const int degree;

	float& operator[](int i) { return coefficients[i]; }

	std::shared_ptr<polynomial> operator+(const polynomial& other) const;
	std::shared_ptr<polynomial> operator-(const polynomial& other) const;
	std::shared_ptr<polynomial> operator+=(const polynomial& other);
	std::shared_ptr<polynomial> operator-=(const polynomial& other);
	std::shared_ptr<polynomial> operator*(const polynomial& other);
	std::shared_ptr<polynomial> operator/(const polynomial& other);

	std::shared_ptr<polynomial> operator=(const float f) { return std::make_shared<polynomial>(f); }

	std::shared_ptr<polynomial> cut(int degree);
};

// for now it supports only floats (ideally it would also support strings, booleans etc.
//template<class T>
class statarray : public std::vector<float>
{
private:
	std::string name = "";
public:
	std::string get_name() const { return std::string(name); }
	void set_name(const std::string& name) { this->name = name; }

	statarray() :vector() {}
	explicit statarray(std::string name) :vector(), name(name) {}
	statarray(float f) :statarray() { this->push_back(f); }
	statarray(const_iterator begin, const_iterator end) :vector(begin, end) {}


	/**
	 * Concatenates the statarray with itself until it has exactly nelements
	 */
	virtual std::shared_ptr<statarray> rep(int nelements) const;

	/**
	 * generates n random values from N(0,1)
	 */
	static std::shared_ptr<statarray> rnorm(int n, float mean = 0, float stdev = 1);
	/**
	 * generates n random integers from U(minvalue, maxvalue)
	 */
	static std::shared_ptr<statarray> randint(int n, int minvalue = INT_MIN, int maxvalue = INT_MAX);
	/**
	 * generates n random floats from U(minvalue, maxvalue)
	 */
	static std::shared_ptr<statarray> randfloat(int n, float minvalue = 0, float maxvalue = 1);
	/**
	 * generates a range of integers [low_incl, high_excl) (python style)
	 */
	static std::shared_ptr<statarray> range(int low_incl, int high_excl);

	virtual float& operator[](const int& i) { return *(this->begin() + i); }
	virtual std::shared_ptr<statarray> operator[](const vector<int>& ids) const;
	virtual std::shared_ptr<statarray> operator[](const statarray& predicate_vector) const;
 
	/**
	 * element-wise multiplication
	 * shorter statarray is repped to match longer's length
	 */
	virtual std::shared_ptr<statarray> operator*(const statarray& other) const;
	/**
	 * element-wise addition
	 * shorter statarray is repped to match longer's length
	 */
	virtual std::shared_ptr<statarray> operator+(const statarray& other) const;
	/**
	 * element-wise subtraction
	 * shorter statarray is repped to match longer's length
	 */
	virtual std::shared_ptr<statarray> operator-(const statarray& other) const;
	/**
	 * element-wise division
	 * shorter statarray is repped to match longer's length
	 */
	virtual std::shared_ptr<statarray> operator/(const statarray& other) const;
	/**
	 * element-wise power function
	 * shorter statarray is repped to match longer's length
	 */
	virtual std::shared_ptr<statarray> operator^(const statarray& other) const;
 
	/**
	 * element-wise less-than
	 * shorter statarray is repped to match longer's length
	 */
	virtual std::shared_ptr<statarray> operator<(const statarray& other) const;
	/**
	 * element-wise less-or-equal
	 * shorter statarray is repped to match longer's length
	 */
	virtual std::shared_ptr<statarray> operator<=(const statarray& other) const;
	/**
	 * element-wise greater-or-equal
	 * shorter statarray is repped to match longer's length
	 */
	virtual std::shared_ptr<statarray> operator>=(const statarray& other) const;
	/**
	 * element-wise greater-than
	 * shorter statarray is repped to match longer's length
	 */
	virtual std::shared_ptr<statarray> operator>(const statarray& other) const;
	/**
	 * element-wise equals
	 * shorter statarray is repped to match longer's length
	 */
	virtual std::shared_ptr<statarray> operator==(const statarray& other) const;

	/**
	 * element-wise multiplication + assignment
	 * shorter statarray is repped to match longer's length
	 */
	virtual std::shared_ptr<statarray> operator*=(const statarray& other);										
	/**
	 * element-wise addition + assignment
	 * shorter statarray is repped to match longer's length
	 */
	virtual std::shared_ptr<statarray> operator+=(const statarray& other);												
	/**
	 * element-wise subtraction + assignment
	 * shorter statarray is repped to match longer's length
	 */
	virtual std::shared_ptr<statarray> operator-=(const statarray& other);
	/**
	 * element-wise division + assignment
	 * shorter statarray is repped to match longer's length
	 */
	virtual std::shared_ptr<statarray> operator/=(const statarray& other);
	/**
	 * element-wise power + assignment
	 * shorter statarray is repped to match longer's length
	 */
	virtual std::shared_ptr<statarray> operator^=(const statarray& other);

	/**
	 * element-wise _logical_ OR
	 * shorter statarray is repped to match longer's length
	 */
	virtual std::shared_ptr<statarray> operator|(const statarray& other) const;
	/**
	 * element-wise _logical_ AND
	 * shorter statarray is repped to match longer's length
	 */
	virtual std::shared_ptr<statarray> operator&(const statarray& other) const;
	/**
	 * element-wise logical negation
	 */
	virtual std::shared_ptr<statarray> operator!() const;

	/**
	 * element-wise arithmetic negation (unary minus)
	 */
	virtual std::shared_ptr<statarray> operator-() const;

	/**
	 * product of all elements
	 */
	virtual float product() const;
 
	/**
	 * sample of n elements from the statarray
	 */
	virtual std::shared_ptr<statarray> sample(int n) const;

	/**
	 * trims the statarray by given fractions of statarray's size
	 * leave right negative to trim both ends by an equal amount
	 * THIS CHANGE IS PERMANENT!
	 */
	virtual std::shared_ptr<statarray> trim(float left, float right = -1);
	/**
	* trims the statarray by given fractions of statarray's size
	* leave right negative to trim both ends by an equal amount
	* THIS CHANGE IS NOT PERMANENT!
	*/
	virtual std::shared_ptr<statarray> trimmed(float left, float right = -1) const;
	/**
	* trims the statarray by given numbers of elements
	* leave right negative to trim both ends by an equal amount
	* THIS CHANGE IS PERMANENT!
	*/
	virtual std::shared_ptr<statarray> trim(int left, int right = -1);
	/**
	* trims the statarray by given numbers of elements
	* leave right negative to trim both ends by an equal amount
	* THIS CHANGE IS NOT PERMANENT!
	*/
	virtual std::shared_ptr<statarray> trimmed(int left, int right = -1) const;

	/**
	 * sorts the statarray by given sort order (ASC|DESC)
	 * THIS CHANGE IS PERMANENT!
	 */
	virtual std::shared_ptr<statarray> sort(sortorder order = ASC);
	/**
	* sorts the statarray by given sort order (ASC|DESC)
	* THIS CHANGE IS NOT PERMANENT!
	*/
	virtual std::shared_ptr<statarray> sorted(sortorder order = ASC) const;
//	virtual std::shared_ptr<vector<int>> order(sortorder order = ASC) const;

	/**
	 * sum of all elements
	 */
	virtual float sum() const;
	/**
	 * arithmetic mean
	 */
	virtual float mean() const;
	/**
	 * returns value that appears most often in the statarray
	 */
	virtual float mode() const;
	/**
	 * inter-quartile mean
	 */
	virtual float iqm() const;
	/**
	 * minimum
	 */
	virtual float min() const;
	/*
	 * maximum
	 */
	virtual float max() const;
	/**
	 * median/second quartile
	 */
	virtual float median() const;
	/**
	 * first quartile
	 */
	virtual float lquart() const;
	/**
	 * third quartile
	 */
	virtual float uquart() const;
	/**
	 * returns quantile specified by 0 <= q <= 1
	 */
	virtual float quantile(float q) const;
	/**
	 * inter-quartile range
	 */
	virtual float iqr() const;
	/**
	 * for a sample it's estimated by its arithmetic mean
	 */
	virtual float expected_value() const;
	/**
	 * standard deviation
	 */
	virtual float stdev() const;
	/**
	 * variance
	 */
	virtual float variance() const;
	/**
	 * skewness
	 */
	virtual float skewness() const;
	/**
	 * kurtosis
	 */
	virtual float kurtosis() const;
	/**
	 * covariance (elements are taken by index order, so it might be useful to sort the array first!)
	 */
	virtual float covariance(const statarray& other) const;
	/**
	 * correlation (Pearson r) (elements are taken by index order, so it might be useful to sort the array first!)
	 */
	virtual float correlation(const statarray& other) const;
	/**
	 * harmonic mean of the array's elements
	 */
	virtual float harmonic_mean() const;
	/**
	 * geometric mean of the array's elements
	 */
	virtual float geometric_mean() const;
	/**
	 * generalized (power) mean with exponent p
	 */
	virtual float generalized_mean(float p) const;
	/**
	 * winsorized mean: given fraction of extremal (smallest or biggest) elements is replaced by the smallest/biggest element that is not replaced
	 */
	virtual float winsorized_mean(float fraction_left, float fraction_right = -1) const;
	/**
	* winsorized mean: given fraction of extremal (smallest or biggest) elements is replaced by the smallest/biggest element that is not replaced
	*/
	virtual float winsorized_mean(int nleft, int nright = -1) const;
	/**
	* winsorized mean: given fraction of extremal (smallest or biggest) elements is removed
	*/
	virtual float truncated_mean(float fraction_left, float fraction_right = -1) const;
	/**
	* winsorized mean: given fraction of extremal (smallest or biggest) elements is removed
	*/
	virtual float truncated_mean(int nleft, int nright = -1) const;
	/**
	 * weighted arithmetic mean of the array's elements
	 */
	virtual float weighted_arithmetic_mean(const vector<float>& weights) const;
	/**
	 * Shapiro-Wilk's test for normality
	 */
	virtual bool shapiro_wilk_test() const;
	/**
	 * t-test for variable difference
	 */
	virtual bool t_test() const;

//	virtual std::shared_ptr<statarray> transform(const statarray& v, const thrust::binary_function<float, float, float>& binary_functor) const;
//	virtual std::shared_ptr<statarray> transform(const thrust::unary_function<float, float>& unary_functor) const;
//	virtual std::shared_ptr<statarray> transform_modify(const statarray& v, const thrust::binary_function<float, float, float>& binary_functor);
//	virtual std::shared_ptr<statarray> transform_modify(const thrust::unary_function<float, float>& unary_functor);
//
	/**
	 * returns standardized version of the statarray
	 */
	virtual std::shared_ptr<statarray> standardized() const;
	/**
	 * replaces given number of elements with min/max element that is not replaced
	 * THIS IS NOT PERMANENT!
	 */
	virtual std::shared_ptr<statarray> winsorized(int nleft, int nright = -1) const;
	/**
	* replaces given number of elements with min/max element that is not replaced
	* THIS IS NOT PERMANENT!
	*/
	virtual std::shared_ptr<statarray> winsorized(float fraction_left, float fraction_right = -1) const;

	/**
	 * returns array's histogram with given number of bins
	 */
	virtual std::shared_ptr<vector<int>> histogram(int nbins) const;

	/**
	 * returns line specification for least squares method
	 */
	virtual std::shared_ptr<polynomial> least_squares(const statarray& other) const;

	/**
	 * writes array to csv given by filename
	 */
	virtual void to_csv(std::string filename) const;
	/**
	 * reads array from csv given by filename
	 */
	static std::shared_ptr<statarray> from_csv(std::string filename);

	friend float correlation(const statarray& v1, const statarray& v2);
	friend float covariance(const statarray& v1, const statarray& v2);
	friend std::shared_ptr<polynomial> least_squares(const statarray& v1, const statarray& v2);
};

std::shared_ptr<std::vector<bool>> or(std::vector<bool>& v1, std::vector<bool>& v2);
std::shared_ptr<std::vector<bool>> and(std::vector<bool>& v1, std::vector<bool>& v2);
std::shared_ptr<std::vector<bool>> not(std::vector<bool>& v1);

/**
 * prints statarray v in human-readable form
 */
void print(const statarray& v);

#endif
