#include <memory>
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


	virtual std::shared_ptr<statarray> rep(int nelements) const;

//	static std::shared_ptr<statarray> rnorm(int n);
//	static std::shared_ptr<statarray> randint(int n, int minvalue = INT_MIN, int maxvalue = INT_MAX);
//	static std::shared_ptr<statarray> randfloat(int n, float minvalue = 0, float maxvalue = 1);

	virtual std::shared_ptr<statarray> operator[](const vector<int>& ids) const;
	virtual std::shared_ptr<statarray> operator[](const statarray& predicate_vector) const;
 
//	virtual std::shared_ptr<statarray> operator*(float f) const;
//	friend std::shared_ptr<statarray> operator*(const float& f, const statarray& right);
	virtual std::shared_ptr<statarray> operator*(const statarray& other) const;
 
//	virtual std::shared_ptr<statarray> operator+(float f) const;
//	friend std::shared_ptr<statarray> operator+(const float& f, const statarray& right);
	virtual std::shared_ptr<statarray> operator+(const statarray& other) const;
 
//	virtual std::shared_ptr<statarray> operator-(float f) const;
//	friend std::shared_ptr<statarray> operator-(const float& f, const statarray& right);
	virtual std::shared_ptr<statarray> operator-(const statarray& other) const;

//	virtual std::shared_ptr<statarray> operator/(float f) const;
//	friend std::shared_ptr<statarray> operator/(const float& f, const statarray& right);
	virtual std::shared_ptr<statarray> operator/(const statarray& other) const;

//	virtual std::shared_ptr<statarray> operator^(float f) const;
//	friend std::shared_ptr<statarray> operator^(const float& f, const statarray& right);
	virtual std::shared_ptr<statarray> operator^(const statarray& other) const;
 
//	virtual std::shared_ptr<statarray> operator<(float f) const;
	virtual std::shared_ptr<statarray> operator<(const statarray& other) const;
//	virtual std::shared_ptr<statarray> operator<=(float f) const;
	virtual std::shared_ptr<statarray> operator<=(const statarray& other) const;
//	virtual std::shared_ptr<statarray> operator>=(float f) const;
	virtual std::shared_ptr<statarray> operator>=(const statarray& other) const;
//	virtual std::shared_ptr<statarray> operator>(float f) const;
	virtual std::shared_ptr<statarray> operator>(const statarray& other) const;
//	virtual std::shared_ptr<statarray> operator==(float f) const;
	virtual std::shared_ptr<statarray> operator==(const statarray& other) const;

//	friend std::shared_ptr<statarray> operator<(const float& f, const statarray& right);
//	friend std::shared_ptr<statarray> operator>(const float& f, const statarray& right);
//	friend std::shared_ptr<statarray> operator<=(const float& f, const statarray& right);
//	friend std::shared_ptr<statarray> operator>=(const float& f, const statarray& right);
//	friend std::shared_ptr<statarray> operator==(const float& f, const statarray& right);

//	virtual std::shared_ptr<statarray> operator*=(float f);
	virtual std::shared_ptr<statarray> operator*=(const statarray& other);
												
//	virtual std::shared_ptr<statarray> operator+=(float f);
	virtual std::shared_ptr<statarray> operator+=(const statarray& other);
												
//	virtual std::shared_ptr<statarray> operator-=(float f);
	virtual std::shared_ptr<statarray> operator-=(const statarray& other);
												
//	virtual std::shared_ptr<statarray> operator/=(float f);
	virtual std::shared_ptr<statarray> operator/=(const statarray& other);
												
//	virtual std::shared_ptr<statarray> operator^=(float f);
	virtual std::shared_ptr<statarray> operator^=(const statarray& other);

	virtual std::shared_ptr<statarray> operator|(const statarray& other) const;
	virtual std::shared_ptr<statarray> operator&(const statarray& other) const;
	virtual std::shared_ptr<statarray> operator!() const;

	virtual std::shared_ptr<statarray> operator-() const;
 
//	virtual std::shared_ptr<statarray> sample(int n) const;
//
//	virtual std::shared_ptr<statarray> sort(sortorder order = ASC);
//	virtual std::shared_ptr<statarray> sorted(sortorder order = ASC) const;
//	virtual std::shared_ptr<vector<int>> order(sortorder order = ASC) const;
//
//	virtual float sum() const;
//	virtual float mean() const;
//	virtual float mode() const;
//	virtual float iqm() const;
//	virtual float min() const;
//	virtual float max() const;
//	virtual float median() const;
//	virtual float lquart() const;
//	virtual float uquart() const;
//	virtual float quantile(float q) const;
//	virtual float iqr() const;
//	virtual float stdev() const;
//	virtual float variance() const;
//	virtual float skewness() const;
//	virtual float kurtosis() const;
//	virtual float covariance(const statarray& other) const;
//	virtual float correlation(const statarray& other) const;
//	virtual float harmonic_mean() const;
//	virtual float geometric_mean(int k) const;
//	virtual float generalized_mean(int k) const;
//	virtual float winsorized_mean(float fraction) const;
//	virtual float truncated_mean(float fraction) const;
//	virtual float weighted_arithmetic_mean(const vector<float>& weights) const;
//	virtual bool shapiro_wilk_test() const;
//	virtual bool t_test() const;

//	virtual std::shared_ptr<statarray> standardized() const;
//
//	virtual std::shared_ptr<vector<int>> histogram(int nbins) const;
//
//	virtual std::shared_ptr<polynomial> least_squares(const statarray& other) const;

//	virtual void to_csv(std::string filename) const;
//	virtual void from_csv(std::string filename) const;

//	friend float correlation(const statarray& v1, const statarray& v2);
//	friend std::shared_ptr<polynomial> least_squares(const statarray& v1, const statarray& v2);
};

std::shared_ptr<std::vector<bool>> or(std::vector<bool>& v1, std::vector<bool>& v2);
std::shared_ptr<std::vector<bool>> and(std::vector<bool>& v1, std::vector<bool>& v2);
std::shared_ptr<std::vector<bool>> not(std::vector<bool>& v1);

void print(const statarray& v);

#endif
