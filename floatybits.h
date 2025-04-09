#ifndef _FLOATYBITS_
#define _FLOATYBITS_

#include <cmath>
#include <iostream>
#include <vector>
#include <random>

using namespace std;


/*
val_bits(bits) parses an array or vector of bits and returns the numerical value
*/
int val_bits(vector<int> bits);

/*
expandfracbits(fr) takes fr in [0,1] on input and returns a 52 bit representation
*/
vector<int> expandfracbits(double fr);

/*
expandexpbits(n) takes an exponent integer n on input and returns an 11 bit representation
*/
vector<int> expandexpbits(int n);

/*
bitform(x) take a double x on input and returns the 64 bit representation
*/
vector<int> bitform(double x);

/*
random_double_bits(signbit,exponent) returns the bit representation of a random double with a given sign bit and exponent
*/
vector<int> random_double_bits(int signbit,int exponent);

/*
double_rep(bits) returns the numerical double of a 64 bit number
*/
double double_rep(vector<int> bits);

/*
random_pd(expmin,expmax) returns a random p-double
*/
vector<double> random_pd(int expmin,int expmax,int p=2);

/*
split4(bits) takes a 64 bit representation on input and returns the quad double in numerical form
*/
vector<double> split4(vector<int> bits);

/*
split4pd(x) applies split4 to a p-double x then returns the combined 4p-double
*/
vector<double> split4pd(vector<double> x);

#endif