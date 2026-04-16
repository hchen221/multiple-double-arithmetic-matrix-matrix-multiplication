#ifndef _FLOATYBITS_
#define _FLOATYBITS_

//#include "../../PHCpack/src/GPU/Norms/double_double_functions.h"

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
vector<int> random_double_bits(int signbit,int exponent,bool fixfirst=true);

/*
double_rep(bits) returns the numerical double of a 64 bit number
*/
double double_rep(vector<int> bits);

/*
random_pd(expmin,expmax) returns a random p-double
*/
vector<double> random_pd(int expmin,int expmax,int p);

/*
is_balanced checks if 2 doubles x and y are balanced with respect to an exponent threshold, adapted from splitting_doubles.cpp from PHCpack
*/
bool is_balanced(double x,double y,int threshold);

/*
balance takes 2 doubles x and y and balances them with a given exponent threshold, adapted from splitting_doubles.cpp from PHCpack 
*/
void balance(double &x,double &y,int threshold);

/*
balance4 takes a split 4p double x (from indices a to b) on input and balances the exponents
*/
void balance4(vector<double> &x,int a,int b);

/*
split4(bits) takes a 64 bit representation on input and returns the quad double in numerical form
*/
vector<double> split4(vector<int> bits);

/*
split4pd(x) applies split4 to a vector of p-doubles x then returns the combined vector of 4p-doubles
*/
vector<double> split4pd(vector<double> x,int p);

/*
balance8 takes a split 8p double x (from indices a to b) on input and balances the exponents
*/
void balance8(vector<double> &x,int a,int b);

/*
split8(bits) takes a 64 bit representation on input and returns the quad double in numerical form
*/
vector<double> split8(vector<int> bits);

/*
split8pd(x) applies split8 to a vector of p-doubles x then returns the combined vector of 8p-doubles
*/
vector<double> split8pd(vector<double> x,int p);

/*
mixbalance(x,a,b) takes a double-double split into 12 parts (4 for high, 8 for low), indicated from indices a to b on input, and balances the exponents
*/
void mixbalance(vector<double> &x, int a,int b);

/*
mixsplit2(x) takes a vector of double doubles  then splits the high part into 4, low into 8
*/
vector<double> mixsplit2(vector<double> x);
#endif
