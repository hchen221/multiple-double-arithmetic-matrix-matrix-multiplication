#include "floatybits.h"

/*
val_bits(bits) parses an array or vector of bits and returns the numerical value
*/
int val_bits(vector<int> bits) {
    int s = 0;
    for (int i=0;i<bits.size();i++)
        s = 2*s+bits[i];
    return s;
}

/*
expandfracbits(fr) takes fr in [0,1] on input and returns a 52 bit representation
*/
vector<int> expandfracbits(double fr) {
    vector<int> bits;
    double temp = fr;
    int i = -1;
    while (temp>0) {
        if (temp >= pow(2.0,i)) {
            bits.push_back(1);
            temp -= pow(2.0,i);
        } else {
            bits.push_back(0);
        }
        i -= 1;
    }
    while (bits.size()<52) {
        bits.push_back(0);
    }
    return bits;
}

/*
expandexpbits(n) takes an exponent integer n on input and returns an 11 bit representation
*/
vector<int> expandexpbits(int n) {
    vector<int> bits;
    int temp = n;
    while (temp>0) {
        bits.insert(bits.begin(),temp%2);
        temp = floor(temp/2);
    }
    while (bits.size() < 11) {
        bits.insert(bits.begin(),0);
    }
    return bits;
}

/*
bitform(x) take a double x on input and returns the 64 bit representation
*/
vector<int> bitform(double x) {
    int exponent;
    double fraction = frexp(x, &exponent );
    vector<int> bits;
    if (x>=0) {
        bits.push_back(0);
    } else {
        bits.push_back(1);
    }
    vector<int> fracbits = expandfracbits(fraction);
    vector<int> expbits = expandexpbits(exponent);
    bits.insert(bits.end(),expbits.begin(),expbits.end());
    bits.insert(bits.end(),fracbits.begin(),fracbits.end());
    return bits;
}

/*
random_double_bits(signbit,exponent) returns the bit representation of a random double with a given sign bit and exponent
*/
vector<int> random_double_bits(int signbit,int exponent) {
    vector<int> bits;
    bits.push_back(signbit);
    vector<int> expbits = expandexpbits(exponent);
    bits.insert(bits.end(),expbits.begin(),expbits.end());
    vector<int> fracbits;
    for (int i=0;i<52;i++) {
        fracbits.push_back(rand()%2);
    }
    bits.insert(bits.end(),fracbits.begin(),fracbits.end());
    return bits;
}

/*
double_rep(bits) returns the numerical double of a 64 bit number
*/
double double_rep(vector<int> bits) {
    vector<int> expbits;
    expbits.insert(expbits.end(),bits.begin()+1,bits.begin()+12);
    int exponent = val_bits(expbits)-1023;
    double fraction = 0;
    for (int i=1;i<=52;i++) {
        fraction += pow(2.0,-i)*bits[11+i];
    }
    double signbit = bits[0];
    return pow(-1.0,signbit)*fraction*pow(2.0,exponent);
}

/*
random_pd(expmin,expmax) returns a random p-double
*/
vector<double> random_pd(int expmin,int expmax,int p) {
    int hiexp;
    if (expmax==expmin) {
        hiexp = expmin+1023;
    } else {
        hiexp = expmin+1023+rand()%(expmax-expmin);
    }
    vector<double> parts;
    for (int i=0;i<p;i++) {
        vector<int> bits = random_double_bits(0,hiexp-52*i);
        parts.push_back(double_rep(bits));
    }
    return parts;
}

/*
split4(bits) takes a 64 bit representation on input and returns the quad double in numerical form
*/
vector<double> split4(vector<int> bits) {
    vector<int> bits1 = bits;
    vector<int> bits2 = bits;
    vector<int> bits3 = bits;
    for (int i=0;i<52;i++) {
        if (i<13) {
            bits2[11+i] = 0;
            bits3[11+i] = 0;
        }
        if (i>=13&&i<26) {
            bits1[11+i] = 0;
            bits3[11+i] = 0;
        }
        if (i>=26&&i<39) {
            bits1[11+i] = 0;
            bits2[11+i] = 0;
        }
    }
    vector<double> D;
    double d1 = double_rep(bits1);
    double d2 = double_rep(bits2);
    double d3 = double_rep(bits3);
    D.push_back(d1);
    D.push_back(d2);
    D.push_back(d3);
    D.push_back(double_rep(bits)-(d1+d2+d3));
    return D;
}

/*
split4pd(x) applies split4 to a p-double x then returns the combined 4p-double
*/
vector<double> split4pd(vector<double> x) {
    vector<double> x4;
    for (int i=0;i<x.size();i++) {
        vector<double> xi4 = split4(bitform(x[i]));
        x4.insert(x4.end(),xi4.begin(),xi4.end());
    }
    return x4;
}
