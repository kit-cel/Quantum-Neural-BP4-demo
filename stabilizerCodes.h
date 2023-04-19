//
// Created by sisi on 1/13/23.
//
#ifndef BPDECODING_STABILIZIERCODES_H
#define BPDECODING_STABILIZIERCODES_H
#include <assert.h>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <string>
#include <vector>
enum class stabilizerCodesType { GeneralizedBicycle = 0, HpergraphProduct = 1, toric = 3 };

class stabilizerCodes {
  public:
    stabilizerCodes(unsigned n, unsigned k, unsigned m, stabilizerCodesType codeType, bool trained = false);

    std::vector<bool> decode(unsigned int L, double epsilon);

    std::vector<bool> floodingDecode(unsigned int L, double epsilon, const std::vector<unsigned> error);

    std::vector<bool> checkSucess(const double *Taux, const double *Tauy, const double *Tauz);
    void loadWeights_cn();
    void loadWeights_vn();
    void loadWeights_llr();
    void readH();
    void readG();
    inline unsigned traceInnerProduct(unsigned a, unsigned b);
    bool checksymplectic();

    void addErrorGivenEpsilon(double epsilon);

    void addErrorGivenPositions(int pos[], int error[], int size);

    void calSyndrome(); // also check if the error is all 0, if true, not decoding needed

    double bielief_quantize(double Taux, double Tauy, double Tauz);

    bool print_msg = false;

    stabilizerCodesType mycodetype;
    unsigned N;
    unsigned K;
    unsigned M;
    unsigned G_rows;
    bool mTrained;
    unsigned trained_iter;
    std::vector<unsigned> error;
    std::vector<unsigned> error_hat;
    std::vector<unsigned> syn;
    std::vector<std::string> errorString;
    std::vector<std::string> errorHatString;
    unsigned maxDc;
    unsigned maxDv;
    std::vector<unsigned> dv;
    std::vector<unsigned> dc;
    std::vector<std::vector<unsigned>> Nv;
    std::vector<std::vector<unsigned>> Mc;
    std::vector<std::vector<unsigned>> checkVal;
    std::vector<std::vector<unsigned>> varVal;

    std::vector<std::vector<unsigned>> Nvk;
    std::vector<std::vector<unsigned>> Mck;

    std::vector<std::vector<unsigned>> G;

    std::vector<std::vector<std::vector<double>>> weights_cn;
    std::vector<std::vector<std::vector<double>>> weights_vn;
    std::vector<std::vector<double>> weights_llr;
};
#endif // BPDECODING_STABILIZIERCODES_H
