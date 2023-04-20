/*
 * Copyright 2023 Sisi Miao, Communications Engineering Lab @ KIT
 *
 * SPDX-License-Identifier: MIT
 *
 * This file accompanies the paper
 *     S. Miao, A. Schnerring, H. Li and L. Schmalen,
 *     "Neural belief propagation decoding of quantum LDPC codes using overcomplete check matrices,"
 *     Proc. IEEE Inform. Theory Workshop (ITW), Saint-Malo, France, Apr. 2023, https://arxiv.org/abs/2212.10245
 */
#ifndef BPDECODING_STABILIZIERCODES_H
#define BPDECODING_STABILIZIERCODES_H
#include <string>
#include <vector>
enum class stabilizerCodesType { GeneralizedBicycle = 0, HypergraphProduct = 1, toric = 3 };

class stabilizerCodes {
  public:
    stabilizerCodes(unsigned n, unsigned k, unsigned m, stabilizerCodesType codeType, bool trained = false);

    std::vector<bool> decode(unsigned int L, double epsilon);

    std::vector<bool> flooding_decode(unsigned int L, double epsilon);

    std::vector<bool> check_success(const double *Taux, const double *Tauy, const double *Tauz);
    void load_cn_weights();
    void load_vn_weights();
    void load_llr_weights();
    void read_H();
    void read_G();
    static inline bool trace_inner_product(unsigned a, unsigned b);
    std::string code_type_string() const;
    bool check_symplectic();

    void add_error_given_epsilon(double epsilon);

    // void add_error_given_positions(int pos[], int error[], int size);

    void calculate_syndrome(); // also check if the error is all 0, if true, not decoding needed

    static double quantize_belief(double Taux, double Tauy, double Tauz);

  private:
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
