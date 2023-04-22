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
#include "stabilizerCodes.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>

stabilizerCodes::stabilizerCodes(unsigned n, unsigned k, unsigned m, stabilizerCodesType codeType, const fileReader &fr,
                                 bool trained) {
    mycodetype = codeType;
    N = n;
    K = k;
    M = m;
    G_rows = N + K;
    mTrained = trained;
    dc = fr.dc;
    dv = fr.dv;
    maxDc = fr.maxDc;
    maxDv = fr.maxDv;
    Nv = fr.Nv;
    Mc = fr.Mc;
    checkVal = fr.checkVal;
    varVal = fr.varVal;
    Nvk = fr.Nvk;
    Mck = fr.Mck;
    G = fr.G;
    if (trained) {
        weights_cn = fr.weights_cn;
        weights_vn = fr.weights_vn;
        weights_llr = fr.weights_llr;
    }
}

std::vector<bool> stabilizerCodes::decode(unsigned int L, double epsilon) {
    if (errorString.empty())
        return {true, true};

    calculate_syndrome();
    error_hat = std::vector<unsigned>(N, 0);
    return flooding_decode(L, epsilon);
}

void stabilizerCodes::add_error_given_epsilon(double epsilon) {
    error.clear();
    errorString.clear();
    std::uniform_real_distribution<double> dist(0, 1);

    auto res = std::random_device()();
    std::ranlux24 generator(res);
    std::uniform_real_distribution<double> distribution(0, 1);
    auto roll = [&distribution, &generator]() { return distribution(generator); };

    for (unsigned i = 0; i < N; i++) {
        double rndValue = roll();
        if (rndValue < epsilon / 3) {
            error.push_back(1);
            errorString.emplace_back("X" + std::to_string(i));
        } else if (rndValue < epsilon * 2 / 3) {
            error.push_back(2);
            errorString.emplace_back("Z" + std::to_string(i));
        } else if (rndValue < epsilon) {
            error.push_back(3);
            errorString.emplace_back("Y" + std::to_string(i));
        } else {
            error.push_back(0);
        }
    }
}

// void stabilizerCodes::add_error_given_positions(int *pos, int *error, int size) {}

double stabilizerCodes::quantize_belief(double Tau, double Tau1, double Tau2) {
    double nom = log1p(exp(-1.0 * Tau));
    double denom = std::max(-1.0 * Tau1, -1.0 * Tau2) + log1p(exp(-1.0 * fabs((Tau1 - Tau2))));
    double ret_val = nom - denom;
    if (std::isnan(ret_val)) {
        throw std::runtime_error("quantize_belief: Log difference is NaN");
    }
    return ret_val;
}

inline bool stabilizerCodes::trace_inner_product(unsigned int a, unsigned int b) {
    return !(a == 0 || b == 0 || a == b);
}

// TODO: reimplement flooding decode
std::vector<bool> stabilizerCodes::flooding_decode(unsigned int L, double epsilon) {
    // Variable num_elements_in_H never used
    //
    // double num_elements_in_H = 0;
    // for (unsigned i = 0; i < N; i++) {
    //     num_elements_in_H += dv[i];
    // }
    std::vector<bool> success(2, false);
    double L0 = log(3.0 * (1 - epsilon) / epsilon);

    double lambda0 = log((1 + exp(-L0)) / (2 * exp(-L0)));
    //
    // Allocate memory for messages
    double **mc2v, **mv2c;
    mc2v = (double **)malloc(M * sizeof(double *));
    mv2c = (double **)malloc(N * sizeof(double *));
    double *Taux, *Tauz, *Tauy;
    Taux = (double *)malloc(N * sizeof(double *));
    Tauz = (double *)malloc(N * sizeof(double *));
    Tauy = (double *)malloc(N * sizeof(double *));
    double *phi_msg;
    phi_msg = (double *)malloc(maxDc * sizeof(double *));
    for (unsigned i = 0; i < M; i++) {
        mc2v[i] = (double *)malloc(dc[i] * sizeof(double *));
        for (unsigned j = 0; j < dc[i]; j++)
            mc2v[i][j] = 0;
    }
    if (mTrained) {
        for (unsigned i = 0; i < N; i++) {
            mv2c[i] = (double *)malloc(dv[i] * sizeof(double *));
            for (unsigned j = 0; j < dv[i]; j++) {
                double lam = log((1 + exp(-L0 * weights_llr[0][i])) /
                                 (exp(-L0 * weights_llr[0][i]) + exp(-L0 * weights_llr[0][i])));
                mv2c[i][j] = lam *= weights_vn[0][i][j];
            }
        }
    } else {
        for (unsigned i = 0; i < N; i++) {
            mv2c[i] = (double *)malloc(dv[i] * sizeof(double *));
            for (unsigned j = 0; j < dv[i]; j++) {
                mv2c[i][j] = lambda0;
            }
        }
    }
    if (print_msg) {
        for (unsigned r = 0; r < 1; r++) {
            for (unsigned c = 0; c < dv[r]; c++) {
                std::cout << mv2c[r][c] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    // start decoding
    for (unsigned decIter = 0; decIter < L; decIter++) {
        // CN update, for i-th check node, calculate mc2v
        // Variable sum_cn_msg never used
        // double sum_cn_msg = 0;
        for (unsigned i = 0; i < M; i++) {
            double phi_sum = 0;
            double sign_prod = (syn[i] == 0 ? 1.0 : -1.0);
            // Sum-Product
            for (unsigned j = 0; j < dc[i]; j++) {
                if (mv2c[Mc[i][j]][Mck[i][j]] != 0.0)
                    phi_msg[j] = -1.0 * log(tanh(fabs(mv2c[Mc[i][j]][Mck[i][j]]) / 2.0));
                else
                    phi_msg[j] = 60;
                phi_sum += phi_msg[j];
                sign_prod *= (mv2c[Mc[i][j]][Mck[i][j]] >= 0.0 ? 1.0 : -1.0);
            }

            for (unsigned j = 0; j < dc[i]; j++) {
                double phi_extrinsic_phi_sum = phi_sum - phi_msg[j];
                double phi_phi_sum = 60;
                if (phi_extrinsic_phi_sum != 0)
                    phi_phi_sum = -1.0 * log(tanh(phi_extrinsic_phi_sum / 2.0));
                mc2v[i][j] = phi_phi_sum * sign_prod * (mv2c[Mc[i][j]][Mck[i][j]] >= 0.0 ? 1.0 : -1.0);
                if (mTrained && decIter < trained_iter)
                    mc2v[i][j] *= weights_cn[decIter][i][j];
                // sum_cn_msg += fabs(mc2v[i][j]);
                if (std::isnan(mc2v[i][j])) {
                    throw std::runtime_error("flooding_decode: mc2v[i][j] is NaN");
                }
                if (std::isinf(mc2v[i][j])) {
                    throw std::runtime_error("flooding_decode: mc2v[i][j] is infinity");
                }
            }
        }

        // unused:
        // double ave_cn_msg = sum_cn_msg / num_elements_in_H;
        if (print_msg) {
            std::cout << "CN messages" << std::endl;
            for (unsigned r = 0; r < 1; r++) {
                for (unsigned c = 0; c < dc[r]; c++) {
                    std::cout << mc2v[r][c] << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
        // VN update
        for (unsigned Vidx = 0; Vidx < N; Vidx++) {
            Taux[Vidx] = L0;
            Tauz[Vidx] = L0;
            Tauy[Vidx] = L0;
            if (mTrained && decIter < trained_iter) {
                Taux[Vidx] *= weights_llr[decIter + 1][Vidx];
                Tauy[Vidx] *= weights_llr[decIter + 1][Vidx];
                Tauz[Vidx] *= weights_llr[decIter + 1][Vidx];
            }
            double Tauxi;
            double Tauyi;
            double Tauzi;

            // jj, index of the neighboring CNs of the VN, sum up the CN messages
            for (unsigned jj = 0; jj < dv[Vidx]; jj++) {
                if (varVal[Vidx][jj] == 1) {
                    Tauz[Vidx] += mc2v[Nv[Vidx][jj]][Nvk[Vidx][jj]];
                    Tauy[Vidx] += mc2v[Nv[Vidx][jj]][Nvk[Vidx][jj]];
                } else if (varVal[Vidx][jj] == 2) {
                    Taux[Vidx] += mc2v[Nv[Vidx][jj]][Nvk[Vidx][jj]];
                    Tauy[Vidx] += mc2v[Nv[Vidx][jj]][Nvk[Vidx][jj]];
                } else if (varVal[Vidx][jj] == 3) {
                    Taux[Vidx] += mc2v[Nv[Vidx][jj]][Nvk[Vidx][jj]];
                    Tauz[Vidx] += mc2v[Nv[Vidx][jj]][Nvk[Vidx][jj]];
                } else
                    throw std::invalid_argument("something is wrong");
            }

            for (unsigned jj = 0; jj < dv[Vidx]; jj++) {
                double temp;
                if (varVal[Vidx][jj] == 1) {
                    Tauxi = Taux[Vidx];
                    Tauzi = Tauz[Vidx] - mc2v[Nv[Vidx][jj]][Nvk[Vidx][jj]];
                    Tauyi = Tauy[Vidx] - mc2v[Nv[Vidx][jj]][Nvk[Vidx][jj]];
                    temp = quantize_belief(Tauxi, Tauyi, Tauzi);
                } else if (varVal[Vidx][jj] == 2) {
                    Tauxi = Taux[Vidx] - mc2v[Nv[Vidx][jj]][Nvk[Vidx][jj]];
                    Tauzi = Tauz[Vidx];
                    Tauyi = Tauy[Vidx] - mc2v[Nv[Vidx][jj]][Nvk[Vidx][jj]];
                    temp = quantize_belief(Tauzi, Tauyi, Tauxi);
                } else if (varVal[Vidx][jj] == 3) {
                    Tauxi = Taux[Vidx] - mc2v[Nv[Vidx][jj]][Nvk[Vidx][jj]];
                    Tauzi = Tauz[Vidx] - mc2v[Nv[Vidx][jj]][Nvk[Vidx][jj]];
                    Tauyi = Tauy[Vidx];
                    temp = quantize_belief(Tauyi, Tauxi, Tauzi);
                } else
                    throw std::invalid_argument("something is wrong");
                double limit = 60;
                if (temp > limit)
                    mv2c[Vidx][jj] = limit;
                else if (temp < -limit)
                    mv2c[Vidx][jj] = -limit;
                else
                    mv2c[Vidx][jj] = temp;

                if (std::isnan(mv2c[Vidx][jj])) {
                    throw std::runtime_error("flooding_decode: mv2c[Vidx][jj] is NaN");
                }
                if (std::isinf(mv2c[Vidx][jj])) {
                    throw std::runtime_error("flooding_decode: mv2c[Vidx][jj] is infinity");
                }
                if (mTrained && decIter < trained_iter)
                    mv2c[Vidx][jj] *= weights_vn[decIter + 1][Vidx][jj];
            }
        }
        if (print_msg) {
            std::cout << "Taux, Tauy, Tauz" << std::endl;
            for (unsigned i = 0; i < N; i++)
                std::cout << Taux[i] << " ";
            std::cout << std::endl;
            for (unsigned i = 0; i < N; i++)
                std::cout << Tauy[i] << " ";
            std::cout << std::endl;
            for (unsigned i = 0; i < N; i++)
                std::cout << Tauz[i] << " ";
            std::cout << std::endl;
            std::cout << "VN messages" << std::endl;
            for (unsigned r = 0; r < 1; r++) {
                for (unsigned c = 0; c < dv[r]; c++) {
                    std::cout << mv2c[r][c] << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
        success = check_success(Taux, Tauy, Tauz);
        if (success[0])
            break;
    }

    free(Taux);
    free(Tauy);
    free(Tauz);
    free(phi_msg);
    for (unsigned i = 0; i < M; i++) {
        free(mc2v[i]);
    }
    for (unsigned i = 0; i < N; i++) {
        free(mv2c[i]);
    }
    free(mc2v);
    free(mv2c);
    return success;
}

void stabilizerCodes::calculate_syndrome() {
    for (unsigned i = 0; i < M; i++) {
        bool check = false;
        for (unsigned j = 0; j < dc[i]; j++) {
            check = trace_inner_product(error[Mc[i][j]], checkVal[i][j]) ? !check : check;
        }
        syn.push_back(check % 2);
    }
}

std::vector<bool> stabilizerCodes::check_success(const double *Taux, const double *Tauy, const double *Tauz) {
    std::vector<bool> success(2, false);
    error_hat = std::vector<unsigned>(N, 0);
    for (unsigned i = 0; i < N; i++) {
        if (Taux[i] > 0 && Tauy[i] > 0 && Tauz[i] > 0) {
            error_hat[i] = 0;
        } else if (Taux[i] < Tauy[i] && Taux[i] < Tauz[i]) {
            error_hat[i] = 1;
        } else if (Tauz[i] < Taux[i] && Tauz[i] < Tauy[i]) {
            error_hat[i] = 2;
        } else {
            error_hat[i] = 3;
        }
    }
    for (unsigned i = 0; i < M; i++) {
        // unsigned check = 0;
        bool check = false;
        for (unsigned j = 0; j < dc[i]; j++) {
            check = trace_inner_product(error_hat[Mc[i][j]], checkVal[i][j]) ? !check : check;
        }
        if ((check) != syn[i]) {
            return success;
        }
    }
    success[0] = true;
    for (unsigned i = 0; i < G_rows; i++) {
        bool check = false;
        for (unsigned j = 0; j < N; j++) {
            check = trace_inner_product(error[j], G[i][j]) ? !check : check;
            check = trace_inner_product(error_hat[j], G[i][j]) ? !check : check;
        }
        if (check) {
            return success;
        }
    }
    success[1] = true;
    return success;
}
