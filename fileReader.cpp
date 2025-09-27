//
// Created by sisi on 4/20/23.
//
#include "fileReader.h"
#include <cmath>
#include <experimental/filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>

fileReader::fileReader(unsigned n, unsigned k, unsigned m, stabilizerCodesType codeType, bool trained, std::string decoderName) {
    mycodetype = codeType;
    N = n;
    K = k;
    M = m;
  	specifier = decoderName;
    G_rows = N + K;
    mTrained = trained;
    read_H();
    read_G();
    if (mTrained) {
        load_cn_weights();
        load_llr_weights();
        load_vn_weights();
    }
}

void fileReader::read_H() {
    std::string codeTypeString = code_type_string();
    std::string filename = "./PCMs/" + codeTypeString + "_" + std::to_string(N) + "_" + std::to_string(K) + "/" +
                           codeTypeString + "_" + std::to_string(N) + "_" + std::to_string(K) + "_H_" +
                           std::to_string(M) + ".alist";
    // Nvk same shape and Nv, but its value k means the i-th VN being the k-th neigbor of j-th CN
    // Mck same shape and Nv, but its value k means the i-th CN being the k-th neigbor of j-th VN
    std::vector<std::vector<unsigned>> checkValues;
    std::vector<std::vector<unsigned>> VariableValues;

    std::string line;
    std::ifstream matrix_file;
    matrix_file.open(filename);
    if (matrix_file.is_open()) {
        // first line, n k
        getline(matrix_file, line, ' ');
        unsigned n = std::stoul(line);
        getline(matrix_file, line);
        unsigned m = std::stoul(line);

        if (n != N) {
            throw std::runtime_error("read_H: file-specified N not as expected");
        }
        if (m != M) {
            throw std::runtime_error("read_H: file-specified M not as expected");
        }
        // second line max dv and dc
        getline(matrix_file, line, ' ');
        maxDv = std::stoul(line);
        getline(matrix_file, line);
        maxDc = std::stoul(line);

        // third line, dv degrees
        for (unsigned i = 0; i < n - 1; i++) {
            getline(matrix_file, line, ' ');
            dv.push_back(std::stoul(line));
            Nvk.emplace_back();
        }
        Nvk.emplace_back();
        getline(matrix_file, line);
        dv.push_back(std::stoul(line));

        // forth line, dc degrees
        for (unsigned i = 0; i < m - 1; i++) {
            getline(matrix_file, line, ' ');
            dc.push_back(std::stoul(line));
            Mck.emplace_back();
        }
        Mck.emplace_back();
        getline(matrix_file, line);
        dc.push_back(std::stoul(line));

        // fifth to n+5-th line, dv neighbors
        for (unsigned i = 0; i < n; i++) {
            Nv.emplace_back(dv[i], 0);
            for (unsigned j = 0; j < dv[i] - 1; j++) {
                getline(matrix_file, line, ' ');
                Nv[i][j] = std::stoul(line) - 1;
                Mck[Nv[i][j]].push_back(j);
            }
            getline(matrix_file, line);
            Nv[i][dv[i] - 1] = std::stoul(line) - 1;
            Mck[Nv[i][dv[i] - 1]].push_back(dv[i] - 1);
        }

        // n+6-th to n+6+m-th line, dc neighbors
        for (unsigned i = 0; i < m; i++) {
            Mc.emplace_back(dc[i], 0);
            for (unsigned j = 0; j < dc[i] - 1; j++) {
                getline(matrix_file, line, ' ');
                Mc[i][j] = std::stoul(line) - 1;
                Nvk[Mc[i][j]].push_back(j);
            }
            getline(matrix_file, line);
            Mc[i][dc[i] - 1] = std::stoul(line) - 1;
            Nvk[Mc[i][dc[i] - 1]].push_back(dc[i] - 1);
        }
        // n+6+m+1-th to n+6+m+n-th line, value of each row
        for (unsigned i = 0; i < m; i++) {
            checkValues.emplace_back(dc[i], 0);
            for (unsigned j = 0; j < dc[i] - 1; j++) {
                getline(matrix_file, line, ' ');
                checkValues[i][j] = std::stoul(line);
            }
            getline(matrix_file, line);
            checkValues[i][dc[i] - 1] = std::stoul(line);
        }
        // last lines, value of each column
        for (unsigned i = 0; i < n; i++) {
            VariableValues.emplace_back(dv[i], 0);
            for (unsigned j = 0; j < dv[i] - 1; j++) {
                getline(matrix_file, line, ' ');
                VariableValues[i][j] = std::stoul(line);
            }
            getline(matrix_file, line);
            VariableValues[i][dv[i] - 1] = std::stoul(line);
        }
        matrix_file.close();
    } else
        std::cout << "Unable to open file" << std::endl;
    checkVal = checkValues;
    varVal = VariableValues;
}
void fileReader::read_G() {
    std::string codeTypeString = code_type_string();
    std::string filename = "./PCMs/" + codeTypeString + "_" + std::to_string(N) + "_" + std::to_string(K) + "/" +
                           codeTypeString + "_" + std::to_string(N) + "_" + std::to_string(K) + "_G.txt";
    std::string line;
    std::ifstream matrix_file;
    matrix_file.open(filename);
    if (matrix_file.is_open()) {
        for (unsigned i = 0; i < G_rows; i++) {
            std::vector<unsigned> row(N, 0);
            for (unsigned j = 0; j < N - 1; j++) {
                getline(matrix_file, line, ' ');
                row[j] = std::stoul(line);
            }
            getline(matrix_file, line);
            row[N - 1] = std::stoul(line);
            G.push_back(row);
        }
        matrix_file.close();
    } else
        std::cout << "Unable to open file" << std::endl;
}

void fileReader::load_cn_weights() {
    auto path = construct_weights_path("weight_cn.txt");
    std::vector<std::vector<std::vector<double>>> weight_cn;
    std::string line;
    std::ifstream weights_file;
    weights_file.open(path.c_str());
    unsigned dec_iter;
    if (!weights_file.is_open()) {
        throw std::runtime_error(std::string("Couldn't open check node weights file ") + path.string());
    }
    // first line, number of iterations
    getline(weights_file, line);
    dec_iter = std::stoul(line);
    trained_iter = dec_iter;
    // second line, n m
    getline(weights_file, line, ' ');
    unsigned n = std::stoul(line);
    getline(weights_file, line);
    unsigned m = std::stoul(line);

    if (n != N) {
        throw std::runtime_error("load_cn_weights: file-specified N not as expected");
    }
    if (m != M) {
        throw std::runtime_error("load_cn_weights: file-specified M not as expected");
    }
    // rest of the lines, value of each rows of all iterations
    for (unsigned iter = 0; iter < dec_iter; iter++) {
        weight_cn.emplace_back();
    }
    for (unsigned iter = 0; iter < dec_iter; iter++) {
        std::vector<std::vector<double>> weight_cn_tmp;
        for (unsigned i = 0; i < m; i++) {
            std::vector<double> weight_cn_row(dc[i], 0);
            for (unsigned j = 0; j < dc[i] - 1; j++) {
                getline(weights_file, line, ' ');
                weight_cn_row[j] = std::stod(line);
            }
            getline(weights_file, line);
            weight_cn_row[dc[i] - 1] = std::stod(line);
            weight_cn_tmp.push_back(weight_cn_row);
        }
        weights_cn.push_back(weight_cn_tmp);
    }

    weights_file.close();
}

void fileReader::load_llr_weights() {
    auto path = construct_weights_path("weight_llr.txt");
    std::vector<std::vector<double>> weight_llr;

    std::string line;
    std::ifstream weights_file;
    weights_file.open(path.c_str());
    unsigned dec_iter;
    if (!weights_file.is_open()) {
        throw std::runtime_error(std::string("Couldn't open check LLR weights file ") + path.string());
    }
    // first line, number of iterations
    getline(weights_file, line);
    dec_iter = std::stoul(line);

    // rest of the lines, value of all llr weights
    for (unsigned iter = 0; iter < dec_iter; iter++) {
        weight_llr.emplace_back(N, 0);
        for (unsigned i = 0; i < N - 1; i++) {
            getline(weights_file, line, ' ');
            weight_llr[iter][i] = std::stod(line);
        }
        getline(weights_file, line);
        weight_llr[iter][N - 1] = std::stod(line);
    }

    weights_file.close();

    weights_llr = weight_llr;
}

void fileReader::load_vn_weights() {
    auto path = construct_weights_path("weight_vn.txt");
    std::vector<std::vector<std::vector<double>>> weight_cn;
    std::string line;
    std::ifstream weights_file;
    weights_file.open(path.c_str());
    unsigned dec_iter;
    if (!weights_file.is_open()) {
        throw std::runtime_error(std::string("Couldn't open variable node weights file ") + path.string());
    }
    // first line, number of iterations
    getline(weights_file, line);
    dec_iter = std::stoul(line);
    // second line, n m
    getline(weights_file, line, ' ');
    unsigned n = std::stoul(line);
    getline(weights_file, line);
    unsigned m = std::stoul(line);

    if (n != N) {
        throw std::runtime_error("load_vn_weights: file-specified N not as expected");
    }
    if (m != M) {
        throw std::runtime_error("load_vn_weights: file-specified M not as expected");
    }
    // rest of the lines, value of each rows of all iterations
    for (unsigned iter = 0; iter < dec_iter; iter++) {
        weight_cn.emplace_back();
    }
    for (unsigned iter = 0; iter < dec_iter; iter++) {
        std::vector<std::vector<double>> weight_cn_tmp;
        for (unsigned i = 0; i < n; i++) {
            std::vector<double> weight_cn_row(dv[i], 0);
            for (unsigned j = 0; j < dv[i] - 1; j++) {
                getline(weights_file, line, ' ');
                weight_cn_row[j] = std::stod(line);
            }
            getline(weights_file, line);
            weight_cn_row[dv[i] - 1] = std::stod(line);
            weight_cn_tmp.push_back(weight_cn_row);
        }
        weights_vn.push_back(weight_cn_tmp);
    }

    weights_file.close();
}

std::experimental::filesystem::path fileReader::construct_weights_path(std::string_view filename) const {
    std::string codeTypeString = code_type_string();
    std::stringstream directory_name_builder;
    directory_name_builder << codeTypeString << "_" << N << "_" << K << "_" << M << "_" << specifier;
    std::experimental::filesystem::path path = "training_results";
    path /= directory_name_builder.str();
    path /= filename;

    return path;
}

std::string fileReader::code_type_string() const {
    switch (mycodetype) {
    case stabilizerCodesType::GeneralizedBicycle:
        return "GB";
    case stabilizerCodesType::HypergraphProduct:
        return "HP";
    case stabilizerCodesType::toric:
        return "toric";
    }
    throw std::invalid_argument("unimplemented codetype");
}

bool fileReader::check_symplectic() {
    //    read_H();
    //    read_G();
    unsigned *vec1;
    vec1 = (unsigned *)malloc(N * sizeof(unsigned *));
    unsigned *vec2;
    vec2 = (unsigned *)malloc(N * sizeof(unsigned *));

    for (unsigned rowid = 0; rowid < M; rowid++) {
        for (unsigned iii = 0; iii < N; iii++)
            vec1[iii] = 0;
        for (unsigned j = 0; j < dc[rowid]; j++) {
            vec1[Mc[rowid][j]] = checkVal[rowid][j];
        }
        // check HH^T=0
        for (unsigned i = 0; i < M; i++) {
            for (unsigned iii = 0; iii < N; iii++)
                vec2[iii] = 0;
            for (unsigned j = 0; j < dc[i]; j++) {
                vec2[Mc[i][j]] = checkVal[i][j];
            }
            bool syn_check = false;
            for (unsigned j = 0; j < N; j++) {
                syn_check = trace_inner_product(vec1[j], vec2[j]) ? !syn_check : syn_check;
            }
            if (syn_check) {
                throw std::runtime_error("check_symplectic: syn_check % 2 != 0");
            }
        }
        // check GH^T=0
        for (unsigned i = 0; i < G_rows; i++) {
            bool syn_check = false;
            for (unsigned j = 0; j < N; j++) {
                syn_check = trace_inner_product(vec1[j], G[i][j]) ? !syn_check : syn_check;
            }
            if (syn_check) {
                throw std::runtime_error("check_symplectic / GH^T: syn_check % 2 != 0");
            }
        }
    }
    free(vec1);
    free(vec2);
    std::cout << "check symplectic ok" << std::endl;
    return true;
}

inline bool fileReader::trace_inner_product(unsigned int a, unsigned int b) { return !(a == 0 || b == 0 || a == b); }
