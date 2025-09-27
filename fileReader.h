//
// Created by sisi on 4/20/23.
//

#ifndef NBP_JUPYTER_FILEREADER_H
#define NBP_JUPYTER_FILEREADER_H

#include <experimental/filesystem>
#endif // NBP_JUPYTER_FILEREADER_H
#include <string>
#include <string_view>
#include <vector>
enum class stabilizerCodesType { GeneralizedBicycle = 0, HypergraphProduct = 1, toric = 3 };
class fileReader {
  public:
    fileReader(unsigned n, unsigned k, unsigned m, stabilizerCodesType codeType, bool trained = false, std::string specifier = "main");
    std::string code_type_string() const;
    std::experimental::filesystem::path construct_weights_path(std::string_view filename) const;

    void load_cn_weights();
    void load_vn_weights();
    void load_llr_weights();
    void read_H();
    void read_G();

    bool check_symplectic();
    static inline bool trace_inner_product(unsigned a, unsigned b);
    stabilizerCodesType mycodetype;
    unsigned N;
    unsigned K;
    unsigned M;
  	std::string specifier;
    unsigned G_rows;
    bool mTrained;
    unsigned trained_iter;
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
