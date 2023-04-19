#include "stabilizerCodes.h"
#include <iomanip>
#include <iostream>
#include <omp.h>
int main() {
    unsigned n = 46;
    unsigned k = 2;
    unsigned m = 800;

    int decIterNum = 6;
    bool trained = true;
    double ep0 = 0.1;
    stabilizerCodesType codeType = stabilizerCodesType::GeneralizedBicycle;
    stabilizerCodes code(n, k, m, codeType);
    code.checksymplectic();

    int limit1 = 300;
    int limit2 = 45000000;
    //    double ep_list[] =
    //    {0.14,0.13,0.12,0.11,0.1,0.09,0.08,0.07,0.06,0.05,0.04,0.03,0.02,0.01,0.009,0.008,0.007,0.006,0.005};
    double ep_list[] = {
        0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01,
    };
    std::cout << "%[[" << n << "," << k << +"]], " << m << " checks, " << decIterNum << " iter ";
    if (trained)
        std::cout << "trained" << std::endl;
    else
        std::cout << std::endl;

    std::cout << "%collect " << limit1 << " FE or " << limit2 << " decoding reached" << std::endl;

    //    omp_set_num_threads(1);
    for (double epsilon : ep_list) {
        double total_decoding = 0;
        double failure = 0;

#pragma omp parallel
        {
            while (failure <= limit1 && total_decoding <= limit2) {
                stabilizerCodes code(n, k, m, codeType, trained);
                code.addErrorGivenEpsilon(epsilon);
                std::vector<bool> success;
                success = code.decode(decIterNum, ep0);
#pragma omp critical
                {
                    if (!success[1])
                        failure += 1;
                    total_decoding += 1;
                }
            }
        }
        std::cout << "%FE " << failure << ", total dec. " << total_decoding << "\\\\" << std::endl;
        std::cout << epsilon << " " << (failure / total_decoding) << "\\\\" << std::endl;
    }
    return 0;
}