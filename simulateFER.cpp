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
#include "ensembleDecoder.h"

#include "helpers.h"

#include <iostream>
#include <omp.h>
#include <vector>

int main(int argc, char *argv[]) {
    unsigned n = 72;
    unsigned k = 2;
    unsigned m = 216;

    int decIterNum = 25;
    bool trained = true;
    double ep0 = 0.4;
    stabilizerCodesType codeType = stabilizerCodesType::toric;

    // Every decoder in the ensemble needs to be specified here
    fileReader first_matrix_supplier(n, k, m, codeType, trained, "first-decoder");
    fileReader second_matrix_supplier(n, k, m, codeType, trained, "second-decoder");
    matrix_supplier.check_symplectic();

    constexpr int default_max_frame_errors = 300;
    constexpr int default_max_decoded_words = 45000000;
    //    double ep_list[] =
    //    {0.14,0.13,0.12,0.11,0.1,0.09,0.08,0.07,0.06,0.05,0.04,0.03,0.02,0.01,0.009,0.008,0.007,0.006,0.005};
    const std::vector<double> default_ep_list{
        0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01,
    };

    const auto arguments =
        helpers::parse_arguments(argc, argv, default_ep_list, default_max_frame_errors, default_max_decoded_words);
    if (arguments.print_help) {
        helpers::print_help(arguments.progname);
        return 0;
    }
    const std::vector<double> &ep_list = std::move(arguments.epsilons);
    const auto max_frame_errors = arguments.maximum_frame_errors;
    const auto max_decoded_words = arguments.maximum_decoded_words;

    std::cout << "% [[" << n << "," << k << +"]], " << m << " checks, " << decIterNum << " iter ";
    if (trained)
        std::cout << ",trained";
    std::cout << "\n"
              << "% collect " << max_frame_errors << " frame errors or " << max_decoded_words
              << " decoded error patterns\n";

    //    omp_set_num_threads(1);
    for (double epsilon : ep_list) {
        double total_decoding = 0;
        double failure = 0;

#pragma omp parallel
        {
            while (failure <= max_frame_errors && total_decoding <= max_decoded_words) {
                std::vector<bool> success;
                // Create the same error pattern for all paths, the supplied matrix does not
                // have to be from a path in the ensemble
                stabilizerCodes errorCreator(n, k, m, codeType, first_matrix_supplier, trained);
          			errorCreator.add_error_given_epsilon(epsilon);

         				ensembleDecoder ens;
                stabilizerCodes first(n, k, m, codeType, first_matrix_supplier, trained);
                stabilizerCodes second(n, k, m, codeType, second_matrix_supplier, trained);

                ens.add_decoder(first);
                ens.add_decoder(second);
                ens.setErrors(errorCreator.getErrorString(), errorCreator.getError());
                // NBED Curve
        				// success = ens.decodeAllPaths(decIterNum, ep0);

				        // LER-NBED curve
								for(int i = 0 ; i < ens.list_of_decoders.size(); i++){
										success = ens.list_of_decoders[i]->decode(decIterNum, ep0);
										if (success[1]) {
											break;
										}
								}
        				
#pragma omp critical
                {
									if (!success[1]) 
									{
											failure += 1;
									}
									total_decoding += 1;
											}
            }
        }
        std::cout << "% FE " << failure << ", total dec. " << total_decoding << "\\\\" << std::endl;
        std::cout << epsilon << " " << (failure / total_decoding) << "\\\\" << std::endl;
    }
    return 0;
}
