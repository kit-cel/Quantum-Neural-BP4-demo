/*
 * Copyright 2023 Marcus Müller, Communications Engineering Lab @ KIT
 *
 * SPDX-License-Identifier: MIT
 *
 * This file accompanies the paper
 *     S. Miao, A. Schnerring, H. Li and L. Schmalen,
 *     "Neural belief propagation decoding of quantum LDPC codes using overcomplete check matrices,"
 *     Proc. IEEE Inform. Theory Workshop (ITW), Saint-Malo, France, Apr. 2023, https://arxiv.org/abs/2212.10245
 */
#include <string>
#include <string_view>
#include <vector>
namespace helpers {
struct parse_result {
    std::vector<double> epsilons;
    std::string progname;
    unsigned maximum_frame_errors = 30;
    unsigned maximum_decoded_words = 45000000;
    bool print_help = false;
};
void print_help(std::string_view progname);
parse_result parse_arguments(int argument_count, char *arguments[], const std::vector<double> &default_epsilons,
                             unsigned maximum_frame_errors = 30, unsigned maximum_decoded_words = 45000000);
} // namespace helpers
