
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
#include "helpers.h"
#include <iostream>
#include <string>
#include <string_view>
#include <vector>
namespace helpers {
void print_help(std::string_view progname) {
    std::cout
        << "Neural Belief Propagation Using Overcomplete Check Matrices\n"
           "\n"
           "Usage:\n"
        << progname
        << " [options] [Epsilon] [Epsilon]...\n"
           "\n"
           "Positional arguments:\n"
           "  Epsilon                     optionally specify a list of epsilons at which to verify.\n"
           "                              If no epsilons are given, use default list of epsilons \n"
           "\n"
           "Options:\n"
           "  -r|--range START STEP STOP  Specify a decreasing range of epsilons to use, e.g.,\n"
           "                              \"-r 0.1 0.025 0.9\" to get 0.1 0.975 0.95 0.925 0.9\n"
           "\n"
           "  -e|--max-frame-errors N     Stop simulating after observing N frame errors\n"
           "  -w|--max-decoded-words N    Stop simulating after decoding N words\n"
           "\n\n\n"
           " This program accompanies the paper\n"
           "     S. Miao, A. Schnerring, H. Li and L. Schmalen,\n"
           "     \"Neural belief propagation decoding of quantum LDPC codes using overcomplete check matrices,\"\n"
           "     Proc. IEEE Inform. Theory Workshop (ITW), Saint-Malo, France, Apr. 2023, "
           "https://arxiv.org/abs/2212.10245\n";
}
parse_result parse_arguments(int argument_count, char *arguments[], const std::vector<double> &default_epsilons,
                             unsigned maximum_frame_errors, unsigned maximum_decoded_words) {
    parse_result result;
    result.maximum_frame_errors = maximum_frame_errors;
    result.maximum_decoded_words = maximum_decoded_words;
    result.progname = arguments[0];
    ++arguments;
    --argument_count;

    if (!argument_count) { // no arguments passed – use default epsilons
        result.epsilons = default_epsilons;
        return result; // return empty list
    }

    while (argument_count) {
        auto &argument = *(arguments++);
        const std::string_view argument_view(argument); // for easier comparison

        if (argument_view == "-h" || argument_view == "--help") {
            result.print_help = true;
            return result;
        }
        if (argument_view == "-r" || argument_view == "--range") {
            if (argument_count < 4) { // need three more arguments
                result.print_help = true;
                return result;
            }
            std::string_view start_(*(arguments++));
            std::string_view step_(*(arguments++));
            std::string_view stop_(*(arguments++));
            argument_count -= 3;
            try {
                char **ignoreme = nullptr;
                auto start = std::strtod(start_.data(), ignoreme);
                auto step = std::strtod(step_.data(), ignoreme);
                auto stop = std::strtod(stop_.data(), ignoreme);
                if (stop <= 0.0f) {
                    throw std::invalid_argument("Stop value needs to be positive");
                }
                if (step <= 0.0f) {
                    throw std::invalid_argument("step value needs to be positive");
                }
                if (start <= stop) {
                    throw std::invalid_argument("Stop value needs to be smaller than start value");
                }
                for (double value = start; value >= stop; value -= step) {
                    result.epsilons.push_back(value);
                }
            } catch (std::invalid_argument &exceptions) {
                std::cerr << "Could not parse range: " << exceptions.what() << "; skipping range.\n";
            }
        } else if (argument_view == "-e" || argument_view == "--max-frame-errors") {
            argument_count--;
            std::string_view sv(*(arguments++));
            result.maximum_frame_errors = std::strtoul(sv.data(), nullptr, 10);
        } else if (argument_view == "-w" || argument_view == "--max-decoded-words") {
            argument_count--;
            std::string_view sv(*(arguments++));
            result.maximum_decoded_words = std::strtoul(sv.data(), nullptr, 10);
        } else {
            // we encountered something that doesn't look like a keyword argument -- proceed to parse numbers
            arguments--;
            break;
        }
        argument_count--;
    }

    if (!argument_count && result.epsilons.empty()) { // no arguments passed, and no range found – use default epsilons
        result.epsilons = default_epsilons;
        return result;
    }

    result.epsilons.reserve(result.epsilons.size() + argument_count);
    for (auto counter = argument_count; counter; --counter) {
        auto argument = *(arguments++);
        auto value = std::atof(argument);
        if (value <= 0.0) { // couldn't parse number, or actually 0.0 (or below) was passed
            std::cerr << "Ignoring argument '" << argument << "' as unparseable or non-positive.\n";
            continue; // skip this
        }
        result.epsilons.push_back(value);
    }
    return result;
}

}; // namespace helpers
