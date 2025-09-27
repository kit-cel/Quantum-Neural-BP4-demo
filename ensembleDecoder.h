#include "stabilizerCodes.h"
#include <string>
#include <vector>

class ensembleDecoder{
	public:
		ensembleDecoder();

		std::vector<unsigned> returnGuess(){return estimatedError;};
		bool updateGuess(const std::vector<unsigned>& newCandidate, int index);

		std::vector<std::string> list_of_specifiers;
		std::vector<stabilizerCodes*> list_of_decoders;
	  void setErrors(std::vector<std::string> FixedErrorString, std::vector<unsigned> FixedError);

		std::vector<bool> decodeAllPaths(unsigned int L, double epsilon);
		void add_decoder(stabilizerCodes& decoder);

	private:
		std::vector<unsigned> estimatedError;
		int bestDecoder;
		std::vector<std::vector<double>> estimatedTaus;
};
