#include <string>
#include <vector>

unsigned int populateSearchVals(const std::string &searchString, std::vector<std::vector<unsigned int> > &searchVals);

void cudaVLikeCWarpWrapper(const char *data, const unsigned int *startIndex, const unsigned int *endIndex, const unsigned int numElements, unsigned int * matchCount,const size_t numBlocks, const size_t numThreads, const std::string &searchString);
