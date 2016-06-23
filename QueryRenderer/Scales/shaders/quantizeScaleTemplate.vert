#define domainType_<name> <domainType>
#define rangeType_<name> <rangeType>

#define numRanges_<name> <numRanges>

uniform double quantizeDiff;
uniform domainType_<name> uDomains_<name>[2];
uniform rangeType_<name> uRanges_<name>[numRanges_<name>];

rangeType_<name> getQuantizeScale_<name>(in domainType_<name> category) {
    double diff = double(category - uDomains_<name>[0]);
    int idx = int(max(min(trunc(diff / quantizeDiff), double(numRanges_<name>-1), double(0))));

    return uRanges_<name>[idx];
}
