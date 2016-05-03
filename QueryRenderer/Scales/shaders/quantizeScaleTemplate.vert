#define domainType_<name> <domainType>
#define rangeType_<name> <rangeType>

#define numRanges_<name> <numRanges>

uniform double quantizeDiff;
uniform domainType_<name> uDomains_<name>[2];
uniform rangeType_<name> uRanges_<name>[numRanges_<name>];

rangeType_<name> getQuantizeScale_<name>(in domainType_<name> category) {
    #if numRanges_<name> == 1
        rangeType_<name> val = uRanges_<name>[0];
    #else
        double diff = double(category - uDomains_<name>[0]);
        int idx = int(max(min(trunc(diff / double(quantizeDiff)), double(numRanges_<name>-1), double(0))));

        rangeType_<name> val = uRanges_<name>[idx];
    #endif

    return val;
}
