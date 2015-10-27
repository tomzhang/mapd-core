#define domainType_<name> <domainType>
#define rangeType_<name> <rangeType>

#define numDomains_<name> <numDomains>
#define numRanges_<name> <numRanges>
#define useClamp_<name> <useClamp>

uniform domainType_<name> uDomains_<name>[numDomains_<name>];
uniform rangeType_<name> uRanges_<name>[numRanges_<name>];



rangeType_<name> getLinearScale_<name>(in domainType_<name> domainVal) {
    domainType_<name> val1;
    domainType_<name> val2;
    int idx1, idx2;

    #if numDomains_<name> == 1 || numRanges_<name> == 1
        idx1 = 0;
        idx2 = 0;
        val1 = uDomains_<name>[0];
        val2 = uDomains_<name>[0];
    #elif numDomains_<name> == 2 || numRanges_<name> == 2
        idx1 = 0;
        idx2 = 1;
        val1 = uDomains_<name>[0];
        val2 = uDomains_<name>[1];
    #else
        int startIdx = 0;
        int endIdx = numDomains_<name> - 1;
        domainType_<name> midVal;

        if (domainVal == uDomains_<name>[startIdx]) {
            idx1 = startIdx;
            idx2 = startIdx+1;
        } else if (domainVal == uDomains_<name>[endIdx]) {
            idx1 = endIdx-1;
            idx2 = endIdx;
        } else {
            while (true) {
                int midIdx = startIdx + (endIdx - startIdx) / 2;
                if (midIdx == startIdx) {
                    idx1 = startIdx;
                    idx2 = endIdx;
                    break;
                } else {
                    midVal = uDomains_<name>[midIdx];
                    if (domainVal == midVal) {
                        idx1 = midIdx;
                        idx2 = midIdx+1;
                        break;
                    } else if (domainVal > midVal) {
                        startIdx = midIdx;
                    } else {
                        endIdx = midIdx;
                    }
                }
            }
        }

        val1 = uDomains_<name>[idx1];
        val2 = uDomains_<name>[idx2];
    #endif

    float t = float(domainVal - val1) / float(val2 - val1);

    #if useClamp_<name> == 1
        t = clamp(t, 0.0, 1.0);
    #endif

    return mix(uRanges_<name>[idx1], uRanges_<name>[idx2], t);
}
