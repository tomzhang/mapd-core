#define domainType_<name> <domainType>
#define rangeType_<name> <rangeType>

#define numDomains_<name> <numDomains>
#define numRanges_<name> <numRanges>

uniform domainType_<name> uDomains_<name>[numDomains_<name>];
uniform rangeType_<name> uRanges_<name>[numRanges_<name>];

uniform rangeType_<name> uDefault_<name>;


rangeType_<name> getOrdinalScale_<name>(in domainType_<name> category) {
    int idx = -1;
    rangeType_<name> val = uDefault_<name>;

    #if numDomains_<name> == 1 || numRanges_<name> == 1
        if (category == uDomains_<name>[0]) {
            val = uRanges_<name>[0];
        }
    // #elif numDomains_<name> == 2 || numRanges_<name> == 2
    //     if (category == uDomains_<name>[0]) {
    //         val = uRanges_<name>[0];
    //     } else if (category == uDomains_<name>[1]) {
    //         val = uRanges_<name>[1];
    //     }
    #else
        // performs a binary search so domain needs to be
        // properly sorted.
        int startIdx = 0;
        int endIdx = numDomains_<name>-1;
        int midIdx = 0;

        domainType_<name> midVal = domainType_<name>(0);

        if (category == uDomains_<name>[startIdx]) {
            idx = startIdx;
        } else if (category == uDomains_<name>[endIdx]) {
            idx = endIdx;
        } else {
            while (true) {
                midIdx = startIdx + ((endIdx - startIdx) / 2);
                if (midIdx == startIdx) {
                    break;
                } else {
                    midVal = uDomains_<name>[midIdx];
                    if (category == midVal) {
                        idx = midIdx;
                        break;
                    } else if (category > midVal) {
                        startIdx = midIdx;
                    } else if (category < midVal) {  // for some reason, just an "else" doesn't work here ????
                        endIdx = midIdx;
                    }
                }
            }
        }

        if (idx >= 0) {
            val = uRanges_<name>[idx % numRanges_<name>];
        }

    #endif

    return val;
}
