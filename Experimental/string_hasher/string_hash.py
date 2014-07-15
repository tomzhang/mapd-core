from sets import Set

def strToInt(string):
    numChars = len(string)
    value = 0
    for c in range(numChars):
        value += pow(256,c)*ord(string[c])
    return value

class Hasher:
    def __init__(self,numBins=67003):
        self.numBins = numBins 
        self.reset()
        self.hashFunc=strToInt

    def reset(self):
        self.binCounts = [0] * self.numBins
        self.binTokens = []
        for t in range(self.numBins):
            self.binTokens.append([])
        self.stringSet = Set([])


    def genBinCountsForStrings(self, stringList):
        self.reset()
        for s in stringList:
            self.binCounts[self.hashFunc(s) % self.numBins] += 1

    def hashStrings(self, stringList):
        self.reset()
        for s in stringList:
            hashBin = self.hashFunc(s) % self.numBins
            self.binCounts[hashBin] += 1
            self.binTokens[hashBin].append(s)

    def hashUniqueStrings(self, stringList): #checks if string has been seen before before hashing
        self.reset()
        for s in stringList:
            hashBin = self.hashFunc(s) % self.numBins
            self.binCounts[hashBin] += 1
            self.binTokens[hashBin].append(s)

        





