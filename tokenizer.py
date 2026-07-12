from array import array

from datetime import datetime

byteTokens = 256

vocabulary_list = [
    "<BOS>",
    "<EOS>",
    "\n",

    # "NAISENT",
    # "NAISENT.",
    # "Hello, ",
    # "Hello",
    # "there",
    # "Hi",
    # "I am ",
]

def getHighestNumberIndex(list):
    highestNumber = None
    highestnumberIndex = 0
    i = 0
    for v in list:
        if highestNumber:
            if v > highestNumber:
                highestNumber = v
                highestnumberIndex = i
        else:
            highestNumber = v
            highestnumberIndex = i
        i += 1
    return highestnumberIndex

def vocabulary_size(): return byteTokens + len(vocabulary_list)

def toToken(tokenIndex):
    token = array("f", [0.0] * vocabulary_size())
    token[tokenIndex] = 1.0

    return token

def getFittingToken(sequence: str, currentIndex):
    maxLength = 1
    wordIndex = None
    token = None
    tokenIndex = 0
    for i in range(len(vocabulary_list)):
        potentionalWord: str = vocabulary_list[i]

        length = len(potentionalWord)
        if length > maxLength \
            and currentIndex+length <= len(sequence) \
                and sequence[currentIndex:currentIndex+length] == potentionalWord:
            maxLength = length
            wordIndex = i
    if wordIndex == None:
        # print(list(sequence[currentIndex].encode("utf-8"))[0])
        tokenIndex = list(sequence[currentIndex].encode("utf-8"))[0]
    else:
        tokenIndex = wordIndex+byteTokens
        # if vocabulary_list[wordIndex] == "\nMODEL: ":
        #     modelResponsePos = tokenziedAmount

    token = toToken(tokenIndex)

    return token, maxLength, tokenIndex

def tokenize(sequence: str):
    tokenSeries = array("f")
    tokenziedAmount = 0
    BOS_positions = array("i")
    EOS_positions = array("i")

    currentIndex = 0
    while True:
        if currentIndex >= len(sequence): break
        token, tokenTextLength, tokenIndex = getFittingToken(sequence, currentIndex)
        tokenSeries.extend(token)
        currentIndex += tokenTextLength

        if toWord(tokenIndex) == "<BOS>":
            BOS_positions.append(tokenziedAmount)
        elif toWord(tokenIndex) == "<EOS>":
            EOS_positions.append(tokenziedAmount)

        tokenziedAmount += 1

    return tokenSeries, tokenziedAmount, BOS_positions, EOS_positions

def toWord(tokenIndex):
    if tokenIndex <= byteTokens-1:
        # print(tokenIndex, bytes(list([tokenIndex])).decode("utf-8", errors="ignore"))
        return bytes(list([tokenIndex])).decode("utf-8", errors="ignore")
    
    return vocabulary_list[tokenIndex-byteTokens]

def formSequence(tokenSeries, tokensAmount):
    sequence = ""

    l = vocabulary_size()

    for i in range(tokensAmount):
        sequence += toWord(getHighestNumberIndex(tokenSeries[i*l:i*l+l]))

    return sequence

def formSequenceSplit(tokenSeries, tokenAmounts):
    sequence = ""

    l = vocabulary_size()

    current = 0
    for i in range(len(tokenAmounts)):
        for j in range(tokenAmounts[i]):
            sequence += toWord(getHighestNumberIndex(tokenSeries[current:current+l]))
            current += l
        sequence +=  "\n--------------------------------\n"

    return sequence

def BPE_learn(sequence: str):
    tokenIndices = array("i")

    tokenizedAmount = 0

    currentIndex = 0
    while True:
        if currentIndex >= len(sequence): break
        _, tokenTextLength, tokenIndex = getFittingToken(sequence, currentIndex)
        tokenIndices.append(tokenIndex)
        currentIndex += tokenTextLength
        tokenizedAmount += 1

    pairScores = {}
    maxScore = 0
    for i in range(len(tokenIndices)-1):
        key = (tokenIndices[i], tokenIndices[i+1])
        if pairScores.__contains__(key):
            pairScores[key] += 1
        else:
            pairScores[key] = 1
        
        if pairScores[key] > maxScore:
            maxScore = pairScores[key]
    
    additions = {}
    for i in range(len(tokenIndices)-1):
        key = (tokenIndices[i], tokenIndices[i+1])

        if (not additions.__contains__(key)) and pairScores[key] == maxScore:
            additions[key] = toWord(tokenIndices[i]) \
                             + toWord(tokenIndices[i+1])

    for (_,__), word in additions.items():
        vocabulary_list.append(word)

    # print(tokenIndices, pairScores, additions)

    # print("TOKENIZED:", tokenizedAmount)

# t = list("h".encode("utf-8"))

# print(t[0], list([t[0]]))
# print(bytes(list([t[0]])).decode("utf-8", errors="ignore"))