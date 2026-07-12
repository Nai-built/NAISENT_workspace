from array import array

from datetime import datetime

vocabulary_list = [
    "USER: ",
    "\nUSER: ",
    "MODEL: ",
    "\nMODEL: ",
    "<END OF SEQUENCE>\n",

    " ",
    "Hello",
    # "Hello, ",
    ",",
    "there",
    # " there",
    # " there,",
    # " there, ",
    "I",
    "am",
    # "I am ",
    "NAISENT",
    "Hi",
    # "Hi ",
    # "Hi,",
    # "Hi, ",
    "who",
    "how",
    "are",
    "you",
    # " who",
    # " how",
    # " are",
    # " you",
    # " you?",
    "?",
    "hi",
    "hello",
    ".",
    "\n",
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

def vocabulary_size():
    return len(vocabulary_list)

def toToken(word):
    token = array("f", [0.0] * vocabulary_size())
    token[vocabulary_list.index(word)] = 1.0

    return token

def toWord(token):
    return vocabulary_list[getHighestNumberIndex(token)]

def tokenize(sequence: str):
    tokenSeries = array("f")
    tokenziedAmount = 0
    modelResponsePos = 0

    currentIndex = 0
    while True:
        if currentIndex >= len(sequence): break

        maxLength = 0
        word = None
        for i in range(len(vocabulary_list)):
            potentionalWord: str = vocabulary_list[i]

            length = len(potentionalWord)
            if length > maxLength \
                and currentIndex+length <= len(sequence) \
                    and sequence[currentIndex:currentIndex+length] == potentionalWord:
                maxLength = length
                word = potentionalWord
        if word == None:
            currentIndex += 1
        else:
            currentIndex += maxLength
            tokenSeries.extend(toToken(word))
            if word == "\nMODEL: ":
                modelResponsePos = tokenziedAmount
            tokenziedAmount += 1

    return tokenSeries, tokenziedAmount, modelResponsePos

def formSequence(tokenSeries, tokensAmount):
    sequence = ""

    l = len(vocabulary_list)

    for i in range(tokensAmount):
        sequence += toWord(tokenSeries[i*l:i*l+l])

    return sequence

Tsequence = "USER: Hello, who are you?\nMODEL: NAISENT<END OF SEQUENCE>"
tokenized, amount, _ = tokenize(Tsequence)
unTokenized = formSequence(tokenized, amount)

# print(Tsequence)
# print(tokenized, amount)
# print(unTokenized)