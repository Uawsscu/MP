from textblob import TextBlob
#from capture import *
#from Kinect_detect import *


#>>>>>>>>>>>>>>>>>> PASS ThisIs! <<<<<<<<<<<<<<<<<<
def get_object_train(text):
    # CUT "END"
    nounP = ''
    np = False
    print "Train ---Obj---"
    ans = text[0:-3]
    #print "!!" + ans
    b = TextBlob(ans)
    for item in b.noun_phrases:
        #print item
        np = True
        nounP =item

    if(np==False) :
        sentence = b.sentences[0]
        #print sentence

        for word, pos in sentence.tags:
            if pos[0:1] == 'N':
                # CAPTURE
                #cap_ture(word)
                #print word + " >>N"
                nounP =word
                break
    return nounP

#>>>>>>>>>>>>>>>> PASS COMMAND! <<<<<<<<<<<<<<<<<<<<
def get_object_command(text):
    print "Command ---Obj---" #Jerry move a ball to the left #"Jerry grab a ball"
    ans = text[6:]

    return ans

#>>>>>>>>>>>>>>>>>> PASS TrainArm! <<<<<<<<<<<<<<<<
def text_STEP(text):
    text = text.split()
    STEP = ""
    for i in range(-2,0):
        STEP = STEP + " " + text[i]
        #print STEP+".."
    return STEP

def get_TrainArm(text):
    STEP = text_STEP(text) #"this is how to grab a ball one step" >>> grab a ball
    print STEP
    test = text.split(STEP,1)[0]
    test = test[15:]
    return test

#>>>>>>>>>>>>>>>>>> PASS Q! <<<<<<<<<<<<<<<<<<
def get_object_question(text):
    print "question--Obj--"
    # CUT "END"
    nounP = ''
    np = False
    b = TextBlob(text)
    for item in b.noun_phrases:
        np = True
        nounP = item

    if (np == False):
        sentence = b.sentences[0]
        for word, pos in sentence.tags:
            if pos[0:1] == 'N':
                nounP = word
                break
    return nounP
#>>>>>>>>>>> Enum <<<<<<<<<<<<<<<<<<<
def text2int(textnum, numwords={}):
    if not numwords:
        units = [
            "zero", "one", "two", "three", "four", "five"
        ]
        for idx, word in enumerate(units):    numwords[word] = (1, idx)

    current = result = 0
    for word in textnum.split():
        if word not in numwords:
            continue
            raise Exception("Illegal word: " + word)

        scale, increment = numwords[word]
        return increment

def get_objectDetect(text):
    np = False
    b = TextBlob(text)
    nounP=""
    for item in b.noun_phrases:
        np = True
        nounP =item

    if(np==False) :
        sentence = b.sentences[0]
        for word, pos in sentence.tags:
            if pos[0:1] == 'N':
                nounP =word
                break
    return nounP
#######################

def get_objectJerry(text):
    np = False
    text = text[5:]
    b = TextBlob(text)
    nounP=""
    for item in b.noun_phrases:
        np = True
        nounP =item

    if(np==False) :
        sentence = b.sentences[0]
        for word, pos in sentence.tags:
            if pos[0:1] == 'N':
                nounP =word
                break
    return nounP

def get_V(text):
    b = TextBlob(text)
    for word, pos in b.tags:
        #print pos
        if pos[0:2] == 'VB':
            vb_NN = word
    if b.words[-2] == "the" and b.words[-3] == "to" :
        vb_NN = vb_NN + " " + b.words[-1]
    return vb_NN

print get_V("jerry grab a ball")

#print get_object_question("do you know a red bottle")
#print get_object_command("jerry grab a ball")

#print get_objectJerry("jerry grab a ball")
#print text2int("test fie is")
#print get_objectDetect("teddy bear")