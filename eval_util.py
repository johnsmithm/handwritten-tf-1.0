# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Provides functions to help with evaluating models."""
import datetime
import numpy as np
import tensorflow as tf

from tensorflow.python.platform import gfile
   
def levenshtein(source, target, mat = None):
    #todos:\/
    #add different score for changing a letter into another
    #add different score for adding a letter before and after another letter ?
    if len(source) < len(target):
        return levenshtein(target, source)

    # So now we have len(source) >= len(target).
    if len(target) == 0:
        return len(source)

    # We call tuple() to force strings to be used as sequences
    # ('c', 'a', 't', 's') - numpy uses them as values by default.
    source = np.array(tuple(source))
    target = np.array(tuple(target))
    
    mm = np.ones((len(source),len(target)))
    #mat = getMat()
    if mat is not None:
        for i,p1 in enumerate(source):
            for j,p2 in enumerate(target):
                mm[i,j] = mat[p1,p2]

    # We use a dynamic programming algorithm, but with the
    # added optimization that we only need the last two rows
    # of the matrix.
    previous_row = np.arange(target.size + 1).astype("float32")
    for i,s in enumerate(source):
        # Insertion (target grows longer than source):
        current_row = previous_row + 1

        # Substitution or matching:
        # Target and source items are aligned, and either
        # are different (cost of 1), or are the same (cost of 0).
        current_row[1:] = np.minimum(
                current_row[1:],
                np.add(previous_row[:-1], mm[i,:]*(target != s)))

        # Deletion (target grows shorter than source):
        current_row[1:] = np.minimum(
                current_row[1:],
                current_row[0:-1] + 2)

        previous_row = current_row

    return previous_row[-1]

def getMat():
    voc = get_characters()
    #del voc[''];del voc[' '];del voc['%%']
    mat = np.ones((len(voc),len(voc)))
    pairs = [['a','o'],['g','y'],['l','t'],['f','t'],['q','p'],['b','d'],['m','n'],
            ['g','q'],['q','j'],['v','w'],['v','r'],['x','y'],['i','j'],['u','r'],
            ['u','a'],['u','o'],['c','u']]
    chains = [['l','t','f'],['o','u','a'],['e','l'],['h','m','n'],['i','j'],['q','p','y','j'],
             ['r','w','v'],['d','a','g'],['x','y'],['b','d'],['s','f']]
    for p1,p2 in pairs:
        mat[voc[p1],voc[p2]] -= 0.02
        mat[voc[p2],voc[p1]] -= 0.02
    for p in chains:
        for p1 in p:
            for p2 in p:
                if p1!=p2:
                    mat[voc[p1],voc[p2]] -= 0.01
                    mat[voc[p2],voc[p1]] -= 0.01
    return mat

def read_vocab(path):#'vocabulary.txt'
    if not tf.gfile.Exists(path):
        print('no file!')
        return []
    f = tf.gfile.GFile(path)
    s=  str(f.readline())
    l = []
    while s:
        l.append( [int(i) for i in s.split()])
        s=  str(f.readline())
    #print (l)
    return l

def getIndex(c,voc):
    for name, age in voc.iteritems():
        if age == c:
            return name
    print("-"*30,"error-",c)
    return None

def get_characters():
    vocabulary = {}
    nrC=1
    vocabulary['%%'] = 0 
    
    c = '0'
    '''
    while ord(c) != ord('9')+1:
        vocabulary[c] = nrC
        nrC = nrC + 1
        c = chr(ord(c)+1)
    c = 'A'
    while ord(c) != ord('Z')+1:
        vocabulary[c] = nrC
        nrC = nrC + 1
        c = chr(ord(c)+1)
    '''
    c = 'a'
    while ord(c) != ord('z')+1:
        vocabulary[c] = nrC
        nrC = nrC + 1
        c = chr(ord(c)+1)
    '''
    cr = [',','.','"','\'',' ','-','#','(',')',';','?',':','*','&','!','/','']
    for c in cr:
        vocabulary[c] = nrC
        nrC = nrC + 1
    '''
    vocabulary[' '] = nrC
    nrC += 1
    vocabulary[''] = nrC
    return vocabulary

def show_prediction(dec, label, lmP = None , top_k=1):
    voc = get_characters()
    p = []
    for i,word in enumerate(label[:top_k]):
        f = [''.join([getIndex(j,voc) for j in word if j])]
        print('corr:',[getIndex(j,voc) for j in word if j])
        for guess_batch in dec:
            print('pred:',[getIndex(j,voc) for j in guess_batch[i] if j])
            f.append(''.join([getIndex(j,voc) for j in guess_batch[i] if j]))
        if lmP is not None:
            print('lmp :',[getIndex(j,voc) for j in lmP[i] if j])
        print('-'*10)
        p.append(f)
    return p
        
def split_sequence(seq,delimiter=27,exclude=[0]):         
    words = []
    word = []
    for c in seq:
            if c == delimiter:
                if len(word) > 0:
                    words.append(word)
                word = []
            elif c  not in exclude:
                word.append(c)
    if len(word) > 0:
                    words.append(word)
    return words
def cut_zeros(word):
    return [c for c in word if c != 0 and c != 28]
def calculate_models_error_withLanguageModel(decodedPr, labels_val, vocabulary,top_k):
    if len(vocabulary) == 0:
        return -1
    #space is 27
    voc_guess = [i for i in decodedPr[0]]
    voc_gesss_v = [len(i) for i in decodedPr[0]]
    for guess_batch in decodedPr:
        for k,guess in enumerate(guess_batch):            
            words = split_sequence(guess)
            w = []
            v1 = 0
            for guessW in words:
                v = voc_gesss_v[k]
                w1 = []
                for i,word in enumerate(vocabulary):
                    #are sorted by length, => cut the for when we can!!!
                    if v < len(word) - len(guessW):
                        break
                    ed = levenshtein(word,guessW)
                    if v>ed:
                        v = ed
                        w1 = word
                if len(w)>0:
                            w.append(27)
                w.extend(w1)
                v1 += v
            if v1<voc_gesss_v[k]:
                        voc_gesss_v[k] = v1
                        voc_guess[k] = w
    err = 0.0
    for k,truth in  enumerate(labels_val):
        thuth = cut_zeros(truth)
        err += levenshtein(truth, voc_guess[k])/float(len(truth))
    err /= len(labels_val)
    return err, voc_guess

def get_trie(vocabulary, sp=56):
    Trie = {}
    def add_trie(trie, w, n = 0):

        if n == len(w):
            trie[0] = 1
            return
        if w[n] == sp:
            add_trie(Trie, w[n+1:] )
            return
        if w[n] not in trie:
            trie[w[n]] = {}
        add_trie(trie[w[n]], w , n + 1)
    for w in vocabulary:
        add_trie(Trie, w)
    return Trie
def trie_exist(trie, w, n = 0):
    
    if n == len(w):
        
        if 0 in trie:
            return True
        return False
    if w[n] not in trie:
        return False
    return trie_exist(trie[w[n]], w, n + 1)

def bi_gram_model(w, tr, bi, on):
    #print(w)
    if len(w)>2:
        return tr[w[-3],w[-2],w[-1]]
    if len(w)>1:
        return bi[w[-2],w[-1]]
    if len(w)==1:
        return on[w[0]]
    return 0.

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return x/np.max(x)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
def get_n_gram(vocab, vocab_size):    
    tri_gram = np.zeros([vocab_size]*3)
    bi_gram = np.zeros([vocab_size]*2)
    one_gram = np.zeros([vocab_size])
    mB = 0
    mT = 0
    nrL = 0
    for w in vocab:
        nrL+=1
        if w[0]>vocab_size:
            print w[0]
        one_gram[w[0]]+=1
        for j in range(1,len(w)):
            nrL+=1
            one_gram[w[j]]+=1
            bi_gram[w[j-1],w[j]] += 1.
            mB = max(mB,bi_gram[w[j-1],w[j]])
            if j>1:
                tri_gram[w[j-2],w[j-1],w[j]] += 1
                mT = max(mT, tri_gram[w[j-2],w[j-1],w[j]])
            
    return softmax(one_gram), softmax(bi_gram), softmax(tri_gram)



def beam_search_dict(preds, trans=bi_gram_model, voc_size=29, k=5, bk = 100):
    """Beam search with dictionary
    
    Args:
        trans: function that takes the word and returns the probs of last caracter
        preds: (steps, batch-size, vocab-size)
    
    Returns:
        float - error
        (batch-size, steps)
    """
    # p(y,x), p(-,y,x), y
    B = [[[0.,1.,[]]] for _ in range(preds.shape[1])]
    for i,pred in enumerate(preds):#each time step
        bn = [[] for _ in range(preds.shape[1])]
        for j,y in enumerate(B[:bk]): #each batch
            for q,w in enumerate(y): #each word
                nv = [0.,0.,w[2]]
                if len(nv[2]) > 0:
                    nv[0] = B[j][q][0]*pred[j,nv[2][-1]]#add again last character                    
                    for p,v in enumerate(y):# looking up for y[:-1] in beam
                        if len(v)==len(nv[2])-1 and len([0 for ii, jj in zip(v, nv[2][:-1]) if ii != jj])==0:
                            #extention of y with y[-1] = ctc(y[-1])*(trandition P)*p(y[:-1],t-1,x)
                            nv[0] = nv[0] + pred[j,nv[2][-1]]*trans(nv[2])*B[j][q][1]
                nv[1] = (B[j][q][0]+B[j][q][1])*pred[j,28]#add blank
                bn[j].append(nv)
                for l in range(1,voc_size-1): #extend with caracters, todo: use n-grams
                    if len(w[2]) == 0 or w[2][-1] != l:
                        #extention of y with l = ctc(l)*(trandition y->l P)*p(y,t-1,x)-totala
                        #print(l,pred[j,l],trans(w[2]+[l]),(B[j][q][0]+B[j][q][1]))
                        bn[j].insert(0,[pred[j,l]*trans(w[2]+[l])*(B[j][q][0]+B[j][q][1]),
                               0.0,
                               w[2]+[l]])
                        if len(bn[j]) > bk:
                            bn[j].sort(key=lambda x: x[0]+x[1], reverse=True)   
                            while len(bn[j])>bk:
                                del bn[j][bk]
                            
                            
                            #bn[j] = bn[j][:k]
                #print(pred)
                #print(bn)
        del B[:]
        B = bn[:]
        #print(B,'-')
        #print(preds[1,2,:])
    return B

def get_closest_word(guessW, vocabulary):
    v = 100
    w1 = None
    for i,word in enumerate(vocabulary):     
                    if len(word) < len(guessW)-1 or len(word) > len(guessW)+1:
                        continue
                    ed = levenshtein(word,guessW)
                    if v>ed:
                        v = ed
                        w1 = word
    return w1, v

def dict_model(bPreds, is_word, labels_val, vocabulary=None, n_gram=None,bk=100):
    nPreds = []
    
    for preds in bPreds[:bk]:
        wordsList = []
        for pred in preds:
            words = []
            word = []
            nr = 0
            try:
                if len(pred[2])>0:
                    pred = pred[2]
            except:
                pass
            for l in pred:
                if l==0:
                    continue
                if l == 27 and len(word) != 0:
                    #print('.'*10)
                    words.append([word, is_word(word)])
                    if words[-1][1]:
                        nr+=100
                    else:# - use distance to find closest word
                        if vocabulary is not None:
                            w1,v = get_closest_word(word, vocabulary)
                            #print(v,[eval_util.getIndex(j1,caracters) for j1 in w1 if j1] )
                            if v <= 1.5:
                                words[-1][0] = w1
                                nr += 5-v
                    word=[]
                else:
                    word.append(l)
            if len(word)!=0:
                #print(word)
                words.append([word, is_word(word)])
                if words[-1][1]:
                    nr+=100
                else:# - use distance to find closest word
                    if vocabulary is not None:
                            w1,v = get_closest_word(word, vocabulary)
                            #print(v,[eval_util.getIndex(j1,caracters) for j1 in w1 if j1] )
                            if v <= 1.5:
                                words[-1][0] = w1
                                nr += 5-v
            wordsList.append([words,nr])
        
        wordsList.sort(key=lambda x: x[1], reverse=True) 
        #print(wordsList)
        #return 9
        npres = []
        for w in wordsList[0][0]:
            if len(npres) > 0 :
                npres+=[27]
            npres+=w[0]
        nPreds.append(npres)
    err = 0.0
    for k,truth in  enumerate(labels_val[:bk]):
        truth = cut_zeros(truth)
        #print(truth)
        #print(nPreds[k])
        err += levenshtein(truth, nPreds[k])/float(len(truth))
    err /= len(labels_val[:bk])*1.0
    return nPreds, err

def get_error(labels_val, nPreds, bk=100):
    err = 0.0
    for k,truth in  enumerate(labels_val[:bk]):
        truth = cut_zeros(truth)
        #print(truth)
        #print(nPreds[k])
        err += levenshtein(truth, nPreds[k])/float(len(truth))
    err /= len(labels_val[:bk])*1.0
    return err

def mkP(decoder):
    pr = [[] for _ in range(decoder[0].shape[0])]
    for i in decoder: # each prediction batch
        for k,j in enumerate(i): #each prediction from batch
            pr[k].append(j)
    return pr