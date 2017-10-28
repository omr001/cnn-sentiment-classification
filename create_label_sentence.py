from nltk.tree import Tree
import codecs
import json
import re
import sys
if __name__ == "__main__":
    '''
    SST dataset to label_sentence.txt
    '''
    f = open('./data/sst'+sys.argv[2]+'_'+sys.argv[1]+'_label_sentence.txt', 'w')
    
    if sys.argv[2] == '5':
        if sys.argv[1] == 'dev' or sys.argv[1] == 'test':
            for line in open('./data/'+sys.argv[1]+'.txt'):
                t = Tree.fromstring(line)
                sentence = ' '.join(t.leaves())
                label = t.label()
                f.write(label+' '+sentence+'\n')
        elif sys.argv[1] == 'train':
            for line in open('./data/'+sys.argv[1]+'.txt'):
                t = Tree.fromstring(line)
                h = t.height()
                for h in range(h-1):
                    for s in t.subtrees(lambda x: x.height() == h+2):
                        phrase = ' '.join(s.leaves())
                        label = s.label()
                        f.write(label+' '+phrase+'\n')
    
    elif sys.argv[2] == '2':
        if sys.argv[1] == 'dev' or sys.argv[1] == 'test':
            for line in open('./data/'+sys.argv[1]+'.txt'):
                t = Tree.fromstring(line)
                sentence = ' '.join(t.leaves())
                label = t.label()
                if int(label) == 0 or int(label) == 1:
                   f.write('0'+' '+sentence+'\n') 
                elif int(label) == 3 or int(label) == 4:
                    f.write('1'+' '+sentence+'\n')
        elif sys.argv[1] == 'train':
            for line in open('./data/'+sys.argv[1]+'.txt'):
                t = Tree.fromstring(line)
                h = t.height()
                for h in range(h-1):
                    for s in t.subtrees(lambda x: x.height() == h+2):
                        phrase = ' '.join(s.leaves())
                        label = s.label()
                        if int(label) == 0 or int(label) == 1:
                            f.write('0'+' '+phrase+'\n')
                        elif int(label) == 3 or int(label) == 4:
                            f.write('1'+' '+phrase+'\n')

