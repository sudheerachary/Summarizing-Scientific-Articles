import re
import string
import sys
f=open(sys.argv[1],'r').read()
f=re.sub('\. ', '.\n', f)
f=re.sub('\n\n','\n',f)
f=f.split('\n')
for sent in f:
	sent=sent.strip()
	sent=re.sub("\. ", ".\n", sent)
	sent=re.sub("\.[0-9]+(,\s?[0-9]\s?)+", ".\n", sent)
	sent=re.sub("[0-9]+(,\s?[0-9]\s?)+", " ", sent)
	sent=re.sub("\n\s*[0-9]+\s*","\n",sent)
	sent=re.sub("\.[0-9]+\n","\.\n",sent)
	sent=re.sub("\s+"," ",sent)
	for c in string.ascii_letters: sent=re.sub(c+"\.[0-9]+\s", c+".\n",sent)
	sent=re.sub("\. ", ".\n", sent)
	sent=re.sub("\n\n","\n",sent)
	sent=sent.strip()
	if '.' not in sent: continue
	if len(sent) != 0: print(sent)
