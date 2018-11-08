import sys
import os

toread=sys.argv[1]
towrite=open(sys.argv[2],'a')
annFile=open(sys.argv[3],'a')

anns={"A":"AIM","T":"TEXTUAL","O":"OTHER","W":"OWN","C":"CONTRAST","B":"BACKGROUND","S":"BASIS","I":"IGNORE","R":"RECHECK"}

def func(line):
	notdone=True
	while notdone:
		os.system("clear")
		print()
		print(line)
		for x in anns.keys():
			print(x+"-"+anns[x])
		print()	
		inp=(str)(input("Annotation: ")).upper()
		notdone=False
		if inp not in anns.keys():
			notdone=True
			continue
		if anns[inp]=="IGNORE": return
		towrite.write(line.strip()+"\n")
		annFile.write(anns[inp]+"\n")

with open(toread) as f:
	for line in f:
		func(line)

towrite.close()
annFile.close()

