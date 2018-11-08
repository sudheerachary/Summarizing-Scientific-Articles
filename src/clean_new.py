import os, re
i = 1
list_of_files = [f1 for f1 in os.listdir('.') if os.path.isfile(f1)]

def removeTags(current_line):
	cleanr = re.compile('<.*?>')
	cleantext = re.sub(cleanr, '', current_line)
	return cleantext

for file in list_of_files:
	corpora = open('./annotated_data/'+str(i)+'.txt', 'w')
	annotation = open('./annotated_data/'+str(i)+'.ann', 'w')
	with open(file) as f:
	    lines = f.readlines()
	    for line in lines:
	    	if 'AIM\'>' in line:
	    		text = line.split("AIM\'>")[1]
	    		if '</A-S>' in text:
	    			text = text.split('</A-S>')[0]
	    		elif '</S>' in text:
	    			text = text.split('</S>')[0]
	    		text = text.strip()
	    		text = removeTags(text)
	    		corpora.write(text)
	    		corpora.write('\n')
	    		annotation.write('aim\n')

	    	elif 'BKG\'>' in line:
	    		text = line.split("BKG\'>")[1]
	    		if '</A-S>' in text:
	    			text = text.split('</A-S>')[0]
	    		elif '</S>' in text:
	    			text = text.split('</S>')[0]
	    		text = text.strip()
	    		text = removeTags(text)
	    		corpora.write(text)
	    		corpora.write('\n')
	    		annotation.write('background\n')

	    	elif 'OTH\'>' in line:
	    		text = line.split("OTH\'>")[1]
	    		if '</A-S>' in text:
	    			text = text.split('</A-S>')[0]
	    		elif '</S>' in text:
	    			text = text.split('</S>')[0]
	    		text = text.strip()
	    		text = removeTags(text)
	    		corpora.write(text)
	    		corpora.write('\n')
	    		annotation.write('other\n')

	    	elif 'CTR\'>' in line:
	    		text = line.split("CTR\'>")[1]
	    		if '</A-S>' in text:
	    			text = text.split('</A-S>')[0]
	    		elif '</S>' in text:
	    			text = text.split('</S>')[0]
	    		text = text.strip()
	    		text = removeTags(text)
	    		corpora.write(text)
	    		corpora.write('\n')
	    		annotation.write('contrast\n')

	    	elif 'OWN\'>' in line:
	    		text = line.split("OWN\'>")[1]
	    		if '</A-S>' in text:
	    			text = text.split('</A-S>')[0]
	    		elif '</S>' in text:
	    			text = text.split('</S>')[0]
	    		text = text.strip()
	    		text = removeTags(text)
	    		corpora.write(text)
	    		corpora.write('\n')
	    		annotation.write('own\n')

	    	elif 'TXT\'>' in line:
	    		text = line.split("TXT\'>")[1]
	    		if '</A-S>' in text:
	    			text = text.split('</A-S>')[0]
	    		elif '</S>' in text:
	    			text = text.split('</S>')[0]
	    		text = text.strip()
	    		text = removeTags(text)
	    		corpora.write(text)
	    		corpora.write('\n')
	    		annotation.write('textual\n')

	    	elif 'BAS\'>' in line:
	    		text = line.split("BAS\'>")[1]
	    		if '</A-S>' in text:
	    			text = text.split('</A-S>')[0]
	    		elif '</S>' in text:
	    			text = text.split('</S>')[0]
	    		text = text.strip()
	    		text = removeTags(text)
	    		corpora.write(text)
	    		corpora.write('\n')
	    		annotation.write('basis\n')
	i += 1