import networkx as nx
import spacy
import numpy as np
import sys
"""
tokens_in = 0
tokens = 0
mean = []
mean_2 = []
with open("semeval_test.tsv") as infile:
	for line in infile:
		line = line.strip().split("\t")
		spd = line[1].split()
		#print (spd)
		#print (len(spd))
		#input("")
		tokens_in += len(spd) - 2
		es = line[2].split()
		e1, e2 = int(es[0]), int(es[1])
		tokens += e2 - e1 - 1
		mean.append(len(spd) - 2)
		mean_2.append(e2 - e1 - 1)
print (tokens_in, tokens)
print (tokens_in / tokens)
print (np.mean(mean))
print (np.mean(mean_2))
sys.exit(0)
		
"""


nlp = spacy.load('en')

def generate_dependency_parses(filename, outfile_name):


	# document = nlp(u'Robots in popular culture are there to remind us of the awesomeness of unbound human agency.')

	counter = 0
	x1, x2, y = [], [], []
	positions = []
	#with open("/home/dominik/Documents/DFKI/clean_dir/Hiwi-master/NemexRelator2010/SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT") as infile:
	tokens_outside = 0
	with open(filename) as infile:
		for doc in infile:
			if counter % 4 == 0:
				# get rid of <e1>...</e1>
				document = nlp(doc.split("\t")[1].strip().replace("<e1>", "e1>").replace("</e1>", "</e1").replace("<e2>", "e2>").replace("</e2>", "</e2")[1:-1])

				new_doc = []
				pos = []

				# get new sentence without entity indicators, but remember positions
				for i,token in enumerate(document):
				    #print (token)
				    if token.text.startswith("e1>"):
				      start_e1 = i
				    if token.text.endswith("</e1"):
				      end_e1 = i
				      e1 = token.text[:-4] + "-" + str(i)
				      pos.append(i)


				    if token.text.startswith("e2>"):
				      start_e2 = i

				    if token.text.endswith("</e2"):
				      end_e2 = i
				      e2 = token.text[:-4] + "-" + str(i)
				      pos.append(i)

				    if token.text.startswith("e1>") or token.text.startswith("e2>"):
				      new_tok = token.text[3:]
				      #print (new_tok)
				    else:
				      new_tok = token.text
				    if new_tok.endswith("</e2") or new_tok.endswith("</e1"):
				      new_tok = new_tok[:-4]

				    new_doc.append(new_tok)
				doc = nlp(" ".join(new_doc))
				for i, token in enumerate(doc):
				    if i == end_e1:
				      e1 = token.text.lower() + "-" + str(i)
				    if i == end_e2:
				      e2 = token.text.lower() + "-" + str(i)
				    #print (i, token)

				#print (start_e1, end_e1, start_e2, end_e2)


				# create graph with all the edges (dependencies)
				edges = []
				for token in doc:
					# FYI https://spacy.io/docs/api/token
					for child in token.children:
						edges.append(('{0}-{1}'.format(token.lower_,token.i),'{0}-{1}'.format(child.lower_,child.i)))
						#print ('{0}-{1}'.format(token.lower_,token.i),'{0}-{1}'.format(child.lower_,child.i))
						#input("")
				graph = nx.Graph(edges)  # Well that was easy
				#print ([doc.text for doc in doc])
				#print ([i for i in edges])
				tokens = [tok.text for tok in doc]

				# compute dependency path
				try:
					sdp = nx.shortest_path(graph, source=e1, target=e2)
					x2.append([i.split("-")[-1] for i in sdp])
				except:
					x2.append([""])
			
				"""
				context_left, e1, context_mid, e2, context_right = self.tokenize(line)
				position_e1 = len(context_left) + len(e1) - 1
				position_e2 = len(context_left) + len(e1) + len(context_mid) + len(e2) - 1
				entity_positions.append((position_e1, position_e2))
			
				sent = context_left + e1 + context_mid + e2 + context_right
				X.append(sent)
				"""
				positions.append(pos)
				x1.append(tokens)
				#x2.append(i.split("-")[-1] for i in sdp)
				for i in sdp:

					if int(i.split("-")[-1]) < int(e1.split("-")[-1]):
						tokens_outside += 1
					if int(i.split("-")[-1]) > int(e2.split("-")[-1]):
						tokens_outside += 1

			elif counter % 4 == 1:
				y.append(doc.strip())

			counter += 1


	print (len(x1), len(x2), len(y), len(positions))
	print (tokens_outside) # 1861 tokens outside, 14327
	
	with open(outfile_name, "w") as outfile:
		for sent, sdp, pos, label in zip(x1, x2, positions, y):
			#print (sent, sdp, pos, label)
			outfile.write(" ".join(sent) + "\t" + " ".join(sdp) + "\t" + " ".join(list(map(str,pos))) + "\t" + label + "\n")



# "/home/dominik/Documents/DFKI/clean_dir/Hiwi-master/NemexRelator2010/SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT"
# "semeval_train.tsv"

generate_dependency_parses(sys.argv[1], sys.argv[2])
