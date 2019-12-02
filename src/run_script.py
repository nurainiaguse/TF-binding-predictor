import os
import subprocess
import sys

rootdir = sys.argv[1] 	
for filename in os.listdir(rootdir):
	if not filename.startswith("."):
		tf_dir = os.path.join(rootdir,filename,filename+"-train-sequence.fa")
		tf_dir2 = os.path.join(rootdir,filename,filename+"-train.bed")
		print(tf_dir,tf_dir2,"python first_model.py " + tf_dir + " " + tf_dir2 )
		os.system("python first_model.py " + tf_dir + " " + tf_dir2 + " " + filename)