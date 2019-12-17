import os
import subprocess
import sys
# from verbose_plots import auprc_all

model_dict = {
	"1": "first_model.py",
	"2": "deepbind_model.py",
}
test_only = False
plot_only = False
if (len(sys.argv) == 4 ):
	if (sys.argv[3] == "test-only"):
		test_only = True
	else:
		plot_only = True

TF_list = []

rootdir = sys.argv[1] 	
for filename in os.listdir(rootdir):
	if not filename.startswith(".") and not filename.startswith("README"):
		tf_dir = os.path.join(rootdir,filename,filename+"-train-sequence.fa")
		tf_dir2 = os.path.join(rootdir,filename,filename+"-train.bed")
		# print(tf_dir,tf_dir2,"python first_model.py " + tf_dir + " " + tf_dir2 )
		# if plot_only:
		# 	continue
		# if not test_only:
		os.system("python " + model_dict[sys.argv[2]]+" " + tf_dir + " " + tf_dir2 + " " + filename)
		# tf_dir = os.path.join(rootdir,filename,filename+"-test-sequence.fa")
		# tf_dir2 = os.path.join(rootdir,filename,filename+"-test.bed")
		# os.system("python " + "predict_seq.py"+" " + tf_dir + " " + tf_dir2 + " " + filename)