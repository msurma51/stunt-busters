import os

curr_dir = os.getcwd()
os.chdir('../..')
proj_dir = os.getcwd()
plots_dir = os.path.join(proj_dir, 'plots')
if not os.path.exists(plots_dir):
	os.mkdir(plots_dir)
os.chdir(curr_dir)