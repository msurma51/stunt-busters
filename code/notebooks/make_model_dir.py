import os

curr_dir = os.getcwd()
os.chdir('../..')
proj_dir = os.getcwd()
model_dir = os.path.join(proj_dir, 'state_dicts')
if not os.path.exists(model_dir):
	os.mkdir(model_dir)
os.chdir(curr_dir)