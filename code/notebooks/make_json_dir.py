import os

curr_dir = os.getcwd()
os.chdir('../..')
proj_dir = os.getcwd()
json_dir = os.path.join(proj_dir, 'json')
if not os.path.exists(json_dir):
	os.mkdir(json_dir)
os.chdir(curr_dir)