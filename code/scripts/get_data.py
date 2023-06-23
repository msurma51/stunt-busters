import os

script_dir = os.getcwd()
os.chdir('..\..')
proj_dir = os.getcwd()
data_dir = os.path.join(proj_dir, "data")
os.system("rmdir /s {}".format(data_dir))
os.system("mkdir data")
os.chdir(data_dir)
os.system("kaggle competitions download -c nfl-big-data-bowl-2023")
os.system('tar -xf nfl-big-data-bowl-2023.zip')
os.chdir(script_dir)
