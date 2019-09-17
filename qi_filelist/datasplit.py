import sys

import os

args = sys.argv
# if len(args) != 3:
#     sys.exit(-1)

output_path = "../ace2005-preprocessing/jmee_data_list.csv"

split_names = ["train", "test", "dev"]
with open(output_path, "w+") as fout:
    fout.write("type,path\n")
    for split_name in split_names:
        with open("../qi_filelist/new_filelist_ACE_%s" % split_name, "r") as flist:
            for fname in flist:
                fname = fname.strip()
                if len(fname) == 0: continue
                fout.write(split_name+","+fname+"\n")

print("Done!")
