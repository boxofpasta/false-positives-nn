import os
import sys

"""
command-line args:
[1] directory
[2] initial counter value

description:
goes through specified directory, taking all images that start with 'Screenshot' and renames
them to falsei.jpg, where i is a counter that starts at a user-provided offset
"""

if len(sys.argv) > 2:
	counter = int(sys.argv[2])
else:
	counter = 0

dir_name = sys.argv[1]
for fname in os.listdir(dir_name):
	if fname[:10] == 'Screenshot':
		orig_name = dir_name + '/' + fname
		new_name = dir_name + '/false' + str(counter) + '.jpg'
		os.rename(orig_name, new_name)
		counter += 1