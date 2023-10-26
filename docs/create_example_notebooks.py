import sys
import glob
import os
import shutil


print('Creating example notebooks...')
string = """
{
    "path": "../../../examples/<file>.ipynb"
}"""

string_examples = """
Examples
========

.. toctree::
   :maxdepth: 3
"""

this_dir = os.path.dirname(os.path.realpath(__file__))

files = glob.glob(os.path.join(this_dir, "../examples/*.ipynb"))
examples_file = open(os.path.join(this_dir, "source/examples.rst"), "w")
examples_file.write(string_examples)

# remove all files in source/examples:
for file in os.listdir(os.path.join(this_dir, "source/examples")):
    if file.endswith(".nblink"):
        os.remove(os.path.join(this_dir, "source/examples", file))

for file in files:
    print(file)
    f = os.path.split(file)[-1].split(".")[0]
    path = os.path.join(this_dir, "source/examples", f + ".nblink")
    # make a file
    if os.path.exists(path):
        os.remove(path)

    with open(path, "w") as d:
        d.write(str(string).replace("<file>", str(f)))

    examples_file.write("\n   examples/" + f + ".nblink")
examples_file.close()