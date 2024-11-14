
import argparse
import xml.etree.ElementTree as ET

parser = argparse.ArgumentParser(
    prog="cell-grep-extract",
    description=
    """
    Make table.txt for cell_grep.py .
    Input: cellosaurus.xml Output: table.txt
    """)
parser.add_argument("filename")
parser.add_argument("output")

args = parser.parse_args()

filename = args.filename
output   = args.output

tree = ET.parse(filename)
root = tree.getroot()

count = 0

fp = open(output, "w")

L = root.find("cell-line-list")
for cell_line in L:
    count += 1
    accession_list = cell_line.find("accession-list")
    primary = None
    for accession in accession_list:
        if accession.get("type") == "primary":
            primary = accession.text
    if primary == None: continue
    sex = cell_line.get("sex")
    if sex == None or "unspec" in sex: sex = "None"
    age = cell_line.get("age")
    if age == None or "unspec" in age: age = "None"
    if "Y" in age:
        # Cut out Y for Year and any month info
        c = age.find("Y")
        age = age[:c]
    # print(sex)
    # print(age)
    aliases = []
    name_list = cell_line.find("name-list")
    for name in name_list:
        # print(name.text)
        alias = name.text
        if "[" in alias:
            c = alias.find("[")
            alias = alias[:c].strip()
        aliases.append(alias)

    xref_list = cell_line.find("xref-list")
    if xref_list is not None:
        # aliases.append("XREF")
        for xref in xref_list:
            # print(xref.attrib["accession"])
            aliases.append(xref.attrib["accession"])

    comment_list = cell_line.find("comment-list")
    if comment_list is None:
        comment_list = []
    population = None
    for comment in comment_list:
        if comment.get("category") == "Population":
            # print(comment.text.strip())
            population = comment.text.strip()
    if population is None:
        population = "None"
    # print("")

    fp.write(primary)
    fp.write("\t")
    fp.write(sex)
    fp.write("\t")
    fp.write(age)
    fp.write("\t")
    fp.write(population)
    fp.write("\t")
    fp.write(",".join(aliases))
    fp.write("\n")

print("count: %i" % count)
