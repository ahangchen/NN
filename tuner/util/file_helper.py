def write(path, content):
    with open(path, "a+") as dst_file:
        dst_file.write(content + '\n')
