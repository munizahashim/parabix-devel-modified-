with open("/home/cprabhu/parabix-devel/build/error","ab") as f:
    f.write(b'\xFF\xFF')
    f.close()
print("The details you have given are successfully written to the corresponding file. You can open a file to check")

