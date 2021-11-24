import sys
offset = sys.argv[1]

with open("/home/cprabhu/parabix-devel/build/original","rb") as f:
    f.seek(int(offset), 1)
    original = f.read()
    with open("/home/cprabhu/parabix-devel/build/decompressed.txt","wb") as finalFile:
        finalFile.write(original)
        finalFile.close()
    f.close()

print("Decompressed data written to decompressed.txt")

