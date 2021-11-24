with open("/home/cprabhu/parabix-devel/build/ztf-phrase-hash.z","rb") as compressedFile:
    compressed = compressedFile.read()
    with open("/home/cprabhu/parabix-devel/build/error","rb") as hashTable:
        fullHashTable = hashTable.read()
        with open("/home/cprabhu/parabix-devel/build/finalCompressed.z","wb") as finalFile:
            finalFile.write(fullHashTable)
            finalFile.write(compressed)
            finalFile.close()
        hashTable.close()
    compressedFile.close()
print("Files merged successfully.")

