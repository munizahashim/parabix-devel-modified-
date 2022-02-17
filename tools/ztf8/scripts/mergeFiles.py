import sys
dictFile = sys.argv[1]
cmpFile = sys.argv[2]
finalCmpFile = sys.argv[3]
with open(cmpFile,"rb") as compressedFile:
    compressed = compressedFile.read()
    with open(dictFile,"rb") as hashTable:
        fullHashTable = hashTable.read()
        with open(finalCmpFile,"wb") as finalFile:
            finalFile.write(fullHashTable)
            finalFile.write(compressed)
            finalFile.close()
        hashTable.close()
    compressedFile.close()
print("Files merged successfully.")

