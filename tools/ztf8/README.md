ZTF-Phrase-Hash is an extended version of the ZTF compression/decompression algorithm developed on the basis of the Parabix framework. 

The main components of ZTF-Phrase-Hash are compression and decompression pipeline. The components of the compression and decompression algorithm are divided across multiple sub-components called kernels, as the programming paradigm of the Parabix framework intends its applications to contain a pipelined execution of kernels.

## Compression

### Definitions:

1. ZTF symbol - any group/s of UTF-8 codepoints word that complies with the rules of word boundaries of the Unicode text segmentation algorithm - https://unicode.org/reports/tr29/#Word_Boundaries
The length of a ZTF symbol is between 3-32 bytes.

2. ZTF phrase - a group of 1 or more ZTF symbols. The length of a ZTF phrase is between 3-32 bytes.

3. ZTF hashcode - any pair, triple, quadruple byte sequence following the below rules.
- First byte/ prefix is always a legal/illegal UTF-8 multi-byte prefix. The range is `0xC0-0xFF`
- The subsequent bytes/ suffix are ASCII bytes in the range `0x00-0x7F`
- Each hashcode is an independent unit of data. It does not depend on its successors or predecessors both at the time of compression and decompression, allowing streaming parallel operations.

4. Encode - The process of translating data from one form to another; this may include compression, or it may refer to other translations done as part of this specification.

5. Decode:  The reverse of "encode"; describes a process of reversing a prior encoding to recover the original content.

6. Bixnum - A transposed representation of a non-negative integer sequence.

### Properties:

1. Any ZTF-phrase compressed by the compression algorithm, has its constituent ZTF-symbols occur in plain text at least once in the compressed file.
- This helps in ensuring that a particular full-word exists in the uncompressed file or not even before proceeding with the complete decompression of the compressed file.
2. All the ZTF-phrases of the same length are encoded in a way to contain the hashcode prefix from a predefined range of prefix bytes. 
- This helps in determining the length of the ZTF-phrase easily at the decompression step.


The overall structure of ZTF-Phrase-Hash compression has 8 phases. Every procedure implementation is explained with an example on the respective Wiki page. A brief explanation of each step is provided in the later section of this Wiki.

1. Unicode word segmentation - Identify the words that comply with the Unicode word boundary definition. 
2. [ZTF-symbol identification](https://cs-git-research.cs.surrey.sfu.ca/cprabhu/parabix-devel/-/wikis/ZTF-symbol-identification) - Among the identified Unicode words, select the words in the length range of 3-32 bytes as ZTF-symbols.
3. [Hashcode calculation](https://cs-git-research.cs.surrey.sfu.ca/cprabhu/parabix-devel/-/wikis/Hashcode-calculation) - Calculate prefix-based hashcodes for every symbol.
4. [Identify compressible hashcodes/phrases](https://cs-git-research.cs.surrey.sfu.ca/cprabhu/parabix-devel/-/wikis/Repeated-phrases/Repeated-phrases-and-hashcodes) - For the hashcodes calculated, identify all the repeated hashcodes in the complete input file. This step helps to avoid storing all the k-symbol ZTF-phrases in the hash table. The advantages of this step are discussed later in detail. 
5. [Select the most number of non-overlapping ZTF phrases for compression](https://cs-git-research.cs.surrey.sfu.ca/cprabhu/parabix-devel/-/wikis/Phrase-selection-logic) - Among the repeated hashcodes identified for all k-symbol phrases, identify the longest k-symbol ZTF-phrases that are non-overlapping and could be considered for compression. This step is comprised of 3 phases to identify the longest and non-overlapping phrases which will be discussed in detail in the specific section.
6. [Length-based compression of ZTF-phrases](https://cs-git-research.cs.surrey.sfu.ca/cprabhu/parabix-devel/-/wikis/Compression-of-ZTF-phrases) - Perform a dictionary-based compression. Store an initial occurrence of all the unique phrases in a hash table that are compressed. This helps to avoid hashcode collision in case more than 1 phrases correspond to the same hashcode value. The length-based compression helps process all the phrases belonging to the same length group. This helps to easily generate the hashcode prefixes.
This step provides us with a mask that indicates the byte positions that are to be compressed. 

6.1 Eliminate any overlapping (k-1)-symbol phrases - Once all the k-symbol phrases are compressed (marked for compression by rewriting the plaintext with codewords), we check if any of the (k-1)-symbol phrases are marked that overlap with already compressed phrases. We eliminate any such phrases in this step.

Steps 3-6.1 continue for phrases with k-symbol till 1-symbol phrases.

7. Delete the bytes marked for compression - With all the accumulated masks that indicate the byte positions to be compressed, we use parallel deletion techniques to compress the data.
8. Print the compressed data - final compressed data is printed to STDOUT


## Implementation details for steps 3-6.1

### 3. Hashcode calculation

- Hashcode calculation is an extended version of the ZTF-hash Bixnum algorithm where bits from previous byte positions are blended into the subsequent bytes to randomize each ZTF-symbol hashcode. The bit-mixing is restricted at the word boundary of each ZTF-symbol.
This is performed in a log-based technique where not all the previous i-1 bits are mixed with the i-th bit of the symbol.
- The maximum `k` value for the current execution of the algorithm is determined at run-time. `k` value indicates at most how many ZTF-symbols comprise a ZTF-phrase during compression.
- We calculate k different hashcodes for each symbol. For every k-symbol phrase, the last byte of the phrase undergoes bit-mixing with hashcodes of previous (k-1) symbols. The result of incremental hashcode calculation increases the randomness of the hashcode of each phrase.
- In order to finalize the hashcode, we calculate the length of k-symbol phrases using parallel bitwise techniques and are concatenated with the hashcodes. 
- The length + hashcode of k-symbol phrase helps with identifying the prefix range and length-based compression step which are discussed in later sections.  

### 4. Identify compressible hashcodes/phrases

- In this step, we process ZTF-phrases with k-symbol and make our way with decreasing the value of `k` to 1.
- This step is performed for all the k-symbol phrases in a segment with a predefined number of strides. For every k-symbol phrase, based on the length of the phrase, we extract a specific # of bits of the hashcode (to be shared as a table) which will be called the encoded form of the phrase. We store these hashcodes in a hashtable and look for any repeated occurrence of the same hashcode.
-- This step does not take into consideration any collisions between phrases with the same hashcodes. We handle the collision at the time of compression.
-- We create a bitstream marking 1's for the last byte of symbols S which have k-symbol phrases ending at S that were seen previously in the same segment.

### 5. Select non-overlapping ZTF phrases for compression

- Every phrase selected for compression has to go through 3 steps to determine if can be compressed. Consider an example for k-symbol phrases of length 12 bytes
1.  Check if there are any other 12-byte phrases marked for compression in the overlapping region.
- Eg: In the sequence "Hello world there ", if "Hello world " and "world there " are marked for compression as both the phrases were seen previously. In this case, we compress only the leftmost (first) 2-symbol phrase "Hello world " and consider "there " as a 1-symbol phrase.
2. Check if any phrase of length 12 from step 1 is in the following overlapping region of a longer phrase (length > 12) already selected for compression.
- Eg: In the sequence "national green month ", if "green month " is marked for compression from step 1, and "national green " is also marked for compression from phrases of length 15, we remove the compression mark for "green month " and prioritize longer length phrase "national green " to be compressed.
- However, "month " will be considered for 1-symbol phrase compression.
3. Check if any phrase of length 12 from step 2 is in the preceding overlapping region of a longer phrase (length > 12) already selected for compression.
- Eg: In the sequence "green month national ", if "green month " is marked for compression from step 2, and "month national " is also marked for compression from phrases of length 15, we remove the compression mark for "green month " and prioritize longer length phrase "month national " to be compressed.

At the end of these 3 steps, we retain only the non-overlapping hashcode marks. The steps start from maximum length phrases and continue with smaller length phrases.

### 6. [Length-based compression of ZTF-phrases](https://cs-git-research.cs.surrey.sfu.ca/cprabhu/parabix-devel/-/wikis/Encoding-schemes)

- This is the core of the compression algorithm. It starts from maximum k-symbol phrases and falls back to 1-symbol phrase compression.
- For every phrase with k-symbols represented by a 1-bit at the last byte of the phrase, we store the phrase in the hash table to ensure no two different phrases with the same hashcode are compressed to the same encoded form.
- Phrases of length 3-8 are encoded as 2-byte codewords. Phrases of length 9-16 are encoded as 3-byte codewords. Phrases of length 17-32 are encoded as 4-byte codewords.
- The below table elaborated prefix ranges for different length phrases.

| Length | Prefix                  |
| -------|:-----------------------:|
|   3    |  0xC0-0xC7              |
|   4    |  0xC8-0xCF              |
|   5    |  0xD0, 0xD4, 0xD8, 0xDC |
|   6    |  0xD1, 0xD5, 0xD9, 0xDD |
|   7    |  0xD2, 0xD6, 0xDA, 0xDE |
|   8    |  0xD3, 0xD7, 0xDB, 0xDF |
|   9    |  0xE0, 0xE8             |
|  10    |  0xE1, 0xE9             |
|  11    |  0xE2, 0xEA             |
|  12    |  0xE3, 0xEB             |
|  13    |  0xE4, 0xEC             |
|  14    |  0xE5, 0xED             |
|  15    |  0xE6, 0xEE             |
|  16    |  0xE7, 0xEF             |
|  17    |  0xF0                   |
|  18    |  0xF1                   |
|  19    |  0xF2                   |
|  20    |  0xF3                   |
|  21    |  0xF4                   |
|  22    |  0xF5                   |
|  23    |  0xF6                   |
|  24    |  0xF7                   |
|  25    |  0xF8                   |
|  26    |  0xF9                   |
|  27    |  0xFA                   |
|  28    |  0xFB                   |
|  29    |  0xFC                   |
|  30    |  0xFD                   |
|  31    |  0xFE                   |
|  32    |  0xFF                   |


### 6.1 Eliminate any overlapping (k-1)-sym phrases

- An extraction mask is produced from the last step that indicates the byte positions to be compressed. Using that eliminate any (k-1)-symbol phrase hashcode positions marked for compression so that redundant compression of sub-phrases and overwrite of encoded symbols does not happen.


### Build and Run

Build steps are the same as mentioned in the README of the Parabix home page.
To run the ztf-phrase-hash application, follow the below steps.


1. Compression: currently works only in single-threaded mode. We are writing the hashtable and the compressed data in the compressed file, where each segment of compressed data has the corresponding dictionary in the preamble.

`bin/ztf-phrase-hash <input_file> > compressed.z -thread-num=1`

2. Decompression: takes in the hashtable and the compressed data. Creates the hashtable first for each segment and then replaces the codeword in the compressed data referring to the hashtable entries.

`bin/ztf-phrase-hash -d compressed.z > decompressed`

