Compression results on an extensively diverse set of Wikibooks files.

1. Basic compression and decompression.
* Min-max symbol length: 3-32 bytes
* Git tag: https://cs-git-research.cs.surrey.sfu.ca/cameron/parabix-devel/-/tags/CMP_DeCMP_v1
* Max # sym in phrase: 2

| Filename | Size (MB) | Compressed size (MB) | Hashtable (MB) | Time (sec) | Decompression time (sec) |
|---------:|----------:|---------------------:|---------------:|----------: |-------------------------:|
|Arabic    | 19.67     | 13.18                | 2.35           | 4.555      |0.502                     |
|German    | 204.98    | 161.39               | 3.7            | 47.672     |5.474                     |
|Greek     | 19.60     | 10.067               | 2.80           | 5.138      |0.681                     |
|Spanish   | 73.91     | 55.78                | 2.21           | 17.3       |2.024                     |
|Persian   | 20.15     | 10.60                | 1.96           | 5.129      |0.681                     |
|Finnish   | 19.96     | 14.01                | 2.27           | 5.229      |0.689                     |
|French    | 91.82     | 67.28                | 2.41           | 21.211     |2.57                      |
|Indonesian| 15.20     | 10.52                | 1.05           | 3.842      |0.546                     |
|Japanese  | 56.48     | 49.44                | 2.38           | 12.791     |1.552                     |
|Korean    | 12.60     | 10.21                | 1.20           | 3.118      |0.457                     |
|Russian   | 59.19     | 38.60                | 3.80           | 14.462     |1.789                     |
|Thai      | 11.58     | 9.75                 | 0.81           | 2.825      |0.407                     |
|Turkish   | 12.35     | 8.78                 | 1.59           | 3.382      |0.468                     |
|Vietnamese| 11.84     | 7.82                 | 0.89           | 3.087      |0.44                      |
|Chinese   | 20.51     | 15.29                | 1.93           | 5.064      |0.704                     |
|All-wiki  | 649.90    | 581.76               | 4.24           | 148.873    |14.191                    |


