Compression results on an extensively diverse set of Wikibooks files.

1. Basic compression and decompression.
* Min-max symbol length: 3-32 bytes
* Git tag: https://cs-git-research.cs.surrey.sfu.ca/cameron/parabix-devel/-/tags/CMP_DeCMP_v1
* Max number of Unicode word (ztf-symbol) in phrase: 2

| Filename | Size (MB) | Compressed size (MB) | Hashtable (MB) | Time (sec) | Decompression time (sec) |
|---------:|----------:|---------------------:|---------------:|----------: |-------------------------:|
|Arabic    | 18.7523   | 12.5694              | 2.2411         | 4.555      |0.502                     |
|German    | 195.48    | 153.91               | 3.5286         | 47.672     |5.474                     |
|Greek     | 18.7002   | 9.6006               | 2.6703         | 5.138      |0.681                     |
|Spanish   | 70.4887   | 53.196               | 2.1076         | 17.3       |2.024                     |
|Persian   | 19.2197   | 10.1089              | 1.8692         | 5.129      |0.681                     |
|Finnish   | 19.0433   | 13.361               | 2.1648         | 5.229      |0.689                     |
|French    | 87.567    | 64.1632              | 2.2984         | 21.211     |2.57                      |
|Indonesian| 14.4981   | 10.0327              | 1.0014         | 3.842      |0.546                     |
|Japanese  | 53.8666   | 47.1497              | 2.2697         | 12.791     |1.552                     |
|Korean    | 12.0246   | 9.737                | 1.1444         | 3.118      |0.457                     |
|Russian   | 56.4551   | 36.8118              | 3.624          | 14.462     |1.789                     |
|Thai      | 11.0458   | 9.2983               | 0.772476       | 2.825      |0.407                     |
|Turkish   | 11.7863   | 8.3733               | 1.5163         | 3.382      |0.468                     |
|Vietnamese| 11.2978   | 7.4577               | 0.84877        | 3.087      |0.44                      |
|Chinese   | 19.5651   | 14.5817              | 1.8406         | 5.064      |0.704                     |
|All-wiki  | 619.79    | 554.81               | 4.0436         | 148.873    |14.191                    |

1. Using scalable hashtable
* Min-max symbol length: 3-32 bytes
* Git commit SHA: 30e3dede9405064973af836fde2ce153df0e4b8b

| Filename | Size (MB) | Compressed size (MB) | Hashtable (MB) |
|---------:|----------:|---------------------:|---------------:|
|Arabic    | 18.7523   | 10.7836              | 3.7036         |
|German    | 195.48    | 145.47               | 8.5545         |
|Greek     | 18.7002   | 7.6456               | 4.3704         |
|Spanish   | 70.4887   | 50.9866              | 3.1941         |
|Persian   | 19.2197   | 9.1646               | 2.532          |
|Finnish   | 19.0433   | 12.1977              | 3.1733         |
|French    | 87.567    | 61.4654              | 3.5301         |
|Indonesian| 14.4981   | 9.5497               | 1.3796         |
|Japanese  | 53.8666   | 46.2149              | 3.1613         |
|Korean    | 12.0246   | 9.6029               | 1.297          |
|Russian   | 56.4551   | 29.6791              | 8.3315         |
|Thai      | 11.0458   | 9.2451               | 0.832972       |
|Turkish   | 11.7863   | 7.7808               | 2.0695         |
|Vietnamese| 11.2978   | 7.2636               | 1.0205         |
|Chinese   | 19.5651   | 14.138               | 2.3369         |
|All-wiki  | 619.79    | 539.53               | 12.505         |