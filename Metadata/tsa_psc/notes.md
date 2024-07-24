###---------------------------------------------------------------
## Wrong Labellings
see: https://www.kaggle.com/c/passenger-screening-algorithm-challenge/discussion/41223
https://www.kaggle.com/c/passenger-screening-algorithm-challenge/discussion/37615
https://www.kaggle.com/c/passenger-screening-algorithm-challenge/discussion/45801

- *ignored: did not change labelling but marked a region for zone
- ignored*: did not change labelling nor mark any region for zone
- corrected: changed labelling of affected zones

### Suspicious labellings
1. 6cbda8596c5c9b1e31d4fab9b5a9e02b should be 0 not 1. No 	suspicious object in Zone 9 [ignored*]
2. 9e067ae96bb10fb62a3b4e7adf4d58ca nothing in Zone 9 [*ignored]
3. ab52f3a07e8d37a5b7120acc81258254 nothing in Zone 9 [ignored*]
4. e48b103b2d8bedb994c0ce62e15d1662 nothing in Zone 3 [ignored*] 
5. e4b560b0f6d2c44535610f38a787df93 should be 0 not 1. No 	suspicious object in Zone 15 [*ignored]
6. 52c8235df3f0552e6c134529ca85d958 nothing in Zone 9 [ignored*]
7. 7235e754185d3321c4b6883d001a35ad nothing in Zone 9 [*ignored]

### Obvious mislabelling
1. swap zone labelling for 623c761b4db398ea2157e6c5cd6c8c58 and 	496ec724cc1f2886aac5840cf890988a [corrected]
2. 56b9c0086836fe2fca86d773cacaf783, threat in zone 2 not 4 	[corrected]
3. cbc6f0a3be3d802fc3d2bd45c183a49d, threat in zone 2 not 4 	[corrected]
4. d904d73f5e53eed05fef89ce0032fc1c, threat in zone 12 not 9 	[corrected]
5. 81aab48146e23ac8f9744f18459ecdbe, threat in zone 17 not 7 	[corrected]

### Corrupt scan
1. 42181583618ce4bbfbc0c4c300108bf5, broken data in frame 15 & 16
	copied frame 0 as 16 for the extracted dataset. Original 	scan images are left unchanged in full_image_threat/ dir. UPDATE: Scan is excluded from training dataset.
###---------------------------------------------------------------

### Size Stats of labelled boxes
- max width: 159
- max height: 101
- min width: 12
- min height: 11
- avg width: 55
- avg height: 56
- num of markings: 14533

### Processing scans with no threats
Saving .aps scans with no threats resulted to 319 of 1229 scans

### Processing all scans and organizing frames with no threats
1. frame 0: 553 
2. frame 1: 563
3. frame 2: 579
4. frame 3: 640
5. frame 4: 798
6. frame 5: 657
7. frame 6: 580
8. frame 7: 561
9. frame 8: 539
10. frame 9: 546
11. frame 10: 565
12. frame 11: 600
13. frame 12: 702
14. frame 13: 578
15. frame 14: 550
16. frame 15: 573

### Cropping regions with threat and no-threat
threat: 14533
no-threat: 14533

