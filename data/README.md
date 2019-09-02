# Polyvore-$U$s Dataset

There are four types of datasets saved in corresponding folders:

- Polyvore-630: `tuples_630`
- Polyvore-53: `tuples_53`
- Polyvore-519: `tuples_519`
- Polyvore-32: `tuples_32`

For simplicity, we use `tuples_n` to represent one of the above datasets.

Each `tuples_n` has the following files:

- Image list:

  - `image_list_top`
  - `image_list_shoe`
  - `image_list_bottom`

- Positive outfits:

  - `tuples_train_posi`
  - `tuples_val_posi`
  - `tuples_test_posi`

- Negative outfits:

  - `tuples_train_nega`
  - `tuples_val_nega`
  - `tuples_test_nega`
  - `tuples_train_nega_hard`
  - `tuples_val_nega_hard`
  - `tuples_test_nega_hard`

- FITB task:

  - `fill_in_blank_test`

## Image list files

Each image file saves the list of image names for corresponding category.

```plain
...
dc90bcdc75a05a2a4314ab816fd9b545a54f88aa.jpg
6c39ae4ad52cbd5535b75cd1a4b99b546b6d7ca3.jpg
72de70f320f6d54ad95355cd7e9859758036ae41.jpg
f1d95fb2414c5dfa163397d41a12c2891e70ddd1.jpg
34b2c7e0073e05d95ae5ce59aa48efa4c505a1fd.jpg
86694788a5699fbd858d019a7f16cc7e3d6bca5e.jpg
2950f1d1b1fe579195a9c30c284925553e10f27a.jpg
6322d5442110ce5e1f23be390269f23ceee0c6dd.jpg
d000a72c7777e4dc488b6c443b9fe875abc3cecc.jpg
beb68720d4f1c0353e1dc906b15ea62d9ef3f54d.jpg
02d8afd73ad3dd6774363ed6f831e3e138a82746.jpg
...
```

## Tuples files

There are two different outfits, one with fixed length and the other with variable length.
Since most of our collected data only have one bottom and one pair of shoes and rare outfits have more than two tops.
For variable-length outfits, we only use those outfits with one or two tops. Each outfit is saved as an index tuple.
The image name of each index can be found in the image list files.


Following is an example of fix-length outfits

```plain
user,top,bottom,shoe
0,19000,48753,37974
0,2224,49908,32656
0,54932,5748,6533
0,52587,32001,55739
0,48612,30775,4043
0,33408,36630,47362
0,43933,12612,18802
0,41211,15299,6439
0,1978,15437,43276
0,60437,4625,8297
0,69928,48765,49147
0,61017,55451,44237
0,23272,20904,42105
0,36426,737,12200
0,13885,7378,64784
...
```

where the first column are the indexes of users.

For variable-length outfits, we use `-1` to indicate the lack of second top.

```plain
user,top,top,bottom,shoe
0,61394,67131,25167,7305
0,44966,61548,22343,20402
0,40509,-1,43001,8635
0,8981,-1,32475,52841
0,7537,48836,26740,46410
0,20295,15862,15510,40988
0,20629,-1,2689,2777
0,41466,55124,27787,27874
0,42258,5203,31654,22994
0,16758,71187,36632,47514
0,51719,6670,28911,10548
0,73125,-1,32364,44325
0,37375,44822,37320,56293
0,24999,67748,23145,30436
0,62567,-1,51357,38127
0,24784,-1,1782,38940
...
```

All positive and negative outfits are saved with the same format.

## Types of negative outfits

There are two types of negative outfits:

- randomly mixed: negative outfits are randomly mixed using the items in the corresponding split, e.g training set.
- hard negative: hard negative outfits are sampled from the positive outfits of other users.

During training phase, one can use the negative outfits pre-prepared or generate before each epoch. In `PolyvoreDataset` class, by calling the method `set_nega_mode()` to change different types of negative outfits.

- `RandomFix`: the randomly mixed negative outfits in the pre-prepared file.
- `HardFix`: the hard negative outfits in the pre-prepared file.
- `RandomOnline`: the randomly mixed negative outfits before each epoch.
- `HardOnline`: the hard negative outfits before each epoch.

## Types of outputs

Each dataset class will return positive-negatie pairs or just positive/negative outfits.
There are three types of outpu. See the source code for details.

- `PairWise`: users, positive , negative
- `PosiOnly`: users, positive
- `NegaOnly`: users, negative

## Fill-in-the-Blank

The `fill_in_blank_test` file is for the FITB task. In each line, we give several outfits. The first is the ground-truth which should be ranked in the first place. The outfits in each line are only different in one items. We only give the list for `test` data and set the number of items to `4`.
You can generate other samples by using the method `make_fill_in_blank` in `FITBDataset` class.

```plain
user,top,bottom,shoe,user,top,bottom,shoe,...
0,2125,46574,11630,0,42251,46574,11630,...
0,25509,31273,43047,0,25509,31273,46786,...
0,42775,24203,22618,0,42775,19304,22618,...
0,48350,37488,63721,0,48350,51013,63721,...
```
