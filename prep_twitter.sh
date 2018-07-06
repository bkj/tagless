# WITHOUT SEEDS
#python tagless/prep.py --inpath twitter.feats --outpath ./target_dir/data/twitter

# WITH SEEDS
python tagless/prep.py --inpath twitter.feats --outpath ./target_dir/data/twitter --inseed twitter_seeds.txt  --outseed ./target_dir/data/twitter_seeds
