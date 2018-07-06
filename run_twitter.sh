# WITHOUT SEEDS
# python -m tagless --outpath ./target_dir/results/my-twitter-labels --inpath ./target_dir/data/twitter --img-dir "" --n-return 1

# WITH SEEDS
python -m tagless --outpath ./target_dir/results/my-twitter-labels --inpath ./target_dir/data/twitter --img-dir "" --n-return 1 --seeds ./target_dir/data/twitter_seeds
