nohup python train_battle.py --algo mfac --map_size 50 &
nohup python train_battle.py --algo ac --map_size 50 &
nohup python train_battle.py --algo mfq --map_size 50 &
nohup python train_battle.py --algo il --map_size 50 &





ac 1998
il 1997
mfac 1998
mfq 1999


nohup python battle.py --algo ac --oppo mfac --idx 1998 1998 --n_round 100 > ac_vs_mfac_100.file 2>&1 &
nohup python battle.py --algo ac --oppo mfac --idx 1998 1998 --n_round 200 > ac_vs_mfac_200.file 2>&1 &
nohup python battle.py --algo ac --oppo mfac --idx 1998 1998 --n_round 500 > ac_vs_mfac_500.file 2>&1 &


nohup python battle.py --algo mfq --oppo il --idx 1999 1997 --n_round 100 > mfq_vs_il_100.file 2>&1 &
nohup python battle.py --algo mfq --oppo il --idx 1999 1997 --n_round 200 > mfq_vs_il_200.file 2>&1 &
nohup python battle.py --algo mfq --oppo il --idx 1999 1997 --n_round 500 > mfq_vs_il_500.file 2>&1 &
nohup python battle.py --algo mfq --oppo il --idx 1999 1997 --n_round 1000 > mfq_vs_il_1000.file 2>&1 &

nohup python battle.py --algo mfq --oppo mfac --idx 1999 1998 --n_round 100 > mfq_vs_mfac_100.file 2>&1 &
nohup python battle.py --algo mfq --oppo mfac --idx 1999 1998 --n_round 200 > mfq_vs_mfac_200.file 2>&1 &
nohup python battle.py --algo mfq --oppo mfac --idx 1999 1998 --n_round 500 > mfq_vs_mfac_500.file 2>&1 &
nohup python battle.py --algo mfq --oppo mfac --idx 1999 1998 --n_round 1000 > mfq_vs_mfac_1000.file 2>&1 &


nohup python battle.py --algo mfq --oppo ac --idx 1999 1998 --n_round 100 > mfq_vs_ac_100.file 2>&1 &
nohup python battle.py --algo mfq --oppo ac --idx 1999 1998 --n_round 200 > mfq_vs_ac_200.file 2>&1 &
nohup python battle.py --algo mfq --oppo ac --idx 1999 1998 --n_round 500 > mfq_vs_ac_500.file 2>&1 &
nohup python battle.py --algo mfq --oppo ac --idx 1999 1998 --n_round 1000 > mfq_vs_ac_1000.file 2>&1 &


nohup python battle.py --algo mfq --oppo mfq --idx 1999 1999 --n_round 100 > mfq_vs_mfq_100.file 2>&1 &
nohup python battle.py --algo mfq --oppo mfq --idx 1999 1999 --n_round 200 > mfq_vs_mfq_200.file 2>&1 &
nohup python battle.py --algo mfq --oppo mfq --idx 1999 1999 --n_round 500 > mfq_vs_mfq_500.file 2>&1 &
nohup python battle.py --algo mfq --oppo mfq --idx 1999 1999 --n_round 1000 > mfq_vs_mfq_1000.file 2>&1 &


#------------------------------mfrl------------------------------------

nohup python battle.py --algo ac --oppo mfac --idx 1998 1998 --n_round 100 --map_size 50 > map_50_ac_vs_mfac_100.file 2>&1 &
nohup python battle.py --algo ac --oppo mfac --idx 1998 1998 --n_round 200 --map_size 50 > map_50_ac_vs_mfac_200.file 2>&1 &
nohup python battle.py --algo ac --oppo mfac --idx 1998 1998 --n_round 500 --map_size 50 > map_50_ac_vs_mfac_500.file 2>&1 &


nohup python battle.py --algo mfq --oppo il --idx 1999 1997 --n_round 100 --map_size 50 > map_50_mfq_vs_il_100.file 2>&1 &
nohup python battle.py --algo mfq --oppo il --idx 1999 1997 --n_round 200 --map_size 50 > map_50_mfq_vs_il_200.file 2>&1 &
nohup python battle.py --algo mfq --oppo il --idx 1999 1997 --n_round 500 --map_size 50 > map_50_mfq_vs_il_500.file 2>&1 &
nohup python battle.py --algo mfq --oppo il --idx 1999 1997 --n_round 1000 --map_size 50 > map_50_mfq_vs_il_1000.file 2>&1 &

nohup python battle.py --algo mfq --oppo mfac --idx 1999 1998 --n_round 100 --map_size 50 > map_50_mfq_vs_mfac_100.file 2>&1 &
nohup python battle.py --algo mfq --oppo mfac --idx 1999 1998 --n_round 200 --map_size 50 > map_50_mfq_vs_mfac_200.file 2>&1 &
nohup python battle.py --algo mfq --oppo mfac --idx 1999 1998 --n_round 500 --map_size 50 > map_50_mfq_vs_mfac_500.file 2>&1 &
nohup python battle.py --algo mfq --oppo mfac --idx 1999 1998 --n_round 1000 --map_size 50 > map_50_mfq_vs_mfac_1000.file 2>&1 &


nohup python battle.py --algo mfq --oppo ac --idx 1999 1998 --n_round 100 --map_size 50 > map_50_mfq_vs_ac_100.file 2>&1 &
nohup python battle.py --algo mfq --oppo ac --idx 1999 1998 --n_round 200 --map_size 50 > map_50_mfq_vs_ac_200.file 2>&1 &
nohup python battle.py --algo mfq --oppo ac --idx 1999 1998 --n_round 500 --map_size 50 > map_50_mfq_vs_ac_500.file 2>&1 &
nohup python battle.py --algo mfq --oppo ac --idx 1999 1998 --n_round 1000 --map_size 50 > map_50_mfq_vs_ac_1000.file 2>&1 &


nohup python battle.py --algo mfq --oppo mfq --idx 1999 1999 --n_round 100 --map_size 50 > map_50_mfq_vs_mfq_100.file 2>&1 &
nohup python battle.py --algo mfq --oppo mfq --idx 1999 1999 --n_round 200 --map_size 50 > map_50_mfq_vs_mfq_200.file 2>&1 &
nohup python battle.py --algo mfq --oppo mfq --idx 1999 1999 --n_round 500 --map_size 50 > map_50_mfq_vs_mfq_500.file 2>&1 &
nohup python battle.py --algo mfq --oppo mfq --idx 1999 1999 --n_round 1000 --map_size 50 > map_50_mfq_vs_mfq_1000.file 2>&1 &
