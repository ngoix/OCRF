#!/bin/sh

#cout << "###################################"
#"\nUsage:"
#"./ocrf -db database_name [-options]"
#"\n\nOptions:"

#"-db \t database name"
#"-dimension: \tnumber of dimensions"
#"-strat \t Folder hierarchy for learning process : strat_i/fold_j/learning_set.arff|test_set.arff"
#"-fold \t Folder hierarchy for learning process : strat_i/fold_j/learning_set.arff|test_set.arff"
#	cout <<"-path_learning"<<endl
#	cout <<"-path_test\n"<<endl
#	
#"-method \tlearning method for OCRF; "
#					"0 (default) for outlier generation in bagging+rsm set (projected bootstrap set);"
#					"1 for outlier generation before the induction of the forest (i.e. befor bagging in Forest-RI), no RSM is applied;"
#					"default with bagging+rsm\n"

#	cout
#			<< "-beta \t Factor controlling the number of outlier data generated according to the number of target data (e.g. 10 for 10x number of target data)\n"
#			<< endl;
#"-alpha \t Factor controlling the extension of the outlier domain used for outlier generation according to the volume of the hyperbox surrounding the target data\n"
#"-rejectOutOfBounds \t Data outside target bounds are considered as outlier data"
#"-optimize \t 0 for uniform distribution for outlier data; 1 for biased roulette-wheel distribution"

#"-krsm \t Number of dimensions for the Random Subscpace Method (RSM)"
#"-krfs \t Number of features randomly selected at each node during the induction of the tree"
# "###################################"

#standard forest params
krfs=-1;
nbTree=100;

#Outlier generation
alpha=1.2;
beta=10;
krsm=-1;

rejectOutOfBounds=0;
method=1;
optimize=1;

#database params
#dim=4;db="iris_versicolour";
dim=4;db="iris_setosa";
#dim=4;db="iris_virginica";
db_proc=$db"_"$dim"D";

#not used as hardcoded condition if -1
if [ $dim -lt 10 ];then val_krsm=$dim;fi

strat=0;
fold=0;

for strat in 0 1 2 3 4;
do
for fold in $(seq 0 9);
do

dir_root="../data/learning";
path_learning="$dir_root/$db_proc/strats/strat_$strat/fold_$fold/app.arff";
path_test="$dir_root/$db_proc/strats/strat_$strat/fold_$fold/test.arff";

echo "~~~~~~~~~~~~~START";date;echo "";

./ocrf \
-path_learning $path_learning -path_test $path_test -strat $strat -fold $fold \
-db $db_proc -dimension $dim \
-krsm $krsm -krfs $krfs -nbTree $nbTree \
-method $method -rejectOutOfBounds $rejectOutOfBounds -optimize $optimize -beta $beta -alpha $alpha;

echo "";date;
echo "~~~~~~~~~~~~~END";

done
done
