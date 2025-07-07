# Using `manhattan` metric
## 0.2 fraction of training set
option 1:
Tuning on max min_sample of D+1
Best parameters: {'min_samples': 10, 'eps': 20.82750400620556}
Best silhouette score: 0.7254556206069943
9 clusters with 82 noise points

option 2:
Tuning on max min_xample of D*2
Best parameters: {'min_samples': 25, 'eps': 19.86083512762457}
Best silhouette score: 0.690743768614763
5 clusters with 154 noise points

## 0.6 fraction of training set
option 1:
Tuning on max min_sample of D+1
Best parameters: {'min_samples': 41, 'eps': 24.21643785367032}
Best silhouette score: 0.7677986214616699
4 clusters with 289 noise points

option 2:
Tuning on max min_xample of D*2
Best parameters: {'min_samples': 113, 'eps': 20.242685021887063}
Best silhouette score: 0.6903768676224799
2 clusters with 725 noise points

## full training set
option 1:
Tuning on max min_sample of D+1
Best parameters: {'min_samples': 101, 'eps': 24.504985754262748}
Best silhouette score: 0.767512271424871
4 clusters with 427 noise points

option 2:
Tuning on max min_xample of D*2
Best parameters: {'min_samples': 19, 'eps': 24.797764936664798}
Best silhouette score: 0.7652262578920606
9 clusters with 167 noise points

# Using `manhattan` metric with PCA
## 0.2 fraction of training set
option 1:
Tuning on max min_sample of D+1
Best parameters: {'min_samples': 87, 'eps': 2.012101485894015}
Best silhouette score: 0.8040126850100613
2 clusters with 6531 noise points

option 2:
Tuning on max min_xample of D*2
Best parameters: {'min_samples': 190, 'eps': 2.251479810909843}
Best silhouette score: 0.8058637316328787
2 clusters with 6652 noise points

## 0.6 fraction of training set
option 1:
Tuning on max min_sample of D+1
Best parameters: {'min_samples': 122, 'eps': 9.338259312723947}
Best silhouette score: 0.6180881124732168
15 clusters with 3630 noise points

option 2:
Tuning on max min_xample of D*2
Best parameters: {'min_samples': 207, 'eps': 10.513896245924082}
Best silhouette score: 0.6419710699275109
10 clusters with 4606 noise points

## full training set
option 1:
Tuning on max min_sample of D+1
Best parameters: {'min_samples': 90, 'eps': 12.89571129990176}
Best silhouette score: 0.6292364968407647
21 clusters with 2399 noise points

option 2:
Tuning on max min_xample of D*2
Best parameters: {'min_samples': 220, 'eps': 9.818441335808252}
Best silhouette score: 0.6409745618863946
14 clusters with 5686 noise points
