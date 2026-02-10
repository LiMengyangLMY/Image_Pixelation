运行app.py
网页登录http://localhost:5000

## 注意
聚类函数不要写为下面的形式(当前scikit-learn版本不支持)：**n_init='auto'** 改为**n_init=10**
```
kmeans = KMeans(
    n_clusters=int(target_cluster_count), 
    random_state=0, 
    n_init='auto'  # <--- 问题的核心
).fit(lab_data)
```