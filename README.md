# 拼豆图纸生成网页

运行app.py
网页登录http://localhost:5000

[TOC]

## 数据结构
### 核心数据库结构定义
#### 颜色库 (colors.db)
用于存储可用的标准颜色色版。
**表名**: colors
**字段**: num (TEXT/INT), R, G, B, lab_l, lab_a, lab_b

#### 图纸源数据库 ({drawing_id}.db)
用于存储单张图纸的像素排布和统计元数据。
**表 grid**: r (INTEGER), c (INTEGER), color_id (TEXT)
**表 metadata**:
key='dimensions': value 为 [rows, cols] 的 JSON 字符串。
key='color_code_count': value 为包含颜色 ID 及其详情的 JSON 字典。


## 注意
聚类函数不要写为下面的形式(当前scikit-learn版本不支持)：**n_init='auto'** 改为**n_init=10**
```
kmeans = KMeans(
    n_clusters=int(target_cluster_count), 
    random_state=0, 
    n_init='auto'  # <--- 问题的核心
).fit(lab_data)
```