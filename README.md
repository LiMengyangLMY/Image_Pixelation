# 拼豆图纸生成网页

运行app.py
网页登录http://localhost:5000

[TOC]
# 文件结构
## 文件树状结构
│  app.py
│  color_data.csv
│  db_manager.py
│  file_manager.py
│  image_utils.py
│  readData.py
│  README.md
│
├─data
│  ├─Color
│  │      colors.db
│  │
│  ├─DrawingData
│  │
│  └─users.db 
│
├─font
│      arial.ttf
│      arialbd.ttf
│      arialbi.ttf
│      ariali.ttf
│      ARIALN.TTF
│      ARIALNB.TTF
│      ARIALNBI.TTF
│      ARIALNI.TTF
│      ariblk.ttf
│      msyh.ttc
│      msyhbd.ttc
│      msyhl.ttc
│
├─static
│  ├─outputs
│  │
│  └─uploads
│
├─templates
│      colors.html
│      draw_page.html
│      image_conversion.html
│      index.html
│
└─__pycache__
        db_manager.cpython-38.pyc
        file_manager.cpython-38.pyc
        image_utils.cpython-38.pyc
## 文件说明
### 核心脚本
* **app.py**: Flask 后端主程序，负责路由分发和各类 API 接口。
* **db_manager.py**: 负责图纸数据库的创建、裁剪、单格/批量颜色更新，以及最多 10 步的滚动快照撤销 (`undo`) 逻辑。
* **file_manager.py**: 文件管理器，用于限制 `static/uploads` 和 `static/outputs` 文件夹内的文件数量（默认最多保留20个），按照修改时间自动清理旧文件。
* **image_utils.py**: 核心图像处理工具。包含图片转像素网格、KMeans 聚类降色、基于 CIELAB 空间的 Pro 版杂色合并算法，以及图纸的可视化渲染（带网格线和图例）。
* **readData.py**: 数据库迁移脚本。用于读取颜色的 RGB 值，将其转换为 Lab 色彩空间值，并覆盖更新到 `colors.db` 中。
#### data 文件夹
用于存放 SQLite 数据库文件。
* **Color**: 存放颜色数据库。`colors.db` 是默认数据库，注意 `colors.db` 数据库不能删。
* **DrawingData**: 用于存放生成的单张图纸源数据库文件及其撤销快照。
#### static 文件夹
由于用于缓存输入输出图片。
* **outputs**: 存放处理后输出的图纸图片。
* **uploads**: 存放用户上传的原始图片。

# 数据结构
## 核心数据库结构定义

### 用户数据库 (users.db)
根据 Day 1 的最终讨论结果，这里为你提供 **Markdown (md)** 格式的用户数据库结构。你可以直接将其覆盖到你的 `README.md` 文件中。

---

## 核心数据库结构定义
### 用户数据库 (users.db)
用于存储用户信息、登录凭证及等级权限。
**表名**: `users`
| 字段名 | 类型 | 说明 |
| --- | --- | --- |
| **id** | INTEGER | 主键，自动递增。 |
| **username** | TEXT | 用户名，全站唯一 (UNIQUE)。 |
| **email** | TEXT | 用户邮箱，用于找回密码及身份标识 (UNIQUE)。 |
| **password_hash** | TEXT | 经过哈希加密后的密码存储。 |
| **user_level** | TEXT | 用户等级：`common` (普通) 或 `vip` (VIP)。 |
| **security_question** | TEXT | 安全问题（可选，用于无邮箱环境找回密码）。 |
| **security_answer** | TEXT | 安全问题答案（哈希存储）。 |
| **created_at** | DATETIME | 账户注册时间，默认为 `CURRENT_TIMESTAMP`。 |

### VIP 卡密数据库 (users.db)
用于存储预生成的 VIP 激活码，支持“售卖-激活”模式。
**表名**: `vip_codes`
| 字段名 | 类型 | 说明 |
| --- | --- | --- |
| **code** | TEXT | 16位随机卡密 (PRIMARY KEY)。 |
| **is_used** | INTEGER | 使用状态：0 为未使用，1 为已使用。 |
| **used_by** | TEXT | 使用该卡密的用户用户名。 |
| **valid_days** | INTEGER | 该卡密提供的 VIP 天数（如 31 天或 365 天）。 |

### 邮箱验证表 (users.db)
用于临时存放注册或找回密码时的验证码。
**表名**: `email_verify`
| 字段名 | 类型 | 说明 |
| --- | --- | --- |
| **email** | TEXT | 目标邮箱。 |
| **code** | TEXT | 6位数字验证码。 |
| **expire_at** | DATETIME | 过期时间（通常为生成后的 5-10 分钟）。 |

### 颜色库 (colors.db)
用于存储可用的标准颜色色版。
**表名**: colors
**字段**: num (TEXT/INT), R, G, B, lab_l, lab_a, lab_b
### 图纸源数据库 ({drawing_id}.db)
用于存储单张图纸的像素排布和统计元数据。
**表 grid**: r (INTEGER), c (INTEGER), color_id (TEXT)
**表 metadata**:
key='dimensions': value 为 [rows, cols] 的 JSON 字符串。
key='color_code_count': value 为包含颜色 ID 及其详情的 JSON 字典。

# 功能
## 用户中心 (login.html)
该页面是系统的门面，采用响应式卡片设计，集成了用户身份验证与权限管理的四大核心模块：
### 1. 登录与身份持久化
* **多维度登录**：支持通过 **用户名** 或 **注册邮箱** 进行登录。
* **安全校验**：后端将采用 `werkzeug.security` 对输入的密码进行哈希比对，确保数据安全。
* **会话管理**：登录成功后，系统通过 `Flask-Login` 保持用户会话，作为访问图纸编辑页面的通行证。
### 2. 注册与邮箱验证
* **邮箱强绑定**：注册时必须填写有效邮箱，并输入系统发送的 **6位数字验证码** 进行核验。
* **双重密码确认**：前端强制要求输入两次密码并进行一致性校验，防止因输入失误导致无法登录。
* **防止重名**：后端自动检测用户名及邮箱的唯一性，避免账号冲突。
### 3. VIP 权限激活（卡密系统）
* **自主激活**：用户输入在第三方平台购买的 **16位唯一激活码** 即可实时升级。
* **等级划分**：激活后用户等级由 `common` 提升至 `vip`，解锁 **Pro 进阶降色算法** 等高级功能。
* **防重复使用**：数据库实时记录卡密使用状态及使用者信息，确保卡密安全。
### 4. 密码管理与找回
* **邮箱自助找回**：当用户忘记密码时，可通过注册邮箱接收验证码，并在输入两次新密码确认后完成重置。
* **安全重置逻辑**：找回密码流程同样包含“二次密码确认”，确保用户修改后的密码准确无误。
* **修改密码**：支持登录用户在个人设置中直接更新登录凭证。
## 首页 (index 页面)
1. **数据库概览**：查看系统 `Color` 目录下现有的颜色数据库列表。
2. **状态显示**：显示当前正在使用的主颜色数据库名称。
3. **导航系统**：作为系统的入口，可跳转至图片转换图纸、编辑图纸和颜色数据库管理页面。
## 图片转图纸 (image_conversion 页面)
1. **图片上传与格式校验**：支持上传 `png`, `jpg`, `jpeg`, `gif`, `bmp` 格式的图片，最大限制 16MB。
2. **自定义尺寸参数**：可自由设定目标图纸的网格宽度 (target_width) 和单格像素大小 (pixel_size)。
3. **基础转换**：直接计算原图像素点的 LAB 值，并匹配颜色库中最相近的色号生成拼豆图纸。
4. **普通降色功能 (reduce_colors)**：利用 KMeans 聚类算法，将图片包含的颜色种类压缩到用户指定的数量 (color_count)。
5. **Pro 进阶降色 (reduce_colors_Pro)**：在 KMeans 聚类的基础上，通过 CIELAB 色彩空间计算颜色距离，自动将占比低于 2% 的“稀有杂色”合并到邻近或全局的最优主流颜色中，大幅提升图纸的纯净度和拼豆实操的可行性。
6. **结果展示与缓存控制**：直观展示原图和生成图纸的对比；每次转换后自动调用清理机制，防止图片堆积占用磁盘空间。
### 图纸编辑 (draw_page 页面)
1. **图纸数据载入与管理**
   * **自动加载**：从转image_conversion页面进入时，会自动加载刚才生成的图纸源数据。
   * **新建空白图纸**：支持输入宽、高尺寸，自动使用 "H2"（默认白色）填满生成新图纸。
   * **保存修改**：通过 API 随时将网页端修改的数据保存为后端的 `.db` 源文件。
2. **图纸导出功能**
   * **下载可视化图纸**：将当前的 `.db` 像素排布数据渲染生成一张包含**坐标轴、网格线和色号/数量图例统计**的高质量 `.png` 图片并下载。
   * **下载图纸源数据**：*（备注：目前后端有加载逻辑，下载源 `.db` 文件等功能可配合前端实现）*
3. **交互修改功能**
   * **快照与撤销机制**：在执行裁切、单格修改、批量修改前，系统会自动创建 `.snap_x` 滚动快照（最多保留 10 步），支持随时一键撤回操作 (`Undo`)。
   * **单格更换**：选定目标颜色后，更新指定行/列坐标的单个网格颜色，并自动更新颜色总数统计。
   * **同色批量替换**：锁定某一旧色号，一键将其在全图中全部替换为指定的新色号，并完成计数的合并/增减。
   * **图纸裁剪**：根据传入的起始和结束坐标框选裁剪保留图纸区域，并重新统计新区域内的颜色数据。
   * *(以下为待对接或完善前端交互的功能)*：上传图纸源数据功能、右键快速取色功能。

### 颜色库管理 (colors 页面)
1. **颜色库展示**：用于提供前端颜色库管理的入口。
2. **安全保护**：禁止随意删除原始颜色数据库 `colors.db`，确保系统基础功能运行不报错。

## 注意
聚类函数不要写为下面的形式(当前scikit-learn版本不支持)：**n_init='auto'** 改为**n_init=10**
```
kmeans = KMeans(
    n_clusters=int(target_cluster_count), 
    random_state=0, 
    n_init='auto'  # <--- 问题的核心
).fit(lab_data)
```




