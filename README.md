[TOC]
# 拼豆图纸生成工坊 (Perler Bead Pattern Generator)

这是一个基于 Flask 的 Web 应用，致力于将用户上传的图片智能转换为拼豆（Perler Beads）图纸。项目集成了图像降色算法、在线像素编辑器、以及完整的用户账户体系（含邮件验证与 VIP 会员机制）。

---

## 🚀 最新架构：数据隔离与分级存储 (Day 4 Update)

本项目已升级为 **生产级数据架构**，实现了严格的用户数据隔离和差异化服务逻辑：

* **🛡️ 全员独立目录**：每个用户（无论 VIP 还是普通用户）在服务器上都拥有独立的专属文件夹 `data/DrawingData/user_{id}/`，彻底解决多用户并发冲突问题。
* **⏳ 智能生命周期管理**：
    * **VIP 用户**：图纸永久保存，永不丢失，支持跨设备、跨时间段继续编辑。
    * **普通用户**：提供 **5小时** 临时存储服务。图纸在最后一次操作后保留 5 小时，超时后由后台任务自动擦除，有效节省服务器资源。
* **🧹 自动化后台清理**：基于 `APScheduler` 的后台守护进程，每小时自动扫描并清理过期的普通用户文件，无需人工干预。
* **⏪ 差异化撤销 (Undo)**：
    * **VIP**：支持 **10步** 历史回滚。
    * **普通用户**：支持 **5步** 历史回滚。

---

## 快速开始

### 1. 初始化数据库

首次运行项目前，必须初始化用户数据库：

bash
python init_user_db.py


*如果后续更新了代码导致数据库报错，可运行 `python fix_db.py` 进行无损修复。*

### 2. 配置邮件服务 (关键)

打开 `app.py`，找到 `邮件配置 (SMTP)` 部分，填入你的邮箱信息。
**注意**：`MAIL_PASSWORD` 必须填写邮箱的 **授权码** (Authorization Code)，而非登录密码。

python
# app.py 示例配置
app.config['MAIL_SERVER'] = 'smtp.qq.com'      # 例如 QQ 邮箱
app.config['MAIL_PORT'] = 465
app.config['MAIL_USE_SSL'] = True
app.config['MAIL_USERNAME'] = '你的邮箱@qq.com'
app.config['MAIL_PASSWORD'] = '你的授权码'      # <--- 注意这里！
app.config['MAIL_DEFAULT_SENDER'] = app.config['MAIL_USERNAME']


### 3. 安装依赖库

本项目依赖 Python 3.8+。为了支持后台自动清理任务，请确保安装了 `APScheduler`：

bash
pip install flask flask-login flask-mail pillow numpy pandas opencv-python scikit-image scikit-learn apscheduler


### 4. 启动服务器

bash
python app.py


### 5. 访问应用

打开浏览器访问：`http://localhost:5000`
默认会跳转至登录页面，请先注册账号。

---

## 核心功能

### 1. 用户中心 (login.html)

该页面是系统的门面，采用响应式卡片设计，集成了用户身份验证与权限管理的四大核心模块：

* **多维度登录**：支持通过 **用户名** 或 **注册邮箱** 进行登录。
* **安全校验**：后端将采用 `werkzeug.security` 对输入的密码进行哈希比对，确保数据安全。
* **会话管理**：登录成功后，系统通过 `Flask-Login` 保持用户会话，并以此为依据路由到用户专属的数据目录。
* **邮箱强绑定**：注册时必须填写有效邮箱，并输入系统发送的 **6位数字验证码** 进行核验。
* **VIP 权限激活**：用户输入激活码即可实时升级，解锁 **永久云存储** 和 **Pro 算法**。

### 2. 图片转图纸 (image_conversion)

* **多格式支持**：支持 JPG, PNG, GIF, BMP 等主流图片格式。
* **智能降色算法**：
    * **基础版**：基于 KMeans 聚类，快速提取主题色。
    * **Pro 进阶版**：结合 CIELAB 色彩空间与空间邻域算法，自动合并占比 <2% 的杂色，显著提升图纸纯净度。
* **自定义参数**：用户可自由设定目标尺寸（网格宽度）、单格像素大小、以及目标颜色数量。

### 3. 在线编辑器 (draw_page)

* **实时交互与防抖**：点击网格即可丝滑修改颜色，内置严格的鼠标事件防抖机制，保障大幅面图纸修改时的浏览器性能。
* **高级辅助拾色工具**：
    * **右键吸管**：支持在画布上右键点击任意像素格，瞬间将其提取为当前目标色。
    * **色号直搜**：通过输入准确的色号，快速在庞大的色卡库中定位。
    * **RGB 智能找色**：输入任意 RGB 数值，底层借助 Lab 空间欧氏距离算法，自动为您匹配最相近的已有拼豆颜色。
* **批量替换**：支持“同色一键替换”，快速调整整体色调。
* **裁剪工具**：自定义坐标或拖拽鼠标裁剪图纸区域。
* **撤销机制 (Undo)**：根据用户等级提供 5步 或 10步 的历史记录回滚。
* **自动保存**：
    * **VIP**：实时保存至云端 `user_{id}` 目录，永久存储。
    * **普通用户**：实时更新至临时文件，超时（5小时）自动销毁。
* **导出下载**：生成包含**网格线**、**坐标轴**和**色号图例统计**的高清图纸图片。

---

## 项目结构

### 文件树状结构

text
Image_Pixelation/
│  app.py                   # [核心] Flask 后端入口，集成了 APScheduler 后台清理任务
│  db_manager.py            # [核心] 负责路径路由、权限判断、撤销快照及数据库操作
│  image_utils.py           # [核心] 图像处理算法 (KMeans, CIELAB转换, 绘图渲染)
│  file_manager.py          # [核心] 文件清理工具 (包含 run_auto_clean 逻辑)
│  init_user_db.py          # 数据库初始化脚本
│  fix_db.py                # 数据库修复脚本
│  debug_mail.py            # 邮件发送调试脚本
│  readData.py              # 颜色库迁移工具
│  README.md                # 项目说明文档
│
├─data/                     # 数据存储目录
│  ├─Color/                 # 存放色卡数据库 (colors.db)
│  ├─DrawingData/           # [核心] 图纸存储根目录
│  │  ├─user_101/           # 用户 ID=101 的专属目录
│  │  ├─user_102/           # 用户 ID=102 的专属目录
│  │  └─public_temp/        # 未登录用户的临时目录
│  └─users.db               # 用户信息数据库
│
├─static/
│  ├─outputs/               # 转换结果图片缓存
│  └─uploads/               # 上传原图缓存
│
└─templates/                # 前端页面
       index.html           # 首页 / 仪表盘
       login.html           # 登录 / 注册
       image_conversion.html# 图片转换设置页
       draw_page.html       # 在线绘图编辑器
       colors.html          # 色卡管理页


### 核心脚本逻辑说明

* **app.py**:
    * 启动时初始化 `BackgroundScheduler`，每 60 分钟调用一次 `file_manager.run_auto_clean`。
    * 废除全局变量 `CURRENT_DRAWING_ID`，改用 `session['drawing_id']` 追踪用户当前操作的图纸。
* **db_manager.py**:
    * **`get_db_path(drawing_id)`**: 核心路由函数。检测 `current_user` 状态，将文件读写请求重定向到 `data/DrawingData/user_{id}/`。
    * **`save_snapshot`**: 在保存快照时，根据 `user_level` 动态决定保留 5 个还是 10 个备份。
* **file_manager.py**:
    * **`run_auto_clean`**: 定时任务入口。查询数据库获取所有 **普通用户** 的 ID，遍历其文件夹，删除修改时间超过 5 小时的 `.db` 文件。VIP 用户的文件夹会被自动跳过。

---

## 数据库结构

### 1. 用户数据库 (users.db)

**表名**: `users` (存储用户信息)
| 字段名 | 类型 | 说明 |
| --- | --- | --- |
| **id** | INTEGER | 主键，自动递增。 |
| **username** | TEXT | 用户名，全站唯一。 |
| **email** | TEXT | 用户邮箱。 |
| **password_hash** | TEXT | 密码哈希。 |
| **user_level** | TEXT | `common` 或 `vip`。决定了图纸的存储寿命和撤销步数。 |
| **vip_expire_at** | DATETIME | VIP 过期时间。 |

*(其他表结构保持不变)*

---

## 配置说明

### Windows 主机名报错修复

在 Windows 环境下，`socket` 模块可能因中文计算机名导致邮件发送失败。本项目已在 `app.py` 头部集成自动修复补丁：

python
import socket
try:
    socket.gethostname = lambda: "localhost"
except:
    pass


### 特别注意：scikit-learn 版本兼容性

在 `image_utils.py` 中，KMeans 聚类函数的初始化参数需注意版本兼容性。
**正确写法**: `n_init=10` (新版 sklearn 要求显式指定或使用 'auto')

---

## 常见问题排查

**Q1: 普通用户生成的图纸为什么第二天不见了？**
这是预期行为。普通用户的图纸仅在云端暂存 5 小时（从最后一次操作算起）。请在编辑完成后及时点击“导出”保存到本地。VIP 用户则无此限制。

**Q2: 为什么 `app.py` 启动时会有两条 "后台清理任务已启动" 的日志？**
Flask 在 Debug 模式下会启动一个主进程和一个重载进程。为了避免任务重复运行，代码中已通过 `os.environ.get('WERKZEUG_RUN_MAIN')` 进行了判断，确保定时任务只在主进程中运行一次。

**Q3: 升级 VIP 后，之前的 5 步撤销会变成 10 步吗？**
是的。升级后，下次操作时系统会自动放宽限制，允许您保存更多历史快照。反之，如果 VIP 过期降级，超出 5 步的旧快照将在下一次操作时被自动清理。