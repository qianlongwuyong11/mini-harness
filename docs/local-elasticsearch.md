# 本机 Elasticsearch（RAG 第三步）

固定 Windows ZIP 版本 9.5.3，使用包内 JDK；不需要 Docker、Kibana。
配置与认证材料保存在 Git 忽略目录 `.mini-harness/elasticsearch/`。
安装脚本下载官方包并校验官方 SHA512，不覆盖已经配置的实例。

在 mini-harness 根目录运行：

```powershell
.\.venv\Scripts\python.exe scripts/local_es.py install
.\start-es.cmd
```

启动后另开终端检查：

```powershell
.\.venv\Scripts\python.exe scripts/local_es.py health
```

- HTTPS 地址：`https://127.0.0.1:9200`，CA 校验开启，必须认证。
- HTTP/transport 仅绑定 IPv4 loopback，端口 9200/9300。
- 初始 JVM 堆 2 GB；不是 ES 总内存上限。机器学习模块关闭。
- data/logs 独立于安装版本目录；停进程不会删除索引。
- connection.json 包含随机本地开发管理员密码，请勿提交或分享。
- Python 客户端允许 MINI_HARNESS_ES_PASSWORD 环境变量覆盖密码。
- 启动器先验证已有实例；复用时不会停止它。自己启动时持有进程对象，Ctrl+C 关闭自己启动的进程树。
- Windows 关闭目前使用 taskkill /T /F（强制停止），不是优雅停机；通过真实重启测试验证已确认写入的恢复。后续 UI 统一启动器接入时可改进关闭协议。
- 不提供基于旧 PID 文件的 stop 命令，避免误杀复用 PID 的进程。
- 当前用项目已有 httpx 调用 ES REST API，不增加官方 Python SDK，未接入论文索引或自动启动 UI。

重启持久性验收（需要 9200 空闲）：

```powershell
.\.venv\Scripts\python.exe scripts/local_es.py smoke
```

只创建随机命名的测试索引；写入、强制停止、重启、验证原记录，再删除该测试索引并关闭自身实例。
数据目录不可直接从资源管理器改写；以后索引重建和备份通过 ES API 处理。

官方参考：https://www.elastic.co/docs/deploy-manage/deploy/self-managed/install-elasticsearch-with-zip-on-windows
