# FactorHub 网络配置指南

## 🌐 网络访问配置

### 默认配置

FactorHub 现在默认运行在 `0.0.0.0:8501`，这意味着：
- 服务监听所有网络接口
- 可以通过本机IP地址或公网IP地址访问
- 适合部署在服务器上供多人使用

### 访问地址

#### 本地访问
```
http://localhost:8501
```

#### 局域网访问
```
http://本机IP地址:8501
例如：http://192.168.1.100:8501
```

#### 公网访问（云服务器）
```
http://服务器公网IP:8501
例如：http://123.45.67.89:8501
```

### 查看本机IP地址

#### Windows
```cmd
ipconfig
```

#### Linux/Mac
```bash
ifconfig
# 或者
ip addr show
```

## 🔒 安全注意事项

### 1. 防火墙配置

确保服务器防火墙允许8501端口的访问：

#### Ubuntu/Debian
```bash
sudo ufw allow 8501
```

#### CentOS/RHEL
```bash
sudo firewall-cmd --permanent --add-port=8501/tcp
sudo firewall-cmd --reload
```

#### Windows
在Windows防火墙中添加8501端口的入站规则

### 2. 安全访问控制

如果需要限制访问，可以修改启动脚本：

#### 仅本地访问
```bash
python scripts/start_app.py --host localhost
```

#### 特定IP访问
```bash
python scripts/start_app.py --host 192.168.1.100
```

### 3. 云服务器配置

#### 阿里云ECS
在安全组中添加8501端口的入站规则

#### 腾讯云CVM
在安全组中添加8501端口的入站规则

#### AWS EC2
在安全组中添加8501端口的入站规则

## 🔧 自定义网络配置

### 修改默认端口

如果8501端口被占用，可以指定其他端口：

```bash
python scripts/start_app.py --port 8502
```

### 修改监听地址

如果只想本地访问：

```bash
python scripts/start_app.py --host localhost
```

### 组合配置

```bash
python scripts/start_app.py --host 0.0.0.0 --port 8502
```

## 🐳 Docker部署（可选）

如果使用Docker部署，确保端口映射正确：

```bash
docker run -d \
  -p 8501:8501 \
  --name factorhub \
  factorhub:latest
```

## 🌍 反向代理配置

如果需要通过域名访问，可以配置反向代理：

### Nginx配置示例
```nginx
server {
    listen 80;
    server_name factorhub.yourdomain.com;

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Apache配置示例
```apache
<VirtualHost *:80>
    ServerName factorhub.yourdomain.com
    ProxyPreserveHost On
    ProxyRequests Off
    ProxyPass / http://127.0.0.1:8501/
    ProxyPassReverse / http://127.0.0.1:8501/
</VirtualHost>
```

## 🔍 网络故障排除

### 检查端口是否监听
```bash
netstat -tlnp | grep 8501
# 或者
ss -tlnp | grep 8501
```

### 检查服务是否运行
```bash
ps aux | grep streamlit
```

### 测试本地访问
```bash
curl http://localhost:8501
```

### 测试远程访问
```bash
curl http://服务器IP:8501
```

### 常见问题

**Q: 无法从外部访问**
A: 检查防火墙设置，确保8501端口已开放

**Q: 访问速度慢**
A: 检查网络带宽，考虑使用CDN

**Q: 多用户并发问题**
A: Streamlit是单用户设计，多用户需要部署多个实例

## 📝 最佳实践

1. **生产环境**: 建议使用反向代理和HTTPS
2. **安全性**: 配置访问限制和监控
3. **性能**: 根据用户量调整服务器配置
4. **备份**: 定期备份数据和配置

---

**注意**: 在公网部署时，请确保遵守相关法律法规，并采取适当的安全措施。