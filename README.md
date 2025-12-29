# 职业紧张分析系统

基于机器学习的职业紧张风险预测与分析平台。

## 功能特点
- 🧠 **职业紧张预测**：基于14个职业生活特征预测紧张风险
- 📊 **数据分析**：交互式数据可视化分析
- 💡 **个性化建议**：针对不同风险等级的健康建议

## 部署到Streamlit Cloud

### 步骤
1. **准备GitHub仓库**
   ```bash
   git init
   git add .
   git commit -m "部署职业紧张分析系统"
   git branch -M main
   git remote add origin https://github.com/你的用户名/occupational-stress-app.git
   git push -u origin main
   ```

2. **部署到Streamlit Cloud**
   - 访问 [share.streamlit.io](https://share.streamlit.io/)
   - 点击 "New App"
   - 配置仓库为：`你的用户名/occupational-stress-app`
   - 设置主文件路径为：`app.py`
   - 点击 "Deploy"

3. **访问应用**
   - 部署完成后获得公共URL，如：`https://occupational-stress-app.streamlit.app/`

## 本地运行
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 文件说明
- `app.py` - 主应用文件
- `analysis.py` - 数据分析模块
- `model_compatible.py` - 机器学习模型
- `stress_data.xlsx` - 训练数据
- `requirements.txt` - Python依赖

## 技术栈
- Streamlit - Web应用框架
- Scikit-learn - 机器学习库
- Plotly - 数据可视化
- Pandas - 数据处理