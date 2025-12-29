import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

class DataAnalyzer:
    def __init__(self, data_path='stress_data.xlsx'):
        self.data_path = data_path
        self.df = None
        self.load_data()
        
    def load_data(self):
        """加载数据"""
        try:
            self.df = pd.read_excel(self.data_path)
            st.success("数据加载成功！")
        except Exception as e:
            st.error(f"数据加载失败: {e}")
            # 生成模拟数据用于演示
            self.create_sample_data()
    
    def create_sample_data(self):
        """创建模拟数据用于演示"""
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            '年龄': np.random.normal(35, 10, n_samples),
            '工龄': np.random.normal(8, 6, n_samples),
            '周均工作时间': np.random.normal(45, 8, n_samples),
            '生活满意度得分': np.random.normal(6, 2, n_samples),
            '疲劳程度分级': np.random.randint(0, 4, n_samples),
            '收入水平': np.random.choice(['低', '中', '高'], n_samples, p=[0.3, 0.5, 0.2]),
            '是否职业紧张': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
        }
        
        self.df = pd.DataFrame(data)
        st.info("使用示例数据进行分析")
    
    def get_data_overview(self):
        """数据概览统计"""
        st.subheader("📊 数据概览")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("总样本数", len(self.df))
        
        with col2:
            st.metric("职业紧张比例", f"{self.df['是否职业紧张'].mean():.1%}")
        
        with col3:
            avg_age = self.df['年龄'].mean()
            st.metric("平均年龄", f"{avg_age:.1f}岁")
        
        with col4:
            avg_hours = self.df['周均工作时间'].mean()
            st.metric("周均工作时间", f"{avg_hours:.1f}小时")
        
        # 显示前几行数据
        st.write("**数据预览:**")
        st.dataframe(self.df.head(), use_container_width=True)
    
    def create_feature_distribution_charts(self):
        """创建特征分布图表"""
        st.subheader("📈 特征分布分析")
        
        # 选择要分析的变量
        feature_options = ['年龄', '工龄', '周均工作时间', '生活满意度得分', '疲劳程度分级']
        selected_feature = st.selectbox("选择要分析的特征", feature_options)
        
        if selected_feature:
            col1, col2 = st.columns(2)
            
            with col1:
                # 直方图
                fig = px.histogram(self.df, x=selected_feature, 
                                 title=f'{selected_feature}分布',
                                 nbins=20,
                                 color_discrete_sequence=['#6f42c1'])
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # 箱线图
                fig = px.box(self.df, y=selected_feature, 
                           title=f'{selected_feature}箱线图',
                           color_discrete_sequence=['#6f42c1'])
                st.plotly_chart(fig, use_container_width=True)
    
    def create_correlation_analysis(self):
        """相关性分析"""
        st.subheader("🔗 相关性分析")
        
        # 选择数值型特征
        numeric_features = ['年龄', '工龄', '周均工作时间', '生活满意度得分', '疲劳程度分级']
        corr_df = self.df[numeric_features].corr()
        
        # 相关性热力图
        fig = px.imshow(corr_df,
                       title="特征相关性热力图",
                       color_continuous_scale='RdBu_r',
                       aspect="auto")
        st.plotly_chart(fig, use_container_width=True)
        
        # 散点图矩阵
        st.write("**散点图矩阵:**")
        fig = px.scatter_matrix(self.df[numeric_features],
                              title="特征散点图矩阵")
        st.plotly_chart(fig, use_container_width=True)
    
    def create_group_analysis(self):
        """分组分析"""
        st.subheader("👥 分组对比分析")
        
        group_by = st.selectbox("按变量分组", 
                              ['收入水平'])
        metric = st.selectbox("分析指标", 
                            ['年龄', '工龄', '周均工作时间', '生活满意度得分'])
        
        if group_by and metric:
            # 检查分组变量是否存在于数据中且有足够的分组数量
            if group_by not in self.df.columns or len(self.df[group_by].dropna().unique()) < 2:
                st.warning(f"⚠️ 无法进行分组分析：变量 '{group_by}' 不存在或分组数量不足")
                return
            
            try:
                # 分组统计
                grouped_stats = self.df.groupby(group_by)[metric].agg(['mean', 'std', 'count'])
                
                # 分组柱状图
                fig = px.bar(grouped_stats.reset_index(), 
                            x=group_by, y='mean',
                            title=f'{metric}按{group_by}分组对比',
                            color=group_by,
                            color_discrete_sequence=px.colors.qualitative.Set3)
                fig.update_traces(texttemplate='%{y:.2f}', textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
                
                # 分组箱线图
                fig = px.box(self.df, x=group_by, y=metric,
                            title=f'{metric}分布（按{group_by}分组）',
                            color=group_by)
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"分组分析过程中出现错误: {e}")
                st.info("系统将显示基础统计分析作为替代")
                # 显示基础的描述性统计
                st.write(f"**{metric} 描述性统计:**")
                st.write(self.df[metric].describe())
    
    def create_stress_risk_analysis(self):
        """职业紧张风险分析"""
        st.subheader("⚠️ 职业紧张风险因素分析")
        
        # 不同变量的职业紧张比例
        variables = ['收入水平']
        
        for var in variables:
            risk_rates = self.df.groupby(var)['是否职业紧张'].mean().reset_index()
            
            fig = px.bar(risk_rates, x=var, y='是否职业紧张',
                        title=f'{var}与职业紧张关系',
                        labels={'是否职业紧张': '职业紧张比例'},
                        color=var)
            fig.update_traces(texttemplate='%{y:.1%}', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
    
    def create_insights(self):
        """数据分析洞察"""
        st.subheader("💡 数据洞察")
        
        insights = []
        
        # 分析洞察点
        stress_rate = self.df['是否职业紧张'].mean()
        if stress_rate > 0.3:
            insights.append(f"⚠️ **高风险预警**: 总体职业紧张比例达到{stress_rate:.1%}，需要关注")
        
        # 工作时间与疲劳关系
        work_fatigue_corr = self.df['周均工作时间'].corr(self.df['疲劳程度分级'])
        if abs(work_fatigue_corr) > 0.3:
            direction = "正相关" if work_fatigue_corr > 0 else "负相关"
            insights.append(f"📊 **工作时间影响**: 周工作时间与疲劳程度存在{direction}关系 (r={work_fatigue_corr:.2f})")
        
        # 收入水平与紧张关系
        income_stress = self.df.groupby('收入水平')['是否职业紧张'].mean()
        if '低' in income_stress.index and '高' in income_stress.index:
            if income_stress['低'] > income_stress['高']:
                insights.append("💸 **收入影响**: 低收入群体的职业紧张风险更高")
        else:
            st.info("⚠️ 收入水平分析：数据中缺少完整收入水平分类")
        
        # 显示洞察
        for insight in insights:
            st.info(insight)
        
        # 关键统计指标
        st.write("**关键统计指标:**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_fatigue = self.df['疲劳程度分级'].mean()
            st.metric("平均疲劳程度", f"{avg_fatigue:.1f}/3")
        
        with col2:
            avg_satisfaction = self.df['生活满意度得分'].mean()
            st.metric("平均生活满意度", f"{avg_satisfaction:.1f}/10")
        
        with col3:
            avg_work_hours = self.df['周均工作时间'].mean()
            st.metric("平均周工作时间", f"{avg_work_hours:.1f}小时")

def create_analysis_page():
    """创建数据分析页面"""
    
    # 页面标题
    st.title("📊 数据分析")
    st.markdown("深入探索职业紧张相关数据的统计分析和可视化洞察")
    
    # 初始化数据分析器
    analyzer = DataAnalyzer()
    
    if analyzer.df is not None:
        # 创建选项卡
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📋 数据概览", "📈 特征分布", "🔗 相关性", "👥 分组分析", "💡 数据洞察"
        ])
        
        with tab1:
            analyzer.get_data_overview()
        
        with tab2:
            analyzer.create_feature_distribution_charts()
        
        with tab3:
            analyzer.create_correlation_analysis()
        
        with tab4:
            analyzer.create_group_analysis()
        
        with tab5:
            analyzer.create_stress_risk_analysis()
            analyzer.create_insights()
    
    else:
        st.error("无法加载数据文件")

# 如果是直接运行，创建独立的数据分析页面
if __name__ == "__main__":
    create_analysis_page()