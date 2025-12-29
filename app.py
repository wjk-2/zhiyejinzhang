import streamlit as st
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu
from model_compatible import CompatibleOccupationalStressModel as OccupationalStressModel, process_user_input
MODEL_TYPE = "compatible"

# 页面配置
st.set_page_config(
    page_title="职业紧张程度分析系统",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义样式
def local_css(file_name):
    try:
        with open(file_name, encoding='utf-8') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("自定义样式文件未找到，使用默认样式")
    except UnicodeDecodeError:
        # 如果UTF-8编码失败，尝试GBK编码
        try:
            with open(file_name, encoding='gbk') as f:
                st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"无法读取样式文件: {e}")

# ---------------------- 关键修改1：封装页面跳转函数 ----------------------
def switch_to_page(page_name):
    """统一页面跳转逻辑，确保状态同步+即时生效"""
    # 更新会话状态
    st.session_state.page = page_name
    # 更新URL参数
    st.query_params = {"page": [page_name]}
    # 强制脚本重新运行，立即刷新页面
    st.rerun()

# 初始化session state
if 'page' not in st.session_state:
    st.session_state.page = "首页"

# 处理页面跳转参数（优先读取URL参数，同步到session_state）
query_params = st.query_params
if 'page' in query_params and query_params['page'][0] in ["首页", "职业紧张预测", "数据分析", "关于系统"]:
    st.session_state.page = query_params['page'][0]

# 侧边栏导航
with st.sidebar:
    # ---------------------- 关键修改2：简化默认索引计算 ----------------------
    page_to_index = {
        "首页": 0,
        "职业紧张预测": 1,
        "数据分析": 2,
        "关于系统": 3
    }
    default_index = page_to_index.get(st.session_state.page, 0)
    
    selected = option_menu(
        menu_title="导航菜单",
        options=["首页", "职业紧张预测", "数据分析", "关于系统"],
        icons=["house", "clipboard-pulse", "bar-chart", "info-circle"],
        menu_icon="cast",
        default_index=default_index,  # 简化后的默认索引
        key="nav_menu",
        styles={
            "container": {"padding": "5px", "background-color": "#6f42c1"},
            "icon": {"color": "white", "font-size": "18px"}, 
            "nav-link": {"color": "white", "font-size": "16px", "text-align": "left", "margin": "0px"},
            "nav-link-selected": {"background-color": "#5a32a3"},
        }
    )

# ---------------------- 关键修改3：同步导航选中项与会话状态 ----------------------
if selected != st.session_state.page:
    st.session_state.page = selected
    st.query_params = {"page": selected}
    st.rerun()

# 页面路由
if selected == "首页":
    st.title("职业紧张程度分析系统")
    st.markdown("### 基于机器学习的职业紧张风险预测与分析平台")
    
    # 功能模块展示
    col1, col2, col3 = st.columns(3)
    
    with col1:
        with st.container():
            st.markdown("""
            <div style='background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);'>
                <h3>📊 数据分析</h3>
                <p>通过交互式图表和统计分析，深入了解职业紧张的特征分布和影响因素。</p>
            </div>
            """, unsafe_allow_html=True)
            # ---------------------- 关键修改4：按钮调用统一跳转函数 ----------------------
            if st.button("查看分析", key="analysis_btn"):
                st.session_state.page = "数据分析"
                st.query_params = {"page": "数据分析"}
                st.rerun()
    
    with col2:
        with st.container():
            st.markdown("""
            <div style='background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);'>
                <h3>🧠 职业紧张预测</h3>
                <p>基于机器学习模型，输入14个职业和生活特征，预测职业紧张风险等级。</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("开始预测", key="predict_btn"):
                st.session_state.page = "职业紧张预测"
                st.query_params = {"page": "职业紧张预测"}
                st.rerun()
    
    with col3:
        with st.container():
            st.markdown("""
            <div style='background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);'>
                <h3>ℹ️ 关于系统</h3>
                <p>了解系统设计理念、技术架构和使用方法，获取更多帮助信息。</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("了解更多", key="about_btn"):
                st.session_state.page = "关于系统"
                st.query_params = {"page": "关于系统"}
                st.rerun()

# 职业紧张预测页面
elif selected == "职业紧张预测":
    # 返回主页按钮
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title("职业紧张风险预测")
    with col2:
        if st.button("🏠 返回主页", key="back_from_predict"):
            st.session_state.page = "首页"
            st.query_params = {"page": "首页"}
            st.rerun()
    st.markdown("请输入您的职业和生活特征信息，系统将预测您的职业紧张风险等级")
    
    with st.form("职业紧张预测表单"):
        st.header("基本信息输入")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("年龄", 18, 65, 30)
            work_years = st.slider("工龄", 0, 50, 5)
            position_years = st.slider("本岗位工龄", 0, 50, 2)
            income = st.selectbox("收入水平", ["低", "中", "高"])
            weekly_hours = st.slider("周均工作时间(小时)", 20, 80, 40)
            alcohol = st.slider("饮酒量(标准杯/周)", 0, 30, 0)
            mid_exercise = st.slider("中强度锻炼(小时/周)", 0, 20, 2)
        
        with col2:
            education = st.selectbox("教育程度", ["高中及以下", "大专", "本科", "硕士及以上"])
            shift_work = st.radio("是否轮班", ["是", "否"])
            night_shift = st.radio("是否夜班", ["是", "否"])
            sleep_disorder = st.radio("是否睡眠障碍", ["是", "否"])
            high_exercise = st.slider("高强度锻炼(小时/周)", 0, 20, 1)
            life_satisfaction = st.slider("生活满意度得分(1-10分)", 1, 10, 6)
            fatigue_level = st.select_slider("疲劳程度分级", options=["无", "轻度", "中度", "重度"])
        
        submitted = st.form_submit_button("预测职业紧张风险", type="primary")

    if submitted:
        try:
            # 收集用户输入
            user_input = {
                'age': age,
                'work_years': work_years,
                'position_years': position_years,
                'income': income,
                'weekly_hours': weekly_hours,
                'alcohol': alcohol,
                'low_exercise': mid_exercise,  # 修改字段名匹配模型期望
                'life_satisfaction': life_satisfaction,
                'fatigue_level': 0 if fatigue_level == "无" else (1 if fatigue_level == "轻度" else (2 if fatigue_level == "中度" else 3)),
                'marital_status': '未婚',  # 默认值，实际应用中可以添加输入
                'smoking': 0,  # 默认值
                'education': education,
                'daily_overtime': 0  # 默认值
            }
            
            # 数据处理
            model_input = process_user_input(user_input)
            
            # 使用模型进行预测
            model = OccupationalStressModel()
            model.load_model()
            result = model.predict_stress(model_input)
            
            # 显示模型类型信息
            if MODEL_TYPE == "compatible":
                st.success("✅ 预测完成（使用兼容优化模型）")
            elif MODEL_TYPE == "optimized":
                st.success("✅ 预测完成（使用优化模型）")
            else:
                st.success("预测完成！")
            
            # 显示预测结果
            st.subheader("预测结果")
            
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.metric("职业紧张风险等级", result['risk_level'])
                st.metric("预测概率", f"{result['probability']:.1%}")
            
            with col2:
                # 显示特征重要性
                try:
                    feature_importance = model.get_feature_importance()
                    st.write("**特征重要性排名:**")
                    for i, (feature, importance) in enumerate(feature_importance[:5], 1):
                        st.write(f"{i}. {feature}: {importance:.3f}")
                except:
                    st.info("🔧 特征重要性分析暂时不可用")
            
            st.subheader("个性化建议")
            if result['risk_level'] == "高风险":
                st.error("🔴 **高风险等级**")
                st.warning("""
                **立即行动建议:**
                - 寻求专业心理咨询服务
                - 调整工作强度和时间安排
                - 加强体育锻炼和健康管理
                - 改善睡眠质量和饮食习惯
                - 考虑与上级沟通工作压力问题
                """)
            elif result['risk_level'] == "中风险":
                st.warning("🟡 **中风险等级**")
                st.info("""
                **预防性建议:**
                - 定期进行压力管理训练
                - 保持工作与生活的平衡
                - 培养积极的应对策略
                - 关注身体信号，及时调整
                - 加强社交支持和家庭沟通
                """)
            else:
                st.success("🟢 **低风险等级**")
                st.info("""
                **维持良好状态建议:**
                - 继续保持健康的生活方式
                - 定期进行自我压力评估
                - 培养积极的心态和情绪管理
                - 平衡工作和休息时间
                - 持续关注身心健康指标
                """)
                
        except Exception as e:
            st.error(f"预测过程中出现错误: {str(e)}")
            st.info("系统将使用模拟数据进行预测...")
            # 备用模拟预测
            risk_level = np.random.choice(["低风险", "中风险", "高风险"], p=[0.4, 0.4, 0.2])
            st.metric("职业紧张风险等级", risk_level)

# 数据分析页面
elif selected == "数据分析":
    # 返回主页按钮
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title("数据分析")
    with col2:
        if st.button("🏠 返回主页", key="back_from_analysis"):
            st.session_state.page = "首页"
            st.query_params = {"page": "首页"}
            st.rerun()
    
    # 导入数据分析模块
    try:
        from analysis import create_analysis_page
        create_analysis_page()
    except Exception as e:
        st.error(f"数据分析模块加载失败: {e}")
        st.info("正在使用简化版数据分析功能...")
        
        # 简化版数据分析作为后备方案
        try:
            df = pd.read_excel('stress_data.xlsx')
            st.success("数据加载成功！")
            
            st.subheader("📊 基础统计分析")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("总样本数", len(df))
            
            with col2:
                stress_rate = df['是否职业紧张'].mean()
                st.metric("职业紧张比例", f"{stress_rate:.1%}")
            
            with col3:
                avg_hours = df['周均工作时间'].mean()
                st.metric("周均工作时间", f"{avg_hours:.1f}小时")
            
        except Exception as data_error:
            st.error(f"数据加载错误: {data_error}")

# 关于系统页面
elif selected == "关于系统":
    # 返回主页按钮
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title("关于职业紧张分析系统")
    with col2:
        if st.button("🏠 返回主页", key="back_from_about"):
            st.session_state.page = "首页"
            st.query_params = {"page": "首页"}
            st.rerun()
    st.markdown("""
    ### 系统简介
    本系统基于机器学习技术，通过对14个职业和生活特征的分析，预测个体的职业紧张风险等级。
    
    ### 主要功能
    - **职业紧张预测**：输入个人特征，预测职业紧张风险
    - **数据分析**：展示职业紧张相关数据的统计分析
    - **个性化建议**：根据预测结果提供针对性的改善建议
    
    ### 技术特点
    - 基于标准化数据训练
    - 采用先进的机器学习算法
    - 提供直观的可视化结果
    """)

# 加载自定义样式
local_css("assets/style.css")