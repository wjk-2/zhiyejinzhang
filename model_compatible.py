import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler
import os

class CompatibleOccupationalStressModel:
    """兼容版本的职业紧张模型，避免版本冲突"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = ['年龄', '工龄', '本岗位工龄', '周均工作时间', '日均加班时间',
                            '生活满意度得分', '疲劳程度分级', '收入水平', '饮酒量', 
                            '低强度锻炼', '吸烟量', '婚姻状况']
        self.model_path = 'models/compatible_stress_model.pkl'
        self.scaler_path = 'models/compatible_scaler.pkl'
        
    def load_data(self, excel_path='stress_data.xlsx'):
        """加载和预处理数据"""
        # 确保models目录存在
        os.makedirs('models', exist_ok=True)
        
        # 读取数据
        df = pd.read_excel(excel_path)
        
        # 处理缺失值
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
        
        # 分离特征和目标变量
        X = df[self.feature_names]
        y = df['是否职业紧张'].astype(int)
        
        # 处理类别不平衡（手动实现重采样）
        class_0_count = (y == 0).sum()
        class_1_count = (y == 1).sum()
        
        if class_1_count < class_0_count:
            # 对少数类进行上采样
            minority_indices = y[y == 1].index
            oversampled_indices = np.random.choice(
                minority_indices, 
                size=class_0_count - class_1_count, 
                replace=True
            )
            X_oversampled = pd.concat([X, X.loc[oversampled_indices]])
            y_oversampled = pd.concat([y, y.loc[oversampled_indices]])
            X, y = X_oversampled, y_oversampled
        
        # 数据标准化
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # 保存标准化器
        with open(self.scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
            
        return X_scaled, y
    
    def optimize_hyperparameters(self, X, y):
        """手动实现超参数优化"""
        best_score = 0
        best_params = {}
        
        # 简单的参数搜索
        param_combinations = [
            {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 2},
            {'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 5},
            {'n_estimators': 300, 'max_depth': 20, 'min_samples_split': 10},
            {'n_estimators': 100, 'max_depth': None, 'min_samples_split': 2},
        ]
        
        for params in param_combinations:
            # 5折交叉验证
            scores = []
            for _ in range(5):
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=0.2, random_state=np.random.randint(1000)
                )
                
                model = RandomForestClassifier(
                    n_estimators=params['n_estimators'],
                    max_depth=params['max_depth'],
                    min_samples_split=params['min_samples_split'],
                    random_state=42,
                    class_weight='balanced'
                )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                score = f1_score(y_val, y_pred, average='weighted')
                scores.append(score)
            
            avg_score = np.mean(scores)
            if avg_score > best_score:
                best_score = avg_score
                best_params = params
        
        print(f"最佳F1分数: {best_score:.3f}")
        print(f"最佳参数: {best_params}")
        
        # 使用最佳参数训练最终模型
        final_model = RandomForestClassifier(
            n_estimators=best_params['n_estimators'],
            max_depth=best_params['max_depth'],
            min_samples_split=best_params['min_samples_split'],
            random_state=42,
            class_weight='balanced'
        )
        
        return final_model
    
    def train_model(self, excel_path='stress_data.xlsx'):
        """训练兼容版本的模型"""
        print("🚀 开始训练兼容模型...")
        
        # 确保models目录存在
        os.makedirs('models', exist_ok=True)
        
        # 加载数据
        X, y = self.load_data(excel_path)
        
        # 划分数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 优化超参数
        self.model = self.optimize_hyperparameters(X_train, y_train)
        
        # 训练最终模型
        self.model.fit(X_train, y_train)
        
        # 评估模型
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"测试集准确率: {accuracy:.3f}")
        print(f"测试集F1分数: {f1:.3f}")
        print(classification_report(y_test, y_pred))
        
        # 保存模型
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        print(f"✅ 模型已保存到: {self.model_path}")
        return accuracy
    
    def predict_stress(self, input_data):
        """预测职业紧张风险"""
        if self.model is None:
            self.load_model()
        
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
            input_df = input_df[self.feature_names]
            
            # 使用保存的标准化器
            if os.path.exists(self.scaler_path):
                with open(self.scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                input_scaled = scaler.transform(input_df)
            else:
                input_scaled = input_df.values
        else:
            input_scaled = input_data
        
        probability = self.model.predict_proba(input_scaled)[0]
        prediction = self.model.predict(input_scaled)[0]
        
        risk_prob = probability[1]
        if risk_prob < 0.3:
            risk_level = "低风险"
        elif risk_prob < 0.7:
            risk_level = "中风险"
        else:
            risk_level = "高风险"
            
        return {
            'prediction': int(prediction),
            'probability': risk_prob,
            'risk_level': risk_level,
            'confidence': max(probability)
        }
    
    def load_model(self):
        """加载模型"""
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
        else:
            print("⚠️ 模型文件不存在，请先训练模型")
            self.train_model()
    
    def get_feature_importance(self):
        """获取特征重要性"""
        if self.model is None:
            self.load_model()
        
        importance = self.model.feature_importances_
        return sorted(zip(self.feature_names, importance), 
                     key=lambda x: x[1], reverse=True)

def process_user_input(user_input):
    """处理用户输入，转换为模型需要的格式"""
    processed = {}
    
    # 年龄、工龄等直接使用
    processed['年龄'] = user_input.get('age', 0)
    processed['工龄'] = user_input.get('work_years', 0)
    processed['本岗位工龄'] = user_input.get('position_years', 0)
    processed['周均工作时间'] = user_input.get('weekly_hours', 40)
    processed['日均加班时间'] = user_input.get('daily_overtime', 0)
    processed['饮酒量'] = user_input.get('alcohol', 0)
    processed['低强度锻炼'] = user_input.get('low_exercise', 0)
    processed['生活满意度得分'] = user_input.get('life_satisfaction', 5)
    processed['疲劳程度分级'] = user_input.get('fatigue_level', 0)
    processed['吸烟量'] = user_input.get('smoking', 0)
    
    # 分类变量转换
    income_mapping = {'低': -1, '中': 0, '高': 1}
    processed['收入水平'] = income_mapping.get(user_input.get('income', '中'), 0)
    
    education_mapping = {'高中及以下': 0, '大专': 1, '本科': 2, '硕士及以上': 3}
    processed['教育程度'] = education_mapping.get(user_input.get('education', '本科'), 2)
    
    marriage_mapping = {'未婚': 1, '已婚同居': 2, '已婚分居': 3, '离婚': 4, '丧偶': 5}
    processed['婚姻状况'] = marriage_mapping.get(user_input.get('marital_status', '未婚'), 1)
    
    return processed

# 训练兼容模型
if __name__ == "__main__":
    print("🎯 职业紧张模型兼容版本")
    print("=" * 50)
    
    model = CompatibleOccupationalStressModel()
    accuracy = model.train_model()
    
    print("=" * 50)
    print(f"✅ 兼容模型训练完成！准确率: {accuracy:.3f}")