import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.feature_selection import mutual_info_classif, f_classif
import joblib
import os
import warnings
from tqdm import tqdm
import pingouin as pg  # 用于偏相关分析
# 设置中文字体显示
from matplotlib import font_manager

# 设置中文字体
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    font_path = font_manager.findfont(font_manager.FontProperties(family=['SimHei', 'Microsoft YaHei']))
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=font_path).get_name()
except:
    print("警告: 未找到中文字体，使用默认字体显示")

warnings.filterwarnings('ignore')


class OptimizedFeatureImportanceAnalyzer:
    """特征重要性分析器 - 使用固定权重和偏相关分析"""

    def __init__(self, max_features_per_symptom=12, global_max_features=30):
        self.max_features_per_symptom = max_features_per_symptom
        self.global_max_features = global_max_features
        self.symptom_feature_importance = {}
        self.global_feature_importance = {}
        self.selected_features = []
        self.symptom_specific_features = {}
        self.feature_names = []
        self.symptom_list = []
        self.analysis_results = {}

    def load_and_preprocess_data(self, file_path):
        """加载和预处理数据"""
        print("=== 加载数据 ===")
        try:
            df = pd.read_excel(file_path)
            print(f"数据读取成功! 形状: {df.shape}")
        except Exception as e:
            print(f"读取数据失败: {e}")
            return None, None

        return self.preprocess_data(df)

    def preprocess_data(self, df):
        """预处理数据"""
        data = df.copy()

        print("原始数据形状:", data.shape)
        print("原始数据列名:", data.columns.tolist())

        # 编码性别
        if '性别' in data.columns:
            print("\n性别分布:")
            print(data['性别'].value_counts())
            gender_mapping = {'男': 0, '女': 1, '男性': 0, '女性': 1, 'M': 0, 'F': 1}
            data['性别'] = data['性别'].map(gender_mapping)
            data['性别'] = data['性别'].fillna(0).astype(int)
            print("编码后性别分布:")
            print(data['性别'].value_counts())

        # 创建多标签目标
        multi_labels = self.create_multi_label_targets(data)

        # 选择特征列
        exclude_columns = ['编号', '症状1', '症状2', '症状3']
        feature_columns = [col for col in data.columns if col not in exclude_columns and col in data.columns]

        print(f"基础特征数量: {len(feature_columns)}")

        # 缺失值处理
        data_processed = self.robust_missing_value_imputation(data, feature_columns)
        features = data_processed[feature_columns]

        self.feature_names = feature_columns
        self.symptom_list = multi_labels.columns.tolist()

        print(f"\n最终特征数量: {len(self.feature_names)}")
        print(f"样本数量: {len(features)}")
        print(f"症状数量: {len(self.symptom_list)}")

        return features, multi_labels

    def create_multi_label_targets(self, df):
        """创建多标签目标变量"""
        print("\n=== 创建多标签目标 ===")

        # 定义所有可能的症状（包括"无以上症状"）
        all_possible_symptoms = [
            '2型糖尿病', '动脉粥样硬化性血脂异常', '高尿酸血症',
            '糖尿病前期', '急性胰腺炎支持', '慢性肾病（CKD）', '无以上症状'
        ]

        # 初始化症状列表
        self.symptom_list = all_possible_symptoms

        print(f"症状列表: {self.symptom_list}")

        # 创建多标签列
        multi_labels = pd.DataFrame(0, index=df.index, columns=self.symptom_list)

        # 处理症状列
        symptom_columns = ['症状1', '症状2', '症状3']

        for idx, row in df.iterrows():
            has_any_symptom = False
            for col in symptom_columns:
                if col in df.columns and pd.notna(row[col]) and row[col] != '无' and row[col] != '':
                    symptom = str(row[col]).strip()
                    if symptom in self.symptom_list:
                        multi_labels.loc[idx, symptom] = 1
                        has_any_symptom = True

            # 如果没有检测到任何症状，标记为"无以上症状"
            if not has_any_symptom:
                multi_labels.loc[idx, '无以上症状'] = 1

        # 检查标签冲突
        conflict_count = ((multi_labels.iloc[:, :-1].sum(axis=1) > 0) & (multi_labels['无以上症状'] == 1)).sum()
        if conflict_count > 0:
            print(f"警告: 发现 {conflict_count} 个样本同时有症状和'无以上症状'标签，将清除'无以上症状'标签")
            mask = multi_labels.iloc[:, :-1].sum(axis=1) > 0
            multi_labels.loc[mask, '无以上症状'] = 0

        # 统计症状分布
        print("\n症状分布统计:")
        for symptom in self.symptom_list:
            count = multi_labels[symptom].sum()
            percentage = (count / len(multi_labels)) * 100
            print(f"{symptom}: {count} 人 ({percentage:.1f}%)")

        return multi_labels

    def robust_missing_value_imputation(self, data, feature_columns):
        """鲁棒的缺失值补偿方法"""
        print("\n=== 开始缺失值补偿 ===")
        data_imputed = data.copy()

        # 统计缺失情况
        missing_stats = []
        for col in feature_columns:
            if data_imputed[col].isnull().any():
                null_count = data_imputed[col].isnull().sum()
                null_percentage = (null_count / len(data_imputed)) * 100
                missing_stats.append({
                    'feature': col,
                    'null_count': null_count,
                    'null_percentage': null_percentage
                })
                if null_count > 0:
                    print(f"特征 {col}: {null_count} 个缺失值 ({null_percentage:.2f}%)")

        if not missing_stats:
            print("没有发现缺失值")
            return data_imputed

        # 对数值型特征使用KNN插值
        numeric_features = [col for col in feature_columns if data_imputed[col].dtype in ['float64', 'int64']]

        if numeric_features:
            print(f"对 {len(numeric_features)} 个数值型特征使用KNN插值...")
            try:
                imputer = KNNImputer(n_neighbors=5, weights='uniform')
                numeric_data = data_imputed[numeric_features]
                imputed_numeric = imputer.fit_transform(numeric_data)
                data_imputed[numeric_features] = imputed_numeric
                print("数值型特征KNN插值完成")
            except Exception as e:
                print(f"KNN插值失败，使用均值填充: {e}")
                for col in numeric_features:
                    if data_imputed[col].isnull().any():
                        mean_val = data_imputed[col].mean()
                        data_imputed[col] = data_imputed[col].fillna(mean_val)

        # 对非数值型特征使用众数填充
        non_numeric_features = [col for col in feature_columns if col not in numeric_features]
        for col in non_numeric_features:
            if data_imputed[col].isnull().any():
                mode_value = data_imputed[col].mode()
                fill_value = mode_value[0] if not mode_value.empty else '未知'
                data_imputed[col] = data_imputed[col].fillna(fill_value)
                print(f"特征 {col} 使用众数 '{fill_value}' 填充")

        # 验证缺失值处理结果
        remaining_missing = data_imputed[feature_columns].isnull().sum().sum()
        print(f"缺失值处理完成，剩余缺失值数量: {remaining_missing}")

        return data_imputed

    def compute_feature_importance(self, features, multi_labels):
        """计算特征重要性 - 使用固定权重和偏相关分析"""
        print("\n=== 计算特征重要性 ===")

        # 固定参数和权重
        rf_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'random_state': 42
        }

        # 固定权重
        weights = {
            'rf': 0.3,
            'ig': 0.3,
            'partial_corr': 0.3,  # 使用偏相关分析
            'f_score': 0.1
        }

        print(f"使用固定权重: {weights}")
        print(f"使用RF参数: {rf_params}")

        symptom_importances = {}

        for symptom in tqdm(multi_labels.columns, desc="计算症状特征重要性"):
            y = multi_labels[symptom]

            # 跳过样本太少的症状
            if y.sum() < 5:
                symptom_importances[symptom] = pd.Series(0.5, index=features.columns)
                continue

            # 使用多种方法计算特征重要性
            importance_scores = {}

            # 1. 随机森林重要性（使用固定参数）
            try:
                rf = RandomForestClassifier(**rf_params)
                rf.fit(features, y)
                rf_importance = pd.Series(rf.feature_importances_, index=features.columns)
                importance_scores['rf'] = rf_importance
            except Exception as e:
                print(f"  随机森林计算失败: {e}")
                importance_scores['rf'] = pd.Series(0, index=features.columns)

            # 2. 信息增益
            try:
                ig_scores = mutual_info_classif(features, y, random_state=42)
                ig_scores = pd.Series(ig_scores, index=features.columns)
                importance_scores['ig'] = ig_scores
            except:
                importance_scores['ig'] = pd.Series(0, index=features.columns)

            # 3. 偏相关分析（替换原来的相关系数）
            try:
                partial_corr_scores = self.compute_partial_correlation(features, y)
                importance_scores['partial_corr'] = partial_corr_scores
            except Exception as e:
                print(f"  偏相关分析失败: {e}")
                importance_scores['partial_corr'] = pd.Series(0, index=features.columns)

            # 4. 方差分析
            try:
                f_scores, _ = f_classif(features.fillna(features.mean()), y)
                f_scores = pd.Series(f_scores, index=features.columns).fillna(0)
                importance_scores['f_score'] = f_scores
            except:
                importance_scores['f_score'] = pd.Series(0, index=features.columns)

            # 使用固定权重组合得分
            combined_score = pd.Series(0.0, index=features.columns)

            for method, scores in importance_scores.items():
                if method in weights:
                    # 归一化
                    if scores.max() > scores.min():
                        normalized_scores = (scores - scores.min()) / (scores.max() - scores.min())
                    else:
                        normalized_scores = scores
                    combined_score += normalized_scores * weights[method]

            # 归一化最终得分
            if combined_score.max() > 0:
                combined_score = combined_score / combined_score.max()

            symptom_importances[symptom] = combined_score

        return symptom_importances

    def compute_partial_correlation(self, features, target):
        """计算偏相关系数"""
        partial_corr_scores = pd.Series(0.0, index=features.columns)

        # 创建包含特征和目标的DataFrame
        data = features.copy()
        data['target'] = target

        for feature in features.columns:
            try:
                # 控制其他特征的影响，计算偏相关
                other_features = [col for col in features.columns if col != feature]

                if len(other_features) > 0:
                    # 使用pingouin计算偏相关
                    result = pg.partial_corr(data=data, x=feature, y='target', covar=other_features)
                    partial_corr = result['r'].iloc[0]
                    p_value = result['p-val'].iloc[0]

                    # 使用相关系数的绝对值，并根据p值调整显著性
                    if p_value < 0.05:  # 统计显著
                        score = abs(partial_corr)
                    else:
                        score = abs(partial_corr) * 0.5  # 不显著则减半
                else:
                    # 如果没有其他特征，使用普通相关
                    corr = data[feature].corr(data['target'])
                    score = abs(corr) if not np.isnan(corr) else 0

                partial_corr_scores[feature] = score

            except Exception as e:
                # 如果偏相关计算失败，使用普通相关作为备选
                try:
                    corr = data[feature].corr(data['target'])
                    partial_corr_scores[feature] = abs(corr) if not np.isnan(corr) else 0
                except:
                    partial_corr_scores[feature] = 0

        return partial_corr_scores

    def run_feature_importance_analysis(self, features, multi_labels):
        """运行特征重要性分析"""
        print("=== 运行特征重要性分析 ===")

        # 计算特征重要性
        symptom_importances = self.compute_feature_importance(features, multi_labels)
        self.symptom_feature_importance = symptom_importances

        # 计算全局特征重要性
        global_importance = pd.Series(0.0, index=features.columns)
        for importance_scores in symptom_importances.values():
            global_importance += importance_scores
        global_importance = global_importance / len(symptom_importances)
        self.global_feature_importance = global_importance.to_dict()

        # 为每个症状选择最重要的特征
        all_selected_features = set()
        symptom_specific_features = {}

        for symptom, importance_scores in symptom_importances.items():
            top_features = importance_scores.sort_values(ascending=False).head(self.max_features_per_symptom)
            selected_features = top_features.index.tolist()
            symptom_specific_features[symptom] = selected_features
            all_selected_features.update(selected_features)

        self.symptom_specific_features = symptom_specific_features

        # 如果总特征数超过限制，使用全局重要性进一步筛选
        initial_feature_count = len(all_selected_features)
        if initial_feature_count > self.global_max_features:
            print(f"\n特征数量 ({initial_feature_count}) 超过限制 ({self.global_max_features})，进行进一步筛选")

            # 计算特征在症状中的覆盖度
            feature_coverage = {}
            for feature in all_selected_features:
                coverage = sum(1 for features_list in symptom_specific_features.values()
                               if feature in features_list)
                feature_coverage[feature] = coverage

            # 综合得分：全局重要性 * (1 + 覆盖度权重)
            combined_scores = {}
            for feature in all_selected_features:
                importance_score = global_importance.get(feature, 0)
                coverage_score = feature_coverage[feature] / len(symptom_specific_features)
                # 覆盖度高的特征给予额外奖励
                combined_scores[feature] = importance_score * (1 + 0.5 * coverage_score)

            # 选择得分最高的特征
            sorted_features = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
            final_features = [feature for feature, score in sorted_features[:self.global_max_features]]

            print(f"最终选择 {len(final_features)} 个特征 (从 {initial_feature_count} 个中筛选)")
            self.selected_features = final_features
        else:
            self.selected_features = list(all_selected_features)
            print(f"最终选择 {len(self.selected_features)} 个特征")

        # 保存分析结果
        self.analysis_results = {
            'global_importance': global_importance,
            'symptom_importances': symptom_importances,
            'selected_features': self.selected_features
        }

        return self.analysis_results

    def plot_symptom_specific_feature_importance(self, top_n=8):
        """绘制每个症状的特征重要性"""
        if not self.symptom_feature_importance:
            print("没有特征重要性数据可绘制")
            return

        n_symptoms = len(self.symptom_feature_importance)
        n_cols = 2
        n_rows = (n_symptoms + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()

        for i, (symptom, importance_scores) in enumerate(self.symptom_feature_importance.items()):
            if i < len(axes):
                ax = axes[i]

                # 选择前top_n个特征
                top_features = importance_scores.sort_values(ascending=False).head(top_n)
                features = top_features.index
                scores = top_features.values

                # 使用统一的颜色
                colors = ['skyblue'] * len(features)

                bars = ax.barh(range(len(features)), scores, color=colors)
                ax.set_yticks(range(len(features)))
                ax.set_yticklabels(features, fontsize=8)
                ax.set_xlabel('特征重要性', fontsize=10)
                ax.set_title(f'{symptom}\n前{top_n}个重要特征', fontsize=12, fontweight='bold')
                ax.grid(True, axis='x', alpha=0.3)

                # 添加数值标签
                for j, bar in enumerate(bars):
                    width = bar.get_width()
                    ax.text(width + 0.001, bar.get_y() + bar.get_height() / 2,
                            f'{width:.3f}', ha='left', va='center', fontsize=7)

                ax.invert_yaxis()

        # 隐藏多余的子图
        for i in range(n_symptoms, n_rows * n_cols):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.subplots_adjust(top=0.95, hspace=0.6, wspace=0.4)
        plt.show()

    def plot_global_feature_importance(self, top_n=20):
        """绘制全局特征重要性"""
        if not self.global_feature_importance:
            print("没有全局特征重要性数据可绘制")
            return

        # 转换为Series并排序
        global_importance_series = pd.Series(self.global_feature_importance)
        global_importance_series = global_importance_series.sort_values(ascending=False).head(top_n)

        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(global_importance_series)), global_importance_series.values, color='skyblue')
        plt.yticks(range(len(global_importance_series)), global_importance_series.index)
        plt.xlabel('全局特征重要性')
        plt.title(f'全局特征重要性 (前{top_n}个特征)')
        plt.gca().invert_yaxis()
        plt.grid(True, axis='x', alpha=0.3)

        # 添加数值标签
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.001, bar.get_y() + bar.get_height() / 2,
                     f'{width:.3f}', ha='left', va='center', fontsize=8)

        plt.tight_layout()
        plt.show()

    def plot_feature_coverage_heatmap(self):
        """绘制特征覆盖度热力图"""
        if not self.symptom_specific_features:
            print("没有症状特定特征数据可绘制")
            return

        # 创建特征-症状矩阵
        features = list(set([f for features_list in self.symptom_specific_features.values() for f in features_list]))
        symptoms = list(self.symptom_specific_features.keys())

        coverage_matrix = pd.DataFrame(0, index=features, columns=symptoms)

        for symptom, feature_list in self.symptom_specific_features.items():
            for feature in feature_list:
                if feature in coverage_matrix.index:
                    coverage_matrix.loc[feature, symptom] = 1

        # 计算每个特征的总覆盖度
        coverage_matrix['总覆盖度'] = coverage_matrix.sum(axis=1)
        coverage_matrix = coverage_matrix.sort_values('总覆盖度', ascending=False)

        # 绘制热力图
        plt.figure(figsize=(12, max(8, len(features) * 0.3)))
        sns.heatmap(coverage_matrix.drop('总覆盖度', axis=1),
                    annot=True, cmap='YlOrRd',
                    cbar_kws={'label': '是否选择'})
        plt.title('特征-症状选择热力图')
        plt.tight_layout()
        plt.show()

        return coverage_matrix

    def save_analysis_results(self, filepath='feature_importance_analysis'):
        """保存分析结果"""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)

        # 保存数据
        analysis_data = {
            'symptom_feature_importance': self.symptom_feature_importance,
            'global_feature_importance': self.global_feature_importance,
            'selected_features': self.selected_features,
            'symptom_specific_features': self.symptom_specific_features,
            'feature_names': self.feature_names,
            'symptom_list': self.symptom_list,
            'analysis_results': self.analysis_results
        }

        joblib.dump(analysis_data, f'{filepath}_results.pkl')

        # 保存为Excel文件
        with pd.ExcelWriter(f'{filepath}_results.xlsx') as writer:
            # 全局特征重要性
            global_importance_df = pd.DataFrame({
                '特征': list(self.global_feature_importance.keys()),
                '重要性': list(self.global_feature_importance.values())
            }).sort_values('重要性', ascending=False)
            global_importance_df.to_excel(writer, sheet_name='全局特征重要性', index=False)

            # 每个症状的特征重要性
            for symptom, importance_scores in self.symptom_feature_importance.items():
                symptom_df = pd.DataFrame({
                    '特征': importance_scores.index,
                    '重要性': importance_scores.values
                }).sort_values('重要性', ascending=False)
                sheet_name = symptom[:31]  # Excel工作表名称限制
                symptom_df.to_excel(writer, sheet_name=sheet_name, index=False)

            # 症状特定特征选择
            symptom_features_df = pd.DataFrame(dict([
                (k, pd.Series(v)) for k, v in self.symptom_specific_features.items()
            ]))
            symptom_features_df.to_excel(writer, sheet_name='症状特定特征', index=False)

        print(f"分析结果已保存到:")
        print(f"  - {filepath}_results.pkl")
        print(f"  - {filepath}_results.xlsx")

    def load_analysis_results(self, filepath='feature_importance_analysis'):
        """加载分析结果"""
        try:
            analysis_data = joblib.load(f'{filepath}_results.pkl')
            self.symptom_feature_importance = analysis_data['symptom_feature_importance']
            self.global_feature_importance = analysis_data['global_feature_importance']
            self.selected_features = analysis_data['selected_features']
            self.symptom_specific_features = analysis_data['symptom_specific_features']
            self.feature_names = analysis_data['feature_names']
            self.symptom_list = analysis_data['symptom_list']
            self.analysis_results = analysis_data.get('analysis_results', {})

            print("分析结果加载成功!")
            return True
        except Exception as e:
            print(f"加载分析结果失败: {e}")
            return False


def run_feature_importance_analysis():
    """运行特征重要性分析"""
    print("=== 特征重要性分析 ===")

    # 初始化分析器
    analyzer = OptimizedFeatureImportanceAnalyzer(
        max_features_per_symptom=8,
        global_max_features=25
    )

    # 加载数据
    file_path = input("请输入数据文件路径 (默认: C:\\Users\\a'y\\Desktop\\shenghua2.xls): ").strip()
    if not file_path:
        file_path = r"C:\Users\a'y\Desktop\shenghua2.xls"

    features, multi_labels = analyzer.load_and_preprocess_data(file_path)

    if features is None or multi_labels is None:
        print("数据加载失败，请检查文件路径")
        return

    # 运行分析
    analysis_result = analyzer.run_feature_importance_analysis(features, multi_labels)

    # 绘制可视化结果
    print("\n=== 生成可视化结果 ===")
    analyzer.plot_symptom_specific_feature_importance(top_n=8)
    analyzer.plot_global_feature_importance(top_n=20)
    analyzer.plot_feature_coverage_heatmap()

    # 保存分析结果
    analyzer.save_analysis_results('feature_importance_analysis')

    print("\n=== 特征重要性分析完成 ===")
    print(f"最终选择了 {len(analyzer.selected_features)} 个特征")
    print("结果已保存到 feature_importance_analysis_results.pkl 和 .xlsx")

    return analyzer, analysis_result


def load_and_visualize_existing_analysis():
    """加载并可视化已有的分析结果"""
    print("=== 加载已有特征重要性分析结果 ===")

    analyzer = OptimizedFeatureImportanceAnalyzer()

    if analyzer.load_analysis_results('feature_importance_analysis'):
        # 绘制可视化结果
        print("\n=== 生成可视化结果 ===")
        analyzer.plot_symptom_specific_feature_importance(top_n=8)
        analyzer.plot_global_feature_importance(top_n=20)
        coverage_matrix = analyzer.plot_feature_coverage_heatmap()

        print("\n=== 分析结果加载和可视化完成 ===")
        return analyzer
    else:
        print("请先运行特征重要性分析")
        return None


if __name__ == "__main__":
    print("=== 特征重要性分析系统 ===")
    print("1. 运行特征重要性分析")
    print("2. 加载并可视化已有分析结果")

    choice = input("请选择操作 (1 或 2): ").strip()

    if choice == "1":
        print("\n开始运行特征重要性分析...")
        analyzer, analysis_result = run_feature_importance_analysis()
    elif choice == "2":
        print("\n加载已有特征重要性分析结果...")
        analyzer = load_and_visualize_existing_analysis()
    else:
        print("无效选择，退出程序")