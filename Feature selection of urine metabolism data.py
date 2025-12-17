import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif, f_classif
import joblib
import os
import warnings
from tqdm import tqdm
import pingouin as pg
from sklearn.impute import KNNImputer
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

# 增大全局字体大小
plt.rcParams['font.size'] = 14  # 从默认的10增加到14
plt.rcParams['axes.titlesize'] = 16  # 标题字体大小
plt.rcParams['axes.labelsize'] = 14  # 坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 12  # x轴刻度标签字体大小
plt.rcParams['ytick.labelsize'] = 12  # y轴刻度标签字体大小
plt.rcParams['legend.fontsize'] = 12  # 图例字体大小

warnings.filterwarnings('ignore')


class SixCategoryFeatureAnalyzer:
    """六大类特征分析器"""

    def __init__(self, max_features_per_disease=30, global_max_features=100, keep_first_n_features=213):
        self.max_features_per_disease = max_features_per_disease
        self.global_max_features = global_max_features
        self.keep_first_n_features = keep_first_n_features  # 保留前213列特征
        self.disease_feature_importance = {}
        self.global_feature_importance = {}
        self.selected_features = []
        self.filtered_selected_features = []  # 新增：筛选后的特征列表
        self.disease_specific_features = {}
        self.filtered_disease_specific_features = {}  # 新增：筛选后的疾病特定特征
        self.feature_names = []
        self.all_feature_columns = []  # 新增：所有特征列（352列）
        self.first_213_feature_columns = []  # 新增：前213列特征
        self.disease_list = []
        self.analysis_results = {}

    def load_and_preprocess_data(self, file_path):
        """加载和预处理数据"""
        print("=== 加载六大类工作表数据 ===")
        try:
            df = pd.read_excel(file_path, sheet_name='六大类')
            print(f"六大类工作表读取成功! 形状: {df.shape}")
            print(f"总列数: {len(df.columns)}")
            print(f"前20列: {df.columns[:20].tolist()}")
            print(f"后20列: {df.columns[-20:].tolist()}")
        except Exception as e:
            print(f"读取数据失败: {e}")
            return None, None

        return self.preprocess_data(df)

    def preprocess_data(self, df):
        """预处理数据 - 使用所有特征进行分析"""
        data = df.copy()

        print("\n=== 数据预处理 ===")
        print(f"原始数据形状: {data.shape}")
        print(f"原始数据总列数: {len(data.columns)}")

        # 创建六大类目标变量
        multi_labels = self.create_six_category_targets(data)

        # =========== 识别所有特征列和前213列特征 ===========
        print(f"\n=== 识别特征列 ===")

        # 排除的目标列
        exclude_columns = ['编号'] + self.disease_list

        # 获取所有特征列（排除编号和疾病标签）
        all_columns = data.columns.tolist()
        self.all_feature_columns = [col for col in all_columns
                                    if col not in exclude_columns and col in data.columns]

        # 获取前213列特征（第1-213列，排除疾病标签）
        if len(data.columns) >= 214:
            # 前213列（第0-212列）
            first_213_cols = data.columns[:213].tolist()
            self.first_213_feature_columns = [col for col in first_213_cols
                                              if col not in exclude_columns and col in data.columns]

            print(f"所有特征列数量: {len(self.all_feature_columns)}")
            print(f"前213列特征数量: {len(self.first_213_feature_columns)}")
            print(f"前213列特征示例: {self.first_213_feature_columns[:10]}")

            # 检查有多少特征不在前213列中
            extra_features = [col for col in self.all_feature_columns if col not in self.first_213_feature_columns]
            print(f"第214列及之后的特征数量: {len(extra_features)}")
            if extra_features:
                print(f"额外特征示例: {extra_features[:10]}")
        else:
            print(f"警告: 数据只有 {len(data.columns)} 列，少于214列")
            self.first_213_feature_columns = self.all_feature_columns

        # =========== 使用所有特征进行分析 ===========
        print(f"\n=== 使用所有特征进行分析 ===")
        print(f"使用的特征数量: {len(self.all_feature_columns)}")
        print(f"前10个特征: {self.all_feature_columns[:10]}")

        # 缺失值处理 - 对所有特征进行处理
        data_processed = self.robust_missing_value_imputation(data, self.all_feature_columns)
        features = data_processed[self.all_feature_columns]

        # =========== 对分类特征进行编码 ===========
        features = self.encode_categorical_features(features)
        # ===========================================

        self.feature_names = self.all_feature_columns
        self.disease_list = multi_labels.columns.tolist()

        print(f"\n预处理完成!")
        print(f"特征数量: {len(self.feature_names)}")
        print(f"样本数量: {len(features)}")
        print(f"疾病类型数量: {len(self.disease_list)}")
        print(f"特征列示例: {self.feature_names[:5]}...")

        return features, multi_labels

    def filter_features_by_first_213(self, features_list):
        """筛选特征，只保留在前213列中存在的特征"""
        if not hasattr(self, 'first_213_feature_columns') or not self.first_213_feature_columns:
            print("警告: 前213列特征未定义，无法筛选")
            return features_list

        filtered_features = [f for f in features_list if f in self.first_213_feature_columns]
        removed_features = [f for f in features_list if f not in self.first_213_feature_columns]

        if removed_features:
            print(f"移除了 {len(removed_features)} 个不在前213列的特征")
            print(f"移除的特征示例: {removed_features[:10]}")

        return filtered_features

    def encode_categorical_features(self, features):
        """对分类特征进行编码 - 最小修改版本"""
        features_encoded = features.copy()

        # 识别分类特征
        categorical_features = []
        for col in features_encoded.columns:
            # 简单的分类特征识别：非数值型或只有少量唯一值
            if features_encoded[col].dtype == 'object' or features_encoded[col].nunique() < 10:
                categorical_features.append(col)

        if categorical_features:
            print(f"识别到 {len(categorical_features)} 个分类特征: {categorical_features[:10]}...")

            for col in categorical_features:
                try:
                    # 尝试转为数值型（如果已经是数值或可转为数值则跳过）
                    features_encoded[col] = pd.to_numeric(features_encoded[col], errors='ignore')

                    # 如果转换失败（仍为object类型），进行标签编码
                    if features_encoded[col].dtype == 'object':
                        # 获取唯一值并映射为数值
                        unique_values = features_encoded[col].unique()
                        value_mapping = {val: idx for idx, val in enumerate(unique_values)}
                        features_encoded[col] = features_encoded[col].map(value_mapping)

                        # 打印编码信息
                        print(f"  特征 '{col}' 已编码: {len(unique_values)}个唯一值")
                except Exception as e:
                    print(f"  编码特征 '{col}' 时出错: {e}")
                    # 如果编码失败，删除该列
                    features_encoded = features_encoded.drop(columns=[col])
                    print(f"  已删除特征: {col}")

        return features_encoded

    def create_six_category_targets(self, df):
        """创建六大类目标变量"""
        print("\n=== 创建六大类目标变量 ===")

        # 六大类疾病名称
        six_categories = [
            '内分泌系统疾病',
            '消化系统疾病',
            '循环系统疾病',
            '泌尿生殖系统疾病',
            '恶性肿瘤',
            '血液及造血器官疾病和涉及免疫机制的某些疾患'
        ]

        self.disease_list = six_categories

        print(f"六大类疾病列表: {self.disease_list}")

        # 创建多标签列
        multi_labels = pd.DataFrame(0, index=df.index, columns=self.disease_list)

        # 检查六大类列是否存在
        for disease in six_categories:
            if disease in df.columns:
                # 将非空值标记为1，空值保持为0
                multi_labels[disease] = df[disease].apply(lambda x: 1 if pd.notna(x) and str(x).strip() != '' else 0)
                print(f"{disease}: {multi_labels[disease].sum()} 个阳性样本")
            else:
                print(f"警告: 未找到列 '{disease}'")
                multi_labels[disease] = 0

        # 统计疾病分布
        print("\n六大类疾病分布统计:")
        for disease in six_categories:
            count = multi_labels[disease].sum()
            percentage = (count / len(multi_labels)) * 100
            print(f"{disease}: {count} 人 ({percentage:.1f}%)")

        return multi_labels

    def robust_missing_value_imputation(self, data, feature_columns):
        """鲁棒的缺失值补偿方法"""
        print("\n=== 开始缺失值补偿 ===")
        data_imputed = data.copy()

        # 统计缺失情况
        missing_stats = []
        total_missing = 0

        for col in feature_columns:
            if data_imputed[col].isnull().any():
                null_count = data_imputed[col].isnull().sum()
                null_percentage = (null_count / len(data_imputed)) * 100
                missing_stats.append({
                    'feature': col,
                    'null_count': null_count,
                    'null_percentage': null_percentage
                })
                total_missing += null_count
                if null_count > 0 and null_percentage > 50:
                    print(f"特征 {col}: {null_count} 个缺失值 ({null_percentage:.2f}%)")

        print(f"总缺失值数量: {total_missing}")

        if not missing_stats:
            print("没有发现缺失值")
            return data_imputed

        # 对数值型特征使用KNN插值
        numeric_features = []
        categorical_features = []

        # 更精确地识别数值型特征
        for col in feature_columns:
            if pd.api.types.is_numeric_dtype(data_imputed[col]):
                numeric_features.append(col)
            else:
                categorical_features.append(col)

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
        for col in categorical_features:
            if data_imputed[col].isnull().any():
                mode_value = data_imputed[col].mode()
                fill_value = mode_value[0] if not mode_value.empty else '未知'
                data_imputed[col] = data_imputed[col].fillna(fill_value)
                if data_imputed[col].isnull().sum() > 0:
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
            'rf': 0.4,
            'ig': 0.3,
            'partial_corr': 0.2,  # 使用偏相关分析
            'f_score': 0.1
        }

        print(f"使用固定权重: {weights}")
        print(f"使用RF参数: {rf_params}")

        disease_importances = {}

        for disease in tqdm(multi_labels.columns, desc="计算疾病特征重要性"):
            y = multi_labels[disease]

            # 跳过样本太少的疾病
            if y.sum() < 5:
                disease_importances[disease] = pd.Series(0.5, index=features.columns)
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

            disease_importances[disease] = combined_score

        return disease_importances

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
        """运行特征重要性分析 - 为每个疾病单独选择特征"""
        print("=== 运行特征重要性分析 ===")
        print(f"分析的特征总数: {len(features.columns)}")

        # 计算特征重要性
        disease_importances = self.compute_feature_importance(features, multi_labels)
        self.disease_feature_importance = disease_importances

        # 计算全局特征重要性（仅用于展示，不用于筛选）
        global_importance = pd.Series(0.0, index=features.columns)
        for importance_scores in disease_importances.values():
            global_importance += importance_scores
        global_importance = global_importance / len(disease_importances)
        self.global_feature_importance = global_importance.to_dict()

        # 为每个疾病选择最重要的特征 - 只选择前213列的特征
        all_selected_features = set()
        disease_specific_features = {}

        for disease, importance_scores in disease_importances.items():
            # 关键修改：只从前213列中选择特征
            # 1. 先筛选出前213列的特征
            first_213_importance = importance_scores[importance_scores.index.isin(self.first_213_feature_columns)]

            # 2. 选择前max_features_per_disease个特征
            top_features = first_213_importance.sort_values(ascending=False).head(self.max_features_per_disease)
            selected_features = top_features.index.tolist()

            # 3. 保存到相应的字典中
            disease_specific_features[disease] = selected_features
            all_selected_features.update(selected_features)

            print(f"{disease}: 从前213列中选择了 {len(selected_features)} 个特征")
            print(f"  前5个特征: {selected_features[:5]}")

        # 保存筛选后的特征
        self.disease_specific_features = disease_specific_features
        self.selected_features = list(all_selected_features)  # 所有疾病的特征合集（都是前213列的）

        # =========== 也保存到filtered版本中（为了兼容性）===========
        self.filtered_selected_features = list(all_selected_features)
        self.filtered_disease_specific_features = disease_specific_features

        # 保存分析结果
        self.analysis_results = {
            'global_importance': global_importance,
            'disease_importances': disease_importances,
            'selected_features': self.selected_features,
            'filtered_selected_features': self.filtered_selected_features,
            'disease_specific_features': disease_specific_features,  # 修改：直接保存筛选后的
            'filtered_disease_specific_features': self.filtered_disease_specific_features,
            'all_feature_columns': self.all_feature_columns,
            'first_213_feature_columns': self.first_213_feature_columns
        }

        return self.analysis_results

    def plot_disease_specific_feature_importance(self, top_n=30):
        """绘制每个疾病的特征重要性 - 只显示前213列中的前top_n个特征"""
        if not self.disease_feature_importance:
            print("没有特征重要性数据可绘制")
            return

        for disease, importance_scores in self.disease_feature_importance.items():
            # 为每个疾病创建单独的图表
            plt.figure(figsize=(16, 12))

            # 只选择前213列的特征
            first_213_importance = importance_scores[importance_scores.index.isin(self.first_213_feature_columns)]

            # 从这些特征中选择前top_n个
            top_features = first_213_importance.sort_values(ascending=False).head(top_n)

            # 检查是否有特征
            if len(top_features) == 0:
                print(f"警告: {disease} 没有前213列的特征")
                continue

            features = top_features.index
            scores = top_features.values

            # 使用统一的颜色（都是前213列的特征）
            colors = ['skyblue'] * len(features)

            bars = plt.barh(range(len(features)), scores, color=colors)
            plt.yticks(range(len(features)), features, fontsize=14)
            plt.xlabel('特征重要性', fontsize=16)

            # 添加标题说明
            plt.title(f'{disease} - 前{top_n}个重要特征', fontsize=18, fontweight='bold')
            plt.grid(True, axis='x', alpha=0.3)

            # 添加数值标签
            for j, bar in enumerate(bars):
                width = bar.get_width()
                plt.text(width + 0.001, bar.get_y() + bar.get_height() / 2,
                         f'{width:.3f}', ha='left', va='center', fontsize=12)

            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()

            # 打印当前绘制的特征列表
            print(f"\n{disease} 绘制的特征列表（前{top_n}个）:")
            for i, (feature, score) in enumerate(zip(features, scores)):
                print(f"  {i + 1}. {feature}: {score:.4f}")

    def plot_global_feature_importance(self, top_n=30):
        """绘制全局特征重要性 - 只显示前213列中的前top_n个特征"""
        if not self.global_feature_importance:
            print("没有全局特征重要性数据可绘制")
            return

        # 转换为Series
        global_importance_series = pd.Series(self.global_feature_importance)

        # =========== 修改：只选择前213列中的特征 ===========
        # 先筛选出前213列的特征
        first_213_importance = global_importance_series[
            global_importance_series.index.isin(self.first_213_feature_columns)]

        # 从这些特征中选择前top_n个
        global_importance_series = first_213_importance.sort_values(ascending=False).head(top_n)
        # =================================================

        plt.figure(figsize=(16, 12))  # 增大图形尺寸
        bars = plt.barh(range(len(global_importance_series)), global_importance_series.values, color='skyblue')
        plt.yticks(range(len(global_importance_series)), global_importance_series.index, fontsize=14)
        plt.xlabel('全局特征重要性', fontsize=16)

        # 添加标题说明
        plt.title(f'全局特征重要性 (前{top_n}个特征)', fontsize=18, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(True, axis='x', alpha=0.3)

        # 添加数值标签
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.001, bar.get_y() + bar.get_height() / 2,
                     f'{width:.3f}', ha='left', va='center', fontsize=12)

        plt.tight_layout()
        plt.show()

    def plot_feature_coverage_heatmap(self):
        """绘制特征覆盖度热力图 - 只显示前213列特征"""
        if not self.filtered_disease_specific_features:
            print("没有筛选后的疾病特定特征数据可绘制")
            return

        # 创建特征-疾病矩阵（使用筛选后的特征）
        features = list(
            set([f for features_list in self.filtered_disease_specific_features.values() for f in features_list]))
        diseases = list(self.filtered_disease_specific_features.keys())

        coverage_matrix = pd.DataFrame(0, index=features, columns=diseases)

        for disease, feature_list in self.filtered_disease_specific_features.items():
            for feature in feature_list:
                if feature in coverage_matrix.index:
                    coverage_matrix.loc[feature, disease] = 1

        # 计算每个特征的总覆盖度
        coverage_matrix['总覆盖度'] = coverage_matrix.sum(axis=1)
        coverage_matrix = coverage_matrix.sort_values('总覆盖度', ascending=False)

        # 绘制热力图
        plt.figure(figsize=(14, max(10, len(features) * 0.4)))  # 增大图形尺寸
        sns.heatmap(coverage_matrix.drop('总覆盖度', axis=1),
                    annot=True, cmap='YlOrRd',
                    cbar_kws={'label': '是否选择'},
                    annot_kws={"size": 12})  # 增大热力图注释字体
        plt.title('特征-疾病选择热力图（仅显示前213列特征）', fontsize=16)  # 增大标题字体
        plt.tight_layout()
        plt.show()

        return coverage_matrix

    def save_analysis_results(self, filepath='six_category_analysis'):
        """保存分析结果 - 只保存筛选后的特征"""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)

        # =========== 关键修改：只保存筛选后的特征 ===========

        # 筛选疾病特征重要性（只保留前213列）
        filtered_disease_importance = {}
        for disease, importance_scores in self.disease_feature_importance.items():
            # 只保留前213列的特征重要性
            filtered_scores = importance_scores[
                importance_scores.index.isin(self.first_213_feature_columns)
            ]
            filtered_disease_importance[disease] = filtered_scores

        # 筛选全局特征重要性（只保留前213列）
        global_importance_series = pd.Series(self.global_feature_importance)
        filtered_global_importance = global_importance_series[
            global_importance_series.index.isin(self.first_213_feature_columns)
        ].to_dict()

        # 重点修改：确保保存的特征列表只包含前213列的特征
        # 保存数据时，只使用筛选后的特征
        analysis_data = {
            'disease_feature_importance': filtered_disease_importance,  # 筛选后的
            'global_feature_importance': filtered_global_importance,  # 筛选后的
            'selected_features': self.filtered_selected_features,  # 修改：只保存筛选后的特征
            'filtered_selected_features': self.filtered_selected_features,  # 筛选后的特征
            'disease_specific_features': self.filtered_disease_specific_features,  # 修改：只保存筛选后的特征
            'filtered_disease_specific_features': self.filtered_disease_specific_features,  # 筛选后的
            'feature_names': self.first_213_feature_columns,  # 修改：只使用前213列特征名称
            'disease_list': self.disease_list,
            'all_feature_columns': self.all_feature_columns,
            'first_213_feature_columns': self.first_213_feature_columns,
            'analysis_results': self.analysis_results
        }

        joblib.dump(analysis_data, f'{filepath}_results.pkl')

        # 保存为Excel文件
        with pd.ExcelWriter(f'{filepath}_results.xlsx') as writer:
            # 全局特征重要性（只显示前213列）
            filtered_global_importance = {k: v for k, v in self.global_feature_importance.items()
                                          if k in self.first_213_feature_columns}
            filtered_global_importance_df = pd.DataFrame({
                '特征': list(filtered_global_importance.keys()),
                '重要性': list(filtered_global_importance.values())
            }).sort_values('重要性', ascending=False)
            filtered_global_importance_df.to_excel(writer, sheet_name='全局特征重要性(前213列)', index=False)

            # 每个疾病的特征重要性（只显示前213列）
            for disease, importance_scores in self.disease_feature_importance.items():
                # 只选择前213列的特征
                filtered_scores = importance_scores[importance_scores.index.isin(self.first_213_feature_columns)]
                # 取前50个特征（和绘图保持一致）
                top_features = filtered_scores.sort_values(ascending=False).head(50)

                disease_df = pd.DataFrame({
                    '特征': top_features.index,
                    '重要性': top_features.values,
                    '排名': range(1, len(top_features) + 1)
                })
                sheet_name = disease[:28] + "前50特征"  # Excel工作表名称限制
                disease_df.to_excel(writer, sheet_name=sheet_name, index=False)

            # 重点：疾病特定特征选择（只保存前213列的前50个特征）
            disease_features_data = {}
            for disease in self.filtered_disease_specific_features:
                # 获取该疾病在前213列的所有特征重要性
                disease_scores = filtered_disease_importance.get(disease, pd.Series())
                # 选择前50个特征
                top_50_features = disease_scores.sort_values(ascending=False).head(50).index.tolist()
                disease_features_data[disease] = top_50_features

            disease_features_df = pd.DataFrame(dict([
                (k, pd.Series(v)) for k, v in disease_features_data.items()
            ]))
            disease_features_df.to_excel(writer, sheet_name='疾病特定特征(前213列前50)', index=False)

            # 前213列特征列表
            first_213_features_df = pd.DataFrame({
                '前213列特征': self.first_213_feature_columns,
                '序号': range(1, len(self.first_213_feature_columns) + 1)
            })
            first_213_features_df.to_excel(writer, sheet_name='前213列特征列表', index=False)

        print(f"分析结果已保存到:")
        print(f"  - {filepath}_results.pkl")
        print(f"  - {filepath}_results.xlsx")

    def load_analysis_results(self, filepath='six_category_analysis'):
        """加载分析结果"""
        try:
            analysis_data = joblib.load(f'{filepath}_results.pkl')
            self.disease_feature_importance = analysis_data['disease_feature_importance']
            self.global_feature_importance = analysis_data['global_feature_importance']
            self.selected_features = analysis_data['selected_features']
            self.filtered_selected_features = analysis_data.get('filtered_selected_features', [])
            self.disease_specific_features = analysis_data['disease_specific_features']
            self.filtered_disease_specific_features = analysis_data.get('filtered_disease_specific_features', {})
            self.feature_names = analysis_data['feature_names']
            self.disease_list = analysis_data['disease_list']
            self.all_feature_columns = analysis_data.get('all_feature_columns', [])
            self.first_213_feature_columns = analysis_data.get('first_213_feature_columns', [])
            self.analysis_results = analysis_data.get('analysis_results', {})

            print("分析结果加载成功!")
            return True
        except Exception as e:
            print(f"加载分析结果失败: {e}")
            return False


def run_six_category_analysis():
    """运行六大类特征分析"""
    print("=== 六大类特征重要性分析 ===")

    # 初始化分析器
    analyzer = SixCategoryFeatureAnalyzer(
        max_features_per_disease=50,
        global_max_features=200,
        keep_first_n_features=213
    )

    # 文件路径
    file_path = r"C:\Users\a'y\Desktop\生化血常规，尿检，全部.xlsx"
    print(f"分析文件: {file_path}")

    # 加载和预处理数据
    features, multi_labels = analyzer.load_and_preprocess_data(file_path)

    if features is None or multi_labels is None:
        print("数据加载失败")
        return None

    # 运行特征重要性分析
    analysis_result = analyzer.run_feature_importance_analysis(features, multi_labels)

    # 绘制可视化结果
    print(f"\n=== 生成可视化结果 ===")
    analyzer.plot_disease_specific_feature_importance(top_n=50)
    analyzer.plot_global_feature_importance(top_n=50)
    analyzer.plot_feature_coverage_heatmap()

    # 保存分析结果
    analyzer.save_analysis_results('six_category_analysis')

    print(f"\n=== 六大类特征分析完成 ===")
    print(f"共分析了 {len(analyzer.disease_list)} 种疾病")
    print(f"从所有特征中选择了 {len(analyzer.selected_features)} 个特征")
    print(f"筛选后保留了 {len(analyzer.filtered_selected_features)} 个特征")

    # 显示筛选后的特征
    print(f"\n筛选后的特征列表（前20个）:")
    for i, feature in enumerate(analyzer.filtered_selected_features[:20]):
        print(f"  {i + 1}. {feature}")
    if len(analyzer.filtered_selected_features) > 20:
        print(f"  ... 还有 {len(analyzer.filtered_selected_features) - 20} 个特征")

    return analyzer, analysis_result


if __name__ == "__main__":
    print("=== 六大类特征重要性分析系统 ===")
    print("开始运行六大类特征分析...")
    analyzer, results = run_six_category_analysis()