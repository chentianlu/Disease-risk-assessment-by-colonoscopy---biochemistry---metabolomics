import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import warnings
import matplotlib.pyplot as plt
import os
import sys
from sklearn.impute import KNNImputer
from matplotlib import font_manager
import warnings

# 设置本地模型路径（请根据实际路径修改）
LOCAL_MODEL_PATH = r"tabpfn-v2.5-classifier-v2.5_default.ckpt"

# TabPFN相关导入
try:
    from tabpfn import TabPFNClassifier
    import torch

    # 尝试修改TabPFN源代码中的下载行为
    try:
        # 导入TabPFN的内部模块
        import tabpfn.models.tabpfn as tabpfn_module

        # 修改下载标志为False
        tabpfn_module.download = False
        print("已禁用TabPFN自动下载")
    except Exception as e:
        print(f"修改TabPFN下载行为失败: {e}")
        print("将尝试其他方法...")

    # 设置环境变量，强制使用本地模型
    os.environ['HF_HOME'] = os.path.dirname(LOCAL_MODEL_PATH) if os.path.exists(LOCAL_MODEL_PATH) else '.'
    os.environ['TRANSFORMERS_CACHE'] = os.path.dirname(LOCAL_MODEL_PATH) if os.path.exists(LOCAL_MODEL_PATH) else '.'
    os.environ['TORCH_HOME'] = os.path.dirname(LOCAL_MODEL_PATH) if os.path.exists(LOCAL_MODEL_PATH) else '.'

    TABPFN_AVAILABLE = True
    TABPFN_VERSION = "available"
except ImportError:
    print("警告: TabPFN库未安装，请使用: pip install tabpfn")
    TABPFN_AVAILABLE = False
    TABPFN_VERSION = "not_installed"


# 定义可序列化的虚拟模型类（移动到try-catch外部）
class DummyTabPFNClassifier:
    def __init__(self, device='cpu'):
        self.device = device
        self.classes_ = None

    def fit(self, X, y):
        print("TabPFN未安装，使用虚拟模型训练")
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.random.randint(0, 2, size=len(X))

    def predict_proba(self, X):
        prob = np.random.rand(len(X), 2)
        return prob / prob.sum(axis=1, keepdims=True)


# 如果TabPFN不可用，设置TabPFNClassifier为虚拟类
if not TABPFN_AVAILABLE:
    TabPFNClassifier = DummyTabPFNClassifier

warnings.filterwarnings('ignore')

# 设置中文字体显示
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    font_path = font_manager.findfont(font_manager.FontProperties(family=['SimHei', 'Microsoft YaHei']))
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=font_path).get_name()
except:
    print("警告: 未找到中文字体，使用默认字体显示")


class WeightedSixCategoryPredictor:
    """六大类预测器 - 根据特征排名添加权重（带特征权重版本）"""

    def __init__(self, random_state=42, weight_method='exponential'):
        """
        初始化预测器

        参数:
        - random_state: 随机种子
        - weight_method: 权重分配方法，可选 'exponential'(指数衰减), 'linear'(线性衰减), 'rank_based'(基于排名)
        """
        self.models = {}
        self.scalers = {}
        self.selected_features = []
        self.disease_specific_features = {}
        self.feature_importances = {}  # 存储特征重要性分数
        self.feature_rankings = {}  # 存储特征排名
        self.feature_weights = {}  # 存储特征权重
        self.random_state = random_state
        self.weight_method = weight_method
        self.X_test = None
        self.y_test = None
        self.test_indices = None
        self.training_histories = {}  # 注意: TabPFN没有训练历史
        self.patient_predictions = None
        self.analysis_results = {}
        self.is_trained = False
        self.model_feature_columns = {}
        self.model_feature_weights = {}  # 存储每个模型使用的特征权重
        self.results = {}  # 存储每个模型的结果

    def calculate_feature_weights(self, features_list, feature_importance_scores=None):
        """
        根据特征排名计算权重

        参数:
        - features_list: 特征列表
        - feature_importance_scores: 特征重要性分数（可选）

        返回:
        - feature_weights_dict: 特征权重字典
        """
        n_features = len(features_list)
        weights = np.zeros(n_features)

        if self.weight_method == 'exponential':
            # 指数衰减权重: 排名第1的特征权重为1.0，后续特征权重指数衰减
            decay_rate = 0.9  # 衰减率
            for i in range(n_features):
                weights[i] = decay_rate ** i

        elif self.weight_method == 'linear':
            # 线性衰减权重: 从1.0线性衰减到0.2
            start_weight = 1.0
            end_weight = 0.3
            for i in range(n_features):
                weights[i] = start_weight - (start_weight - end_weight) * (i / max(n_features - 1, 1))

        elif self.weight_method == 'rank_based':
            # 基于重要性分数的权重（如果有重要性分数）
            if feature_importance_scores is not None and len(feature_importance_scores) == n_features:
                # 归一化重要性分数作为权重
                importance_array = np.array(feature_importance_scores)
                if importance_array.max() > 0:
                    weights = importance_array / importance_array.max()
                else:
                    weights = np.ones(n_features)
            else:
                # 如果没有重要性分数，使用指数衰减
                decay_rate = 0.9
                for i in range(n_features):
                    weights[i] = decay_rate ** i

        elif self.weight_method == 'inverse_rank':
            # 反排名权重: 排名越靠前权重越高
            for i in range(n_features):
                weights[i] = 1.0 / (i + 1)

        elif self.weight_method == 'equal_top_n':
            # 前N个特征权重相等，后面的特征权重较低
            top_n = min(10, n_features // 2)
            for i in range(n_features):
                if i < top_n:
                    weights[i] = 1.0
                else:
                    weights[i] = 0.3

        else:
            # 默认：所有特征权重相等
            weights = np.ones(n_features)

        # 创建特征权重字典
        feature_weights_dict = {}
        for i, feature in enumerate(features_list):
            feature_weights_dict[feature] = weights[i]

        return feature_weights_dict

    def load_selected_features(self):
        """加载特征选择结果，包括重要性分数"""
        print("=== 加载特征选择结果 ===")
        try:
            results_path = 'six_category_analysis_results.pkl'
            if not os.path.exists(results_path):
                print(f"警告: 找不到 {results_path}")
                return False

            analysis_data = joblib.load(results_path)
            print("成功加载六大类分析结果")

            self.selected_features = analysis_data.get('selected_features', [])
            self.disease_specific_features = analysis_data.get('disease_specific_features', {})
            self.feature_importances = analysis_data.get('feature_importances', {})
            self.feature_rankings = analysis_data.get('feature_rankings', {})
            self.analysis_results = analysis_data

            print(f"总共选择了 {len(self.selected_features)} 个特征")

            # 为每种疾病计算特征权重
            for disease, features_list in self.disease_specific_features.items():
                print(f"  {disease}: {len(features_list)} 个特定特征")
                if disease in self.feature_importances:
                    importance_scores = self.feature_importances[disease]
                    self.feature_weights[disease] = self.calculate_feature_weights(
                        features_list, importance_scores
                    )
                else:
                    self.feature_weights[disease] = self.calculate_feature_weights(features_list)

            # 为全局特征计算权重
            if self.selected_features:
                self.feature_weights['global'] = self.calculate_feature_weights(self.selected_features)

            print(f"\n特征权重分配方法: {self.weight_method}")
            return True

        except Exception as e:
            print(f"加载特征选择结果失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def apply_feature_weights(self, X, disease, top_n=None):
        """
        对特征数据应用权重

        参数:
        - X: 特征数据（DataFrame）
        - disease: 疾病名称
        - top_n: 只应用前N个特征的权重（None表示应用所有特征）

        返回:
        - X_weighted: 加权后的特征数据
        """
        if disease not in self.feature_weights:
            print(f"  警告: 未找到 {disease} 的特征权重，使用等权重")
            return X

        feature_weights = self.feature_weights[disease]
        X_weighted = X.copy()

        # 如果指定了top_n，只对前top_n个特征应用权重
        if top_n is not None:
            features_to_weight = list(feature_weights.keys())[:top_n]
        else:
            features_to_weight = list(feature_weights.keys())

        applied_count = 0
        for feature in features_to_weight:
            if feature in X_weighted.columns:
                weight = feature_weights[feature]
                X_weighted[feature] = X_weighted[feature] * weight
                applied_count += 1

        print(f"  对 {applied_count}/{len(features_to_weight)} 个特征应用了权重")

        return X_weighted

    def apply_knn_imputation(self, X_train, X_test=None, n_neighbors=5):
        """应用KNN插值处理缺失值"""
        imputer = KNNImputer(n_neighbors=n_neighbors)

        # 对训练集进行拟合和转换
        X_train_imputed = imputer.fit_transform(X_train)
        X_train_imputed_df = pd.DataFrame(X_train_imputed, columns=X_train.columns)

        if X_test is not None:
            # 对测试集进行转换（使用训练集的模型）
            X_test_imputed = imputer.transform(X_test)
            X_test_imputed_df = pd.DataFrame(X_test_imputed, columns=X_test.columns)
            return X_train_imputed_df, X_test_imputed_df

        return X_train_imputed_df

    def load_and_preprocess_data(self, train_file_path, test_file_path=None):
        """加载和预处理训练集和测试集数据"""
        print("=== 加载数据 ===")
        try:
            train_df = pd.read_excel(train_file_path, sheet_name='训练集')
            print(f"训练集数据读取成功! 形状: {train_df.shape}")

            if test_file_path and os.path.exists(test_file_path):
                test_df = pd.read_excel(test_file_path, sheet_name='六大类')
                print(f"测试集数据读取成功! 形状: {test_df.shape}")
            else:
                test_df = None
                if test_file_path:
                    print(f"警告: 测试集文件不存在: {test_file_path}")

        except Exception as e:
            print(f"读取数据失败: {e}")
            return None, None, None, None

        return self.preprocess_data(train_df, test_df)

    def preprocess_data(self, train_df, test_df=None):
        """预处理训练集和测试集数据"""

        # 创建六大类目标变量
        def create_targets(df):
            six_categories = [
                '内分泌系统疾病', '消化系统疾病', '循环系统疾病',
                '泌尿生殖系统疾病', '恶性肿瘤', '血液及造血器官疾病和涉及免疫机制的某些疾患'
            ]
            multi_labels = pd.DataFrame(0, index=df.index, columns=six_categories)

            for disease in six_categories:
                if disease in df.columns:
                    multi_labels[disease] = df[disease].apply(
                        lambda x: 1 if pd.notna(x) and str(x).strip() != '' else 0)
                else:
                    multi_labels[disease] = 0

            return multi_labels

        # 选择特征列
        def select_features(df):
            exclude_columns = ['编号', '内分泌系统疾病', '消化系统疾病', '循环系统疾病',
                               '泌尿生殖系统疾病', '恶性肿瘤', '血液及造血器官疾病和涉及免疫机制的某些疾患']
            feature_columns = [col for col in df.columns if col not in exclude_columns and col in df.columns]
            return df[feature_columns]

        train_features = select_features(train_df)
        train_multi_labels = create_targets(train_df)

        if test_df is not None:
            test_features = select_features(test_df)
            test_multi_labels = create_targets(test_df)
        else:
            test_features, test_multi_labels = None, None

        print(f"\n预处理完成!")
        print(f"训练集特征数量: {len(train_features.columns)}")
        print(f"训练集样本数量: {len(train_features)}")
        print(f"训练集疾病数量: {len(train_multi_labels.columns)}")

        if test_features is not None:
            print(f"测试集特征数量: {len(test_features.columns)}")
            print(f"测试集样本数量: {len(test_features)}")
            print(f"测试集疾病数量: {len(test_multi_labels.columns)}")

        # =========== 编码分类特征 ===========
        gender_columns = [col for col in train_features.columns if '性别' in col or 'sex' in col.lower()]

        for gender_col in gender_columns:
            if gender_col in train_features.columns:
                print(f"\n编码分类特征: {gender_col}")
                # 编码性别：男->1, 女->0
                train_features[gender_col] = train_features[gender_col].map({'男': 1, '女': 0})
                train_features[gender_col] = train_features[gender_col].fillna(0)
                print(f"  {gender_col}: '男'->1, '女'->0")

                if test_features is not None and gender_col in test_features.columns:
                    test_features[gender_col] = test_features[gender_col].map({'男': 1, '女': 0})
                    test_features[gender_col] = test_features[gender_col].fillna(0)
        # ===============================================

        return train_features, train_multi_labels, test_features, test_multi_labels

    def create_tabpfn_model(self):
        """创建TabPFN模型，强制使用本地模型"""
        if not TABPFN_AVAILABLE:
            print("警告: TabPFN未安装，使用虚拟模型")
            return DummyTabPFNClassifier()

        try:
            print(f"  尝试创建TabPFN模型，使用本地文件: {LOCAL_MODEL_PATH}")

            # 检查本地模型文件是否存在
            if os.path.exists(LOCAL_MODEL_PATH):
                print(f"  找到本地模型文件: {LOCAL_MODEL_PATH}")
            else:
                print(f"  警告: 本地模型文件不存在: {LOCAL_MODEL_PATH}")
                print("  将尝试使用虚拟模型")
                return DummyTabPFNClassifier()

            # 尝试多种方法创建模型，避免自动下载
            try:
                # 方法1: 直接指定模型路径
                print("  尝试方法1: 直接指定模型路径")
                model = TabPFNClassifier(device='cpu')

                # 尝试设置模型路径
                try:
                    # 尝试访问模型的内部属性来设置路径
                    if hasattr(model, 'model_path'):
                        model.model_path = LOCAL_MODEL_PATH
                    elif hasattr(model, 'transformer'):
                        if hasattr(model.transformer, 'model_path'):
                            model.transformer.model_path = LOCAL_MODEL_PATH
                except:
                    pass

            except Exception as e:
                print(f"  方法1失败: {e}")

                # 方法2: 使用try-catch包裹的虚拟模型
                print("  尝试方法2: 创建带错误处理的模型")

                class LocalTabPFNClassifier(TabPFNClassifier):
                    def __init__(self, model_path=LOCAL_MODEL_PATH, **kwargs):
                        # 设置环境变量和路径
                        os.environ['HF_HOME'] = os.path.dirname(model_path)
                        os.environ['TRANSFORMERS_CACHE'] = os.path.dirname(model_path)

                        # 尝试修改下载行为
                        try:
                            import tabpfn.models.tabpfn as tabpfn_module
                            tabpfn_module.download = False
                        except:
                            pass

                        super().__init__(**kwargs)

                        # 尝试设置模型路径
                        try:
                            if hasattr(self, 'model_path'):
                                self.model_path = model_path
                        except:
                            pass

                model = LocalTabPFNClassifier(device='cpu')

            print("  TabPFN模型创建成功")
            return model

        except Exception as e:
            print(f"创建TabPFN模型失败: {e}")
            print("  回退到虚拟模型")
            return DummyTabPFNClassifier()

    def prepare_data_with_smote(self, features, target, test_features=None, test_target=None):
        """准备训练数据 - 对训练集应用SMOTE，使用独立测试集"""
        if test_features is not None and test_target is not None:
            X_train = features
            y_train = target
            X_test = test_features
            y_test = test_target
            print("使用独立测试集进行验证")
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                features, target,
                test_size=0.2,
                random_state=self.random_state,
                stratify=target
            )
            print("从训练集分割测试集进行验证")

        print(f"  SMOTE前训练集样本分布: {pd.Series(y_train).value_counts().to_dict()}")

        try:
            X_train_resampled, y_train_resampled = X_train, y_train

        except Exception as e:
            print(f"  数据准备失败，使用原始数据: {e}")
            X_train_resampled, y_train_resampled = X_train, y_train

        return X_train_resampled, X_test, y_train_resampled, y_test

    def align_features(self, features, required_features):
        """对齐特征，确保特征顺序和完整性"""
        aligned_features = pd.DataFrame()

        for feature in required_features:
            if feature in features.columns:
                aligned_features[feature] = features[feature]
            else:
                # 如果特征不存在，用0填充
                print(f"  警告: 特征 '{feature}' 在数据中不存在，用0填充")
                aligned_features[feature] = 0

        return aligned_features

    def train_models_with_weighted_features(self, train_features, train_multi_labels, test_features=None,
                                            test_multi_labels=None, epochs=100, batch_size=32,
                                            apply_weight_top_n=None):
        """
        使用加权的特征训练模型

        参数:
        - apply_weight_top_n: 只对前N个重要特征应用权重
        """
        diseases = [
            '内分泌系统疾病', '消化系统疾病', '循环系统疾病',
            '泌尿生殖系统疾病', '恶性肿瘤', '血液及造血器官疾病和涉及免疫机制的某些疾患'
        ]

        results = {}
        self.models = {}
        self.scalers = {}
        self.test_indices = None
        self.model_feature_columns = {}
        self.model_feature_weights = {}

        for disease in diseases:
            print(f"\n{'=' * 60}")
            print(f"训练 {disease} 分类器 (带特征权重，使用TabPFN)")
            print(f"{'=' * 60}")

            # 获取该疾病的特定特征
            if disease in self.disease_specific_features:
                disease_features = self.disease_specific_features[disease][:50]  # 只取前30个特征
                print(f"为 {disease} 选择前30个特定特征")
            else:
                # 如果没有特定特征，使用全局选择的特征
                disease_features = self.selected_features[:50]  # 只取前30个特征
                print(f"使用全局选择的前30个特征")

            # 检查特征是否存在
            available_features = [f for f in disease_features if f in train_features.columns]
            missing_features = [f for f in disease_features if f not in train_features.columns]

            if missing_features:
                print(f"  警告: 缺失 {len(missing_features)} 个特征: {missing_features[:5]}...")

            if len(available_features) < 5:
                print(f"  可用特征过少 ({len(available_features)})，跳过")
                continue

            # 使用可用的特征，并按照原始顺序排序
            available_features = [f for f in disease_features if f in available_features]
            print(f"  使用 {len(available_features)} 个可用特征")

            # 保存模型使用的特征列
            model_key = disease
            self.model_feature_columns[model_key] = available_features

            # 准备数据 - 确保使用正确的特征顺序
            X_disease_train = train_features[available_features].copy()

            # =========== 应用特征权重 ===========
            print(f"  应用特征权重 (方法: {self.weight_method})")
            X_disease_train_weighted = self.apply_feature_weights(
                X_disease_train, disease, top_n=apply_weight_top_n
            )

            # 保存特征权重信息
            feature_weights_dict = {}
            for feature in available_features:
                if disease in self.feature_weights and feature in self.feature_weights[disease]:
                    feature_weights_dict[feature] = self.feature_weights[disease][feature]
                else:
                    feature_weights_dict[feature] = 1.0  # 默认权重

            self.model_feature_weights[disease] = feature_weights_dict
            # =====================================

            # 检查数据质量
            X_disease_train_weighted = self._check_data_quality(X_disease_train_weighted, disease)

            # 如果有测试集，使用测试集的特征
            if test_features is not None:
                # 对齐测试集特征
                X_disease_test = self.align_features(test_features, available_features)
                # 测试集也应用相同的权重
                X_disease_test_weighted = self.apply_feature_weights(
                    X_disease_test, disease, top_n=apply_weight_top_n
                )
                y_test = test_multi_labels[disease]
            else:
                X_disease_test_weighted = None
                y_test = None

            # 准备数据
            X_train, X_test, y_train, y_test = self.prepare_data_with_smote(
                X_disease_train_weighted, train_multi_labels[disease],
                X_disease_test_weighted, y_test
            )

            # 保存测试集信息
            if self.test_indices is None:
                self.X_test = X_test
                self.y_test = y_test
                self.test_indices = X_test.index

            # 训练TabPFN模型
            print(f"开始训练TabPFN模型: {disease}")
            model, scaler, metrics = self.train_single_tabpfn_model(
                X_train, X_test, y_train, y_test, disease
            )

            if model is not None:
                self.models[disease] = model
                self.scalers[disease] = scaler  # TabPFN不需要标准化器，但为了兼容性保留
                results[disease] = metrics

                # 打印特征权重信息
                top_features = list(feature_weights_dict.items())[:10]  # 只显示前10个
                print(f"  前10个特征的权重:")
                for feature, weight in top_features:
                    print(f"    {feature}: {weight:.4f}")
            else:
                print(f"  {disease} 模型训练失败，跳过")

        if self.models:
            self.is_trained = True
            print(f"\n成功训练了 {len(self.models)} 个带权重的TabPFN模型")
        else:
            print("\n所有模型训练都失败!")

        # 保存结果
        self.results = results
        return results

    def _check_data_quality(self, X, disease_name):
        """检查数据质量"""
        print(f"  检查 {disease_name} 数据质量:")

        # 检查NaN
        nan_count = X.isna().sum().sum()
        if nan_count > 0:
            print(f"    发现 {nan_count} 个NaN值，应用KNN插值")
            # 使用KNN插值而不是填充0
            X = self.apply_knn_imputation(X, n_neighbors=5)
        else:
            print(f"    未发现NaN值")

        # 检查无穷值
        inf_count = 0
        for col in X.columns:
            if np.any(np.isinf(X[col])):
                inf_count += 1
                print(f"    特征 {col} 包含无穷值，进行替换")
                X[col] = X[col].replace([np.inf, -np.inf], np.nan)

        if inf_count > 0:
            # 如果有无穷值替换为NaN，再次应用KNN插值
            X = self.apply_knn_imputation(X, n_neighbors=5)

        # 检查零方差特征
        zero_var_features = []
        for col in X.columns:
            if X[col].std() < 1e-10:
                zero_var_features.append(col)

        if zero_var_features:
            print(f"    发现 {len(zero_var_features)} 个零方差特征: {zero_var_features[:5]}...")
            # 添加微小噪声避免零方差
            for col in zero_var_features:
                X[col] = X[col] + np.random.normal(0, 1e-6, size=len(X))

        return X

    def train_single_tabpfn_model(self, X_train, X_test, y_train, y_test, disease):
        """使用TabPFN训练单个模型"""
        try:
            # 保存特征列顺序
            feature_columns = X_train.columns.tolist()

            # 检查数据
            if len(X_train) == 0 or len(y_train) == 0:
                print(f"  错误: 训练数据为空")
                return None, None, None

            # TabPFN不需要标准化，但为了兼容性创建一个空的scaler
            scaler = type('DummyScaler', (), {
                'fit_transform': lambda self, X: X.values if hasattr(X, 'values') else X,
                'transform': lambda self, X: X.values if hasattr(X, 'values') else X,
                'feature_names_': feature_columns
            })()

            # 检查是否有缺失值
            train_has_nan = X_train.isna().any().any()
            test_has_nan = X_test.isna().any().any() if X_test is not None else False

            if train_has_nan or test_has_nan:
                print(f"  检测到缺失值，应用KNN插值...")
                print(f"    训练集缺失值: {X_train.isna().sum().sum()} 个")
                if X_test is not None:
                    print(f"    测试集缺失值: {X_test.isna().sum().sum()} 个")

                # 应用KNN插值
                if X_test is not None:
                    X_train, X_test = self.apply_knn_imputation(X_train, X_test, n_neighbors=5)
                else:
                    X_train = self.apply_knn_imputation(X_train, n_neighbors=5)

                print(f"    KNN插值完成")
            else:
                print(f"  未检测到缺失值，跳过KNN插值")

            # 转换为numpy数组
            X_train_np = X_train.values if hasattr(X_train, 'values') else X_train
            X_test_np = X_test.values if hasattr(X_test, 'values') else X_test
            y_train_np = y_train.values if hasattr(y_train, 'values') else y_train

            # 创建并训练TabPFN模型
            print(f"  创建TabPFN模型...")
            model = self.create_tabpfn_model()

            print(f"  训练数据形状: {X_train_np.shape}")
            print(f"  测试数据形状: {X_test_np.shape}")

            # 检查是否需要使用虚拟模型
            if isinstance(model, DummyTabPFNClassifier):
                print("  使用虚拟模型进行训练...")
                model.fit(X_train_np, y_train_np)
                print("  虚拟模型训练完成")
            else:
                print(f"  开始训练TabPFN模型...")
                # TabPFN训练（非常快，通常只需要几秒钟）
                model.fit(X_train_np, y_train_np)
                print(f"  TabPFN模型训练完成")

            # 预测和评估
            y_pred_proba = model.predict_proba(X_test_np)[:, 1]  # 获取正类的概率
            y_pred = (y_pred_proba > 0.5).astype(int)

            # 计算指标
            metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)
            metrics['feature_count'] = X_train_np.shape[1]
            metrics['feature_columns'] = feature_columns
            metrics['weight_method'] = self.weight_method
            metrics['model_type'] = 'TabPFN' if not isinstance(model, DummyTabPFNClassifier) else 'DummyTabPFN'

            # 添加类别分布信息
            metrics['train_positive_count'] = np.sum(y_train_np)
            metrics['train_negative_count'] = len(y_train_np) - np.sum(y_train_np)
            metrics['test_positive_count'] = np.sum(y_test)
            metrics['test_negative_count'] = len(y_test) - np.sum(y_test)

            # 检查预测值是否合理
            unique_preds = np.unique(y_pred_proba)
            if len(unique_preds) == 1:
                print(f"  ⚠️ 警告: {disease} 预测值完全相同!")
            elif len(unique_preds) < 5:
                print(f"  ⚠️ 注意: {disease} 预测值变化较小 ({len(unique_preds)} 个不同值)")

            print(f"{disease} 测试集准确率: {metrics['accuracy']:.3f}")
            print(f"  特征数量: {X_train_np.shape[1]}")
            print(f"  训练集: {len(y_train_np)} 样本")
            print(f"  测试集: {len(y_test)} 样本")
            print(f"  使用的特征: {len(feature_columns)} 个")
            print(f"  权重方法: {self.weight_method}")
            print(f"  模型类型: {metrics['model_type']}")

            return model, scaler, metrics

        except Exception as e:
            print(f"  训练 {disease} TabPFN模型时出错: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None

    def calculate_metrics(self, y_true, y_pred, y_pred_proba):
        """计算评估指标"""
        try:
            auc = roc_auc_score(y_true, y_pred_proba)
        except:
            auc = 0.5

        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'auc': auc
        }

    def predict_with_weighted_features(self, test_features, test_multi_labels):
        """使用加权的特征进行快速批量预测"""
        if not self.is_trained:
            print("错误: 模型尚未训练，无法进行预测")
            return None

        diseases = [
            '内分泌系统疾病', '消化系统疾病', '循环系统疾病',
            '泌尿生殖系统疾病', '恶性肿瘤', '血液及造血器官疾病和涉及免疫机制的某些疾患'
        ]

        print(f"\n开始快速批量预测...")
        print(f"测试集大小: {len(test_features)} 个样本")

        # 预先准备所有患者的数据
        patient_results = []

        # 为每种疾病批量处理
        for disease in diseases:
            print(f"\n处理疾病: {disease}")

            if disease not in self.models:
                print(f"  跳过: 未找到 {disease} 的模型")
                continue

            model = self.models[disease]

            # 获取该模型使用的特征列
            if disease in self.model_feature_columns:
                required_features = self.model_feature_columns[disease]
            else:
                print(f"  跳过: 未找到 {disease} 的特征列")
                continue

            print(f"  使用 {len(required_features)} 个特征")

            # 对齐所有患者的特征
            X_aligned = self.align_features(test_features, required_features)

            # 应用特征权重
            if disease in self.model_feature_weights:
                feature_weights = self.model_feature_weights[disease]
                for feature in X_aligned.columns:
                    if feature in feature_weights:
                        X_aligned[feature] = X_aligned[feature] * feature_weights[feature]

            # 处理NaN值
            if X_aligned.isna().any().any():
                print(f"  处理缺失值...")
                X_aligned = X_aligned.fillna(0)

            try:
                # 转换为numpy数组
                X_np = X_aligned.values

                # 批量预测
                print(f"  进行批量预测...")
                pred_proba = model.predict_proba(X_np)[:, 1]  # 获取所有患者的正类概率
                pred_labels = (pred_proba > 0.5).astype(int)

                print(f"  预测完成: {len(pred_proba)} 个预测")

                # 初始化患者结果列表（如果是第一次循环）
                if disease == diseases[0]:
                    for i, idx in enumerate(test_features.index):
                        patient_results.append({'患者ID': idx})

                        # 添加真实标签
                        true_labels = test_multi_labels.loc[idx]
                        for d in diseases:
                            patient_results[i][f'真实_{d}'] = true_labels[d]

                # 添加该疾病的预测结果
                for i, idx in enumerate(test_features.index):
                    patient_results[i][f'预测概率_{disease}'] = pred_proba[i]
                    patient_results[i][f'预测标签_{disease}'] = pred_labels[i]

            except Exception as e:
                print(f"  批量预测失败: {e}")
                print(f"  尝试逐个预测...")
                # 回退到逐个预测
                for i, idx in enumerate(test_features.index):
                    if i >= len(patient_results):
                        patient_results.append({'患者ID': idx})

                    patient_data = test_features.loc[idx:idx]

                    # 对齐特征
                    X_aligned_single = self.align_features(patient_data, required_features)

                    # 应用特征权重
                    if disease in self.model_feature_weights:
                        feature_weights = self.model_feature_weights[disease]
                        for feature in X_aligned_single.columns:
                            if feature in feature_weights:
                                X_aligned_single[feature] = X_aligned_single[feature] * feature_weights[feature]

                    # 处理NaN值
                    X_aligned_single = X_aligned_single.fillna(0)

                    try:
                        X_np_single = X_aligned_single.values
                        pred_proba_single = model.predict_proba(X_np_single)[0][1]
                        pred_label_single = 1 if pred_proba_single > 0.5 else 0

                        patient_results[i][f'预测概率_{disease}'] = pred_proba_single
                        patient_results[i][f'预测标签_{disease}'] = pred_label_single
                    except Exception as e2:
                        print(f"    患者 {idx} 预测失败: {e2}")
                        patient_results[i][f'预测概率_{disease}'] = np.nan
                        patient_results[i][f'预测标签_{disease}'] = np.nan

        self.patient_predictions = pd.DataFrame(patient_results)
        print(f"\n预测完成! 共处理 {len(self.patient_predictions)} 个患者")
        return self.patient_predictions

    def save_feature_weight_summary(self, filename='feature_weight_summary'):
        """保存特征权重汇总"""
        if not self.model_feature_weights:
            print("没有特征权重信息可保存")
            return

        all_weight_data = []

        for disease, weight_dict in self.model_feature_weights.items():
            for feature, weight in weight_dict.items():
                all_weight_data.append({
                    '疾病': disease,
                    '特征': feature,
                    '权重': weight,
                    '权重方法': self.weight_method
                })

        weight_df = pd.DataFrame(all_weight_data)

        # 按疾病和权重排序
        weight_df = weight_df.sort_values(['疾病', '权重'], ascending=[True, False])

        # 保存到Excel
        weight_df.to_excel(f'{filename}.xlsx', index=False)
        print(f"特征权重汇总已保存到 {filename}.xlsx")

        return weight_df

    def save_results(self, test_features, test_multi_labels, filename='six_category_predictions_weighted'):
        """保存预测结果"""
        if not hasattr(self, 'patient_predictions') or self.patient_predictions.empty:
            if not self.is_trained:
                print("错误: 模型尚未训练，无法生成预测结果")
                return None
            print("使用快速批量预测方法...")
            self.patient_predictions = self.predict_with_weighted_features_fast(test_features, test_multi_labels)

        if self.patient_predictions is None or self.patient_predictions.empty:
            print("错误: 预测结果为空")
            return None

        # TabPFN没有训练历史，所以跳过绘制训练历史

        output_file = f'{filename}.xlsx'

        try:
            with pd.ExcelWriter(output_file) as writer:
                self.patient_predictions.to_excel(writer, sheet_name='测试集患者预测结果', index=False)

                # 计算性能指标（如果测试集有真实标签）
                if any(col.startswith('真实_') for col in self.patient_predictions.columns):
                    performance_summary = []
                    diseases = [
                        '内分泌系统疾病', '消化系统疾病', '循环系统疾病',
                        '泌尿生殖系统疾病', '恶性肿瘤', '血液及造血器官疾病和涉及免疫机制的某些疾患'
                    ]

                    for disease in diseases:
                        true_col = f'真实_{disease}'
                        pred_col = f'预测标签_{disease}'
                        proba_col = f'预测概率_{disease}'

                        if true_col in self.patient_predictions.columns and pred_col in self.patient_predictions.columns:
                            # 移除NaN值
                            valid_mask = self.patient_predictions[true_col].notna() & self.patient_predictions[
                                pred_col].notna()
                            if valid_mask.sum() > 0:
                                y_true = self.patient_predictions.loc[valid_mask, true_col]
                                y_pred = self.patient_predictions.loc[valid_mask, pred_col]

                                if proba_col in self.patient_predictions.columns:
                                    y_pred_proba = self.patient_predictions.loc[valid_mask, proba_col]
                                    try:
                                        auc = roc_auc_score(y_true, y_pred_proba)
                                    except:
                                        auc = 0.5
                                else:
                                    auc = 0.5

                                # 计算指标
                                accuracy = accuracy_score(y_true, y_pred)
                                precision = precision_score(y_true, y_pred, zero_division=0)
                                recall = recall_score(y_true, y_pred, zero_division=0)
                                f1 = f1_score(y_true, y_pred, zero_division=0)

                                performance_summary.append({
                                    '疾病': disease,
                                    '准确率': accuracy,
                                    '精确率': precision,
                                    '召回率': recall,
                                    'F1分数': f1,
                                    'AUC': auc,
                                    '样本数量': len(y_true),
                                    '阳性样本': int(y_true.sum()),
                                    '阴性样本': len(y_true) - int(y_true.sum())
                                })

                    if performance_summary:
                        performance_df = pd.DataFrame(performance_summary)
                        performance_df.to_excel(writer, sheet_name='模型性能汇总', index=False)

                # 保存特征权重信息
                if self.model_feature_weights:
                    weight_summary = []
                    for disease, weight_dict in self.model_feature_weights.items():
                        for i, (feature, weight) in enumerate(weight_dict.items()):
                            weight_summary.append({
                                '疾病': disease,
                                '特征': feature,
                                '权重': weight,
                                '权重排名': i + 1
                            })
                    weight_df = pd.DataFrame(weight_summary)
                    weight_df.to_excel(writer, sheet_name='特征权重汇总', index=False)

            print(f"结果已保存到 {output_file}")
            return self.patient_predictions

        except Exception as e:
            print(f"保存结果失败: {e}")
            # 尝试只保存预测结果
            try:
                self.patient_predictions.to_excel(output_file, index=False)
                print(f"预测结果已保存到 {output_file}（简化版）")
                return self.patient_predictions
            except:
                print(f"保存简化版结果也失败")
                return None

    def save_models(self, model_dir='six_category_models_weighted'):
        """保存所有训练好的模型和标准化器"""
        os.makedirs(model_dir, exist_ok=True)

        # TabPFN模型使用joblib保存
        for disease, model in self.models.items():
            model_path = os.path.join(model_dir, f'{disease}_model.pkl')
            try:
                joblib.dump(model, model_path)
                print(f"TabPFN模型已保存: {model_path}")
            except Exception as e:
                print(f"保存模型 {disease} 失败: {e}")
                # 如果模型无法序列化，保存模型权重信息
                weight_path = os.path.join(model_dir, f'{disease}_weights.txt')
                with open(weight_path, 'w') as f:
                    f.write(f"模型类型: {type(model).__name__}\n")
                    f.write(f"疾病: {disease}\n")
                    f.write(f"特征数量: {len(self.model_feature_columns.get(disease, []))}\n")

        # 保存特征选择结果和权重
        features_path = os.path.join(model_dir, 'selected_features_and_weights.pkl')
        joblib.dump({
            'selected_features': self.selected_features,
            'disease_specific_features': self.disease_specific_features,
            'feature_importances': self.feature_importances,
            'feature_rankings': self.feature_rankings,
            'feature_weights': self.feature_weights,
            'model_feature_weights': self.model_feature_weights,
            'weight_method': self.weight_method,
            'results': self.results
        }, features_path)
        print(f"特征选择和权重信息已保存: {features_path}")

        # 保存模型特征列信息
        feature_columns_path = os.path.join(model_dir, 'model_feature_columns.pkl')
        joblib.dump(self.model_feature_columns, feature_columns_path)
        print(f"模型特征列信息已保存: {feature_columns_path}")

        return model_dir

    def load_models(self, model_dir='six_category_models_weighted'):
        """加载已保存的模型和标准化器"""
        if not os.path.exists(model_dir):
            print(f"模型目录不存在: {model_dir}")
            return False

        try:
            for file in os.listdir(model_dir):
                if file.endswith('_model.pkl'):
                    disease = file.replace('_model.pkl', '')
                    model_path = os.path.join(model_dir, file)
                    self.models[disease] = joblib.load(model_path)
                    print(f"TabPFN模型已加载: {model_path}")

                elif file == 'selected_features_and_weights.pkl':
                    features_path = os.path.join(model_dir, file)
                    features_data = joblib.load(features_path)
                    self.selected_features = features_data.get('selected_features', [])
                    self.disease_specific_features = features_data.get('disease_specific_features', {})
                    self.feature_importances = features_data.get('feature_importances', {})
                    self.feature_rankings = features_data.get('feature_rankings', {})
                    self.feature_weights = features_data.get('feature_weights', {})
                    self.model_feature_weights = features_data.get('model_feature_weights', {})
                    self.weight_method = features_data.get('weight_method', 'exponential')
                    self.results = features_data.get('results', {})
                    print(f"特征选择和权重信息已加载: {features_path}")

                elif file == 'model_feature_columns.pkl':
                    feature_columns_path = os.path.join(model_dir, file)
                    self.model_feature_columns = joblib.load(feature_columns_path)
                    print(f"模型特征列信息已加载: {feature_columns_path}")

            self.is_trained = True
            return True

        except Exception as e:
            print(f"加载模型失败: {e}")
            import traceback
            traceback.print_exc()
            return False


def run_weighted_six_category_prediction(weight_method='exponential', force_retrain=True, apply_weight_top_n=None,
                                         use_top_features=50):  # 新增参数
    """运行带特征权重的六大类预测分析"""
    print("=== 六大类疾病预测分析（带特征权重，使用TabPFN） ===")
    print(f"权重分配方法: {weight_method}")
    print(f"使用前 {use_top_features} 个重要特征")

    # 禁用所有可能的下载
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    os.environ['HF_EVALUATE_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'

    # 检查本地模型文件是否存在
    if os.path.exists(LOCAL_MODEL_PATH):
        print(f"找到本地TabPFN模型文件: {LOCAL_MODEL_PATH}")
    else:
        print(f"警告: 未找到本地TabPFN模型文件: {LOCAL_MODEL_PATH}")
        print("请从以下链接下载:")
        print("https://huggingface.co/Prior-Labs/tabpfn_2_5/resolve/main/tabpfn-v2.5-classifier-v2.5_default.ckpt")
        print("将使用虚拟模型运行...")

    predictor = WeightedSixCategoryPredictor(random_state=42, weight_method=weight_method)

    # 检查TabPFN是否可用
    if not TABPFN_AVAILABLE:
        print("警告: TabPFN库未安装，将使用虚拟模型")
        print("请安装TabPFN: pip install tabpfn")

    model_dir = f'six_category_models_weighted_{weight_method}_top{use_top_features}'

    if not force_retrain and os.path.exists(model_dir) and predictor.load_models(model_dir):
        print("使用已保存的模型进行预测")
        predictor.is_trained = True
    else:
        print("训练新TabPFN模型（带特征权重）")

        # 加载特征选择结果
        if not predictor.load_selected_features():
            print("错误: 无法加载特征选择结果，请先运行特征选择分析")
            return None, None, None

        # 修改类内部的疾病特定特征和全局特征，只保留前30个
        if predictor.disease_specific_features:
            for disease in predictor.disease_specific_features:
                predictor.disease_specific_features[disease] = predictor.disease_specific_features[disease][
                    :use_top_features]

        if predictor.selected_features:
            predictor.selected_features = predictor.selected_features[:use_top_features]

        train_file_path = r"C:\Users\a'y\Desktop\生化血常规，尿检，全部.xlsx"
        test_file_path = r"C:\Users\a'y\Desktop\测试集.xlsx"

        print("正在加载训练集和测试集数据...")
        train_features, train_multi_labels, test_features, test_multi_labels = predictor.load_and_preprocess_data(
            train_file_path, test_file_path
        )

        if train_features is None or train_multi_labels is None:
            print("数据加载失败")
            return None, None, None

        print(f"\n数据加载成功!")
        print(f"训练集特征数量: {len(train_features.columns)}")
        print(f"训练集样本数量: {len(train_features)}")

        if test_features is not None:
            print(f"测试集特征数量: {len(test_features.columns)}")
            print(f"测试集样本数量: {len(test_features)}")

        print("\n开始训练带权重的TabPFN模型...")
        results = predictor.train_models_with_weighted_features(
            train_features, train_multi_labels, test_features, test_multi_labels,
            epochs=200, batch_size=32, apply_weight_top_n=apply_weight_top_n
        )

        if not results:
            print("模型训练失败，没有生成任何结果")
            return predictor, None, None

        print("\n保存TabPFN模型...")
        model_dir = predictor.save_models()

    print("\n加载测试集进行预测...")
    train_file_path = r"C:\Users\a'y\Desktop\生化血常规，尿检，全部.xlsx"
    test_file_path = r"C:\Users\a'y\Desktop\测试集.xlsx"

    _, _, test_features, test_multi_labels = predictor.load_and_preprocess_data(
        train_file_path, test_file_path
    )

    if test_features is None or test_multi_labels is None:
        print("测试集数据加载失败")
        return None, None, None

    print("\n生成预测结果...")
    predictions = predictor.predict_with_weighted_features(test_features, test_multi_labels)

    if predictions is None:
        print("预测失败")
        return None, None, None

    print("\n保存结果...")
    filename = f'six_category_predictions_weighted_{weight_method}'
    if apply_weight_top_n:
        filename += f'_top{apply_weight_top_n}'

    final_predictions = predictor.save_results(test_features, test_multi_labels, filename)

    # 保存特征权重汇总
    predictor.save_feature_weight_summary(f'feature_weights_{weight_method}')

    print(f"\n=== 带特征权重的TabPFN分析完成 ===")
    print(f"权重分配方法: {weight_method}")
    print(f"模型已保存到: {model_dir}")
    print(f"结果已保存到 {filename}.xlsx")

    # 检查预测结果的合理性
    if final_predictions is not None:
        print(f"\n预测结果检查:")
        pred_cols = [col for col in final_predictions.columns if '预测概率_' in col]
        for col in pred_cols:
            values = final_predictions[col].dropna()
            if len(values) > 0:
                unique_vals = values.nunique()
                if unique_vals == 1:
                    print(f"  ⚠️ {col}: 所有预测值相同 ({values.iloc[0]:.6f})")
                elif unique_vals < 5:
                    print(f"  ⚠️ {col}: 预测值变化较小 ({unique_vals} 个不同值)")
                else:
                    print(f"  ✅ {col}: 预测值正常变化 ({unique_vals} 个不同值)")

    return predictor, final_predictions, results if 'results' in locals() else predictor.results


def compare_weight_methods():
    """比较不同的权重分配方法"""
    weight_methods = ['exponential', 'linear', 'inverse_rank', 'equal_top_n']

    comparison_results = {}

    for method in weight_methods:
        print(f"\n{'=' * 60}")
        print(f"测试权重分配方法: {method}")
        print(f"{'=' * 60}")

        predictor, predictions, results = run_weighted_six_category_prediction(
            weight_method=method, force_retrain=True
        )

        if results:
            comparison_results[method] = results

    # 比较结果
    print(f"\n{'=' * 60}")
    print("不同权重分配方法比较结果")
    print(f"{'=' * 60}")

    for disease in ['内分泌系统疾病', '消化系统疾病', '循环系统疾病', '泌尿生殖系统疾病', '恶性肿瘤',
                    '血液及造血器官疾病和涉及免疫机制的某些疾患']:
        print(f"\n{disease}:")
        print(f"{'方法':<15} {'准确率':<10} {'F1分数':<10} {'AUC':<10}")
        print("-" * 45)

        for method in weight_methods:
            if method in comparison_results and disease in comparison_results[method]:
                metrics = comparison_results[method][disease]
                print(f"{method:<15} {metrics['accuracy']:.4f}    {metrics['f1']:.4f}    {metrics['auc']:.4f}")

    return comparison_results


if __name__ == "__main__":
    print("=== 六大类疾病预测系统（带特征权重，使用TabPFN） ===")
    print("开始运行带特征权重的六大类预测分析...")
    TOP_FEATURES = 50

    # 检查TabPFN是否可用
    if not TABPFN_AVAILABLE:
        print("警告: TabPFN库未安装")
        print("请使用以下命令安装: pip install tabpfn")
        print("将继续使用虚拟模型运行...")

    # 运行带特征权重的预测
    weight_method = 'equal_top_n'  # 可以改为 'linear', 'inverse_rank', 'equal_top_n'
    apply_weight_top_n = None  # 设置为数字，如20，表示只对前20个重要特征应用权重

    predictor, predictions, results = run_weighted_six_category_prediction(
        weight_method=weight_method,
        force_retrain=True,
        apply_weight_top_n=apply_weight_top_n
    )

    if predictions is not None:
        print(f"\n测试集 {len(predictions)} 个患者的预测结果:")
        display_columns = ['患者ID']
        for col in predictions.columns:
            if '预测概率_' in col or '真实_' in col:
                display_columns.append(col)

        print(predictions[display_columns].head().to_string(index=False))

    # 可选：比较不同的权重分配方法
    #comparison_results = compare_weight_methods()

    print("\n分析完成!")


