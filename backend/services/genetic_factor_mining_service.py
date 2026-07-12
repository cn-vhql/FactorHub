"""
遗传算法因子挖掘服务 - 使用遗传算法自动发现最优因子
"""
import logging
from typing import List, Dict, Callable, Optional
import pandas as pd
import numpy as np
import random
import re

# 配置日志
logger = logging.getLogger(__name__)

try:
    from deap import base, creator, tools, algorithms
    DEAP_AVAILABLE = True
except ImportError:
    DEAP_AVAILABLE = False
    logger.warning("DEAP库未安装，遗传算法功能将不可用")

from backend.services.factor_generator_service import factor_generator_service
from backend.services.factor_validation_service import factor_validation_service
from backend.formula_engine.code_normalizer import normalize_formula_code


class GeneticFactorMiningService:
    """遗传算法因子挖掘服务"""

    def __init__(
        self,
        base_factors: List[str],
        data: pd.DataFrame,
        return_column: str = "return",
        population_size: int = 50,
        n_generations: int = 20,
        cx_prob: float = 0.7,
        mut_prob: float = 0.3,
        factor_calculator=None,
    ):
        """
        初始化遗传算法挖掘服务

        Args:
            base_factors: 基础因子代码列表（如 ["RSI(close, 14)", "close / open"]）
            data: 数据DataFrame（包含OHLCV列）
            return_column: 收益率列名
            population_size: 种群大小
            n_generations: 迭代代数
            cx_prob: 交叉概率
            mut_prob: 变异概率
            factor_calculator: 因子计算器实例（可选）
        """
        if not DEAP_AVAILABLE:
            raise ImportError("DEAP库未安装，请运行: pip install DEAP")

        self.base_factor_codes = base_factors
        self.data = data
        self.return_column = return_column
        self.population_size = population_size
        self.n_generations = n_generations
        self.cx_prob = cx_prob
        self.mut_prob = mut_prob
        self.factor_calculator = factor_calculator

        # 准备收益率数据
        self.return_values = data[return_column] if return_column in data.columns else None

        # 预计算基础因子值（存储为字典，方便表达式计算）
        self.base_factor_values = {}
        self._precompute_base_factors()
        if not self.base_factor_values:
            raise ValueError("所选基础因子均无法转换为可组合的可执行表达式")

        # 初始化遗传算法
        self._setup_genetic_algorithm()

    def _precompute_base_factors(self):
        """预计算所有基础因子的值"""
        if self.factor_calculator is None:
            # 如果没有提供计算器，使用默认的
            from backend.services.factor_service import factor_service
            self.factor_calculator = factor_service.calculator

        logger.info(f"预计算 {len(self.base_factor_codes)} 个基础因子...")

        for i, factor_code in enumerate(self.base_factor_codes):
            try:
                composable_code = self._build_composable_factor_code(factor_code)
                if not composable_code:
                    logger.warning(
                        "  [%s/%s] %s: 跳过，不支持组合为候选表达式",
                        i + 1,
                        len(self.base_factor_codes),
                        factor_code,
                    )
                    continue

                factor_values = self.factor_calculator.calculate(self.data, factor_code)
                if factor_values is not None and len(factor_values.dropna()) > 0:
                    # 使用唯一的变量名（避免代码中的特殊字符）
                    var_name = f"factor_{i}"
                    self.base_factor_values[var_name] = {
                        "code": composable_code,
                        "source_code": factor_code,
                        "values": factor_values
                    }
                    logger.info(
                        f"  [{i+1}/{len(self.base_factor_codes)}] {factor_code}: "
                        f"{len(factor_values.dropna())} 个有效值，可组合表达式={composable_code}"
                    )
                else:
                    logger.warning(f"  [{i+1}/{len(self.base_factor_codes)}] {factor_code}: 计算失败或无有效值")
            except Exception as e:
                logger.warning(f"  [{i+1}/{len(self.base_factor_codes)}] {factor_code}: 计算出错 - {e}")

        logger.info(f"成功预计算 {len(self.base_factor_values)} 个基础因子")

    def _setup_genetic_algorithm(self):
        """设置遗传算法"""
        # 定义适应度函数（最大化IC绝对值和IR）
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))

        # 定义个体（因子表达式）
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)

        # 创建工具箱
        self.toolbox = base.Toolbox()

        # 注册个体生成函数
        self.toolbox.register(
            "individual",
            self._generate_random_individual,
        )

        # 注册种群生成函数
        self.toolbox.register(
            "population",
            tools.initRepeat,
            list,
            self.toolbox.individual,
        )

        # 注册遗传操作
        self.toolbox.register("mate", self._crossover)
        self.toolbox.register("mutate", self._mutate, indpb=0.2)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

        # 注册评估函数
        self.toolbox.register("evaluate", self._evaluate_factor)

        # 统计信息
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)

        # 进度回调函数（可选）
        self.progress_callback = None

    def set_progress_callback(self, callback):
        """设置进度回调函数

        Args:
            callback: 回调函数，签名为 callback(generation, total_generations, best_fitness, avg_fitness)
        """
        self.progress_callback = callback

    def _generate_random_individual(self):
        """生成随机个体（因子表达式）"""
        # 使用预计算的因子变量名
        var_names = list(self.base_factor_values.keys())

        if not var_names:
            # 如果没有可用的基础因子，返回空个体
            individual = creator.Individual()
            individual.extend(["close"])
            return individual

        # 随机选择表达式类型
        expr_type = random.choice(["single", "binary", "unary"])

        if expr_type == "single" or len(var_names) == 1:
            # 单个因子
            var = random.choice(var_names)
            individual = creator.Individual()
            individual.extend([var])
            return individual

        elif expr_type == "binary" and len(var_names) >= 2:
            # 二元运算组合
            var1, var2 = random.sample(var_names, 2)
            op = random.choice(["+", "-", "*", "/"])
            individual = creator.Individual()
            individual.extend([f"({var1} {op} {var2})"])
            return individual

        else:  # unary
            # 一元运算
            var = random.choice(var_names)
            func = random.choice(["np.log", "np.sqrt", "np.abs", "rank"])
            if func == "rank":
                expr = f"({var}.rank(pct=True))"
            else:
                expr = f"{func}({var})"
            individual = creator.Individual()
            individual.extend([expr])
            return individual

    def _evaluate_factor(self, individual: list) -> tuple:
        """
        评估因子适应度

        Args:
            individual: 个体（因子表达式列表）

        Returns:
            适应度值元组
        """
        expr = individual[0]

        # 尝试计算因子值
        try:
            factor_values = self._compute_factor_expression(expr)

            if factor_values is None or len(factor_values.dropna()) < 10:
                return (0.0,)

            # 验证因子
            if self.return_values is not None:
                validation = factor_validation_service.validate_factor(
                    factor_values=factor_values,
                    return_values=self.return_values,
                    existing_factors=None,
                )

                # 适应度 = 综合得分
                fitness = validation["score"] / 100.0
            else:
                # 如果没有收益率数据，使用因子的标准差作为适应度
                fitness = factor_values.std() / (factor_values.mean() + 1e-8)

            return (fitness,)

        except Exception as e:
            return (0.0,)

    def _compute_factor_expression(self, expr: str) -> Optional[pd.Series]:
        """
        计算因子表达式的值

        Args:
            expr: 因子表达式（包含变量名如 factor_0, factor_1）

        Returns:
            因子值序列
        """
        try:
            # 构建安全的执行环境
            # 将变量名映射到预计算的因子值Series
            safe_dict = {}

            # 添加预计算的基础因子值到环境
            for var_name, factor_info in self.base_factor_values.items():
                safe_dict[var_name] = factor_info["values"]

            # 添加常用的numpy和pandas函数
            safe_dict["np"] = np
            safe_dict["pd"] = pd

            # 添加原始数据列（close, open, high, low, volume）
            for col in ["open", "high", "low", "close", "volume"]:
                if col in self.data.columns:
                    safe_dict[col] = self.data[col]

            # 如果没有可用的数据，返回None
            if not safe_dict:
                logger.warning("没有可用的因子数据用于表达式计算")
                return None

            # 使用eval计算表达式
            try:
                result = eval(expr, {"__builtins__": {}}, safe_dict)

                # 确保返回的是Series
                if isinstance(result, pd.Series):
                    # 如果表达式包含log或sqrt，处理无效值
                    if "log" in expr or "sqrt" in expr:
                        # 替换-inf和inf为NaN，然后用前向填充
                        result = result.replace([np.inf, -np.inf], np.nan)
                        # 可选：进行简单的处理，例如用0填充NaN
                        # result = result.fillna(0)
                elif isinstance(result, (int, float, np.number)):
                    # 如果是标量值，返回与数据长度相同的Series
                    return pd.Series([float(result)] * len(self.data), index=self.data.index)
                else:
                    logger.warning(f"表达式返回了不支持的类型: {type(result)}")
                    return None

                # 检查结果是否有效
                if isinstance(result, pd.Series):
                    valid_count = result.notna().sum()
                    if valid_count == 0:
                        logger.warning(f"表达式计算结果全部为NaN: {expr}")
                        return None
                    # 如果大部分值都是NaN，也认为无效
                    if valid_count < len(result) * 0.1:  # 少于10%有效值
                        logger.warning(f"表达式计算结果大部分为NaN ({valid_count}/{len(result)}): {expr}")
                        return None

                return result

            except NameError as e:
                logger.warning(f"表达式中包含未定义的变量: {e}")
                return None
            except Exception as e:
                logger.warning(f"计算表达式失败: {e}")
                return self._compute_binary_operation(expr)

        except Exception as e:
            logger.warning(f"计算因子表达式失败 {expr}: {e}")
            return None

    def _compute_binary_operation(self, expr: str) -> Optional[pd.Series]:
        """
        计算简单的二元运算表达式（备用方法）

        Args:
            expr: 因子表达式

        Returns:
            因子值序列
        """
        try:
            # 去除外层括号
            expr = expr.strip()
            if expr.startswith("(") and expr.endswith(")"):
                expr = expr[1:-1].strip()

            # 尝试匹配二元运算模式: factor1 op factor2
            import re
            pattern = r'^(\w+)\s*([+\-*/])\s*(\w+)$'
            match = re.match(pattern, expr)

            if match:
                left_factor = match.group(1)
                operator = match.group(2)
                right_factor = match.group(3)

                left = self._get_factor_value(left_factor)
                right = self._get_factor_value(right_factor)

                if left is not None and right is not None:
                    # 对齐索引
                    aligned_data = pd.DataFrame({
                        'left': left,
                        'right': right
                    }).dropna()

                    if len(aligned_data) == 0:
                        return None

                    if operator == '+':
                        result = aligned_data['left'] + aligned_data['right']
                    elif operator == '-':
                        result = aligned_data['left'] - aligned_data['right']
                    elif operator == '*':
                        result = aligned_data['left'] * aligned_data['right']
                    elif operator == '/':
                        result = aligned_data['left'] / (aligned_data['right'] + 1e-8)
                    else:
                        return None

                    return result

            # 如果无法解析为二元运算，尝试直接获取因子值
            return self._get_factor_value(expr)

        except Exception as e:
            logger.warning(f"计算二元运算失败 {expr}: {e}")
            return None

    def _get_factor_value(self, factor_name: str) -> Optional[pd.Series]:
        """获取因子值"""
        # 去除空格
        factor_name = factor_name.strip()

        # 检查是否是预计算的因子变量名（如 factor_0, factor_1）
        if factor_name in self.base_factor_values:
            return self.base_factor_values[factor_name]["values"]

        # 检查是否是原始数据列
        if factor_name in self.data.columns:
            return self.data[factor_name]

        # 如果都不匹配，返回None
        logger.warning(f"未找到因子: {factor_name}")
        return None

    def _extract_inner_expression(self, expr: str) -> str:
        """提取最内层的括号表达式"""
        # 找到第一个完整的括号对
        start = expr.find("(")
        if start == -1:
            return expr

        count = 1
        end = start + 1
        while end < len(expr) and count > 0:
            if expr[end] == "(":
                count += 1
            elif expr[end] == ")":
                count -= 1
            end += 1

        return expr[start + 1:end - 1]

    def _split_binary_operation(self, expr: str) -> List[str]:
        """分割二元运算表达式"""
        operators = ["+", "-", "*", "/"]

        for op in operators:
            if op in expr:
                # 简单分割（实际需要更复杂的解析）
                parts = expr.split(op)
                if len(parts) == 2:
                    return [p.strip() for p in parts]

        return []

    def _crossover(self, ind1, ind2):
        """交叉操作"""
        # 由于每个Individual只有一个表达式元素，我们简单地交换表达式中的因子
        expr1 = ind1[0]
        expr2 = ind2[0]

        # 获取变量名列表
        var_names = list(self.base_factor_values.keys())

        if not var_names:
            # 如果没有可用的因子变量，直接交换整个表达式
            return (ind2, ind1)

        # 提取表达式中的变量并交换
        vars1 = [v for v in var_names if v in expr1]
        vars2 = [v for v in var_names if v in expr2]

        if vars1 and vars2 and random.random() < 0.7:
            # 70%概率交换变量
            var1 = random.choice(vars1)
            var2 = random.choice(vars2)

            new_expr1 = expr1.replace(var1, var2)
            new_expr2 = expr2.replace(var2, var1)

            # 创建新的Individual对象
            child1 = creator.Individual()
            child1.extend([new_expr1])
            child2 = creator.Individual()
            child2.extend([new_expr2])
            return (child1, child2)
        else:
            # 30%概率直接交换整个表达式
            return (ind2, ind1)

    def _mutate(self, individual, indpb: float) -> tuple:
        """变异操作"""
        expr = individual[0]

        # 获取变量名列表
        var_names = list(self.base_factor_values.keys())

        # 可能的变异操作：
        # 1. 更换运算符
        # 2. 更换因子变量
        # 3. 添加统计函数

        if random.random() < 0.3:
            # 更换运算符
            operators = ["+", "-", "*", "/"]
            for op in operators:
                if op in expr and random.random() < indpb:
                    new_op = random.choice([o for o in operators if o != op])
                    expr = expr.replace(op, new_op, 1)
                    break

        if random.random() < 0.3 and var_names:
            # 更换因子变量
            for var in var_names:
                if var in expr and random.random() < indpb:
                    other_vars = [v for v in var_names if v != var]
                    if other_vars:
                        new_var = random.choice(other_vars)
                        expr = expr.replace(var, new_var, 1)
                        break

        if random.random() < 0.2 and var_names:
            # 添加/移除一元函数
            if random.random() < 0.5:
                # 添加函数
                matched_vars = [v for v in var_names if v in expr]
                if matched_vars:
                    var = random.choice(matched_vars)
                    func = random.choice(["np.log", "np.sqrt", "np.abs"])
                    expr = self._replace_variable(expr, var, f"{func}({var})", 1)
            else:
                # 如果表达式整体被一元函数包裹，则移除外层函数
                for func in ["np.log", "np.sqrt", "np.abs"]:
                    if expr.startswith(f"{func}(") and expr.endswith(")"):
                        # 提取内部表达式
                        inner = expr[len(func)+1:-1]
                        expr = inner
                        break

        # 创建新的Individual对象并返回
        mutated = creator.Individual()
        mutated.extend([expr])
        return (mutated,)

    def _convert_expression_to_code(self, expr: str) -> str:
        """
        将内部变量表达式转换为实际因子代码

        Args:
            expr: 包含内部变量名的表达式（如 "(factor_0 * 1.5)"）

        Returns:
            实际因子代码表达式（如 "(RSI(close, 14) * 1.5)"）
        """
        converted_expr = expr

        # 将内部变量名替换为实际因子代码
        for var_name, factor_info in sorted(
            self.base_factor_values.items(),
            key=lambda item: len(item[0]),
            reverse=True,
        ):
            actual_code = factor_info["code"]
            converted_expr = self._replace_variable(
                converted_expr,
                var_name,
                f"({actual_code})",
            )

        return converted_expr

    def mine_factors(self) -> Dict:
        """
        执行因子挖掘

        Returns:
            挖掘结果
        """
        if not DEAP_AVAILABLE:
            return {
                "success": False,
                "message": "DEAP库未安装",
                "best_factors": [],
            }

        logger.info(f"开始遗传算法因子挖掘...")
        logger.info(f"种群大小: {self.population_size}")
        logger.info(f"迭代代数: {self.n_generations}")

        # 初始化种群
        population = self.toolbox.population(n=self.population_size)

        # 评估初始种群
        fitnesses = list(map(self.toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        # 创建Hall of Fame保存最优个体
        halloffame = tools.HallOfFame(10)
        halloffame.update(population)

        # 记录统计信息
        logbook = tools.Logbook()
        logbook.record(gen=0, **self.stats.compile(population))

        # 开始进化循环
        for gen in range(1, self.n_generations + 1):
            # 选择下一代
            offspring = self.toolbox.select(population, len(population))
            offspring = list(map(self.toolbox.clone, offspring))

            # 交叉
            for i in range(1, len(offspring), 2):
                if random.random() < self.cx_prob:
                    offspring[i - 1], offspring[i] = self.toolbox.mate(offspring[i - 1], offspring[i])
                    del offspring[i - 1].fitness.values
                    del offspring[i].fitness.values

            # 变异
            for i in range(len(offspring)):
                if random.random() < self.mut_prob:
                    offspring[i], = self.toolbox.mutate(offspring[i])
                    del offspring[i].fitness.values

            # 评估新的个体
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(map(self.toolbox.evaluate, invalid_ind))
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # 替换种群
            population[:] = offspring

            # 更新Hall of Fame
            halloffame.update(population)

            # 记录统计信息
            record = self.stats.compile(population)
            logbook.record(gen=gen, **record)

            # 调用进度回调
            if self.progress_callback:
                best_fitness = record.get("max", 0.0)
                avg_fitness = record.get("avg", 0.0)
                self.progress_callback(gen, self.n_generations, best_fitness, avg_fitness)

            logger.info(f"Generation {gen}/{self.n_generations} - Best: {record.get('max', 0):.4f}, Avg: {record.get('avg', 0):.4f}")

        # 提取最优因子
        best_factors = []
        for i, individual in enumerate(halloffame):
            # 获取内部表达式
            internal_expr = individual[0]

            # 转换为实际因子代码
            actual_expr = self._convert_expression_to_code(internal_expr)

            factor_info = {
                "rank": i + 1,
                "expression": actual_expr,
                "internal_expression": internal_expr,
                "fitness": float(individual.fitness.values[0]),
            }

            # 重新计算详细指标
            try:
                factor_values = self._compute_factor_expression(internal_expr)
                if factor_values is not None and self.return_values is not None:
                    validation = factor_validation_service.validate_factor(
                        factor_values=factor_values,
                        return_values=self.return_values,
                    )
                    factor_info["validation"] = validation
            except Exception as e:
                # 记录个体评估失败的异常，但继续处理其他个体
                import logging
                logging.getLogger(__name__).warning(f"因子个体评估失败: {e}")

            best_factors.append(factor_info)

        return {
            "success": True,
            "best_factors": best_factors,
            "logbook": logbook,
            "final_population": population,
        }

    def evolve_factor(
        self,
        initial_expression: str,
        n_generations: int = 10,
    ) -> Dict:
        """
        基于初始表达式进化优化

        Args:
            initial_expression: 初始因子表达式
            n_generations: 进化代数

        Returns:
            进化结果
        """
        if not DEAP_AVAILABLE:
            return {
                "success": False,
                "message": "DEAP库未安装",
            }

        # 创建初始种群
        population = [self._generate_random_individual() for _ in range(self.population_size - 1)]
        # 将初始表达式转换为Individual对象
        initial_individual = creator.Individual()
        initial_individual.extend([initial_expression])
        population.insert(0, initial_individual)

        # 创建Hall of Fame
        halloffame = tools.HallOfFame(5)

        # 运行进化
        population, logbook = algorithms.eaSimple(
            population,
            self.toolbox,
            cxpb=self.cx_prob,
            mutpb=self.mut_prob,
            ngen=n_generations,
            stats=self.stats,
            halloffame=halloffame,
            verbose=False,
        )

        # 返回最优个体
        best = halloffame[0]

        return {
            "success": True,
            "original_expression": initial_expression,
            "evolved_expression": self._convert_expression_to_code(best[0]),
            "internal_evolved_expression": best[0],
            "original_fitness": self._evaluate_factor([initial_expression])[0],
            "evolved_fitness": float(best.fitness.values[0]),
            "improvement": float(best.fitness.values[0]) - self._evaluate_factor([initial_expression])[0],
        }

    def _replace_variable(
        self,
        expression: str,
        variable_name: str,
        replacement: str,
        count: int = 0,
    ) -> str:
        """按变量边界安全替换内部表达式变量。"""
        pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(variable_name)}(?![A-Za-z0-9_])")
        return pattern.sub(replacement, expression, count=count)

    def _build_composable_factor_code(self, factor_code: str) -> Optional[str]:
        """将基础因子代码转换为可继续组合的单表达式。"""
        stripped = (factor_code or "").strip()
        if not stripped:
            return None

        normalized = normalize_formula_code(
            stripped,
            self.factor_calculator.infer_formula_type(stripped)
            if self.factor_calculator is not None
            else None,
        )
        resolved_type = (
            self.factor_calculator.infer_formula_type(normalized)
            if self.factor_calculator is not None
            else None
        )

        if resolved_type == "python":
            if normalized.startswith("def calculate_factor"):
                logger.warning("Python 函数型因子暂不支持作为可保存候选因子的基础表达式: %s", stripped)
                return None
            return normalized

        if resolved_type != "mylanguage":
            return None

        return normalized


# 全局遗传算法挖掘服务实例（需要初始化参数）
def create_genetic_mining_service(
    base_factors: List[str],
    data: pd.DataFrame,
    factor_calculator=None,
    **kwargs
) -> GeneticFactorMiningService:
    """
    创建遗传算法挖掘服务

    Args:
        base_factors: 基础因子代码列表（如 ["RSI(close, 14)", "close / open"]）
        data: 包含OHLCV的数据
        factor_calculator: 因子计算器实例（可选）
        **kwargs: 其他参数

    Returns:
        遗传算法挖掘服务实例
    """
    return GeneticFactorMiningService(
        base_factors=base_factors,
        data=data,
        factor_calculator=factor_calculator,
        **kwargs
    )
