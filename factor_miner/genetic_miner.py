"""
遗传算法因子挖掘器
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import random
import copy
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from utils.logger import logger
from utils.config import DEFAULT_CONFIG
from utils.helpers import calculate_ic
from .factor_generator import FactorGenerator

@dataclass
class FactorIndividual:
    """因子个体"""
    name: str
    expression: str
    factor_values: pd.Series
    fitness: float
    generation: int = 0

class GeneticFactorMiner:
    """遗传算法因子挖掘器"""

    def __init__(self,
                 population_size: int = 50,
                 generations: int = 20,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.7,
                 elitism_rate: float = 0.1,
                 max_complexity: int = 5):
        self.logger = logger
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_rate = elitism_rate
        self.max_complexity = max_complexity

        self.factor_generator = FactorGenerator()
        self.operators = self._initialize_operators()
        self.base_factors = ['close', 'volume', 'high', 'low', 'open', 'close_ma5', 'close_ma20', 'volume_ma5']

    def _initialize_operators(self) -> List[Dict]:
        """初始化操作符"""
        return [
            {'name': 'add', 'symbol': '+', 'arity': 2, 'weight': 1.0},
            {'name': 'subtract', 'symbol': '-', 'arity': 2, 'weight': 1.0},
            {'name': 'multiply', 'symbol': '*', 'arity': 2, 'weight': 0.8},
            {'name': 'divide', 'symbol': '/', 'arity': 2, 'weight': 0.6},
            {'name': 'power', 'symbol': '^', 'arity': 2, 'weight': 0.4, 'params': [2, 3]},
            {'name': 'log', 'symbol': 'log', 'arity': 1, 'weight': 0.5},
            {'name': 'sqrt', 'symbol': 'sqrt', 'arity': 1, 'weight': 0.3},
            {'name': 'abs', 'symbol': 'abs', 'arity': 1, 'weight': 0.3},
            {'name': 'rank', 'symbol': 'rank', 'arity': 1, 'weight': 0.6},
            {'name': 'ma', 'symbol': 'ma', 'arity': 1, 'weight': 0.7, 'params': [5, 10, 20]},
            {'name': 'std', 'symbol': 'std', 'arity': 1, 'weight': 0.5, 'params': [10, 20]}
        ]

    def mine_factors(self,
                    data: pd.DataFrame,
                    returns: pd.Series,
                    target_count: int = 10) -> List[FactorIndividual]:
        """使用遗传算法挖掘因子"""
        try:
            self.logger.info("开始遗传算法因子挖掘")

            # 准备基础因子
            base_factor_data = self._prepare_base_factors(data)
            if not base_factor_data:
                return []

            # 初始化种群
            population = self._initialize_population(base_factor_data, returns)

            if not population:
                self.logger.error("无法初始化种群")
                return []

            best_factors = []

            # 进化循环
            for generation in range(self.generations):
                self.logger.info(f"进化第 {generation + 1}/{self.generations} 代")

                # 评估适应度
                population = self._evaluate_population(population, returns)

                # 记录最佳个体
                current_best = max(population, key=lambda x: x.fitness)
                best_factors.append(current_best)

                self.logger.debug(f"第{generation+1}代最佳因子: {current_best.name}, 适应度: {current_best.fitness:.4f}")

                # 选择
                selected_population = self._selection(population)

                # 交叉
                offspring_population = self._crossover(selected_population, base_factor_data)

                # 变异
                offspring_population = self._mutation(offspring_population, base_factor_data)

                # 合并种群
                population = self._merge_population(population, offspring_population)

                # 限制种群大小
                population = sorted(population, key=lambda x: x.fitness, reverse=True)[:self.population_size]

            # 最终评估和筛选
            final_population = self._evaluate_population(population, returns)
            final_population.sort(key=lambda x: x.fitness, reverse=True)

            # 返回最优因子
            top_factors = final_population[:target_count]
            self.logger.info(f"因子挖掘完成，获得 {len(top_factors)} 个优质因子")

            return top_factors

        except Exception as e:
            self.logger.error(f"遗传算法因子挖掘失败: {str(e)}")
            return []

    def _prepare_base_factors(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """准备基础因子"""
        try:
            base_factors = {}

            # 基础价格和成交量因子
            for factor_name in ['close', 'volume', 'high', 'low', 'open']:
                if factor_name in data.columns:
                    base_factors[factor_name] = data[factor_name]

            # 生成技术指标作为基础因子
            technical_factors = self.factor_generator.generate_simple_factors(data)
            base_factors.update(technical_factors)

            # 标准化基础因子
            for name, factor in base_factors.items():
                if factor.std() > 0:
                    base_factors[name] = (factor - factor.mean()) / factor.std()

            self.logger.info(f"准备基础因子: {len(base_factors)}个")
            return base_factors

        except Exception as e:
            self.logger.error(f"准备基础因子失败: {str(e)}")
            return {}

    def _initialize_population(self,
                             base_factors: Dict[str, pd.Series],
                             returns: pd.Series) -> List[FactorIndividual]:
        """初始化种群"""
        try:
            population = []

            # 生成随机因子表达式
            for i in range(self.population_size):
                try:
                    expression = self._generate_random_expression(base_factors.keys())
                    factor_values = self._evaluate_expression(expression, base_factors)

                    if factor_values is not None and self._validate_factor(factor_values):
                        # 计算适应度
                        fitness = self._calculate_fitness(factor_values, returns)

                        individual = FactorIndividual(
                            name=f"factor_{i}",
                            expression=expression,
                            factor_values=factor_values,
                            fitness=fitness,
                            generation=0
                        )
                        population.append(individual)

                except Exception as e:
                    self.logger.debug(f"生成个体{i}失败: {str(e)}")
                    continue

            self.logger.info(f"初始化种群: {len(population)}个个体")
            return population

        except Exception as e:
            self.logger.error(f"初始化种群失败: {str(e)}")
            return []

    def _generate_random_expression(self, factor_names: List[str], max_depth: int = 3) -> str:
        """生成随机因子表达式"""
        if max_depth == 0 or (max_depth > 1 and random.random() < 0.3):
            # 返回基础因子
            return random.choice(factor_names)
        else:
            # 返回操作符表达式
            operator = self._select_random_operator()
            if operator['arity'] == 1:
                operand = self._generate_random_expression(factor_names, max_depth - 1)
                if operator['name'] == 'ma' or operator['name'] == 'std':
                    param = random.choice(operator['params'])
                    return f"{operator['symbol']}({operand}, {param})"
                else:
                    return f"{operator['symbol']}({operand})"
            else:
                operand1 = self._generate_random_expression(factor_names, max_depth - 1)
                operand2 = self._generate_random_expression(factor_names, max_depth - 1)
                if operator['name'] == 'power':
                    param = random.choice(operator['params'])
                    return f"{operand1} {operator['symbol']} {param}"
                else:
                    return f"({operand1} {operator['symbol']} {operand2})"

    def _select_random_operator(self) -> Dict:
        """随机选择操作符"""
        weights = [op['weight'] for op in self.operators]
        return random.choices(self.operators, weights=weights)[0]

    def _evaluate_expression(self,
                           expression: str,
                           base_factors: Dict[str, pd.Series]) -> Optional[pd.Series]:
        """评估因子表达式"""
        try:
            # 简化的表达式解析器
            # 这里应该使用更完善的解析器，如AST

            # 替换操作符为实际函数
            eval_context = {}
            for name, factor in base_factors.items():
                eval_context[name] = factor

            # 添加数学函数
            eval_context.update({
                'abs': np.abs,
                'sqrt': np.sqrt,
                'log': np.log,
                'rank': lambda x: x.rank(pct=True),
                'ma': lambda x, p: x.rolling(p).mean(),
                'std': lambda x, p: x.rolling(p).std()
            })

            # 安全评估表达式
            result = eval(expression, {"__builtins__": {}}, eval_context)

            if isinstance(result, pd.Series):
                return result
            else:
                return pd.Series(result)

        except Exception as e:
            self.logger.debug(f"评估表达式失败: {expression}, 错误: {str(e)}")
            return None

    def _validate_factor(self, factor_values: pd.Series) -> bool:
        """验证因子质量"""
        try:
            if factor_values.isna().all():
                return False

            if factor_values.std() == 0:
                return False

            valid_ratio = factor_values.notna().sum() / len(factor_values)
            if valid_ratio < 0.3:
                return False

            # 检查异常值
            factor_clean = factor_values.dropna()
            if len(factor_clean) == 0:
                return False

            z_scores = np.abs((factor_clean - factor_clean.mean()) / factor_clean.std())
            extreme_ratio = (z_scores > 5).sum() / len(factor_clean)
            if extreme_ratio > 0.05:
                return False

            return True

        except Exception as e:
            self.logger.debug(f"因子验证失败: {str(e)}")
            return False

    def _calculate_fitness(self, factor_values: pd.Series, returns: pd.Series) -> float:
        """计算适应度"""
        try:
            # 对齐数据
            aligned_data = pd.DataFrame({
                'factor': factor_values,
                'returns': returns
            }).dropna()

            if len(aligned_data) < 50:
                return 0.0

            # 计算IC值
            ic = calculate_ic(aligned_data['factor'], aligned_data['returns'], 'spearman')

            if np.isnan(ic):
                return 0.0

            # 使用IC绝对值作为适应度
            fitness = abs(ic)

            # 复杂度惩罚
            expression_complexity = len(str(factor_values.name)) if hasattr(factor_values, 'name') else 10
            complexity_penalty = expression_complexity / 1000

            return max(0.0, fitness - complexity_penalty)

        except Exception as e:
            self.logger.debug(f"计算适应度失败: {str(e)}")
            return 0.0

    def _evaluate_population(self,
                           population: List[FactorIndividual],
                           returns: pd.Series) -> List[FactorIndividual]:
        """评估种群适应度"""
        for individual in population:
            if individual.fitness == 0:
                individual.fitness = self._calculate_fitness(individual.factor_values, returns)
        return population

    def _selection(self, population: List[FactorIndividual]) -> List[FactorIndividual]:
        """选择操作"""
        # 精英保留
        elite_size = int(len(population) * self.elitism_rate)
        elite = sorted(population, key=lambda x: x.fitness, reverse=True)[:elite_size]

        # 轮盘赌选择
        total_fitness = sum(ind.fitness for ind in population)
        if total_fitness == 0:
            return elite

        selected = elite.copy()
        while len(selected) < len(population):
            r = random.uniform(0, total_fitness)
            cumsum = 0
            for individual in population:
                cumsum += individual.fitness
                if cumsum >= r:
                    selected.append(individual)
                    break

        return selected

    def _crossover(self,
                  selected_population: List[FactorIndividual],
                  base_factors: Dict[str, pd.Series]) -> List[FactorIndividual]:
        """交叉操作"""
        offspring = []

        for i in range(0, len(selected_population) - 1, 2):
            parent1 = selected_population[i]
            parent2 = selected_population[i + 1]

            if random.random() < self.crossover_rate:
                # 交叉操作
                child1_expr, child2_expr = self._crossover_expressions(
                    parent1.expression, parent2.expression
                )

                for expr in [child1_expr, child2_expr]:
                    factor_values = self._evaluate_expression(expr, base_factors)
                    if factor_values is not None and self._validate_factor(factor_values):
                        child = FactorIndividual(
                            name=f"factor_{len(offspring)}",
                            expression=expr,
                            factor_values=factor_values,
                            fitness=0.0
                        )
                        offspring.append(child)

        return offspring

    def _crossover_expressions(self, expr1: str, expr2: str) -> Tuple[str, str]:
        """表达式交叉"""
        # 简化的交叉操作
        # 实际应该使用更复杂的语法树交叉

        # 随机选择交叉点
        split_point = min(len(expr1), len(expr2)) // 2

        child1 = expr1[:split_point] + expr2[split_point:]
        child2 = expr2[:split_point] + expr1[split_point:]

        return child1, child2

    def _mutation(self,
                 offspring_population: List[FactorIndividual],
                 base_factors: Dict[str, pd.Series]) -> List[FactorIndividual]:
        """变异操作"""
        mutated_population = []

        for individual in offspring_population:
            if random.random() < self.mutation_rate:
                # 变异操作
                mutated_expr = self._mutate_expression(individual.expression, base_factors.keys())
                factor_values = self._evaluate_expression(mutated_expr, base_factors)

                if factor_values is not None and self._validate_factor(factor_values):
                    mutated_individual = FactorIndividual(
                        name=individual.name + "_mut",
                        expression=mutated_expr,
                        factor_values=factor_values,
                        fitness=0.0
                    )
                    mutated_population.append(mutated_individual)
                else:
                    mutated_population.append(individual)
            else:
                mutated_population.append(individual)

        return mutated_population

    def _mutate_expression(self, expression: str, factor_names: List[str]) -> str:
        """表达式变异"""
        # 简化的变异操作
        mutations = [
            lambda expr: self._replace_operator(expr),
            lambda expr: self._replace_factor(expr, factor_names),
            lambda expr: self._add_parentheses(expr),
            lambda expr: self._remove_parentheses(expr)
        ]

        mutation = random.choice(mutations)
        return mutation(expression)

    def _replace_operator(self, expression: str) -> str:
        """替换操作符"""
        operators = ['+', '-', '*', '/']
        for op in operators:
            if op in expression:
                new_op = random.choice(operators)
                return expression.replace(op, new_op, 1)
        return expression

    def _replace_factor(self, expression: str, factor_names: List[str]) -> str:
        """替换因子"""
        for factor_name in factor_names:
            if factor_name in expression:
                new_factor = random.choice(factor_names)
                return expression.replace(factor_name, new_factor, 1)
        return expression

    def _add_parentheses(self, expression: str) -> str:
        """添加括号"""
        return f"({expression})"

    def _remove_parentheses(self, expression: str) -> str:
        """移除括号"""
        if expression.startswith('(') and expression.endswith(')'):
            return expression[1:-1]
        return expression

    def _merge_population(self,
                         parent_population: List[FactorIndividual],
                         offspring_population: List[FactorIndividual]) -> List[FactorIndividual]:
        """合并种群"""
        return parent_population + offspring_population

    def get_factor_statistics(self, factors: List[FactorIndividual]) -> Dict:
        """获取挖掘因子统计信息"""
        if not factors:
            return {}

        fitness_values = [f.fitness for f in factors]
        expressions = [f.expression for f in factors]

        # 计算表达式复杂度
        complexities = [len(expr) for expr in expressions]

        return {
            'factor_count': len(factors),
            'fitness_stats': {
                'mean': np.mean(fitness_values),
                'std': np.std(fitness_values),
                'min': np.min(fitness_values),
                'max': np.max(fitness_values)
            },
            'complexity_stats': {
                'mean': np.mean(complexities),
                'std': np.std(complexities),
                'min': np.min(complexities),
                'max': np.max(complexities)
            },
            'best_factor': factors[0].name if factors else None,
            'best_fitness': fitness_values[0] if fitness_values else 0
        }

    def export_factors(self, factors: List[FactorIndividual], file_path: str) -> bool:
        """导出挖掘的因子"""
        try:
            export_data = []
            for factor in factors:
                export_data.append({
                    'name': factor.name,
                    'expression': factor.expression,
                    'fitness': factor.fitness,
                    'generation': factor.generation
                })

            df = pd.DataFrame(export_data)
            df.to_csv(file_path, index=False, encoding='utf-8')
            self.logger.info(f"因子导出成功: {file_path}")
            return True

        except Exception as e:
            self.logger.error(f"导出因子失败: {str(e)}")
            return False