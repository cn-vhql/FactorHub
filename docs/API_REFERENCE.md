# FactorHub API 完整参考

**API基础URL**: `http://localhost:8000/api`
**API版本**: v1 (当前)
**总端点数**: 48个

---

## 📋 目录

1. [Factors Router](#factors-router) - 因子管理 (10个端点)
2. [Analysis Router](#analysis-router) - 因子分析 (9个端点)
3. [Backtest Router](#backtest-router) - 策略回测 (4个端点)
4. [Mining Router](#mining-router) - 因子挖掘 (4个端点)
5. [Portfolio Router](#portfolio-router) - 组合优化 (3个端点)
6. [Data Router](#data-router) - 数据管理 (4个端点)
7. [Scoring Router](#scoring-router) - 评分系统 (4个端点)
8. [Search Router](#search-router) - 策略搜索 (10个端点)

---

## Factors Router

**前缀**: `/api/factors`

### 1. 列表所有因子
```
GET /api/factors
Query Parameters:
  - category: string (optional) 因子分类 e.g. "momentum"
  - source: string (optional) 因子来源 e.g. "preset" | "user"
  - page: integer (optional) 分页页码
  - limit: integer (optional) 每页数量

Response:
{
  "success": true,
  "data": [
    {
      "id": 1,
      "name": "rsi_14",
      "code": "RSI(close, 14)",
      "description": "RSI相对强弱指标",
      "category": "momentum",
      "source": "preset",
      "formula_type": "expression",
      "is_active": 1,
      "created_at": "2025-01-01T00:00:00Z",
      "updated_at": "2025-03-27T10:30:00Z"
    }
  ],
  "total": 70
}

Example:
  GET /api/factors?category=momentum&source=preset
```

### 2. 获取因子统计
```
GET /api/factors/stats

Response:
{
  "success": true,
  "data": {
    "preset_count": 70,
    "user_count": 15,
    "total_count": 85,
    "categories": {
      "momentum": 8,
      "trend": 10,
      ...
    },
    "cache_stats": {
      "hit_rate": 0.85,
      "size_mb": 234.5
    }
  }
}
```

### 3. 获取单个因子
```
GET /api/factors/{factor_id}

URL Parameters:
  - factor_id: integer 因子ID

Response:
{
  "success": true,
  "data": {
    "id": 1,
    "name": "rsi_14",
    "code": "RSI(close, 14)",
    ...
  }
}

Error:
  404 - 因子不存在
```

### 4. 创建因子
```
POST /api/factors
Content-Type: application/json

Body:
{
  "name": "my_momentum_factor",
  "code": "close/SMA(close, 20) - 1",
  "category": "momentum",
  "description": "自定义动量因子",
  "formula_type": "expression"  # "expression" | "function"
}

Response:
{
  "success": true,
  "data": {
    "id": 86,
    "name": "my_momentum_factor",
    "source": "user",
    "created_at": "2025-03-27T14:00:00Z"
  },
  "message": "因子创建成功"
}

Error Codes:
  400 - 因子名称已存在
  422 - 参数验证失败
  500 - 创建失败
```

### 5. 更新因子
```
PUT /api/factors/{factor_id}
Content-Type: application/json

Body:
{
  "code": "close/SMA(close, 25) - 1",  # optional
  "description": "更新描述",            # optional
  ...
}

Response:
{
  "success": true,
  "message": "因子已更新"
}

Note:
  - 预置因子 (source="preset") 无法更新
  - 仅支持更新code和description
```

### 6. 删除因子
```
DELETE /api/factors/{factor_id}

Response:
{
  "success": true,
  "message": "因子已删除"
}

Restrictions:
  - 仅可删除用户因子 (source="user")
  - 预置因子无法删除
```

### 7. 复制因子
```
POST /api/factors/{factor_id}/copy
Content-Type: application/json

Body:
{
  "new_name": "my_rsi_14_copy"  # optional，默认为原名_copy
}

Response:
{
  "success": true,
  "data": {
    "id": 87,
    "name": "my_rsi_14_copy",
    "source": "user"
  },
  "message": "因子已复制"
}
```

### 8. 验证因子公式
```
POST /api/factors/validate
Content-Type: application/json

Body:
{
  "code": "RSI(close, 14) > 30",
  "formula_type": "expression"  # "expression" | "function"
}

Response:
{
  "success": true,
  "data": {
    "code": "RSI(close, 14) > 30",
    "formula_type": "expression",
    "valid": true,
    "errors": []
  },
  "message": "公式有效"
}

Error Cases:
{
  "success": false,
  "data": {
    "valid": false,
    "errors": ["SyntaxError: invalid syntax"]
  }
}
```

### 9. 批量生成因子
```
POST /api/factors/batch-generate
Content-Type: application/json

Body:
{
  "base_factors": ["RSI(close, 14)", "close/SMA(close, 20)"],
  "generate_methods": ["arithmetic", "statistics", "technical"],
  "ic_threshold": 0.03,
  "ir_threshold": 0.5,
  "min_valid_ratio": 0.7
}

Response:
{
  "success": true,
  "data": {
    "generated_count": 45,
    "total_possible": 150,
    "factors": [
      {
        "expression": "(RSI(close, 14) + close/SMA(close, 20)) / 2",
        "method": "arithmetic",
        "generated_at": "..."
      }
    ]
  }
}

Methods:
  - arithmetic: 四则运算组合
  - statistics: 统计函数组合
  - technical: 技术指标组合
  - hybrid: 混合组合
```

### 10. 因子预筛选
```
POST /api/factors/preselect
Content-Type: application/json

Body:
{
  "factors": ["rsi_14", "momentum_20", "price_vs_sma20"],
  "ic_threshold": 0.02,
  "ir_threshold": 0.4,
  "min_valid_ratio": 0.6
}

Response:
{
  "success": true,
  "data": {
    "total": 3,
    "selected": 2,
    "factors": [
      {
        "name": "rsi_14",
        "passed": true,
        "reasons": []
      },
      {
        "name": "momentum_20",
        "passed": false,
        "reasons": ["IC低于阈值"]
      }
    ]
  }
}
```

---

## Analysis Router

**前缀**: `/api/analysis`

### 1. 因子计算
```
POST /api/analysis/calculate
Content-Type: application/json

Body:
{
  "factor_name": "RSI(close, 14)",
  "stock_codes": ["000001", "600519"],
  "start_date": "2023-01-01",
  "end_date": "2024-12-31"
}

Response:
{
  "success": true,
  "data": {
    "000001": {
      "dates": ["2023-01-01", "2023-01-02", ...],
      "factor_values": [35.2, 36.1, ...],
      "statistics": {
        "mean": 50.3,
        "std": 15.4,
        "min": 12.5,
        "max": 88.7,
        "count": 480
      }
    },
    "600519": {...}
  },
  "warnings": null
}

Error Codes:
  404 - 因子不存在
  400 - 股票代码无效
  500 - 计算失败
```

### 2. IC/IR分析
```
POST /api/analysis/ic
Content-Type: application/json

Body:
{
  "factors": ["rsi_14", "momentum_20"],
  "stock_codes": ["000001", "000002", "000004"],
  "start_date": "2023-01-01",
  "end_date": "2024-12-31",
  "method": "ts"  # "ts" | "cs" | "monthly"
}

Response:
{
  "success": true,
  "data": {
    "factor_name": "rsi_14",
    "ic_stats": {
      "ic_mean": 0.032,
      "ic_std": 0.12,
      "ir": 0.47,
      "ic_positive_ratio": 0.62,
      "by_period": [
        {
          "date": "2023-01-01",
          "ic": 0.045,
          "valid": true
        }
      ]
    }
  }
}

Method Definitions:
  - ts: 时间序列IC (各日期的截面IC)
  - cs: 截面IC (各股票的时间序列IC)
  - monthly: 月度IC聚合
```

### 3. 稳定性测试
```
POST /api/analysis/stability
Content-Type: application/json

Body: (same as IC analysis)

Response:
{
  "success": true,
  "data": {
    "factor_name": "rsi_14",
    "stability_score": 0.75,
    "grade": "B",
    "ic_consistency": 0.62,
    "rolling_ic": [...]
  }
}

Grades:
  A: > 0.85  (非常稳定)
  B: 0.70-0.85 (稳定)
  C: 0.50-0.70 (中等)
  D: < 0.50 (不稳定)
```

### 4. 衰减分析
```
POST /api/analysis/decay
Content-Type: application/json

Body: (same as IC analysis)

Response:
{
  "success": true,
  "data": {
    "factor_name": "rsi_14",
    "decay_analysis": [
      {
        "horizon_days": 1,
        "ic_mean": 0.045,
        "ic_std": 0.12
      },
      {
        "horizon_days": 5,
        "ic_mean": 0.038,
        "ic_std": 0.15
      },
      ...
    ]
  }
}

Horizons: [1, 3, 5, 10, 20] (固定)
```

### 5. 暴露度分析
```
POST /api/analysis/exposure
Content-Type: application/json

Body: (same as calculate)

Response:
{
  "success": true,
  "data": {
    "000001": {
      "current_value": 52.3,
      "percentile": 0.62,
      "rolling_mean": 50.1,
      "rolling_std": 15.4,
      "distribution": {
        "min": 12.5,
        "q1": 35.2,
        "median": 50.3,
        "q3": 65.1,
        "max": 88.7
      }
    }
  }
}
```

### 6. 有效性分析
```
POST /api/analysis/effectiveness
Content-Type: application/json

Body: (same as IC analysis)

Response:
{
  "success": true,
  "data": {
    "factor_name": "rsi_14",
    "event_analysis": {
      "horizons": [1, 5, 10, 20],
      "long_returns": [0.008, 0.035, 0.062, 0.110],
      "short_returns": [-0.005, -0.032, -0.058, -0.105]
    },
    "correlation_with_returns": 0.18
  }
}
```

### 7. 归因分析
```
POST /api/analysis/attribution
Content-Type: application/json

Body: (same as IC analysis)

Response:
{
  "success": true,
  "data": {
    "factor_name": "rsi_14",
    "alpha": 0.025,
    "beta": 0.85,
    "factor_contribution": 0.045,
    "contribution_pct": 0.42,
    "by_stock": [...]
  }
}
```

### 8. 监控分析
```
POST /api/analysis/monitoring
Content-Type: application/json

Body: (same as IC analysis)

Response:
{
  "success": true,
  "data": {
    "factor_name": "rsi_14",
    "time_series_ic": [...],
    "rolling_window_ic": [...],
    "decay_trend": "stable",
    "alerts": [
      {
        "type": "IC_DROP",
        "severity": "warning",
        "message": "最近30天IC平均值下降"
      }
    ]
  }
}
```

### 9. 多周期分析
```
POST /api/analysis/multi-period
Content-Type: application/json

Body: (same as IC analysis)

Response:
{
  "success": true,
  "data": {
    "periods": [
      {
        "name": "2023",
        "ic_mean": 0.035,
        "ic_std": 0.14
      },
      {
        "name": "2024",
        "ic_mean": 0.028,
        "ic_std": 0.16
      }
    ]
  }
}
```

---

## Backtest Router

**前缀**: `/api/backtest`

### 1. 单因子回测
```
POST /api/backtest/single
Content-Type: application/json

Body:
{
  "stock_codes": ["000001"],
  "factors": ["rsi_14"],  # or factor_name (deprecated)
  "strategy_type": "single_factor",  # or "multi_factor"
  "start_date": "2023-01-01",
  "end_date": "2024-12-31",
  "initial_capital": 1000000,
  "commission_rate": 0.0003,
  "slippage": 0.0002,
  "percentile": 50,
  "direction": "long",  # "long" | "short"
  "n_quantiles": 5,
  "weight_method": "equal_weight"
}

Response:
{
  "success": true,
  "data": {
    "metrics": {
      "total_return": 0.256,
      "annual_return": 0.128,
      "volatility": 0.15,
      "sharpe_ratio": 0.85,
      "max_drawdown": -0.12,
      "calmar_ratio": 1.07,
      "win_rate": 0.62,
      "sortino_ratio": 1.15,
      "var_95": -0.025,
      "cvar_95": -0.035
    },
    "chart_data": {
      "kline": {...},
      "factor": {...},
      "signals": {...},
      "equity": {...}
    },
    "monthly_returns": {...}
  }
}

Parameters:
  - percentile: 1-99 (因子值阈值)
  - n_quantiles: 2-10 (分层数)
  - weight_method: equal_weight | market_cap_weight | ic_ir_optimization
```

### 2. 策略对比
```
POST /api/backtest/comparison
Content-Type: application/json

Body:
{
  "stock_codes": ["000001"],
  "strategies": [
    {
      "name": "Strategy_RSI",
      "factors": ["rsi_14"],
      "weights": [1.0]
    },
    {
      "name": "Strategy_Momentum",
      "factors": ["momentum_20"],
      "weights": [1.0]
    }
  ],
  "start_date": "2023-01-01",
  "end_date": "2024-12-31",
  "initial_capital": 1000000
}

Response:
{
  "success": true,
  "data": {
    "results": {
      "Strategy_RSI": {
        "total_return": 0.256,
        "annual_return": 0.128,
        "sharpe_ratio": 0.85,
        ...
      },
      "Strategy_Momentum": {
        "total_return": 0.221,
        ...
      }
    },
    "best_strategy": "Strategy_RSI"
  }
}
```

### 3. 回测历史
```
GET /api/backtest/history
Query Parameters:
  - limit: integer (default 10) 最近N条记录
  - offset: integer (default 0) 分页偏移

Response:
{
  "success": true,
  "data": [
    {
      "id": "abc123",
      "stock_codes": ["000001"],
      "factors": ["rsi_14"],
      "start_date": "2023-01-01",
      "end_date": "2024-12-31",
      "metrics": {...},
      "created_at": "2025-03-27T10:00:00Z"
    }
  ],
  "total": 125
}
```

### 4. 删除历史记录
```
DELETE /api/backtest/history/{record_id}

Response:
{
  "success": true,
  "message": "历史记录已删除"
}
```

---

## Mining Router

**前缀**: `/api/mining`

### 1. 最近有价值因子
```
POST /api/mining/recent-valuable
Content-Type: application/json

Body:
{
  "stock_codes": ["000001"],
  "end_date": "2025-03-27",
  "eval_days": 90,  # 最近N天评估窗口
  "lookback_days": 240,  # 预热期
  "horizons": [1, 5, 10],
  "top_k": 20,
  "min_samples": 25,
  "correlation_threshold": 0.85
}

Response:
{
  "success": true,
  "data": {
    "global_top_factors": [
      {
        "name": "rsi_14",
        "ic_mean": 0.048,
        "rank": 1
      }
    ],
    "per_stock_top_factors": {
      "000001": [...]
    },
    "correlation_matrix": {...}
  }
}

Parameters:
  - eval_days: 10-730 (评估窗口)
  - lookback_days: 30-1825 (预热期)
  - top_k: 1-200 (返回数量)
  - correlation_threshold: 0-1 (相关性阈值)
```

### 2. 遗传算法挖掘 (异步)
```
POST /api/mining/genetic
Content-Type: application/json

Body:
{
  "stock_code": "000001",
  "base_factors": ["RSI(close, 14)", "close/SMA(close, 20)"],
  "start_date": "2023-01-01",
  "end_date": "2024-12-31",
  "population_size": 50,
  "n_generations": 10,
  "cx_prob": 0.7,
  "mut_prob": 0.3,
  "elite_size": 5,
  "fitness_objective": "ic",
  "ic_threshold": 0.03
}

Response (202 Accepted):
{
  "success": true,
  "data": {
    "task_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "status": "pending"
  },
  "message": "因子挖掘任务已启动"
}

Parameters:
  - population_size: 10-500
  - n_generations: 5-100
  - fitness_objective: "ic" | "ir" | "hybrid"
```

### 3. 挖掘任务状态
```
GET /api/mining/status/{task_id}

Response:
{
  "success": true,
  "data": {
    "task_id": "a1b2c3d4...",
    "status": "running",  # pending | running | completed | failed
    "progress": 45,
    "current_generation": 4,
    "total_generations": 10,
    "best_fitness": 0.045,
    "avg_fitness": 0.032,
    "fitness_history": {
      "best": [0.03, 0.035, 0.04, 0.045],
      "average": [0.02, 0.025, 0.03, 0.032]
    },
    "error": null
  }
}

Status:
  - pending: 等待开始
  - running: 运行中
  - completed: 完成
  - failed: 失败
```

### 4. 挖掘结果
```
GET /api/mining/results/{task_id}

Response:
{
  "success": true,
  "data": {
    "task_id": "a1b2c3d4...",
    "status": "completed",
    "factors": [
      {
        "rank": 1,
        "expression": "(RSI(close, 14) * 1.5) + 0.2",
        "ic": 0.047,
        "ir": 0.54,
        "fitness": 0.047
      }
    ],
    "best_fitness": 0.047,
    "generations": 10,
    "creation_time": "2025-03-27T10:00:00Z",
    "completion_time": "2025-03-27T10:15:00Z"
  }
}
```

---

## Portfolio Router

**前缀**: `/api/portfolio`

### 1. 权重优化
```
POST /api/portfolio/optimize-weights
Content-Type: application/json

Body:
{
  "stock_codes": ["000001"],
  "factors": ["rsi_14", "momentum_20"],
  "start_date": "2023-01-01",
  "end_date": "2024-12-31",
  "method": "max_sharpe",  # See methods below
  "rebalance_freq": "W-FRI"  # Weekly rebalance on Friday
}

Response:
{
  "success": true,
  "data": {
    "weights": {
      "rsi_14": 0.45,
      "momentum_20": 0.55
    },
    "method": "max_sharpe",
    "factors": ["rsi_14", "momentum_20"],
    "metrics": {
      "return": 0.128,
      "ic": 0.048,
      "ir": 0.62,
      "volatility": 0.14,
      "sharpe_ratio": 0.91
    },
    "composite_score": 0.78,
    "composite_stats": {...}
  }
}

Methods:
  1. equal_weight - 等权 (每个因子1/n)
  2. ic_weight - IC加权 (按IC排名)
  3. ir_weight - IR加权 (按信息比率)
  4. max_sharpe - 最大化夏普比
  5. max_return - 最大化收益
  6. min_variance - 最小化波动率
```

### 2. 复合评分
```
POST /api/portfolio/composite-score
Content-Type: application/json

Body:
{
  "stock_codes": ["000001"],
  "factors": ["rsi_14", "momentum_20"],
  "start_date": "2023-01-01",
  "end_date": "2024-12-31"
}

Response:
{
  "success": true,
  "data": {
    "dates": ["2023-01-01", "2023-01-02", ...],
    "scores": [0.52, 0.55, ...],  # 0-100
    "statistics": {
      "mean": 0.52,
      "std": 0.12,
      "min": 0.10,
      "max": 0.95
    }
  }
}
```

### 3. 方法对比
```
POST /api/portfolio/compare-methods
Content-Type: application/json

Body:
{
  "stock_codes": ["000001"],
  "factors": ["rsi_14", "momentum_20"],
  "start_date": "2023-01-01",
  "end_date": "2024-12-31",
  "methods": ["equal_weight", "ic_weight", "max_sharpe"]
}

Response:
{
  "success": true,
  "data": {
    "results": {
      "equal_weight": {
        "weights": {"rsi_14": 0.5, "momentum_20": 0.5},
        "metrics": {...}
      },
      "ic_weight": {...},
      "max_sharpe": {...}
    },
    "best_method": "max_sharpe"
  }
}
```

---

## Data Router

**前缀**: `/api/data`

### 1. 股票OHLCV数据
```
GET /api/data/stock/{stock_code}
Query Parameters:
  - start_date: string (YYYY-MM-DD)
  - end_date: string (YYYY-MM-DD)

Response:
{
  "success": true,
  "data": {
    "index": ["2023-01-01", "2023-01-02", ...],
    "columns": ["open", "high", "low", "close", "volume"],
    "data": [
      [123.45, 124.50, 122.80, 124.00, 45230000],
      [124.00, 125.20, 123.90, 125.10, 52100000],
      ...
    ]
  }
}

Error Codes:
  404 - 股票代码不存在
  400 - 日期范围无效
```

### 2. 缓存统计
```
GET /api/data/cache/stats

Response:
{
  "success": true,
  "data": {
    "total_entries": 1250,
    "hit_count": 45230,
    "miss_count": 8900,
    "hit_rate": 0.835,
    "size_mb": 234.5,
    "ttl_days": 7
  }
}
```

### 3. 清理缓存
```
POST /api/data/cache/cleanup
(Remove expired entries)

Response:
{
  "success": true,
  "data": {
    "cleaned_count": 45,
    "freed_mb": 23.5
  },
  "message": "缓存已清理"
}
```

### 4. 清空缓存
```
POST /api/data/cache/clear
(Remove all cache)

Response:
{
  "success": true,
  "data": {
    "cleared_count": 1250,
    "freed_mb": 234.5
  },
  "message": "缓存已清空"
}
```

---

## Scoring Router

**前缀**: `/api/scoring`

### 1. 时间序列评分 (同步/异步)
```
POST /api/scoring/temporal-score
Content-Type: application/json

Body:
{
  "codes": ["000001", "000002"],
  "trade_date": "2025-03-27",  # optional, 默认今天
  "profile_id": "temporal_default",  # optional
  "per_stock_profiles": {}  # optional dict of code->profile_id
}

Response:
自动选择同步或异步 (>20个代码时自动异步)

Sync Response:
{
  "success": true,
  "data": [
    {
      "date": "2025-03-27",
      "code": "000001",
      "score": 75
    }
  ],
  "timestamp": "2025-03-27T14:00:00Z"
}

Async Response (202):
{
  "task_id": "xyz789",
  "status": "running"
}
```

### 2. 创建异步评分任务
```
POST /api/scoring/temporal-score/jobs
(Same body as above)

Response (202):
{
  "task_id": "xyz789",
  "status": "pending"
}
```

### 3. 获取评分任务状态
```
GET /api/scoring/temporal-score/jobs/{task_id}

Response:
{
  "task_id": "xyz789",
  "status": "completed",  # pending | running | completed | failed
  "trade_date": "2025-03-27",
  "total": 100,
  "processed": 100,
  "success_count": 98,
  "failed_count": 2,
  "progress": 100,
  "created_at": "...",
  "started_at": "...",
  "completed_at": "..."
}
```

### 4. 获取评分结果
```
GET /api/scoring/temporal-score/jobs/{task_id}/results
Query Parameters:
  - detail: boolean (default false) 返回详细信息

Response:
{
  "success": true,
  "data": [
    {
      "date": "2025-03-27",
      "code": "000001",
      "score": 75,
      "detail": {...}  # if detail=true
    }
  ]
}
```

---

## Search Router

**前缀**: `/api/search`

### 1. 冠军因子搜索 - 时间序列池
```
POST /api/search/champion/temporal-pool
Content-Type: application/json

Body:
{
  "codes": ["000001", "000002"],
  "search_space_id": "temporal_default",
  "start_date": "2023-01-01",
  "end_date": "2024-12-31",
  "max_workers": 4
}

Response (202):
{
  "task_id": "champion_001",
  "status": "pending"
}
```

### 2. 冠军因子搜索 - 截面
```
POST /api/search/champion/cross-sectional
Content-Type: application/json

Body:
{
  "universe": "all",  # or specific stock list
  "search_space_id": "cross_sectional_default",
  "start_date": "2023-01-01",
  "end_date": "2024-12-31"
}

Response (202):
{
  "task_id": "champion_002",
  "status": "pending"
}
```

### 3. 冠军因子搜索 - 单只股票
```
POST /api/search/champion/single-stock
Content-Type: application/json

Body:
{
  "code": "000001",
  "search_space_id": "single_stock_default",
  "start_date": "2023-01-01",
  "end_date": "2024-12-31"
}

Response (202):
{
  "task_id": "champion_003",
  "status": "pending"
}
```

### 4. 搜索任务状态
```
GET /api/search/champion/jobs/{task_id}

Response:
{
  "task_id": "champion_001",
  "status": "completed",
  "created_at": "...",
  "updated_at": "...",
  "message": null
}
```

### 5. 搜索结果
```
GET /api/search/champion/jobs/{task_id}/results

Response:
{
  "task_id": "champion_001",
  "status": "completed",
  "champion": {
    "factors": ["rsi_14", "momentum_20"],
    "weights": [0.6, 0.4],
    "annual_return": 0.18,
    "sharpe_ratio": 1.02,
    "stability_score": 0.87
  },
  "error": null
}
```

### 6. 获取已保存的冠军配置
```
GET /api/search/champions/{scope_type}/{scope_key}

Scope Types:
  - temporal_pool/{pool_name}
  - cross_sectional/{sector}
  - single_stock/{code}

Response:
{
  "success": true,
  "data": {
    "factors": [...],
    "weights": [...],
    ...
  }
}
```

### 7. 应用冠军配置
```
POST /api/search/champions/{scope_type}/{scope_key}/apply
Content-Type: application/json

Body:
{
  "factors": ["rsi_14", "momentum_20"],
  "weights": [0.6, 0.4],
  ...
}

Response:
{
  "success": true,
  "scope_type": "single_stock",
  "scope_key": "000001",
  "message": "配置已应用"
}
```

### 8. 获取可用因子档案
```
GET /api/search/profiles
Query Parameters:
  - mode: string (optional)

Response:
{
  "profiles": [
    {
      "id": "temporal_default",
      "name": "时间序列默认档案",
      "description": "...",
      "scope_type": "temporal_pool"
    }
  ]
}
```

### 9. 获取搜索空间配置
```
GET /api/search/search-spaces

Response:
{
  "search_spaces": [
    {
      "id": "temporal_default",
      "name": "时间序列默认",
      "description": "...",
      "default_params": {...}
    }
  ]
}
```

### 10. 获取股票池
```
GET /api/search/pools

Response:
{
  "pools": [
    {
      "id": "top100",
      "name": "TOP100",
      "description": "按市值排名的前100只股票",
      "codes": ["000001", "000002", ...]
    }
  ]
}
```

---

## 错误码参考

### HTTP 状态码
- `200` - 成功
- `202` - 已接受 (异步任务)
- `400` - 请求参数错误
- `404` - 资源不存在
- `422` - 验证错误
- `500` - 服务器错误
- `503` - 服务暂时不可用

### 常见错误响应
```json
{
  "success": false,
  "error": {
    "code": "INVALID_STOCK_CODE",
    "message": "股票代码格式错误",
    "details": {
      "received": "ABC",
      "expected": "^\d{6}$"
    }
  }
}
```

---

## 常见用例

### 用例1: 快速验证因子
```
1. POST /api/factors/validate
   检查公式语法

2. POST /api/analysis/calculate
   计算因子值

3. POST /api/analysis/ic
   IC/IR分析

4. POST /api/backtest/single
   回测验证
```

### 用例2: 多因子组合
```
1. GET /api/factors?category=momentum
   查找动量类因子

2. POST /api/portfolio/optimize-weights
   权重优化

3. POST /api/backtest/single with multiple factors
   多因子回测

4. POST /api/portfolio/compare-methods
   不同优化方法对比
```

### 用例3: 自动因子发现
```
1. POST /api/mining/genetic
   启动遗传算法挖掘

2. GET /api/mining/status/{task_id}
   监控进度

3. GET /api/mining/results/{task_id}
   获取结果

4. POST /api/backtest/single
   对发现的因子进行回测
```

---

**更新日期**: 2025-03-27
**API版本**: 1.0
