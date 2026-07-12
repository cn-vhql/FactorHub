# FactorHub

**FactorHub** is an open-source modern quantitative factor analysis platform designed specifically for the Chinese A-share market.

> FactorHub = Factor + Hub

A full-stack quantitative investment research system integrating factor management, analysis, mining, portfolio optimization, and strategy backtesting.

---

## Language Options

- 🇨🇳 **中文 (Chinese)** - [README_ZH.md](README_ZH.md)
- 🇯🇵 **日本語 (Japanese)** - [README_JP.md](README_JP.md)

---

## Core Value Proposition

| Value Pillar | Description |
|--------------|-------------|
| 🎯 **Complete Factor Lifecycle Management** | Full support from factor creation, validation, analysis to deployment |
| 🧪 **Scientific Factor Evaluation System** | Professional indicators including IC/IR analysis, monotonicity test, turnover analysis |
| 🧬 **Intelligent Factor Mining** | Genetic algorithm-based automated factor mining to discover alpha signals |
| 🤖 **AI Research Assistant** | Supports natural-language factor generation, AI interpretation of analysis results, and report drafting |
| 📊 **Professional Backtesting Engine** | Support for multi-factor combination, strategy comparison, and performance attribution analysis |

---

## Core Features

![1773500695533](image/README_ZH/1773500695533.png)

### 1. Factor Management
- ✅ **Custom Factor Definition** - Supports Tongda Xinhua (MyLanguage) syntax and TALib functions
- ✅ **Formula Validation** - Real-time syntax checking and logical verification
- ✅ **Version Control** - Factor modification history and version rollback
- ✅ **Pre-built Factor Library** - Built-in common technical factors (MA, RSI, MACD, Bollinger Bands, etc.)

![1773500664940](image/README_ZH/1773500664940.png)

### 2. Factor Analysis
- ✅ **IC/IR Analysis** - Information Coefficient and Information Ratio calculation (supports 1-day, 5-day, 10-day prediction cycles)
- ✅ **Factor Exposure Analysis** - Analyze stock exposure distribution on factors
- ✅ **Factor Effectiveness Testing** - Multi-dimensional assessment of factor predictive power
- ✅ **Factor Attribution Analysis** - Decompose factor contribution to returns
- ✅ **Dynamic Monitoring** - Factor performance tracking across time series dimensions

![1773500638704](image/README_ZH/1773500638704.png)

### 3. Factor Mining
- ✅ **Genetic Algorithm Mining** - DEAP-based evolutionary algorithm for automatic factor search
- ✅ **Multi-objective Optimization** - Simultaneously optimize IC, IR, monotonicity, and other objectives
- ✅ **Factor Generation** - Supports basic operators, function calls, and time window operations
- ✅ **Parallel Computing** - Parallel population evaluation for acceleration

![1773500591756](image/README_ZH/1773500591756.png)

### 4. Portfolio Analysis
- ✅ **Multi-factor Portfolio** - Supports equal weight, market cap weighting, IC_IR maximization, etc.
- ✅ **Risk Modeling** - Factor neutralization processing
- ✅ **Optimization Configuration** - Factor weight optimization based on historical performance
- ✅ **Portfolio Performance** - Annual return, Sharpe ratio, maximum drawdown, and other metrics

![1773500498447](image/README_ZH/1773500498447.png)

### 5. Strategy Backtesting
- ✅ **Single Factor Backtesting** - Factor quantile-based stock selection backtesting
- ✅ **Multi-factor Strategies** - Composite factor signal generation
- ✅ **Strategy Comparison** - Multi-strategy parallel backtesting and comparison analysis
- ✅ **Performance Metrics** - Complete metric system including returns, risk, and turnover
- ✅ **Visualization Charts** - Equity curves, drawdowns, factor performance charts, etc.

![1773500440529](image/README_ZH/1773500440529.png)

### 6. AI Capabilities
- ✅ **AI Factor Generation** - Turn research ideas written in plain language into candidate factors for further validation
- ✅ **AI-assisted Formula Refinement** - Help improve factor expressions into clearer, library-ready definitions
- ✅ **AI Interpretation of Factor Analysis** - Produce structured research commentary based on factor definitions, analysis results, and chart context
- ✅ **AI Support for Use-case and Risk Review** - Help researchers quickly understand market intuition, limitations, and potential risks behind a factor
- ✅ **AI Report Export** - Organize AI interpretation output into shareable Markdown research notes

---

## Technical Architecture

### Tech Stack

**Backend:**
- FastAPI 0.135+ - High-performance web framework
- SQLAlchemy 2.0 - ORM database operations
- SQLite - Lightweight data storage
- Pandas 2.0+ / NumPy - Data processing
- TA-Lib - Technical analysis library
- VectorBT 0.25+ - Backtesting engine
- DEAP 1.3+ - Genetic algorithm framework
- XGBoost 2.0+ - Machine learning models
- SHAP 0.42+ - Model interpretation
- akshare 1.12+ - Chinese A-share data source

**Frontend:**
- React 19 - UI framework
- TypeScript - Type safety
- Ant Design 6 - UI component library
- ECharts 6 - Data visualization
- React Router 7 - Routing management
- Axios - HTTP client
- Vite - Build tool

---

## Project Structure

```
FactorHub/
├── backend/                    # Backend code
│   ├── api/                   # API layer
│   │   ├── main.py           # FastAPI main application
│   │   └── routers/          # API routers
│   │       ├── factors.py    # Factor management interface
│   │       ├── analysis.py   # Factor analysis interface
│   │       ├── mining.py     # Factor mining interface
│   │       ├── portfolio.py  # Portfolio analysis interface
│   │       ├── backtest.py   # Strategy backtesting interface
│   │       └── data.py       # Data management interface
│   ├── services/              # Business logic layer
│   ├── strategies/            # Strategy implementation
│   ├── repositories/          # Data access layer
│   ├── models/                # ORM models
│   └── core/                  # Core configuration
├── frontend/                   # Frontend code
│   └── react-antd/            # React + Ant Design version
├── config/                     # Configuration files
├── data/                       # Data directory
├── docs/                       # Documentation
├── tests/                      # Tests
├── scripts/                    # Utility scripts
└── README.md
```

---

## Quick Start

### Prerequisites

- **Python 3.11+**
- **Node.js 18+**
- **pnpm** (package manager)
- **TA-Lib** (technical analysis library)

### Installation

```bash
# Install pnpm
npm install -g pnpm

# Install Python dependencies (using uv)
uv sync

# Install frontend dependencies
cd frontend/react-antd
pnpm install
```

### One-click Startup

```bash
python start_all.py
```

This script will automatically:
1. Check environment prerequisites
2. Install dependencies if needed
3. Start backend service (http://localhost:8000)
4. Start frontend development server (http://localhost:5173)
5. Open browser automatically

### Manual Startup

#### Backend

```bash
uv run python start_api.py
# API available at http://localhost:8000
# API docs at http://localhost:8000/docs
```

#### Frontend

```bash
cd frontend/react-antd
pnpm dev
# Frontend available at http://localhost:5173
```

### Docker Deployment

If you prefer to deploy the full application with Docker, use the built-in Docker configuration:

```bash
cd docker
docker compose build
docker compose up -d
```

After startup, access:

| Service | URL |
|---------|-----|
| Frontend | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |

Common commands:

```bash
# View logs
docker compose logs -f

# Stop services
docker compose down
```

Notes:
- The Docker image includes the application code and frontend build artifacts required for runtime
- The `data/` directory is mounted for persistent storage of factor data and runtime outputs
- Local AI model configuration, cache files, test logs, and database backups are not baked into the image

---

## License

### Dual License

**Personal Use:**

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.

You are free to:
- ✅ Use this software for personal learning, research, and non-commercial purposes
- ✅ Modify and improve this software
- ✅ Distribute modified versions (must retain the same license)
- ✅ Reference this project in your own projects

**Commercial Use:**

⚠️ **Important Note:** Any commercial use (including but not limited to:
- Integrating this project into commercial products
- Using this project to provide paid services
- Using this project for production quantitative trading
- Using this project internally within companies for investment research)

**Requires separate commercial authorization.**

---

### Contact for Commercial Authorization

**Email:** yl_zhangqiang@foxmail.com

When contacting, please specify:
1. Your company/organization name
2. Your usage scenario and requirements
3. Expected scale of use
4. Contact information

We will respond within 3 business days.

---

## Contact

**Project Maintainer:** FactorHub Team

**Email:** yl_zhangqiang@foxmail.com

**Feedback Welcome:**
- Bug reports
- Feature suggestions
- Technical discussions
- Cooperation inquiries

---

**Last Updated:** 2026-03-14
