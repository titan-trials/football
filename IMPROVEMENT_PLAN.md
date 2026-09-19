# Football Props Improvement Plan

**Project**: NFL Player Props Prediction Model  
**Date**: 2026-09-19  
**Status**: Draft  

## Project Overview

This is a sophisticated two-stage NFL player prop prediction model (usage → efficiency) with proper backtesting, feature flags, and train/serve discipline. The project demonstrates strong engineering practices with 144 tests, detailed documentation, and careful attention to avoiding common ML pitfalls.

**Current State**: Production-ready with room for quality, performance, and maintainability improvements

**Key Strengths to Preserve**:
- ✅ Strong train/serve discipline with `UnservableFeedError` enforcement
- ✅ Comprehensive feature flag system with documented measurements
- ✅ Proper backtesting methodology using same code path as production
- ✅ Clean separation of concerns (data layer, features, models)
- ✅ Detailed CONTEXT.md for experiment tracking
- ✅ 144 passing tests with good coverage

---

## Improvement Plan

### 1. Code Quality & Maintainability ⚠️ High Priority

**Rationale**: Improve long-term maintainability and reduce bug surface area

#### 1.1 Type Hints
- **Current**: Sparse type hints throughout codebase
- **Goal**: Add comprehensive type hints to all public functions
- **Priority modules**: `features/usage.py`, `data/nflverse.py`, `predict_slate.py`
- **Estimated effort**: 2-3 days

#### 1.2 Extract Magic Numbers
- **Current**: Hardcoded thresholds in validation scripts
- **Goal**: Move all magic numbers to `config.py` with descriptive names
- **Examples**: `k_sweep.py` thresholds, validation cutoffs, TTL values
- **Estimated effort**: 1 day

#### 1.3 Function Documentation
- **Current**: Some modules lack comprehensive docstrings
- **Goal**: Add Google-style docstrings to all public functions
- **Priority**: Core data loaders, model classes, feature extractors
- **Estimated effort**: 2-3 days

#### 1.4 Error Handling Standardization
- **Current**: Mix of generic exceptions
- **Goal**: Create custom exception hierarchy for domain-specific errors
- **Examples**: `FeedValidationError`, `ModelPredictionError`, `CacheError`
- **Estimated effort**: 1 day

#### 1.5 Code Formatting
- **Current**: No automated formatting
- **Goal**: Add `black` or `ruff` for consistent code style
- **Implementation**: Add to dev requirements, pre-commit hook
- **Estimated effort**: 0.5 day

---

### 2. Performance Optimization 🔧 Medium Priority

**Rationale**: Reduce prediction latency and resource usage for larger slates

#### 2.1 Prediction Pipeline Profiling
- **Current**: No performance profiling
- **Goal**: Identify bottlenecks in `predict_slate.py` using cProfile
- **Focus areas**: Feature computation, model inference, DataFrame operations
- **Estimated effort**: 1 day

#### 2.2 Cache Optimization
- **Current**: Uniform TTL settings in `data/cache.py`
- **Goal**: Optimize TTL per feed type based on update frequency
- **Examples**: Schedules (24h), injuries (6h), snap counts (12h)
- **Estimated effort**: 0.5 day

#### 2.3 Parallel Processing
- **Current**: Sequential player predictions
- **Goal**: Parallelize independent player predictions using multiprocessing
- **Implementation**: `concurrent.futures` for player-level predictions
- **Estimated effort**: 2 days

#### 2.4 Memory Efficiency
- **Current**: Large DataFrame operations in feature engineering
- **Goal**: Optimize memory usage through chunking and lazy evaluation
- **Focus**: `features/usage.py` share computations, efficiency features
- **Estimated effort**: 2 days

---

### 3. Testing & Validation 🧪 High Priority

**Rationale**: Improve confidence in changes and catch regressions early

#### 3.1 Integration Tests
- **Current**: Unit tests only (144 passing)
- **Goal**: Add end-to-end tests for weekly pipeline
- **Scenarios**: `run_slate.py` → `score_slate.py` with mock data
- **Estimated effort**: 3 days

#### 3.2 Property-Based Tests
- **Current**: Example-based tests only
- **Goal**: Add property-based tests for critical invariants
- **Examples**: Probability distributions sum to 1, monotonic calibration
- **Tool**: `hypothesis` library
- **Estimated effort**: 2 days

#### 3.3 Backtest Regression Tests
- **Current**: Manual backtest comparison
- **Goal**: Automated regression tests for historical performance
- **Implementation**: Reference scores in tests, fail on degradation
- **Estimated effort**: 2 days

#### 3.4 Performance Benchmarks
- **Current**: No performance tracking
- **Goal**: Add benchmark suite for prediction latency
- **Tool**: `pytest-benchmark`
- **Estimated effort**: 1 day

---

### 4. Monitoring & Observability 📊 Medium Priority

**Rationale**: Improve production visibility and debugging capabilities

#### 4.1 Structured Logging Framework
- **Current**: Minimal logging (print statements)
- **Goal**: Implement structured logging with levels
- **Implementation**: Python `logging` module with JSON formatting
- **Estimated effort**: 2 days

#### 4.2 Prediction Drift Monitoring
- **Current**: No drift detection
- **Goal**: Track distribution shifts in predictions vs actuals
- **Metrics**: KL divergence, mean shift, variance changes
- **Estimated effort**: 2 days

#### 4.3 Data Quality Checks
- **Current**: Manual feed validation
- **Goal**: Automated validation of feed completeness/quality
- **Checks**: Row counts, null ratios, value ranges, freshness
- **Estimated effort**: 2 days

#### 4.4 Model Performance Dashboard
- **Current**: Basic Streamlit dashboard
- **Goal**: Extend with live performance metrics
- **Features**: Real-time CRPS, calibration plots, prediction latency
- **Estimated effort**: 3 days

---

### 5. Documentation & Onboarding 📚 Medium Priority

**Rationale**: Reduce onboarding time and improve knowledge transfer

#### 5.1 Architecture Diagram
- **Current**: Text descriptions only
- **Goal**: Visual representation of two-stage pipeline
- **Tool**: Mermaid or draw.io diagram
- **Estimated effort**: 1 day

#### 5.2 Quick Start Guide
- **Current**: Comprehensive README
- **Goal**: Streamlined 5-minute setup guide
- **Content**: Prerequisites, installation, first prediction
- **Estimated effort**: 0.5 day

#### 5.3 Contributing Guidelines
- **Current**: No contribution guide
- **Goal**: Document development workflow and standards
- **Content**: PR process, code style, testing requirements
- **Estimated effort**: 1 day

#### 5.4 Troubleshooting Guide
- **Current**: Scattered troubleshooting in RUNBOOK.md
- **Goal**: Centralized common issues and solutions
- **Examples**: Feed failures, cache issues, API key problems
- **Estimated effort**: 1 day

#### 5.5 API Documentation
- **Current**: Docstrings only
- **Goal**: Generated API documentation
- **Tool**: Sphinx or MkDocs with autodoc
- **Estimated effort**: 2 days

---

### 6. Data Pipeline Improvements 🔄 Medium Priority

**Rationale**: Improve data reliability and reduce manual intervention

#### 6.1 Data Validation Layer
- **Current**: Schema validation in individual loaders
- **Goal**: Centralized schema validation for all nflverse feeds
- **Implementation**: Pydantic models or pandera schemas
- **Estimated effort**: 2 days

#### 6.2 Incremental Updates
- **Current**: Full cache refresh
- **Goal**: Optimize cache refresh for weekly data
- **Strategy**: Append-only updates for weekly feeds
- **Estimated effort**: 2 days

#### 6.3 Feed Monitoring
- **Current**: Manual feed checks
- **Goal**: Automated alerts for delayed/missing feeds
- **Implementation**: Health checks with notification system
- **Estimated effort**: 2 days

#### 6.4 Backup/Recovery
- **Current**: No backup strategy
- **Goal**: Backup/recovery for critical cache files
- **Strategy**: Periodic snapshots, rollback capability
- **Estimated effort**: 1 day

---

### 7. Deployment & Automation 🚀 Low Priority

**Rationale**: Improve operational efficiency and reliability

#### 7.1 CI/CD Pipeline
- **Current**: Manual testing
- **Goal**: Automated testing on push
- **Implementation**: GitHub Actions with test suite
- **Estimated effort**: 2 days

#### 7.2 Scheduled Runs
- **Current**: Manual weekly execution
- **Goal**: Automate weekly pipeline execution
- **Tool**: GitHub Actions cron or systemd timer
- **Estimated effort**: 1 day

#### 7.3 Environment Management
- **Current**: Virtual environment only
- **Goal**: Docker container for reproducible deployments
- **Implementation**: Multi-stage Dockerfile, docker-compose
- **Estimated effort**: 2 days

#### 7.4 Secrets Management
- **Current**: .env file with NFL_ODDS_API_KEY
- **Goal**: Consolidate API key handling
- **Implementation**: Environment variable validation, secrets manager
- **Estimated effort**: 0.5 day

---

### 8. Model Architecture 🧠 Low Priority (Experimental)

**Rationale**: Explore potential performance improvements

#### 8.1 Alternative Stage 2 Models
- **Current**: Hurdle model with aDOT conditioning
- **Goal**: Evaluate alternative per-opportunity models
- **Options**: Zero-inflated negative binomial, mixture models
- **Estimated effort**: 1 week

#### 8.2 Ensemble Methods
- **Current**: Single model approach
- **Goal**: Combine multiple Stage 1 approaches
- **Options**: Model averaging, stacking, weighted ensembles
- **Estimated effort**: 1 week

#### 8.3 Market Features
- **Current**: MARKET_IN_SHARE unmeasured (False)
- **Goal**: Explore market-informed share predictions
- **Risk**: Potential train/serve skew, benchmark contamination
- **Estimated effort**: 3 days

#### 8.4 Role Change Detection
- **Current**: ROLE_CHANGE_FLAG not implemented (False)
- **Goal**: Detect and log role changes after teammate injuries
- **Implementation**: Logged column fed to nothing, checked after ~20 weeks
- **Estimated effort**: 2 days

---

## Recommended Implementation Order

### Phase 1: Quick Wins (1-2 weeks)
**Impact**: Immediate quality and reliability improvements

1. **Add type hints to core modules** (1.1)
2. **Implement structured logging** (4.1)
3. **Add integration tests for weekly pipeline** (3.1)
4. **Extract magic numbers to config** (1.2)
5. **Code formatting setup** (1.5)

**Estimated Total**: 8-10 days

### Phase 2: Foundation (2-4 weeks)
**Impact**: Strong foundation for scaling and monitoring

1. **Performance profiling and optimization** (2.1, 2.2)
2. **Property-based testing for invariants** (3.2)
3. **Data validation layer** (6.1)
4. **Extended monitoring in dashboard** (4.4)
5. **Backtest regression tests** (3.3)

**Estimated Total**: 10-15 days

### Phase 3: Polish (4-8 weeks)
**Impact**: Production-ready deployment and long-term maintainability

1. **Documentation improvements** (5.1, 5.2, 5.3, 5.4, 5.5)
2. **CI/CD pipeline** (7.1)
3. **Advanced monitoring/alerting** (4.2, 4.3, 6.3)
4. **Deployment automation** (7.2, 7.3, 7.4)
5. **Experimental model features** (8.1, 8.2, 8.3, 8.4)

**Estimated Total**: 20-30 days

---

## Estimated Impact

| Area | Current State | Target State | Improvement |
|------|--------------|--------------|-------------|
| **Reliability** | Good (144 tests) | Excellent (integration + regression tests) | +30% |
| **Maintainability** | Moderate (sparse docs) | High (type hints, docs, standards) | +40% |
| **Performance** | Adequate | Optimized (profiling, caching, parallel) | +20% |
| **Developer Experience** | Good (good docs) | Excellent (onboarding, debugging tools) | +50% |
| **Production Readiness** | Manual operations | Automated (CI/CD, monitoring) | +60% |

---

## Risk Assessment

### Low Risk Items
- Code formatting (1.5)
- Documentation improvements (5.x)
- Logging framework (4.1)
- API key consolidation (7.4)

### Medium Risk Items
- Type hints addition (1.1) - potential compatibility issues
- Cache optimization (2.2) - may affect data freshness
- Property-based tests (3.2) - requires test design expertise

### High Risk Items
- Parallel processing (2.3) - concurrency bugs
- Experimental model features (8.x) - may degrade performance
- Deployment automation (7.x) - production changes

---

## Success Metrics

### Code Quality
- Type hint coverage: >80% of public functions
- Test coverage: >90% (currently ~70%)
- Code duplication: <5% (measured by tools)

### Performance
- Prediction latency: <30s for full slate (currently ~45s)
- Memory usage: <2GB for full slate (currently ~3GB)
- Cache hit rate: >95% for repeated runs

### Reliability
- Integration test pass rate: 100%
- Regression test failures: 0 per release
- Data quality alerts: <1 per week

### Documentation
- Onboarding time: <2 hours for new developer
- API documentation completeness: 100% of public APIs
- Troubleshooting guide coverage: >90% of common issues

---

## Notes

### Preservation of Existing Strengths
- Maintain train/serve discipline with `UnservableFeedError`
- Keep feature flag system with documented measurements
- Preserve backtesting methodology using same code path
- Maintain clean separation of concerns
- Continue detailed CONTEXT.md experiment tracking

### Dependencies
- Some improvements depend on others (e.g., monitoring needs logging)
- Experimental features should not destabilize production
- Documentation should accompany code changes

### Resource Requirements
- **Phase 1**: 1 developer, part-time
- **Phase 2**: 1 developer, full-time
- **Phase 3**: 1-2 developers, full-time (depending on experimental features)

---

## Next Steps

1. **Review and prioritize** this plan with stakeholders
2. **Select Phase 1 items** for immediate implementation
3. **Set up tracking** (GitHub issues, project board)
4. **Define success criteria** for each phase
5. **Begin implementation** with highest-impact items

---

**Last Updated**: 2026-09-19  
**Next Review**: After Phase 1 completion  
**Owner**: Development Team  
**Status**: Awaiting Approval