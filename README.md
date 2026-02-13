# Wind Turbine Blade Optimizer

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **ML-accelerated wind turbine blade design optimization achieving 5-15% energy production improvements**

An end-to-end system combining physics-based simulation with machine learning to optimize wind turbine blade designs **500× faster** than traditional CFD methods while maintaining high accuracy (R² > 0.95).

---

## Overview

This project demonstrates how machine learning can accelerate renewable energy engineering. By creating fast surrogate models of expensive aerodynamic simulations, we can explore thousands of blade designs in minutes instead of weeks.

**Real-world impact:** For a 100-turbine wind farm, these optimizations could generate:
- **$7.5M+** additional revenue per year
- **150,000 MWh** extra clean energy annually  
- **75,000 tons** CO₂ offset per year

---

## Key Features

| Feature | Description |
|---------|-------------|
| **500× Speedup** | ML surrogate models (0.001s) vs. traditional BEM simulation (0.5s) |
| **5-15% AEP Gains** | Demonstrated improvements over NREL 5MW baseline design |
| **Ensemble ML** | Neural Networks + XGBoost + Random Forest (R² > 0.95) |
| **Physics-Based** | BEM aerodynamics + structural analysis + fatigue modeling |
| **Interactive UI** | Streamlit dashboard for real-time design exploration |
| **Production Ready** | 5,900+ lines, fully documented, tested on Windows/Linux/Mac |

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Letsapatiiso07/wind-turbine-optimizer.git
cd wind-turbine-optimizer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install "scipy>=1.9.0,<1.14.0"  # Important: compatible version
pip install -r requirements.txt
```

### Run the Demo (5 minutes)

```bash
python quickstart.py
```

**Expected output:**
```
======================================================================
Wind Turbine Blade Optimizer - Quick Start
======================================================================

[1/6] Creating NREL 5MW baseline blade...
  ✓ Blade span: 63.0 m
  ✓ Aspect ratio: 16.2
  
[2/6] Running BEM aerodynamic analysis...
  ✓ Power: 1834.1 kW
  ✓ Cp: 0.447
  
...

QUICK START COMPLETE!
  • Annual Energy: 14,920 MWh/year
  • Capacity Factor: 34.0%
  • Revenue (@$50/MWh): $746,000/year
```

### Launch Dashboard

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

### Run Optimization

```bash
# With ML surrogate (fast - recommended)
python main.py --wind-speed 8.5 --n-samples 5000 --method GA

# Direct evaluation (slower but no training needed)
python main.py --skip-surrogate --generations 100
```

---

## How It Works

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   BLADE PARAMETERIZATION                     │
│   15 Parameters → Chord, Twist, Airfoil Family               │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  PHYSICS SIMULATION                          │
│  BEM (Aerodynamics) → Power, Thrust, Loads                   │
│  Beam Theory (Structural) → Deflections, Stresses            │
│  Fatigue Analysis → Damage, Lifetime                         │
│  AEP Calculation → Annual Energy Production                  │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              ML SURROGATE (500× FASTER)                      │
│  Database: 10k-100k Latin Hypercube Samples                  │
│  Models: Neural Net + XGBoost + Random Forest                │
│  Accuracy: R² > 0.95, MAPE < 5%                              │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                   OPTIMIZATION                               │
│  Genetic Algorithm (NSGA-II): Pop=100, Gen=500               │
│  PSO: Swarm=50, Iterations=300                               │
│  Constraints: Fatigue D<1, Deflection<5%, Stress OK          │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              OPTIMIZED BLADE DESIGN                          │
│  AEP: +5-15% | Cp: ~0.48 | Mass: ±10% | Feasible: ✓        │
└─────────────────────────────────────────────────────────────┘
```

---

## Results

### Performance Comparison

| Metric | Baseline | Optimized | Improvement |
|--------|----------|-----------|-------------|
| **AEP** | 14,850 MWh/yr | 16,320 MWh/yr | **+9.9%** |
| **Capacity Factor** | 33.8% | 37.2% | **+10.1%** |
| **Blade Mass** | 17,740 kg | 16,890 kg | **-4.8%** |
| **Max Cp** | 0.465 | 0.487 | **+4.7%** |

### Computational Performance

- **ML Surrogate:** 0.001s per evaluation
- **Direct BEM:** 0.5s per evaluation
- **Speedup:** **500×** faster
- **Accuracy:** R² > 0.95 (< 5% error)

### Economic Impact (100-Turbine Farm)

- Additional Revenue: **$7.5M/year** (@$50/MWh)
- Extra Energy: **150,000 MWh/year**
- CO₂ Offset: **75,000 tons/year**
- 20-Year Value: **$150M+**

---

## Technical Stack

**Core:**
- Python 3.10+ • PyTorch 2.0+ • XGBoost 1.7+ • scikit-learn 1.3+

**Simulation:**
- SciPy 1.9-1.13 (BEM solver) • NumPy 1.24+ (numerics)

**Optimization:**
- PyMOO 0.6+ (NSGA-II) • Custom PSO implementation

**Visualization:**
- Streamlit 1.28+ (dashboard) • Plotly 5.14+ (interactive) • Matplotlib 3.7+ (static)

**Data:**
- h5py 3.9+ (HDF5 storage) • Pandas 2.0+ (data manipulation) • Joblib 1.3+ (parallel)

---

## Usage Examples

### Evaluate Baseline Design

```python
from src import create_baseline_blade, BEMSolver, AEPCalculator, create_weibull_distribution

# Create NREL 5MW baseline
blade = create_baseline_blade()

# Run BEM analysis
bem = BEMSolver()
result = bem.solve(blade, wind_speed=8.0, rotor_speed=12.1)
print(f"Power: {result.power:.1f} kW, Cp: {result.cp:.3f}")

# Calculate AEP
wind_dist = create_weibull_distribution(mean_speed=8.5, k=2.0)
power_curve = bem.compute_power_curve(blade, wind_speeds=range(3, 26))
aep_calc = AEPCalculator(wind_dist)
aep = aep_calc.compute_aep(power_curve)
print(f"AEP: {aep['aep_net_mwh']:.0f} MWh/year")
```

### Train ML Surrogate

```python
from src import DatabaseGenerator, SurrogateEnsemble

# Generate training database
generator = DatabaseGenerator(n_samples=5000)
inputs, outputs = generator.generate_database("training_data.h5")

# Train ensemble
surrogate = SurrogateEnsemble(n_features=15, n_outputs=5)
surrogate.train(X_train, y_train, X_val, y_val, epochs=200)

# Evaluate
metrics = surrogate.evaluate(X_test, y_test)
print(f"R² Score: {metrics['r2_mean']:.4f}")  # Should be > 0.95
```

### Run Optimization

```python
from src import GeneticAlgorithmOptimizer

# Define bounds
param_bounds = np.array([[2.0, 5.0]] * 6 + [[0.0, 20.0]] * 6 + [[0, 2]] * 3)

# Optimize
optimizer = GeneticAlgorithmOptimizer(population_size=100, n_generations=300)
result = optimizer.optimize(objective_func, param_bounds, constraints)

print(f"Best AEP: {-result.best_fitness:.0f} MWh/year")
print(f"Time: {result.total_time:.1f}s")
```

---

## Technical Details

### BEM Aerodynamics
- Prandtl tip loss correction
- Glauert high-induction model
- Variable speed + pitch control
- ~0.5s per full power curve

### Structural Analysis
- 1D Euler-Bernoulli beam model
- Composite materials (E=40 GPa)
- Deflection + stress constraints
- Fatigue via Miner's rule

### Machine Learning
- **Architecture:** NN [128,256,128] + XGBoost + RF
- **Training:** 10k-100k LHS samples
- **Features:** 15 geometry parameters
- **Targets:** AEP, Cp, mass, deflection, stress
- **Ensemble:** 0.5 NN + 0.3 XGB + 0.2 RF

### Optimization
- **NSGA-II:** Multi-objective genetic algorithm
- **PSO:** Swarm intelligence
- **Constraints:** Physics-based (fatigue, deflection, stress)
- **Runtime:** ~30 min with surrogate

---

## Documentation

- **Installation:** [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md)
- **Troubleshooting:** [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- **API Reference:** Docstrings in code (Google style)
- **Examples:** See `notebooks/` directory

---

## Testing

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/ -v

# With coverage
pytest --cov=src tests/
```

**Coverage:** 75%+ on core modules

---

## Deployment

### Streamlit Cloud (Free!)

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repository → Deploy `app.py`
4. Get live URL to share!

### Local Server

```bash
streamlit run app.py --server.port=8501
```

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/amazing`)
3. Commit changes (`git commit -m 'Add feature'`)
4. Push (`git push origin feature/amazing`)
5. Open a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

MIT License - see [LICENSE](LICENSE) file.

**TL;DR:** Use freely, even commercially. Just include the license.

---

## Acknowledgments

**Data & References:**
- NREL - 5MW Reference Turbine
- NREL WIND Toolkit - Wind distributions
- Airfoil Tools - Polar database

**Inspiration:**
- Burton et al. - Wind Energy Handbook
- IEC 61400-1 Standards

**Tools:**
- [PyTorch](https://pytorch.org/) • [Streamlit](https://streamlit.io/) • [PyMOO](https://pymoo.org/)

---

## Status

- Core functionality: Complete
- Documentation: Comprehensive
- Dashboard: Fully functional
- Tests: 75%+ coverage
- Advanced features: In development

**Latest:** v1.2 (Windows-compatible)

---

## Contact

**Questions?**
- [Open an issue](https://github.com/YOUR-USERNAME/wind-turbine-optimizer/issues)
- Email: your.email@example.com
- LinkedIn: [Your Profile](https://linkedin.com/in/tiiso-letsapa-664990209)

**Found a bug?** Please report with Python version, OS, error message, and steps to reproduce.

---

## Show Your Support

If you find this project useful:
- Star the repository
- Share on social media
- Write a blog post
- Use in your work (with attribution)

---

## Impact

**This project contributes to renewable energy by:**
- Demonstrating ML acceleration of engineering simulation
- Enabling faster, better wind turbine designs
- Supporting transition to clean energy
- Providing open-source tools for researchers

**Together, we can accelerate the renewable energy transition!**

---

<div align="center">

**Built with ❤️ for renewable energy**

Made by Tiiso Letsapa

</div>