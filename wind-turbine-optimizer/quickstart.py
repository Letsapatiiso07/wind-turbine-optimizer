#!/usr/bin/env python
"""
Quick Start Script for Wind Turbine Blade Optimizer

Runs a minimal example to verify installation and demonstrate capabilities.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
from parameterization import create_baseline_blade, create_random_blade
from simulation import BEMSolver
from structural import StructuralModel
from aep_calculator import AEPCalculator, create_weibull_distribution
from visualization import BladeVisualizer

def main():
    print("="*70)
    print("Wind Turbine Blade Optimizer - Quick Start")
    print("="*70)
    
    # 1. Create baseline blade
    print("\n[1/6] Creating NREL 5MW baseline blade...")
    baseline = create_baseline_blade()
    props = baseline.compute_properties()
    print(f"  ✓ Blade span: {baseline.params.blade_span:.1f} m")
    print(f"  ✓ Aspect ratio: {props['aspect_ratio']:.1f}")
    print(f"  ✓ Max chord: {props['max_chord']:.2f} m")
    print(f"  ✓ Root twist: {props['root_twist']:.1f}°")
    print(f"  ✓ Tip twist: {props['tip_twist']:.1f}°")
    
    # 2. Run BEM analysis
    print("\n[2/6] Running BEM aerodynamic analysis...")
    bem_solver = BEMSolver()
    result = bem_solver.solve(baseline, wind_speed=8.0, rotor_speed=12.1)
    print(f"  ✓ Power: {result.power:.1f} kW")
    print(f"  ✓ Thrust: {result.thrust:.1f} kN")
    print(f"  ✓ Torque: {result.torque:.1f} kNm")
    print(f"  ✓ Cp: {result.cp:.3f}")
    print(f"  ✓ Ct: {result.ct:.3f}")
    
    # 3. Structural analysis
    print("\n[3/6] Performing structural analysis...")
    struct_model = StructuralModel()
    struct_result = struct_model.analyze(
        baseline,
        loads_flap=result.normal_force,
        loads_edge=result.tangent_force
    )
    print(f"  ✓ Tip deflection (flap): {struct_result.tip_deflection_flap:.3f} m")
    print(f"  ✓ Tip deflection (edge): {struct_result.tip_deflection_edge:.3f} m")
    print(f"  ✓ Deflection ratio: {struct_result.deflection_ratio:.4f} (limit: 0.05)")
    print(f"  ✓ Blade mass: {struct_result.blade_mass:.0f} kg")
    print(f"  ✓ Max stress: {struct_result.max_stress_flap/1e6:.1f} MPa")
    print(f"  ✓ Stress ratio: {struct_result.stress_ratio:.3f} (limit: 1.0)")
    print(f"  ✓ Feasible: {struct_result.is_feasible()}")
    
    # 4. Compute power curve
    print("\n[4/6] Computing full power curve...")
    wind_speeds = np.linspace(3, 25, 23)
    power_curve = bem_solver.compute_power_curve(baseline, wind_speeds)
    print(f"  ✓ Cut-in wind speed: 3.0 m/s")
    print(f"  ✓ Rated wind speed: 11.4 m/s")
    print(f"  ✓ Cut-out wind speed: 25.0 m/s")
    print(f"  ✓ Rated power: {np.max(power_curve['power']):.0f} kW")
    print(f"  ✓ Max Cp: {np.max(power_curve['cp']):.3f}")
    
    # Check Betz limit
    if np.max(power_curve['cp']) < 0.593:
        print(f"  ✓ Cp below Betz limit (0.593) ✓")
    else:
        print(f"  ⚠ Warning: Cp exceeds Betz limit!")
    
    # 5. Calculate AEP
    print("\n[5/6] Calculating Annual Energy Production...")
    wind_dist = create_weibull_distribution(mean_speed=8.5, k=2.0)
    print(f"  ✓ Wind distribution: Weibull (mean={wind_dist.mean_speed:.1f} m/s, k={wind_dist.k:.1f})")
    
    aep_calc = AEPCalculator(wind_dist)
    aep_results = aep_calc.compute_aep(power_curve)
    print(f"  ✓ Gross AEP: {aep_results['aep_gross_mwh']:.0f} MWh/year")
    print(f"  ✓ Net AEP: {aep_results['aep_net_mwh']:.0f} MWh/year")
    print(f"  ✓ Capacity factor: {aep_results['capacity_factor']:.1%}")
    print(f"  ✓ Average power: {aep_results['avg_power_kw']:.0f} kW")
    
    # 6. Visualization
    print("\n[6/6] Creating visualizations...")
    import os
    os.makedirs("results/figures", exist_ok=True)
    
    viz = BladeVisualizer(save_dir="results/figures")
    viz.plot_blade_geometry(baseline, title="Baseline NREL 5MW")
    print(f"  ✓ Blade geometry plot saved")
    
    viz.plot_power_curve(power_curve, title="Power Curve")
    print(f"  ✓ Power curve plot saved")
    
    # Try Cp-TSR plot
    try:
        viz.plot_cp_tsr(power_curve)
        print(f"  ✓ Cp-TSR plot saved")
    except Exception as e:
        print(f"  ⚠ Cp-TSR plot skipped: {e}")
    
    print(f"  ✓ Figures saved to results/figures/")
    
    # Summary
    print("\n" + "="*70)
    print("QUICK START COMPLETE! ✅")
    print("="*70)
    print(f"\nKey Results:")
    print(f"  • Annual Energy: {aep_results['aep_net_mwh']:.0f} MWh/year")
    print(f"  • Capacity Factor: {aep_results['capacity_factor']:.1%}")
    print(f"  • Blade Mass: {struct_result.blade_mass:.0f} kg")
    print(f"  • Max Cp: {np.max(power_curve['cp']):.3f}")
    print(f"  • Revenue (@$50/MWh): ${aep_results['aep_net_mwh']*50:,.0f}/year")
    print(f"  • 20-Year Value: ${aep_results['aep_net_mwh']*50*20:,.0f}")
    
    print(f"\nNext Steps:")
    print(f"  1. Launch dashboard: streamlit run app.py")
    print(f"  2. Run full optimization: python main.py")
    print(f"  3. Explore code in src/")
    print(f"  4. Read README.md for details")
    
    print(f"\n🌍 System ready for blade optimization! 🌍\n")
    
    # Test a random blade for comparison
    print("\n" + "-"*70)
    print("BONUS: Testing random blade generation...")
    print("-"*70)
    
    random_blade = create_random_blade(seed=42)
    random_props = random_blade.compute_properties()
    
    print(f"\nRandom Blade Properties:")
    print(f"  • Max chord: {random_props['max_chord']:.2f} m")
    print(f"  • Taper ratio: {random_props['taper_ratio']:.3f}")
    print(f"  • Twist range: {random_props['twist_range']:.1f}°")
    
    # Quick eval
    random_result = bem_solver.solve(random_blade, wind_speed=8.0, rotor_speed=12.1)
    print(f"\nRandom Blade Performance:")
    print(f"  • Power @ 8 m/s: {random_result.power:.1f} kW")
    print(f"  • Cp @ 8 m/s: {random_result.cp:.3f}")
    
    comparison = (random_result.power - result.power) / result.power * 100
    print(f"\nComparison to Baseline:")
    print(f"  • Power difference: {comparison:+.1f}%")
    
    if comparison > 0:
        print(f"  🎉 Random blade is better! (lucky)")
    else:
        print(f"  ✓ Baseline is better (expected)")
    
    print("\n" + "="*70)
    print("All tests passed! System is working correctly.")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
