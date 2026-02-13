import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Set style
sns.set_palette("husl")
plt.style.use('seaborn-v0_8-darkgrid')


class BladeVisualizer:
    """Visualization tools for wind turbine blade analysis."""
    
    def __init__(self, save_dir: Optional[str] = None):
        """
        Initialize visualizer.
        
        Args:
            save_dir: Directory to save figures
        """
        self.save_dir = Path(save_dir) if save_dir else None
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_blade_geometry(
        self,
        blade_geometry,
        title: str = "Blade Geometry",
        interactive: bool = False
    ):
        """Plot blade planform and twist distribution."""
        
        if interactive:
            # Plotly interactive version
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Chord Distribution', 'Twist Distribution')
            )
            
            fig.add_trace(
                go.Scatter(
                    x=blade_geometry.span_positions,
                    y=blade_geometry.chord,
                    mode='lines+markers',
                    name='Chord',
                    line=dict(color='blue', width=3)
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=blade_geometry.span_positions,
                    y=blade_geometry.twist,
                    mode='lines+markers',
                    name='Twist',
                    line=dict(color='red', width=3)
                ),
                row=1, col=2
            )
            
            fig.update_xaxes(title_text="Span [m]", row=1, col=1)
            fig.update_xaxes(title_text="Span [m]", row=1, col=2)
            fig.update_yaxes(title_text="Chord [m]", row=1, col=1)
            fig.update_yaxes(title_text="Twist [deg]", row=1, col=2)
            
            fig.update_layout(
                title=title,
                showlegend=False,
                height=400
            )
            
            return fig
        
        else:
            # Matplotlib static version
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            ax1.plot(blade_geometry.span_positions, blade_geometry.chord, 'b-', linewidth=2)
            ax1.set_xlabel('Span [m]')
            ax1.set_ylabel('Chord [m]')
            ax1.set_title('Chord Distribution')
            ax1.grid(True, alpha=0.3)
            
            ax2.plot(blade_geometry.span_positions, blade_geometry.twist, 'r-', linewidth=2)
            ax2.set_xlabel('Span [m]')
            ax2.set_ylabel('Twist [deg]')
            ax2.set_title('Twist Distribution')
            ax2.grid(True, alpha=0.3)
            
            plt.suptitle(title)
            plt.tight_layout()
            
            if self.save_dir:
                plt.savefig(self.save_dir / 'blade_geometry.png', dpi=300, bbox_inches='tight')
            
            return fig
    
    def plot_power_curve(
        self,
        power_curve_data: Dict,
        baseline_data: Optional[Dict] = None,
        title: str = "Power Curve",
        interactive: bool = False
    ):
        """Plot power curve with optional baseline comparison."""
        
        if interactive:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=power_curve_data['wind_speed'],
                y=power_curve_data['power'],
                mode='lines+markers',
                name='Optimized',
                line=dict(color='green', width=3)
            ))
            
            if baseline_data:
                fig.add_trace(go.Scatter(
                    x=baseline_data['wind_speed'],
                    y=baseline_data['power'],
                    mode='lines+markers',
                    name='Baseline',
                    line=dict(color='gray', width=3, dash='dash')
                ))
            
            fig.update_layout(
                title=title,
                xaxis_title='Wind Speed [m/s]',
                yaxis_title='Power [kW]',
                hovermode='x unified',
                height=500
            )
            
            return fig
        
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.plot(
                power_curve_data['wind_speed'],
                power_curve_data['power'],
                'g-', linewidth=2.5, label='Optimized'
            )
            
            if baseline_data:
                ax.plot(
                    baseline_data['wind_speed'],
                    baseline_data['power'],
                    'k--', linewidth=2, label='Baseline'
                )
            
            ax.set_xlabel('Wind Speed [m/s]', fontsize=12)
            ax.set_ylabel('Power [kW]', fontsize=12)
            ax.set_title(title, fontsize=14)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            
            if self.save_dir:
                plt.savefig(self.save_dir / 'power_curve.png', dpi=300, bbox_inches='tight')
            
            return fig
    
    def plot_cp_tsr(
        self,
        power_curve_data: Dict,
        interactive: bool = False
    ):
        """Plot Cp vs TSR curve."""
        
        # Compute TSR
        R = 63.5  # Approximate tip radius
        tsr = power_curve_data['rotor_speed'] * 2 * np.pi / 60 * R / power_curve_data['wind_speed']
        cp = power_curve_data['cp']
        
        # Filter valid range
        valid = (tsr > 0) & (tsr < 15) & (cp > 0)
        
        if interactive:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=tsr[valid],
                y=cp[valid],
                mode='lines+markers',
                name='Cp',
                line=dict(color='purple', width=3)
            ))
            
            # Add Betz limit
            fig.add_hline(
                y=0.593,
                line_dash="dash",
                line_color="red",
                annotation_text="Betz Limit"
            )
            
            fig.update_layout(
                title='Power Coefficient vs. Tip Speed Ratio',
                xaxis_title='Tip Speed Ratio [-]',
                yaxis_title='Power Coefficient Cp [-]',
                height=500
            )
            
            return fig
        
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.plot(tsr[valid], cp[valid], 'purple', linewidth=2.5)
            ax.axhline(y=0.593, color='r', linestyle='--', label='Betz Limit')
            
            ax.set_xlabel('Tip Speed Ratio [-]', fontsize=12)
            ax.set_ylabel('Power Coefficient Cp [-]', fontsize=12)
            ax.set_title('Power Coefficient vs. Tip Speed Ratio', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            if self.save_dir:
                plt.savefig(self.save_dir / 'cp_tsr.png', dpi=300, bbox_inches='tight')
            
            return fig
    
    def plot_spanwise_loads(
        self,
        bem_result,
        blade_geometry,
        interactive: bool = False
    ):
        """Plot spanwise distributions of loads and aerodynamics."""
        
        if interactive:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Angle of Attack',
                    'Lift/Drag Coefficients',
                    'Normal Force',
                    'Tangential Force'
                )
            )
            
            span = blade_geometry.span_positions
            
            # Alpha
            fig.add_trace(
                go.Scatter(x=span, y=bem_result.angle_of_attack, name='Alpha'),
                row=1, col=1
            )
            
            # Cl, Cd
            fig.add_trace(
                go.Scatter(x=span, y=bem_result.cl, name='Cl'),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter(x=span, y=bem_result.cd, name='Cd'),
                row=1, col=2
            )
            
            # Forces
            fig.add_trace(
                go.Scatter(x=span, y=bem_result.normal_force, name='Normal'),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(x=span, y=bem_result.tangent_force, name='Tangent'),
                row=2, col=2
            )
            
            fig.update_xaxes(title_text="Span [m]")
            fig.update_layout(height=800, showlegend=True)
            
            return fig
        
        else:
            fig, axs = plt.subplots(2, 2, figsize=(12, 10))
            span = blade_geometry.span_positions
            
            axs[0, 0].plot(span, bem_result.angle_of_attack)
            axs[0, 0].set_ylabel('Alpha [deg]')
            axs[0, 0].set_title('Angle of Attack')
            axs[0, 0].grid(True, alpha=0.3)
            
            axs[0, 1].plot(span, bem_result.cl, label='Cl')
            axs[0, 1].plot(span, bem_result.cd, label='Cd')
            axs[0, 1].set_ylabel('Coefficient [-]')
            axs[0, 1].set_title('Lift/Drag Coefficients')
            axs[0, 1].legend()
            axs[0, 1].grid(True, alpha=0.3)
            
            axs[1, 0].plot(span, bem_result.normal_force)
            axs[1, 0].set_xlabel('Span [m]')
            axs[1, 0].set_ylabel('Force [N/m]')
            axs[1, 0].set_title('Normal Force')
            axs[1, 0].grid(True, alpha=0.3)
            
            axs[1, 1].plot(span, bem_result.tangent_force)
            axs[1, 1].set_xlabel('Span [m]')
            axs[1, 1].set_ylabel('Force [N/m]')
            axs[1, 1].set_title('Tangential Force')
            axs[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if self.save_dir:
                plt.savefig(self.save_dir / 'spanwise_loads.png', dpi=300, bbox_inches='tight')
            
            return fig
    
    def plot_optimization_history(
        self,
        history: List[Dict],
        title: str = "Optimization Convergence",
        interactive: bool = False
    ):
        """Plot optimization convergence history."""
        
        iterations = [h['iteration'] for h in history]
        best_fitness = [h['best_fitness'] for h in history]
        
        if interactive:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=iterations,
                y=best_fitness,
                mode='lines',
                name='Best Fitness',
                line=dict(color='blue', width=2)
            ))
            
            if 'mean_fitness' in history[0]:
                mean_fitness = [h['mean_fitness'] for h in history]
                fig.add_trace(go.Scatter(
                    x=iterations,
                    y=mean_fitness,
                    mode='lines',
                    name='Mean Fitness',
                    line=dict(color='lightblue', width=2, dash='dash')
                ))
            
            fig.update_layout(
                title=title,
                xaxis_title='Iteration',
                yaxis_title='Fitness',
                height=500
            )
            
            return fig
        
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.plot(iterations, best_fitness, 'b-', linewidth=2, label='Best')
            
            if 'mean_fitness' in history[0]:
                mean_fitness = [h['mean_fitness'] for h in history]
                ax.plot(iterations, mean_fitness, 'b--', linewidth=1.5, alpha=0.6, label='Mean')
            
            ax.set_xlabel('Iteration', fontsize=12)
            ax.set_ylabel('Fitness', fontsize=12)
            ax.set_title(title, fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            if self.save_dir:
                plt.savefig(self.save_dir / 'optimization_history.png', dpi=300, bbox_inches='tight')
            
            return fig
    
    def plot_pareto_front(
        self,
        pareto_front: np.ndarray,
        objective_names: List[str] = ['AEP', 'Mass'],
        interactive: bool = False
    ):
        """Plot Pareto front for multi-objective optimization."""
        
        if interactive:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=pareto_front[:, 0],
                y=pareto_front[:, 1],
                mode='markers',
                marker=dict(size=10, color='red'),
                name='Pareto Front'
            ))
            
            fig.update_layout(
                title='Pareto Front',
                xaxis_title=f'{objective_names[0]} (maximize)',
                yaxis_title=f'{objective_names[1]} (minimize)',
                height=600
            )
            
            return fig
        
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.scatter(pareto_front[:, 0], pareto_front[:, 1], c='red', s=100, alpha=0.6)
            ax.set_xlabel(f'{objective_names[0]} (maximize)', fontsize=12)
            ax.set_ylabel(f'{objective_names[1]} (minimize)', fontsize=12)
            ax.set_title('Pareto Front', fontsize=14)
            ax.grid(True, alpha=0.3)
            
            if self.save_dir:
                plt.savefig(self.save_dir / 'pareto_front.png', dpi=300, bbox_inches='tight')
            
            return fig
    
    def plot_wind_rose(
        self,
        wind_distribution,
        interactive: bool = False
    ):
        """Plot wind speed distribution."""
        
        if interactive:
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=wind_distribution.wind_speeds,
                y=wind_distribution.probabilities,
                name='Wind Distribution',
                marker_color='skyblue'
            ))
            
            fig.update_layout(
                title=f'Wind Distribution (Mean: {wind_distribution.mean_speed:.1f} m/s)',
                xaxis_title='Wind Speed [m/s]',
                yaxis_title='Probability',
                height=500
            )
            
            return fig
        
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.bar(
                wind_distribution.wind_speeds,
                wind_distribution.probabilities,
                width=0.4,
                color='skyblue',
                edgecolor='navy'
            )
            
            ax.set_xlabel('Wind Speed [m/s]', fontsize=12)
            ax.set_ylabel('Probability', fontsize=12)
            ax.set_title(
                f'Wind Distribution (Mean: {wind_distribution.mean_speed:.1f} m/s)',
                fontsize=14
            )
            ax.grid(True, alpha=0.3, axis='y')
            
            if self.save_dir:
                plt.savefig(self.save_dir / 'wind_distribution.png', dpi=300, bbox_inches='tight')
            
            return fig
    
    def create_comparison_dashboard(
        self,
        baseline_results: Dict,
        optimized_results: Dict,
        comparison_metrics: Dict
    ):
        """Create comprehensive comparison dashboard."""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Power Curves',
                'AEP Comparison',
                'Key Metrics',
                'Improvement Summary'
            ),
            specs=[
                [{'type': 'scatter'}, {'type': 'bar'}],
                [{'type': 'bar'}, {'type': 'table'}]
            ]
        )
        
        # Power curves
        fig.add_trace(
            go.Scatter(
                x=baseline_results['power_curve']['wind_speed'],
                y=baseline_results['power_curve']['power'],
                name='Baseline',
                line=dict(dash='dash')
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=optimized_results['power_curve']['wind_speed'],
                y=optimized_results['power_curve']['power'],
                name='Optimized'
            ),
            row=1, col=1
        )
        
        # AEP comparison
        fig.add_trace(
            go.Bar(
                x=['Baseline', 'Optimized'],
                y=[
                    comparison_metrics['baseline_aep_mwh'],
                    comparison_metrics['optimized_aep_mwh']
                ],
                marker_color=['gray', 'green']
            ),
            row=1, col=2
        )
        
        # Key metrics
        metrics = ['Capacity Factor', 'Avg Power']
        baseline_vals = [
            comparison_metrics['baseline_cf'],
            comparison_metrics['baseline_avg_power']
        ]
        optimized_vals = [
            comparison_metrics['optimized_cf'],
            comparison_metrics['optimized_avg_power']
        ]
        
        fig.add_trace(
            go.Bar(name='Baseline', x=metrics, y=baseline_vals),
            row=2, col=1
        )
        fig.add_trace(
            go.Bar(name='Optimized', x=metrics, y=optimized_vals),
            row=2, col=1
        )
        
        # Summary table
        fig.add_trace(
            go.Table(
                header=dict(values=['Metric', 'Improvement']),
                cells=dict(values=[
                    ['AEP', 'Capacity Factor'],
                    [
                        f"{comparison_metrics['aep_improvement_percent']:.1f}%",
                        f"{comparison_metrics['cf_improvement_percent']:.1f}%"
                    ]
                ])
            ),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=True, title_text="Optimization Results Dashboard")
        
        return fig


if __name__ == "__main__":
    # Test visualizations
    from parameterization import create_baseline_blade
    from simulation import BEMSolver
    from aep_calculator import create_weibull_distribution
    
    print("Testing visualizations...")
    
    # Create test data
    blade = create_baseline_blade()
    bem = BEMSolver()
    result = bem.solve(blade, 8.0, 12.1)
    
    # Create visualizer
    viz = BladeVisualizer(save_dir="results/figures")
    
    # Test plots
    print("Plotting blade geometry...")
    viz.plot_blade_geometry(blade)
    
    print("Plotting spanwise loads...")
    viz.plot_spanwise_loads(result, blade)
    
    print("Visualizations saved to results/figures/")