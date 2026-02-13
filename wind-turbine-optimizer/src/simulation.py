"""
Blade Element Momentum (BEM) Simulation Module

Implements BEM theory for wind turbine aerodynamic analysis.
Includes structural analysis and fatigue calculations.
"""

import numpy as np
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings


@dataclass
class AirfoilPolar:
    """Airfoil aerodynamic polar data."""
    
    alpha: np.ndarray  # Angle of attack [degrees]
    cl: np.ndarray     # Lift coefficient
    cd: np.ndarray     # Drag coefficient
    cm: np.ndarray     # Moment coefficient (optional)
    
    def interpolate_coefficients(self, alpha_query: float) -> Tuple[float, float]:
        """
        Interpolate Cl and Cd at a given angle of attack.
        
        Args:
            alpha_query: Angle of attack [degrees]
            
        Returns:
            (cl, cd) at the query angle
        """
        # Linear interpolation, extrapolate for out-of-bounds
        cl_interp = np.interp(alpha_query, self.alpha, self.cl)
        cd_interp = np.interp(alpha_query, self.alpha, self.cd)
        
        return cl_interp, cd_interp


class AirfoilDatabase:
    """Database of airfoil polars for different families."""
    
    def __init__(self):
        """Initialize with standard airfoil families."""
        self.polars = self._create_default_polars()
    
    def _create_default_polars(self) -> Dict[int, AirfoilPolar]:
        """
        Create default airfoil polars (simplified).
        
        In production, load from files (e.g., NREL airfoils).
        Returns:
            Dictionary mapping family index to AirfoilPolar
        """
        # Alpha range
        alpha = np.linspace(-10, 20, 61)
        
        # Family 0: Thick root airfoil (DU-W-405, 40% thick)
        cl_0 = np.where(
            alpha < 12,
            0.1 * alpha,
            1.2 - 0.05 * (alpha - 12)  # Stall
        )
        cd_0 = 0.015 + 0.0002 * alpha**2
        
        # Family 1: Mid-span airfoil (DU-96-W-180, 18% thick)
        cl_1 = np.where(
            alpha < 14,
            0.11 * alpha,
            1.5 - 0.1 * (alpha - 14)  # Stall
        )
        cd_1 = 0.008 + 0.0003 * alpha**2
        
        # Family 2: Thin tip airfoil (NACA 64-618, 18% thick)
        cl_2 = np.where(
            alpha < 15,
            0.105 * alpha,
            1.6 - 0.12 * (alpha - 15)  # Stall
        )
        cd_2 = 0.006 + 0.00035 * alpha**2
        
        return {
            0: AirfoilPolar(alpha, cl_0, cd_0, np.zeros_like(alpha)),
            1: AirfoilPolar(alpha, cl_1, cd_1, np.zeros_like(alpha)),
            2: AirfoilPolar(alpha, cl_2, cd_2, np.zeros_like(alpha))
        }
    
    def get_polar(self, family_idx: int) -> AirfoilPolar:
        """Get airfoil polar by family index."""
        return self.polars.get(family_idx, self.polars[1])  # Default to mid


@dataclass
class BEMResult:
    """Results from BEM analysis."""
    
    # Global quantities
    power: float           # [kW]
    thrust: float          # [kN]
    torque: float          # [kNm]
    cp: float              # Power coefficient
    ct: float              # Thrust coefficient
    
    # Spanwise distributions
    axial_induction: np.ndarray      # a
    tangential_induction: np.ndarray # a'
    angle_of_attack: np.ndarray      # [deg]
    inflow_angle: np.ndarray         # phi [deg]
    cl: np.ndarray                   # Lift coefficient
    cd: np.ndarray                   # Drag coefficient
    
    # Loads
    normal_force: np.ndarray   # [N/m] per span
    tangent_force: np.ndarray  # [N/m] per span
    
    # Bending moments (at root)
    flapwise_moment: float  # [kNm]
    edgewise_moment: float  # [kNm]


class BEMSolver:
    """
    Blade Element Momentum solver for wind turbine analysis.
    
    Implements classical BEM with Prandtl tip loss correction.
    """
    
    def __init__(
        self,
        airfoil_db: Optional[AirfoilDatabase] = None,
        rho: float = 1.225,  # Air density [kg/m³]
        max_iterations: int = 100,
        tolerance: float = 1e-5
    ):
        """
        Initialize BEM solver.
        
        Args:
            airfoil_db: Airfoil database
            rho: Air density
            max_iterations: Max BEM iterations per section
            tolerance: Convergence tolerance
        """
        self.airfoil_db = airfoil_db or AirfoilDatabase()
        self.rho = rho
        self.max_iterations = max_iterations
        self.tolerance = tolerance
    
    def solve(
        self,
        blade_geometry,  # BladeGeometry instance
        wind_speed: float,
        rotor_speed: float,  # [RPM]
        pitch: float = 0.0,  # [deg] collective pitch
        yaw: float = 0.0     # [deg] yaw misalignment
    ) -> BEMResult:
        """
        Solve BEM equations for given operating conditions.
        
        Args:
            blade_geometry: BladeGeometry object
            wind_speed: Free stream wind speed [m/s]
            rotor_speed: Rotor speed [RPM]
            pitch: Collective pitch angle [deg]
            yaw: Yaw misalignment [deg]
            
        Returns:
            BEMResult object
        """
        # Convert units
        omega = rotor_speed * 2 * np.pi / 60  # [rad/s]
        
        # Extract blade data
        r = blade_geometry.radial_positions
        chord = blade_geometry.chord
        twist = blade_geometry.twist + pitch  # Total twist
        airfoil_families = blade_geometry.airfoil_family
        n_blades = blade_geometry.params.n_blades
        
        # Initialize arrays
        n_sections = len(r)
        a = np.zeros(n_sections)      # Axial induction
        a_prime = np.zeros(n_sections)  # Tangential induction
        alpha = np.zeros(n_sections)
        phi = np.zeros(n_sections)
        cl = np.zeros(n_sections)
        cd = np.zeros(n_sections)
        f_normal = np.zeros(n_sections)
        f_tangent = np.zeros(n_sections)
        
        # Account for yaw (simple cosine correction)
        wind_speed_eff = wind_speed * np.cos(np.radians(yaw))
        
        # Solve BEM for each section
        for i in range(n_sections):
            if r[i] < 0.01:  # Skip hub region
                continue
            
            # Get airfoil polar
            polar = self.airfoil_db.get_polar(airfoil_families[i])
            
            # Prandtl tip loss factor
            R = blade_geometry.radial_positions[-1]
            F = self._prandtl_tip_loss(r[i], R, n_blades, a[i], wind_speed_eff, omega)
            
            # Solve induction factors iteratively
            a[i], a_prime[i], alpha[i], phi[i], cl[i], cd[i] = self._solve_section(
                r[i], chord[i], twist[i], polar, wind_speed_eff, omega, F
            )
            
            # Compute loads
            V_rel = wind_speed_eff * (1 - a[i]) / np.sin(np.radians(phi[i]))
            L = 0.5 * self.rho * V_rel**2 * chord[i] * cl[i]
            D = 0.5 * self.rho * V_rel**2 * chord[i] * cd[i]
            
            f_normal[i] = L * np.cos(np.radians(phi[i])) + D * np.sin(np.radians(phi[i]))
            f_tangent[i] = L * np.sin(np.radians(phi[i])) - D * np.cos(np.radians(phi[i]))
        
        # Integrate to get global quantities
        dr = np.diff(r)
        dr = np.append(dr, dr[-1])  # Extend for trapz
        
        # Power and thrust per blade
        dQ = f_tangent * r * dr
        torque_per_blade = np.trapz(f_tangent * r, r) / 1000  # [kNm]
        thrust_per_blade = np.trapz(f_normal, r) / 1000      # [kN]
        
        # Total (all blades)
        torque = torque_per_blade * n_blades
        thrust = thrust_per_blade * n_blades
        power = torque * omega  # [kW]
        
        # Coefficients
        A = np.pi * R**2
        cp = power / (0.5 * self.rho * A * wind_speed**3)
        ct = thrust * 1000 / (0.5 * self.rho * A * wind_speed**2)
        
        # Root bending moments
        flapwise_moment = np.trapz(f_normal * (r - r[0]), r) / 1000  # [kNm]
        edgewise_moment = np.trapz(f_tangent * (r - r[0]), r) / 1000 # [kNm]
        
        return BEMResult(
            power=power,
            thrust=thrust,
            torque=torque,
            cp=cp,
            ct=ct,
            axial_induction=a,
            tangential_induction=a_prime,
            angle_of_attack=alpha,
            inflow_angle=phi,
            cl=cl,
            cd=cd,
            normal_force=f_normal,
            tangent_force=f_tangent,
            flapwise_moment=flapwise_moment,
            edgewise_moment=edgewise_moment
        )
    
    def _solve_section(
        self,
        r: float,
        chord: float,
        twist: float,
        polar: AirfoilPolar,
        wind_speed: float,
        omega: float,
        F: float
    ) -> Tuple[float, float, float, float, float, float]:
        """
        Solve BEM equations for a single blade section.
        
        Returns:
            (a, a_prime, alpha, phi, cl, cd)
        """
        # Initial guess
        a = 0.2
        a_prime = 0.01
        
        # Relaxation factor for stability
        relax = 0.5
        
        for iteration in range(self.max_iterations):
            # Inflow angle
            if omega * r * (1 + a_prime) < 1e-6:
                phi = 90.0
            else:
                phi = np.degrees(np.arctan(
                    wind_speed * (1 - a) / (omega * r * (1 + a_prime))
                ))
            
            # Angle of attack
            alpha = phi - twist
            
            # Aerodynamic coefficients
            cl, cd = polar.interpolate_coefficients(alpha)
            
            # Solidity
            sigma = chord / (2 * np.pi * r)
            
            # New induction factors (momentum theory)
            cn = cl * np.cos(np.radians(phi)) + cd * np.sin(np.radians(phi))
            ct_local = cl * np.sin(np.radians(phi)) - cd * np.cos(np.radians(phi))
            
            # Glauert correction for high induction
            if cn > 0:
                a_new = 1 / (4 * F * np.sin(np.radians(phi))**2 / (sigma * cn) + 1)
            else:
                a_new = 0
            
            # Empirical correction for a > 0.4
            if a_new > 0.4:
                a_new = 0.5 * (1 + np.sqrt(1 - ct_local))
            
            if ct_local > 0:
                a_prime_new = 1 / (4 * F * np.sin(np.radians(phi)) * np.cos(np.radians(phi)) / (sigma * ct_local) - 1)
            else:
                a_prime_new = 0
            
            # Relaxation
            a_new = relax * a_new + (1 - relax) * a
            a_prime_new = relax * a_prime_new + (1 - relax) * a_prime
            
            # Check convergence
            if abs(a_new - a) < self.tolerance and abs(a_prime_new - a_prime) < self.tolerance:
                return a_new, a_prime_new, alpha, phi, cl, cd
            
            a = a_new
            a_prime = a_prime_new
        
        # If not converged, return last values with warning
        warnings.warn(f"BEM not converged at r={r:.2f}m after {self.max_iterations} iterations")
        return a, a_prime, alpha, phi, cl, cd
    
    def _prandtl_tip_loss(
        self,
        r: float,
        R: float,
        n_blades: int,
        a: float,
        wind_speed: float,
        omega: float
    ) -> float:
        """
        Prandtl tip loss factor.
        
        Args:
            r: Local radius
            R: Tip radius
            n_blades: Number of blades
            a: Axial induction factor
            wind_speed: Wind speed
            omega: Rotor speed [rad/s]
            
        Returns:
            Tip loss factor F (0-1)
        """
        if r >= R:
            return 1.0
        
        if abs(omega * r) < 1e-6:
            return 1.0
        
        f_tip = n_blades / 2 * (R - r) / r * np.sqrt(1 + (omega * r / wind_speed)**2 / (1 - a)**2)
        F = 2 / np.pi * np.arccos(np.exp(-f_tip))
        
        # Clamp to avoid numerical issues
        F = np.clip(F, 0.01, 1.0)
        
        return F
    
    def compute_power_curve(
        self,
        blade_geometry,
        wind_speeds: np.ndarray,
        rated_power: float = 5000.0,  # [kW]
        rated_wind_speed: float = 11.4,  # [m/s]
        cut_in: float = 3.0,
        cut_out: float = 25.0
    ) -> Dict[str, np.ndarray]:
        """
        Compute full power curve with control strategy.
        
        Args:
            blade_geometry: BladeGeometry object
            wind_speeds: Array of wind speeds [m/s]
            rated_power: Rated power [kW]
            rated_wind_speed: Rated wind speed [m/s]
            cut_in: Cut-in wind speed [m/s]
            cut_out: Cut-out wind speed [m/s]
            
        Returns:
            Dictionary with power, thrust, cp, etc. vs. wind speed
        """
        n_speeds = len(wind_speeds)
        power = np.zeros(n_speeds)
        thrust = np.zeros(n_speeds)
        torque = np.zeros(n_speeds)
        cp = np.zeros(n_speeds)
        rotor_speed = np.zeros(n_speeds)
        pitch = np.zeros(n_speeds)
        
        # Optimal TSR for region 2
        tsr_opt = 7.0
        R = blade_geometry.radial_positions[-1]
        
        for i, v in enumerate(wind_speeds):
            if v < cut_in or v > cut_out:
                continue
            
            if v < rated_wind_speed:
                # Region 2: Variable speed, fixed pitch
                omega = tsr_opt * v / R
                rpm = omega * 60 / (2 * np.pi)
                pitch_angle = 0.0
            else:
                # Region 3: Fixed speed, variable pitch
                omega = tsr_opt * rated_wind_speed / R
                rpm = omega * 60 / (2 * np.pi)
                
                # Simple pitch controller (pitch to limit power)
                pitch_angle = 5.0 * (v - rated_wind_speed)  # deg
                pitch_angle = np.clip(pitch_angle, 0, 30)
            
            # Solve BEM
            result = self.solve(blade_geometry, v, rpm, pitch_angle)
            
            power[i] = min(result.power, rated_power)
            thrust[i] = result.thrust
            torque[i] = result.torque
            cp[i] = result.cp
            rotor_speed[i] = rpm
            pitch[i] = pitch_angle
        
        return {
            'wind_speed': wind_speeds,
            'power': power,
            'thrust': thrust,
            'torque': torque,
            'cp': cp,
            'rotor_speed': rotor_speed,
            'pitch': pitch
        }


if __name__ == "__main__":
    # Test BEM solver
    from parameterization import create_baseline_blade
    
    print("Testing BEM solver...")
    blade = create_baseline_blade()
    solver = BEMSolver()
    
    # Single operating point
    result = solver.solve(blade, wind_speed=8.0, rotor_speed=12.1)
    
    print(f"\nResults at 8 m/s:")
    print(f"  Power: {result.power:.1f} kW")
    print(f"  Thrust: {result.thrust:.1f} kN")
    print(f"  Cp: {result.cp:.3f}")
    print(f"  Ct: {result.ct:.3f}")
    print(f"  Flap moment: {result.flapwise_moment:.1f} kNm")
    
    # Power curve
    wind_speeds = np.linspace(3, 25, 23)
    power_curve = solver.compute_power_curve(blade, wind_speeds)
    
    print(f"\nPower curve computed for {len(wind_speeds)} wind speeds")
    print(f"  Max power: {np.max(power_curve['power']):.1f} kW")
    print(f"  Max Cp: {np.max(power_curve['cp']):.3f}")
