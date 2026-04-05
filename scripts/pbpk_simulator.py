"""
pbpk_simulator.py
Physiologically-Based Pharmacokinetic (PBPK) Simulator

Generates physiologically realistic concentration-time curves from 
molecular descriptors and patient physiology using a 7-compartment ODE model.
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PhysiologyModel:
    """
    Human physiological parameters for PBPK modeling.
    Based on standard 70kg reference human, scaled allometrically.
    
    References:
    - Brown RP, et al. Physiological parameter values for physiologically 
      based pharmacokinetic models. Toxicol Ind Health. 1997.
    - Davies B, Morris T. Physiological parameters in laboratory animals 
      and humans. Pharm Res. 1993.
    """
    
    # Reference values for 70kg adult
    REFERENCE_WEIGHT_KG = 70.0
    
    # Organ volumes as fraction of body weight (L/kg)
    ORGAN_VOLUME_FRACTIONS = {
        'liver': 0.0257,    # ~1.8 L
        'kidney': 0.0043,   # ~0.3 L
        'brain': 0.0200,    # ~1.4 L
        'fat': 0.2143,      # ~15.0 L
        'muscle': 0.4000,   # ~28.0 L
        'blood': 0.0714,    # ~5.0 L (total blood volume)
    }
    
    # Blood flows as fraction of cardiac output (L/h for 70kg, CO ~330 L/h)
    CARDIAC_OUTPUT_L_H = 330.0
    
    BLOOD_FLOW_FRACTIONS = {
        'liver': 0.273,     # ~90 L/h (hepatic, includes portal vein)
        'kidney': 0.218,    # ~72 L/h (renal)
        'brain': 0.136,     # ~45 L/h
        'fat': 0.045,       # ~15 L/h
        'muscle': 0.170,    # ~56 L/h (resting)
    }
    
    def __init__(self, patient: Dict):
        """
        Initialize physiology from patient parameters.
        
        Args:
            patient: Dict with keys: age, sex, weight_kg, height_cm,
                     hepatic_impairment, renal_impairment, cardiac_output_l_min
        """
        self.patient = patient
        self.weight = patient['weight_kg']
        self.height = patient['height_cm']
        self.age = patient['age']
        self.sex = patient['sex']
        
        # Scale physiology
        self._compute_organ_volumes()
        self._compute_blood_flows()
        self._apply_impairments()
    
    def _compute_organ_volumes(self):
        """Scale organ volumes to patient size."""
        weight_ratio = self.weight / self.REFERENCE_WEIGHT_KG
        
        # Organ volumes scale linearly with body weight
        self.organ_volumes = {}
        for organ, fraction in self.ORGAN_VOLUME_FRACTIONS.items():
            self.organ_volumes[organ] = fraction * self.weight
        
        # Gut volume (for oral absorption)
        self.organ_volumes['gut'] = 0.25 * weight_ratio  # ~0.25 L
        
    def _compute_blood_flows(self):
        """Scale blood flows allometrically."""
        # Cardiac output scales with weight^0.75
        co_scaling = (self.weight / self.REFERENCE_WEIGHT_KG) ** 0.75
        cardiac_output = self.CARDIAC_OUTPUT_L_H * co_scaling
        
        # Apply patient-specific cardiac output if provided
        if 'cardiac_output_l_min' in self.patient:
            cardiac_output = self.patient['cardiac_output_l_min'] * 60.0
        
        self.cardiac_output = cardiac_output
        
        self.blood_flows = {}
        for organ, fraction in self.BLOOD_FLOW_FRACTIONS.items():
            self.blood_flows[organ] = fraction * cardiac_output
        
        # Portal vein flow to liver (from gut)
        self.blood_flows['portal'] = 0.15 * cardiac_output  # ~15% of CO
        # Total hepatic flow = hepatic artery + portal vein
        self.blood_flows['liver'] = self.blood_flows['liver'] + self.blood_flows['portal']
    
    def _apply_impairments(self):
        """Apply organ impairment effects."""
        # Hepatic impairment reduces liver blood flow and metabolic capacity
        hepatic_imp = self.patient.get('hepatic_impairment', 0)
        if hepatic_imp > 0:
            # Blood flow reduction proportional to impairment
            flow_reduction = 1 - 0.4 * hepatic_imp
            self.blood_flows['liver'] *= flow_reduction
        
        # Renal impairment reduces kidney blood flow and GFR
        renal_imp = self.patient.get('renal_impairment', 1.0)
        self.blood_flows['kidney'] *= renal_imp
        
        # Albumin affects protein binding
        self.albumin = self.patient.get('albumin_gdl', 4.0)
        self.aag = self.patient.get('alpha1_acid_glycoprotein', 1.0)
        
        # Gastric pH affects oral absorption
        self.gastric_ph = self.patient.get('gastric_ph', 1.5)
        
        # Intestinal transit time
        self.intestinal_transit_time = self.patient.get('intestinal_transit_time_h', 3.0)
        
        # Hematocrit (affects blood:plasma ratio)
        self.hematocrit = self.patient.get('hematocrit', 0.45)
    
    def get_compartment_volume(self, compartment: str) -> float:
        """Get volume of a compartment in liters."""
        mapping = {
            'gut': 'gut',
            'liver': 'liver',
            'blood': 'blood',
            'kidney': 'kidney',
            'brain': 'brain',
            'fat': 'fat',
            'muscle': 'muscle',
        }
        return self.organ_volumes[mapping[compartment]]
    
    def get_blood_flow(self, compartment: str) -> float:
        """Get blood flow to a compartment in L/h."""
        return self.blood_flows.get(compartment, 0)
    
    def get_gfr(self) -> float:
        """Get glomerular filtration rate in L/h."""
        # Normal GFR ~125 mL/min = 7.5 L/h
        base_gfr = 7.5
        renal_imp = self.patient.get('renal_impairment', 1.0)
        # Allometric scaling
        gfr = base_gfr * (self.weight / self.REFERENCE_WEIGHT_KG) ** 0.75 * renal_imp
        return gfr


class QSPRPredictor:
    """
    Quantitative Structure-Property Relationship predictor.
    
    Estimates mechanistic PK parameters from molecular descriptors
    using established QSPR relationships with realistic noise.
    
    References:
    - Obach RS, et al. J Pharmacol Exp Ther. 1997;283(1):46-58.
    - Rodgers T, Rowland M. J Pharm Sci. 2007;96(12):3329-3349.
    - Lipinski CA, et al. Adv Drug Deliv Rev. 2001;46(1-3):3-26.
    """
    
    def __init__(self, rng: np.random.Generator):
        self.rng = rng
    
    def predict_absorption_rate(self, mol: Dict, physiology: PhysiologyModel) -> float:
        """
        Predict absorption rate constant (ka) in h^-1.
        
        Depends on: TPSA (permeability), logP (membrane partitioning),
                    HBD count, gastric pH, intestinal transit time.
        
        Reference: 
        - Clark DE. Eur J Med Chem. 1999;34(12):981-989.
        """
        tpsa = mol['tpsa']
        logp = mol['logp']
        hbd = mol['hbd']
        
        # Base relationship: log(ka) decreases with TPSA, increases with logP
        log_ka = (
            0.8                         # baseline
            - 0.008 * tpsa              # polar surface area reduces permeability
            + 0.15 * logp               # lipophilicity helps (up to a point)
            - 0.1 * hbd                 # H-bond donors reduce permeability
        )
        
        # Gastric pH effect on ionization (affects absorption)
        pka = mol.get('pka_basic', 7.4)
        if pka > 5:  # Basic drugs are ionized in stomach, absorbed in intestine
            ionized_fraction = 1 / (1 + 10 ** (physiology.gastric_ph - pka))
            log_ka -= 0.3 * ionized_fraction  # Ionized drugs absorb slower
        
        # Intestinal transit time effect
        transit_factor = physiology.intestinal_transit_time / 3.0  # normalize to 3h
        log_ka += 0.1 * (transit_factor - 1)  # longer transit = more absorption time
        
        # Add noise (20% CV on log scale)
        log_ka += self.rng.normal(0, 0.2)
        
        ka = np.exp(log_ka)
        return max(ka, 0.01)  # Minimum ka
    
    def predict_intrinsic_clearance(self, mol: Dict, patient: Dict) -> float:
        """
        Predict intrinsic hepatic clearance (CLint) in L/h.
        
        Depends on: logP (CYP affinity), MW (steric hindrance),
                    CYP genotype status.
        
        Reference:
        - Obach RS. Drug Metab Dispos. 1999;27(12):1350-1359.
        """
        logp = mol['logp']
        mw = mol['mw']
        
        # Base relationship
        log_cl_int = (
            1.5                         # baseline
            + 0.3 * logp                # lipophilic drugs cleared faster
            - 0.002 * mw                # larger drugs cleared slower
        )
        
        # CYP3A4 status (major metabolic enzyme)
        cyp3a4 = patient.get('cyp3a4_status', 'NM')
        cyp_multipliers = {
            'PM': 0.1,   # Poor metabolizer
            'IM': 0.5,   # Intermediate
            'NM': 1.0,   # Normal
            'UM': 2.0,   # Ultra-rapid
        }
        log_cl_int += np.log10(cyp_multipliers.get(cyp3a4, 1.0))
        
        # CYP2D6 status (secondary pathway for basic drugs)
        if mol.get('pka_basic', 0) > 7:  # Basic drugs are CYP2D6 substrates
            cyp2d6 = patient.get('cyp2d6_status', 'NM')
            cyp2d6_multipliers = {
                'PM': 0.3,
                'IM': 0.6,
                'NM': 1.0,
                'UM': 1.8,
            }
            log_cl_int += 0.3 * np.log10(cyp2d6_multipliers.get(cyp2d6, 1.0))
        
        # Add noise (40% CV)
        log_cl_int += self.rng.normal(0, 0.4)
        
        cl_int = 10 ** log_cl_int
        return max(cl_int, 0.01)
    
    def predict_fraction_unbound(self, mol: Dict, physiology: PhysiologyModel) -> float:
        """
        Predict fraction unbound in plasma (fu).
        
        Depends on: logP (albumin binding), pKa (AAG binding),
                    albumin and AAG levels.
        
        Reference:
        - Berthier V, et al. J Pharm Sci. 2013;102(6):1881-1893.
        """
        logp = mol['logp']
        
        # Base fu: lipophilic drugs bind more to albumin
        log_fu = -0.5 * logp + 0.5  # baseline relationship
        
        # Albumin effect (binds acidic/neutral drugs)
        albumin_factor = physiology.albumin / 4.0  # normalize to 4.0 g/dL
        log_fu -= 0.2 * (albumin_factor - 1)
        
        # AAG effect (binds basic drugs)
        if mol.get('pka_basic', 0) > 7:
            aag_factor = physiology.aag / 1.0  # normalize to 1.0 g/L
            log_fu -= 0.3 * (aag_factor - 1)
        
        # Add noise
        log_fu += self.rng.normal(0, 0.15)
        
        fu = 10 ** log_fu
        return np.clip(fu, 0.01, 1.0)
    
    def predict_tissue_partitioning(self, mol: Dict) -> Dict[str, float]:
        """
        Predict tissue:blood partition coefficients (Kp).
        
        Depends on: logP (lipophilicity), pKa (ion trapping),
                    tissue composition.
        
        Reference:
        - Rodgers T, Rowland M. J Pharm Sci. 2007;96(12):3329-3349.
        """
        logp = mol['logp']
        pka = mol.get('pka_basic', 7.4)
        fu = mol.get('fraction_unbound', 0.5)
        
        # Simplified Rodgers-Rowland approach
        # Kp = (tissue binding) / (blood binding)
        
        # Liver: moderate lipophilicity preference
        kp_liver = 1.5 + 0.4 * logp + 0.2 * fu
        
        # Kidney: similar to liver but slightly higher
        kp_kidney = 2.0 + 0.3 * logp + 0.3 * fu
        
        # Brain: depends on TPSA and lipophilicity (BBB)
        tpsa = mol['tpsa']
        bbb_factor = 1.0 / (1 + np.exp(0.05 * tpsa - 2.0))
        kp_brain = bbb_factor * (0.5 + 1.5 * fu)
        
        # Fat: highly lipophilic drugs partition here
        kp_fat = 3.0 + 1.5 * logp
        
        # Muscle: moderate partitioning
        kp_muscle = 1.0 + 0.3 * logp + 0.5 * fu
        
        # Add noise (15% CV)
        noise_factors = self.rng.lognormal(0, 0.15, size=5)
        
        return {
            'liver': max(kp_liver * noise_factors[0], 0.1),
            'kidney': max(kp_kidney * noise_factors[1], 0.1),
            'brain': max(kp_brain * noise_factors[2], 0.01),
            'fat': max(kp_fat * noise_factors[3], 0.1),
            'muscle': max(kp_muscle * noise_factors[4], 0.1),
        }
    
    def predict_bioavailability(self, mol: Dict, ka: float, 
                                cl_hepatic: float, physiology: PhysiologyModel) -> float:
        """
        Predict oral bioavailability (F).
        
        F = Fa × Fg × Fh
        where Fa = fraction absorbed, Fg = gut availability, Fh = hepatic first-pass
        
        Reference:
        - Zhao YH, et al. J Pharm Sci. 2001;90(3):277-287.
        """
        tpsa = mol['tpsa']
        logp = mol['logp']
        
        # Fraction absorbed (depends on TPSA, logP)
        fa = 1.0 / (1 + np.exp(0.03 * tpsa - 3.5 + 0.2 * logp))
        fa = max(fa, 0.1)  # Minimum 10%
        
        # Gut availability (assume minimal gut metabolism)
        fg = 0.9 + self.rng.normal(0, 0.05)
        fg = np.clip(fg, 0.7, 1.0)
        
        # Hepatic first-pass (well-stirred model)
        q_h = physiology.blood_flows.get('liver', 90.0)
        fu = mol.get('fraction_unbound', 0.5)
        
        if q_h > 0:
            extraction_ratio = (fu * cl_hepatic) / (q_h + fu * cl_hepatic)
            fh = 1 - extraction_ratio
        else:
            fh = 1.0
        
        fh = np.clip(fh, 0, 1)
        
        f_oral = fa * fg * fh
        return np.clip(f_oral, 0.01, 1.0)


class PBPKModel:
    """
    7-Compartment Physiologically-Based Pharmacokinetic Model.
    
    Compartments:
    - Gut (oral absorption site)
    - Liver (primary metabolism)
    - Blood (central compartment)
    - Kidney (primary excretion)
    - Brain (BBB penetration test)
    - Fat (lipophilic reservoir)
    - Muscle (rest of body)
    
    State vector: [A_gut, A_liver, A_blood, A_kidney, A_brain, A_fat, A_muscle]
    """
    
    COMPARTMENT_NAMES = ['gut', 'liver', 'blood', 'kidney', 'brain', 'fat', 'muscle']
    COMPARMENT_IDX = {name: i for i, name in enumerate(COMPARTMENT_NAMES)}
    
    def __init__(self, physiology: PhysiologyModel):
        self.physiology = physiology
    
    def _ode_rhs(self, t: float, state: np.ndarray, 
                 pk_params: Dict, route: str) -> np.ndarray:
        """
        Right-hand side of the PBPK ODE system.
        
        Args:
            t: Time (h)
            state: State vector [A_gut, A_liver, A_blood, A_kidney, A_brain, A_fat, A_muscle]
            pk_params: Dict with ka, cl_hepatic, cl_renal, kp values, fu
            route: 'oral' or 'iv'
        """
        A = state.copy()
        
        ka = pk_params['ka']
        cl_hepatic = pk_params['cl_hepatic']
        cl_renal = pk_params['cl_renal']
        fu = pk_params['fu']
        kp = pk_params['kp']
        
        physiology = self.physiology
        
        # Amounts
        A_gut = A[self.COMPARMENT_IDX['gut']]
        A_liver = A[self.COMPARMENT_IDX['liver']]
        A_blood = A[self.COMPARMENT_IDX['blood']]
        A_kidney = A[self.COMPARMENT_IDX['kidney']]
        A_brain = A[self.COMPARMENT_IDX['brain']]
        A_fat = A[self.COMPARMENT_IDX['fat']]
        A_muscle = A[self.COMPARMENT_IDX['muscle']]
        
        # Concentrations (amount / volume)
        V = physiology.organ_volumes
        C_gut = max(A_gut / V['gut'], 0) if V['gut'] > 0 else 0
        C_liver = max(A_liver / V['liver'], 0) if V['liver'] > 0 else 0
        C_blood = max(A_blood / V['blood'], 0) if V['blood'] > 0 else 0
        C_kidney = max(A_kidney / V['kidney'], 0) if V['kidney'] > 0 else 0
        C_brain = max(A_brain / V['brain'], 0) if V['brain'] > 0 else 0
        C_fat = max(A_fat / V['fat'], 0) if V['fat'] > 0 else 0
        C_muscle = max(A_muscle / V['muscle'], 0) if V['muscle'] > 0 else 0
        
        # Blood flows
        Q = physiology.blood_flows
        
        # Derivatives
        dA = np.zeros(7)
        
        # ---- Gut (oral absorption) ----
        if route == 'oral':
            dA[self.COMPARMENT_IDX['gut']] = -ka * A_gut
        
        # ---- Liver (metabolism) ----
        # Inflow from blood (hepatic artery) + portal vein from gut
        inflow_liver = Q.get('liver', 90.0) * C_blood
        
        # Outflow from liver (venous return)
        outflow_liver = Q.get('liver', 90.0) * (C_liver / kp['liver'])
        
        # Metabolism (well-stirred model)
        metabolism = cl_hepatic * C_liver
        
        dA[self.COMPARMENT_IDX['liver']] = inflow_liver - outflow_liver - metabolism
        
        # ---- Kidney (excretion) ----
        inflow_kidney = Q.get('kidney', 72.0) * C_blood
        outflow_kidney = Q.get('kidney', 72.0) * (C_kidney / kp['kidney'])
        excretion = cl_renal * C_blood  # Renal clearance from blood
        
        dA[self.COMPARMENT_IDX['kidney']] = inflow_kidney - outflow_kidney - excretion
        
        # ---- Brain (BBB) ----
        inflow_brain = Q.get('brain', 45.0) * C_blood
        outflow_brain = Q.get('brain', 45.0) * (C_brain / kp['brain'])
        
        dA[self.COMPARMENT_IDX['brain']] = inflow_brain - outflow_brain
        
        # ---- Fat ----
        inflow_fat = Q.get('fat', 15.0) * C_blood
        outflow_fat = Q.get('fat', 15.0) * (C_fat / kp['fat'])
        
        dA[self.COMPARMENT_IDX['fat']] = inflow_fat - outflow_fat
        
        # ---- Muscle ----
        inflow_muscle = Q.get('muscle', 56.0) * C_blood
        outflow_muscle = Q.get('muscle', 56.0) * (C_muscle / kp['muscle'])
        
        dA[self.COMPARMENT_IDX['muscle']] = inflow_muscle - outflow_muscle
        
        # ---- Blood (mass balance) ----
        # Everything flows through blood
        dA[self.COMPARMENT_IDX['blood']] = -(
            dA[self.COMPARMENT_IDX['liver']] +
            dA[self.COMPARMENT_IDX['kidney']] +
            dA[self.COMPARMENT_IDX['brain']] +
            dA[self.COMPARMENT_IDX['fat']] +
            dA[self.COMPARMENT_IDX['muscle']]
        )
        
        # Add oral absorption input to blood
        if route == 'oral':
            dA[self.COMPARMENT_IDX['blood']] += ka * A_gut
        
        # Ensure no negative derivatives that would cause negative amounts
        # (numerical stability)
        for i in range(7):
            if A[i] <= 0 and dA[i] < 0:
                dA[i] = 0
        
        return dA
    
    def simulate(self, pk_params: Dict, dose_mg: float, route: str,
                 time_points: Optional[List[float]] = None) -> Tuple[np.ndarray, Dict]:
        """
        Run PBPK simulation.
        
        Args:
            pk_params: Dict with ka, cl_hepatic, cl_renal, fu, kp
            dose_mg: Administered dose in mg
            route: 'oral' or 'iv'
            time_points: Array of time points (default: standard PK schedule)
        
        Returns:
            solution: scipy ODE solution object
            pk_params: Updated pk_params with computed values
        """
        if time_points is None:
            time_points = [0, 0.25, 0.5, 1, 2, 4, 6, 8, 12, 16, 24, 36, 48, 72]
        
        # Initial conditions
        state0 = np.zeros(7)
        if route == 'oral':
            state0[self.COMPARMENT_IDX['gut']] = dose_mg
        else:  # IV bolus
            state0[self.COMPARMENT_IDX['blood']] = dose_mg
        
        # Solve ODE
        sol = solve_ivp(
            fun=lambda t, y: self._ode_rhs(t, y, pk_params, route),
            t_span=[0, 72],
            y0=state0,
            t_eval=time_points,
            method='LSODA',
            rtol=1e-8,
            atol=1e-10,
            max_step=0.1,  # Ensure accuracy
        )
        
        if not sol.success:
            logger.warning(f"ODE solver warning: {sol.message}")
        
        return sol, pk_params


class PKParameterExtractor:
    """
    Extract standard PK parameters from concentration-time curves
    via Non-Compartmental Analysis (NCA).
    """
    
    MEASURED_COMPARTMENTS = ['blood', 'liver', 'kidney', 'brain', 'fat']
    
    @staticmethod
    def compute_auc(time: np.ndarray, concentration: np.ndarray) -> float:
        """Compute AUC using linear trapezoidal rule."""
        from scipy.integrate import trapezoid
        return trapezoid(concentration, time)
    
    @staticmethod
    def compute_terminal_half_life(time: np.ndarray, 
                                     concentration: np.ndarray) -> float:
        """
        Compute terminal elimination half-life from log-linear regression
        of the terminal phase (last 3-5 points where concentration is declining).
        """
        # Find the terminal phase (last points where conc is declining)
        n_points = min(5, len(time))
        
        # Use last n_points where concentration is non-zero
        valid_mask = concentration > 0.01
        valid_time = time[valid_mask]
        valid_conc = concentration[valid_mask]
        
        if len(valid_time) < 3:
            return 24.0  # Default if insufficient data
        
        # Take last points
        t_term = valid_time[-n_points:]
        c_term = valid_conc[-n_points:]
        
        # Log-linear regression
        log_c = np.log(c_term)
        
        # Check if declining
        if len(t_term) < 2 or np.std(log_c) < 0.01:
            return 24.0
        
        coeffs = np.polyfit(t_term, log_c, 1)
        slope = coeffs[0]
        
        if slope >= 0:  # Not declining
            return 24.0
        
        lambda_z = -slope
        half_life = np.log(2) / lambda_z
        
        return max(half_life, 0.5)
    
    @classmethod
    def extract_parameters(cls, curves_df, pk_params: Dict, 
                           dose_mg: float, route: str) -> Dict:
        """
        Extract all standard PK parameters from simulation results.
        
        Args:
            curves_df: DataFrame with compartment, time_h, concentration_ng_ml
            pk_params: Mechanistic PK parameters
            dose_mg: Administered dose
            route: 'oral' or 'iv'
        
        Returns:
            Dict with Cmax, Tmax, AUC, half-life, clearance, etc.
        """
        result = {}
        
        # Blood compartment
        blood = curves_df[curves_df['compartment'] == 'blood']
        blood_conc = blood['concentration_ng_ml'].values
        time = blood['time_h'].values
        
        # Cmax and Tmax
        idx_max = np.argmax(blood_conc)
        result['cmax_blood'] = blood_conc[idx_max]
        result['tmax_blood'] = time[idx_max]
        
        # AUC (0-72h)
        result['auc_0_72_blood'] = cls.compute_auc(time, blood_conc)
        
        # Terminal half-life
        result['half_life'] = cls.compute_terminal_half_life(time, blood_conc)
        
        # AUC to infinity
        lambda_z = np.log(2) / result['half_life']
        c_last = blood_conc[-1]
        result['auc_inf_blood'] = result['auc_0_72_blood'] + c_last / lambda_z
        
        # Clearance
        if route == 'iv':
            result['clearance_l_h'] = dose_mg / result['auc_inf_blood'] * 1e6
        else:
            f = pk_params.get('f_oral', 0.5)
            result['clearance_l_h'] = f * dose_mg / result['auc_inf_blood'] * 1e6
        
        result['clearance_l_h'] = max(result['clearance_l_h'], 0.01)
        
        # Volume of distribution
        result['vss_l'] = result['clearance_l_h'] / lambda_z
        result['vss_l'] = max(result['vss_l'], 1.0)
        
        # Bioavailability
        result['bioavailability'] = pk_params.get('f_oral', 1.0) if route == 'oral' else 1.0
        
        # Brain penetration (brain AUC / blood AUC)
        brain = curves_df[curves_df['compartment'] == 'brain']
        brain_conc = brain['concentration_ng_ml'].values
        result['brain_auc_ratio'] = cls.compute_auc(time, brain_conc) / result['auc_0_72_blood']
        
        # Fraction excreted vs metabolized
        total_clearance = result['clearance_l_h']
        renal_clearance = pk_params.get('cl_renal', 0) * 1e6  # Convert to L/h
        result['fraction_excreted_urine'] = min(renal_clearance / total_clearance, 1.0) if total_clearance > 0 else 0
        result['fraction_metabolized'] = 1.0 - result['fraction_excreted_urine']
        
        return result


def add_measurement_noise(concentration: np.ndarray, 
                           analytical_cv: float = 0.1,
                           biological_cv: float = 0.2,
                           lod: float = 0.01,
                           rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Add realistic analytical and biological noise to concentrations.
    
    Args:
        concentration: True concentration array
        analytical_cv: Coefficient of variation for analytical noise (~10%)
        biological_cv: Coefficient of variation for inter-individual variability (~20%)
        lod: Limit of detection (ng/mL)
        rng: Random number generator
    
    Returns:
        Noisy concentration array
    """
    if rng is None:
        rng = np.random.default_rng()
    
    total_cv = np.sqrt(analytical_cv ** 2 + biological_cv ** 2)
    
    # Log-normal noise
    sigma = np.sqrt(np.log(1 + total_cv ** 2))
    noise = rng.lognormal(0, sigma, size=len(concentration))
    
    noisy_conc = concentration * noise
    
    # Apply limit of detection
    noisy_conc = np.maximum(noisy_conc, lod)
    
    return noisy_conc
